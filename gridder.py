"""
Utilities for gridding station time-series data into temporal raster cubes.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from .ads_client import ADSFetchResult
from .cdo_client import StationRecord

gdal.UseExceptions()

ProgressCallback = Callable[[int, str], None]


@dataclass
class GridParameters:
    """User-configurable settings for raster cube generation."""

    variable: str
    variable_label: str
    target_crs: QgsCoordinateReferenceSystem
    resolution: float
    buffer_cells: int
    extent_rect: QgsRectangle
    clip_layer: Optional[QgsVectorLayer]
    clip_enabled: bool
    method: str
    auto_radius: bool
    radius: float
    auto_radius_multiplier: float
    idw_power: float
    idw_min_points: int
    skip_sparse_slices: bool
    derive_average_temperature: bool
    log_transform: bool
    burn_stations: bool
    burn_radius_cells: float
    output_dir: Path
    write_vrt: bool
    write_cog: bool
    write_qa: bool


@dataclass
class GridOutputs:
    """Artifacts produced by the gridding workflow."""

    variable: str
    variable_label: str
    tif_paths: List[Path]
    vrt_path: Optional[Path]
    coverage_csv: Optional[Path]
    cog_paths: List[Path]
    warnings: List[str]
    band_labels: List[str]


class GridError(RuntimeError):
    """Raised when gridding fails."""


def build_raster_cube(
    result: ADSFetchResult,
    params: GridParameters,
    progress_cb: Optional[ProgressCallback] = None,
) -> GridOutputs:
    """Convert station data to a stack of rasters (one per time slice)."""

    if progress_cb:
        progress_cb(0, "Preparing gridding inputs...")

    variable = params.variable.upper()
    if not result.rows:
        raise GridError("No station data is available to grid.")

    target_crs = params.target_crs
    if not target_crs.isValid():
        raise GridError("Target CRS is invalid.")

    resolution = max(params.resolution, 0.0001)
    method_uses_radius = params.method in {"idw", "nearest", "average", "gwr"}
    search_radius = params.radius if not params.auto_radius else (params.auto_radius_multiplier * resolution)
    if method_uses_radius:
        search_radius = max(search_radius, resolution)
    else:
        search_radius = 0.0

    radius_buffer_cells = math.ceil(search_radius / resolution) if method_uses_radius else 0
    buffer_cells = max(params.buffer_cells, radius_buffer_cells)

    base_extent = QgsRectangle(params.extent_rect)
    if not base_extent.isFinite():
        raise GridError("Grid extent is invalid or empty.")

    buffer_distance = buffer_cells * resolution
    buffered_extent = _expanded_rect(base_extent, buffer_distance)

    transform_context = QgsProject.instance().transformContext()
    clip_layer_path: Optional[Path] = None

    with tempfile.TemporaryDirectory(prefix="ncei_grid_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        if params.clip_enabled and params.clip_layer:
            clip_layer_path = _export_layer_to_gpkg(
                params.clip_layer, target_crs, tmp_dir, transform_context
            )

        outputs_dir = params.output_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        variable_dir = outputs_dir / variable.lower()
        tif_dir = variable_dir / "tif"
        tif_dir.mkdir(parents=True, exist_ok=True)
        cog_dir = variable_dir / "cog"
        if params.write_cog:
            cog_dir.mkdir(parents=True, exist_ok=True)

        station_map = _build_station_lookup(result.station_records)
        values_by_date = _group_rows_by_date(result.rows, variable, params)
        date_keys = sorted(values_by_date.keys())
        if not date_keys:
            raise GridError(f"No usable data found for {variable}.")

        total = len(date_keys)
        nodata = -9999.0
        tif_paths: List[Path] = []
        cog_paths: List[Path] = []
        warnings: List[str] = []
        coverage_rows: List[Tuple[str, int, Optional[float], Optional[float], Optional[float], Optional[float]]] = []
        band_labels: List[str] = []

        grid_width, grid_height = _grid_dimensions(buffered_extent, resolution)
        if grid_width <= 0 or grid_height <= 0:
            raise GridError("Computed grid size is zero; adjust resolution or extent.")

        for index, date_key in enumerate(date_keys, start=1):
            raw_entries: List[Tuple[float, float, float]] = []

            for station_id, value in values_by_date[date_key]:
                record = station_map.get(station_id)
                coords = None
                if record:
                    coords = (record.longitude, record.latitude)
                else:
                    coords = _extract_coordinates_from_rows(result.rows, station_id)
                if coords is None:
                    continue
                raw_entries.append((coords[0], coords[1], value))

            station_count = len(raw_entries)
            stats = _compute_statistics([entry[2] for entry in raw_entries])
            coverage_rows.append(
                (
                    date_key,
                    station_count,
                    stats["min"],
                    stats["max"],
                    stats["mean"],
                    stats["std"],
                )
            )

            if progress_cb:
                pct = int(5 + (index - 1) / total * 70)
                progress_cb(pct, f"Gridding {date_key} ({station_count} stations)...")

            output_name = f"{variable.lower()}_{date_key.replace('-', '')}.tif"
            dest_path = tif_dir / output_name

            if station_count < params.idw_min_points:
                message = (
                    f"{date_key}: insufficient stations ({station_count}); "
                    + ("skipping slice." if params.skip_sparse_slices else "output set to nodata.")
                )
                warnings.append(message)
                if params.skip_sparse_slices:
                    continue
                _create_empty_raster(dest_path, buffered_extent, resolution, grid_width, grid_height, target_crs, nodata)
                _trim_to_extent(dest_path, base_extent, target_crs, nodata)
                if clip_layer_path and params.clip_enabled:
                    _clip_to_polygon(dest_path, clip_layer_path, target_crs, nodata)
                tif_paths.append(dest_path)
                band_labels.append(date_key)
                if params.write_cog:
                    cog_path = cog_dir / output_name
                    _translate_to_cog(dest_path, cog_path)
                    cog_paths.append(cog_path)
                continue

            vector_path, transformed_points = _create_station_layer(
                raw_entries,
                params.log_transform,
                tmp_dir,
                target_crs,
                transform_context,
                date_key,
            )

            if params.method == "gwr":
                _grid_slice_gwr(
                    transformed_points,
                    dest_path,
                    buffered_extent,
                    target_crs,
                    resolution,
                    grid_width,
                    grid_height,
                    nodata,
                    params,
                    search_radius,
                )
            else:
                _grid_slice(
                    vector_path,
                    dest_path,
                    buffered_extent,
                    target_crs,
                    resolution,
                    grid_width,
                    grid_height,
                    nodata,
                    params,
                    search_radius,
                )

            _trim_to_extent(dest_path, base_extent, target_crs, nodata)

            if clip_layer_path and params.clip_enabled:
                _clip_to_polygon(dest_path, clip_layer_path, target_crs, nodata)

            if params.log_transform:
                _apply_inverse_transform(dest_path, nodata)

            if params.burn_stations:
                burn_distance = max(params.burn_radius_cells, 0.0) * resolution
                _burn_station_values(dest_path, vector_path, burn_distance, tmp_dir, transform_context)

            tif_paths.append(dest_path)
            band_labels.append(date_key)

            if params.write_cog:
                cog_path = cog_dir / output_name
                _translate_to_cog(dest_path, cog_path)
                cog_paths.append(cog_path)

        if not tif_paths:
            raise GridError(
                "No rasters were generated. Reduce the minimum station requirement or verify station coverage."
            )

        vrt_path: Optional[Path] = None
        if params.write_vrt and tif_paths:
            if progress_cb:
                progress_cb(90, "Building VRT...")
            vrt_path = variable_dir / f"{variable.lower()}_cube.vrt"
            _build_vrt(tif_paths, vrt_path, band_labels)

        coverage_csv: Optional[Path] = None
        if params.write_qa and coverage_rows:
            coverage_csv = variable_dir / f"{variable.lower()}_coverage.csv"
            _write_coverage_csv(coverage_csv, coverage_rows)

        if progress_cb:
            progress_cb(100, "Gridding complete.")

        return GridOutputs(
            variable=variable,
            variable_label=params.variable_label,
            tif_paths=tif_paths,
            vrt_path=vrt_path,
            coverage_csv=coverage_csv,
            cog_paths=cog_paths,
            warnings=warnings,
            band_labels=band_labels,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _expanded_rect(rect: QgsRectangle, distance: float) -> QgsRectangle:
    expanded = QgsRectangle(rect)
    expanded.setXMinimum(expanded.xMinimum() - distance)
    expanded.setXMaximum(expanded.xMaximum() + distance)
    expanded.setYMinimum(expanded.yMinimum() - distance)
    expanded.setYMaximum(expanded.yMaximum() + distance)
    return expanded


def _grid_dimensions(rect: QgsRectangle, resolution: float) -> Tuple[int, int]:
    width = max(int(math.ceil(rect.width() / resolution)), 1)
    height = max(int(math.ceil(rect.height() / resolution)), 1)
    return width, height


def _build_station_lookup(records: Sequence[StationRecord]) -> Dict[str, StationRecord]:
    lookup: Dict[str, StationRecord] = {}
    for record in records:
        stripped = _strip_station_id(record.id)
        lookup[stripped] = record
    return lookup


def _group_rows_by_date(
    rows: Sequence[Dict[str, str]],
    variable: str,
    params: GridParameters,
) -> Dict[str, List[Tuple[str, float]]]:
    grouped: Dict[str, List[Tuple[str, float]]] = {}
    variable_upper = variable.upper()

    for row in rows:
        date_value = row.get("DATE")
        if not date_value:
            continue

        raw_entry = row.get(variable_upper)
        value: Optional[float] = None
        if raw_entry not in (None, ""):
            try:
                numeric = float(raw_entry)
            except ValueError:
                numeric = None
            else:
                if params.log_transform and numeric < 0:
                    numeric = None
                value = numeric

        if value is None and params.derive_average_temperature and variable_upper == "TAVG":
            tmin = row.get("TMIN")
            tmax = row.get("TMAX")
            if tmin not in (None, "") and tmax not in (None, ""):
                try:
                    numeric = (float(tmin) + float(tmax)) / 2.0
                except ValueError:
                    numeric = None
                else:
                    if params.log_transform and numeric < 0:
                        numeric = None
                    value = numeric

        if value is None:
            continue

        station_id = row.get("STATION")
        if not station_id:
            continue

        grouped.setdefault(date_value, []).append((station_id, value))

    return grouped


def _extract_coordinates_from_rows(
    rows: Sequence[Dict[str, str]],
    station_id: str,
) -> Optional[Tuple[float, float]]:
    for row in rows:
        if row.get("STATION") != station_id:
            continue
        lon = row.get("LONGITUDE")
        lat = row.get("LATITUDE")
        if lon not in (None, "") and lat not in (None, ""):
            try:
                return float(lon), float(lat)
            except ValueError:
                return None
    return None


def _strip_station_id(station_id: str) -> str:
    return station_id.split(":", 1)[1] if ":" in station_id else station_id


def _compute_statistics(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _export_layer_to_gpkg(
    layer: QgsVectorLayer,
    target_crs: QgsCoordinateReferenceSystem,
    temp_dir: Path,
    transform_context,
) -> Path:
    path = temp_dir / f"{layer.name().replace(' ', '_').lower()}_{len(os.listdir(temp_dir))}.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
    options.destCRS = target_crs
    options.ct = QgsCoordinateTransform(layer.crs(), target_crs, transform_context)

    error, _, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, str(path), transform_context, options
    )
    if error != QgsVectorFileWriter.NoError:
        raise GridError("Failed to export layer for intermediate processing.")
    return path


def _create_station_layer(
    entries: Sequence[Tuple[float, float, float]],
    log_transform: bool,
    temp_dir: Path,
    target_crs: QgsCoordinateReferenceSystem,
    transform_context,
    date_key: str,
) -> Tuple[Path, List[Tuple[float, float, float, float]]]:
    layer = QgsVectorLayer("Point?crs=EPSG:4326", "slice_points", "memory")
    provider = layer.dataProvider()
    provider.addAttributes(
        [
            QgsField("value", QVariant.Double),
            QgsField("raw", QVariant.Double),
        ]
    )
    layer.updateFields()

    features: List[QgsFeature] = []
    transformed_points: List[Tuple[float, float, float, float]] = []
    transformer = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:4326"), target_crs, transform_context)
    for lon, lat, raw_value in entries:
        point_4326 = QgsPointXY(lon, lat)
        point_target = transformer.transform(point_4326)
        transformed = math.log1p(raw_value) if log_transform else raw_value

        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromPointXY(point_4326))
        feature.setAttributes([transformed, raw_value])
        features.append(feature)

        transformed_points.append((point_target.x(), point_target.y(), transformed, raw_value))

    provider.addFeatures(features)
    layer.updateExtents()

    save_path = temp_dir / f"stations_{date_key.replace('-', '')}.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
    options.destCRS = target_crs
    options.ct = QgsCoordinateTransform(layer.crs(), target_crs, transform_context)

    error, _, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, str(save_path), transform_context, options
    )
    if error != QgsVectorFileWriter.NoError:
        raise GridError("Failed to export station layer for gridding.")
    return save_path, transformed_points


def _resolve_algorithm(method: str, power: float, radius: float, min_points: int) -> str:
    radius = max(radius, 0.001)
    method = (method or "idw").lower()
    if method == "nearest":
        return f"nearest:radius1={radius}:radius2={radius}"
    if method == "average":
        return f"average:radius1={radius}:radius2={radius}:min_points={max(min_points, 1)}"
    if method == "linear":
        return "linear"
    # Default to IDW
    return (
        f"invdist:power={power}:smoothing=0.0:radius1={radius}:radius2={radius}"
        f":min_points={max(min_points, 1)}"
    )


def _grid_slice(
    vector_path: Path,
    dest_path: Path,
    buffered_extent: QgsRectangle,
    target_crs: QgsCoordinateReferenceSystem,
    resolution: float,
    width: int,
    height: int,
    nodata: float,
    params: GridParameters,
    search_radius: float,
) -> None:
    xmin = buffered_extent.xMinimum()
    xmax = buffered_extent.xMaximum()
    ymin = buffered_extent.yMinimum()
    ymax = buffered_extent.yMaximum()

    algorithm = _resolve_algorithm(params.method, params.idw_power, search_radius, params.idw_min_points)

    creation_options = ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    grid_options = gdal.GridOptions(
        format="GTiff",
        outputBounds=(xmin, ymin, xmax, ymax),
        width=width,
        height=height,
        algorithm=algorithm,
        zfield="value",
        outputType=gdal.GDT_Float32,
        noData=nodata,
        creationOptions=creation_options,
    )

    ds = gdal.Grid(str(dest_path), str(vector_path), options=grid_options)
    if ds:
        ds = None


def _grid_slice_gwr(
    transformed_points: Sequence[Tuple[float, float, float, float]],
    dest_path: Path,
    buffered_extent: QgsRectangle,
    target_crs: QgsCoordinateReferenceSystem,
    resolution: float,
    width: int,
    height: int,
    nodata: float,
    params: GridParameters,
    search_radius: float,
) -> None:
    if not transformed_points:
        _create_empty_raster(dest_path, buffered_extent, resolution, width, height, target_crs, nodata)
        return

    xs = np.array([pt[0] for pt in transformed_points], dtype=np.float64)
    ys = np.array([pt[1] for pt in transformed_points], dtype=np.float64)
    values = np.array([pt[2] for pt in transformed_points], dtype=np.float64)

    arr = np.full((height, width), nodata, dtype=np.float32)
    bandwidth = search_radius if search_radius > 0 else resolution * 5.0
    bandwidth = max(bandwidth, resolution)
    bw2 = bandwidth * bandwidth
    min_points = max(params.idw_min_points, 1)

    xmin = buffered_extent.xMinimum()
    ymax = buffered_extent.yMaximum()

    ones = np.ones(xs.shape[0], dtype=np.float64)

    for row in range(height):
        y = ymax - (row + 0.5) * resolution
        dy = ys - y
        for col in range(width):
            x = xmin + (col + 0.5) * resolution
            dx = xs - x
            dist2 = dx * dx + dy * dy
            weights = np.exp(-0.5 * dist2 / bw2)
            mask = weights > 1e-8
            if np.count_nonzero(mask) < min_points:
                continue

            W = weights[mask]
            if W.sum() <= 0:
                continue

            X = np.vstack((ones[mask], xs[mask], ys[mask])).T
            Y = values[mask]

            W_sqrt = np.sqrt(W)
            Xw = X * W_sqrt[:, None]
            yw = Y * W_sqrt

            try:
                coeffs, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                pred = coeffs[0] + coeffs[1] * x + coeffs[2] * y
            except np.linalg.LinAlgError:
                pred = np.average(Y, weights=W)

            arr[row, col] = np.float32(pred)

    _write_array_to_raster(arr, dest_path, buffered_extent, resolution, target_crs, nodata)


def _create_empty_raster(
    dest_path: Path,
    extent: QgsRectangle,
    resolution: float,
    width: int,
    height: int,
    target_crs: QgsCoordinateReferenceSystem,
    nodata: float,
) -> None:
    creation_options = ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(dest_path), width, height, 1, gdal.GDT_Float32, options=creation_options)
    geotransform = (
        extent.xMinimum(),
        resolution,
        0.0,
        extent.yMaximum(),
        0.0,
        -resolution,
    )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(target_crs.toWkt())
    band = dataset.GetRasterBand(1)
    band.Fill(nodata)
    band.SetNoDataValue(nodata)
    dataset.FlushCache()
    dataset = None


def _write_array_to_raster(
    array: np.ndarray,
    dest_path: Path,
    extent: QgsRectangle,
    resolution: float,
    target_crs: QgsCoordinateReferenceSystem,
    nodata: float,
) -> None:
    height, width = array.shape
    creation_options = ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(dest_path), width, height, 1, gdal.GDT_Float32, options=creation_options)
    geotransform = (
        extent.xMinimum(),
        resolution,
        0.0,
        extent.yMaximum(),
        0.0,
        -resolution,
    )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(target_crs.toWkt())
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def _trim_to_extent(
    raster_path: Path,
    target_extent: QgsRectangle,
    target_crs: QgsCoordinateReferenceSystem,
    nodata: float,
) -> None:
    temp_path = raster_path.with_suffix(".trim.tif")
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        projWin=(
            target_extent.xMinimum(),
            target_extent.yMaximum(),
            target_extent.xMaximum(),
            target_extent.yMinimum(),
        ),
        noData=nodata,
        creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    ds = gdal.Translate(str(temp_path), str(raster_path), options=translate_options)
    if ds:
        ds = None
    os.replace(temp_path, raster_path)


def _clip_to_polygon(
    raster_path: Path,
    clip_layer_path: Path,
    target_crs: QgsCoordinateReferenceSystem,
    nodata: float,
) -> None:
    temp_path = raster_path.with_suffix(".clip.tif")
    warp_options = gdal.WarpOptions(
        cutlineDSName=str(clip_layer_path),
        cropToCutline=True,
        dstNodata=nodata,
        dstSRS=target_crs.toWkt(),
        creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    ds = gdal.Warp(str(temp_path), str(raster_path), options=warp_options)
    if ds:
        ds = None
    os.replace(temp_path, raster_path)


def _apply_inverse_transform(raster_path: Path, nodata: float) -> None:
    dataset = gdal.Open(str(raster_path), gdal.GA_Update)
    if not dataset:
        return
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    mask = array != nodata
    array[mask] = np.expm1(array[mask])
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def _burn_station_values(
    raster_path: Path,
    vector_path: Path,
    burn_distance: float,
    temp_dir: Path,
    transform_context,
) -> None:
    points_layer = QgsVectorLayer(str(vector_path), "stations", "ogr")
    if not points_layer.isValid():
        return

    attribute_name = "raw"
    rasterize_source_path = vector_path

    if burn_distance > 0.0:
        buffer_layer = QgsVectorLayer(f"Polygon?crs={points_layer.crs().authid()}", "station_buffers", "memory")
        provider = buffer_layer.dataProvider()
        provider.addAttributes([QgsField(attribute_name, QVariant.Double)])
        buffer_layer.updateFields()

        for feature in points_layer.getFeatures():
            raw_value = feature[attribute_name]
            geom = feature.geometry()
            buffer_geom = geom.buffer(burn_distance, 16)
            if buffer_geom is None or buffer_geom.isEmpty():
                continue
            new_feature = QgsFeature(buffer_layer.fields())
            new_feature.setGeometry(buffer_geom)
            new_feature.setAttributes([raw_value])
            provider.addFeature(new_feature)

        buffer_layer.updateExtents()
        rasterize_source_path = _export_layer_to_gpkg(
            buffer_layer,
            points_layer.crs(),
            temp_dir,
            transform_context,
        )

    rasterize_options = gdal.RasterizeOptions(attribute=attribute_name, allTouched=True)
    ds = gdal.Rasterize(str(raster_path), str(rasterize_source_path), options=rasterize_options)
    if ds:
        ds = None


def _translate_to_cog(src_path: Path, dst_path: Path) -> None:
    translate_options = gdal.TranslateOptions(format="COG")
    ds = gdal.Translate(str(dst_path), str(src_path), options=translate_options)
    if ds:
        ds = None


def _build_vrt(tif_paths: Sequence[Path], vrt_path: Path, band_labels: Sequence[str]) -> None:
    options = gdal.BuildVRTOptions(separate=True)
    ds = gdal.BuildVRT(str(vrt_path), [str(path) for path in tif_paths], options=options)
    if ds:
        ds = None

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(vrt_path)
    except ET.ParseError:
        return

    root = tree.getroot()
    bands = root.findall("VRTRasterBand")

    for band_elem, label in zip(bands, band_labels):
        band_elem.set("name", label)
        desc_elem = band_elem.find("Description")
        if desc_elem is None:
            desc_elem = ET.SubElement(band_elem, "Description")
        desc_elem.text = label

        metadata_elem = band_elem.find("Metadata")
        if metadata_elem is None:
            metadata_elem = ET.SubElement(band_elem, "Metadata")

        timestamp_found = False
        for item in metadata_elem.findall("MDI"):
            if item.get("key") == "TIMESTAMP":
                item.text = label
                timestamp_found = True
                break
        if not timestamp_found:
            item = ET.SubElement(metadata_elem, "MDI", key="TIMESTAMP")
            item.text = label

    tree.write(vrt_path, encoding="utf-8", xml_declaration=True)


def _write_coverage_csv(
    csv_path: Path,
    rows: Sequence[Tuple[str, int, Optional[float], Optional[float], Optional[float], Optional[float]]],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "station_count", "min", "max", "mean", "std_dev"])
        for row in rows:
            writer.writerow(row)
