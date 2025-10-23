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

from osgeo import gdal
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFields,
    QgsField,
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
    auto_radius: bool
    radius: float
    auto_radius_multiplier: float
    idw_power: float
    idw_min_points: int
    skip_sparse_slices: bool
    derive_average_temperature: bool
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


class GridError(RuntimeError):
    """Raised when gridding fails."""


def build_raster_cube(
    result: ADSFetchResult,
    params: GridParameters,
    progress_cb: Optional[ProgressCallback] = None,
) -> GridOutputs:
    """
    Convert station data to a stack of rasters (one per time slice).
    """
    if progress_cb:
        progress_cb(0, "Preparing gridding inputs...")

    variable = params.variable.upper()
    if not result.rows:
        raise GridError("No station data is available to grid.")

    target_crs = params.target_crs
    if not target_crs.isValid():
        raise GridError("Target CRS is invalid.")

    resolution = max(params.resolution, 0.0001)

    base_extent = QgsRectangle(params.extent_rect)
    if not base_extent.isFinite():
        raise GridError("Grid extent is invalid or empty.")

    buffer_distance = max(params.buffer_cells, 0) * resolution
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

        grid_width, grid_height = _grid_dimensions(buffered_extent, resolution)
        if grid_width <= 0 or grid_height <= 0:
            raise GridError("Computed grid size is zero; adjust resolution or extent.")

        auto_radius = params.auto_radius_multiplier * resolution
        search_radius = params.radius if not params.auto_radius else auto_radius

        for index, date_key in enumerate(date_keys, start=1):
            slice_points = values_by_date[date_key]
            station_values: List[Tuple[float, float, float]] = []

            for station_id, value in slice_points:
                record = station_map.get(station_id)
                if record:
                    station_values.append((record.longitude, record.latitude, value))
                else:
                    fallback = _extract_coordinates_from_rows(result.rows, station_id)
                    if fallback:
                        station_values.append((*fallback, value))

            station_count = len(station_values)
            stats = _compute_statistics([v for _, _, v in station_values])
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
            else:
                _grid_slice(
                    station_values,
                    dest_path,
                    tmp_dir,
                    buffered_extent,
                    target_crs,
                    resolution,
                    grid_width,
                    grid_height,
                    nodata,
                    params.idw_power,
                    search_radius,
                    params.idw_min_points,
                    transform_context,
                )

            _trim_to_extent(dest_path, base_extent, target_crs, nodata)

            if clip_layer_path and params.clip_enabled:
                _clip_to_polygon(dest_path, clip_layer_path, target_crs, nodata)

            tif_paths.append(dest_path)

            if params.write_cog:
                cog_path = cog_dir / output_name
                _translate_to_cog(dest_path, cog_path)
                cog_paths.append(cog_path)

        vrt_path: Optional[Path] = None
        if params.write_vrt and tif_paths:
            if progress_cb:
                progress_cb(90, "Building VRT...")
            vrt_path = variable_dir / f"{variable.lower()}_cube.vrt"
            _build_vrt(tif_paths, vrt_path)

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

        raw = row.get(variable_upper)
        value: Optional[float] = None
        if raw is not None and raw != "":
            try:
                value = float(raw)
            except ValueError:
                value = None
        elif (
            params.derive_average_temperature
            and variable_upper == "TAVG"
        ):
            tmin = row.get("TMIN")
            tmax = row.get("TMAX")
            if tmin not in (None, "") and tmax not in (None, ""):
                try:
                    value = (float(tmin) + float(tmax)) / 2.0
                except ValueError:
                    value = None

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
    path = temp_dir / "clip_layer.gpkg"
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
        raise GridError("Failed to export AOI layer for clipping.")
    return path


def _grid_slice(
    station_values: Sequence[Tuple[float, float, float]],
    dest_path: Path,
    temp_dir: Path,
    buffered_extent: QgsRectangle,
    target_crs: QgsCoordinateReferenceSystem,
    resolution: float,
    width: int,
    height: int,
    nodata: float,
    power: float,
    radius: float,
    min_points: int,
    transform_context,
) -> None:
    temp_layer = QgsVectorLayer("Point?crs=EPSG:4326", "slice", "memory")
    provider = temp_layer.dataProvider()
    provider.addAttributes(
        [
            QgsField("value", QVariant.Double),
        ]
    )
    temp_layer.updateFields()

    features: List[QgsFeature] = []
    for lon, lat, value in station_values:
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
        feature.setAttributes([value])
        features.append(feature)

    provider.addFeatures(features)
    temp_layer.updateExtents()

    reprojected_path = temp_dir / "slice_points.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
    options.destCRS = target_crs
    options.ct = QgsCoordinateTransform(temp_layer.crs(), target_crs, transform_context)

    error, _, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
        temp_layer, str(reprojected_path), transform_context, options
    )
    if error != QgsVectorFileWriter.NoError:
        raise GridError("Failed to export temporary point layer for gridding.")

    xmin = buffered_extent.xMinimum()
    xmax = buffered_extent.xMaximum()
    ymin = buffered_extent.yMinimum()
    ymax = buffered_extent.yMaximum()

    algorithm = f"invdist:power={power}:smoothing=0.0:radius1={radius}:radius2={radius}:min_points={min_points}"
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
    ds = gdal.Grid(str(dest_path), str(reprojected_path), options=grid_options)
    if ds:
        ds = None


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


def _translate_to_cog(src_path: Path, dst_path: Path) -> None:
    translate_options = gdal.TranslateOptions(format="COG")
    ds = gdal.Translate(str(dst_path), str(src_path), options=translate_options)
    if ds:
        ds = None


def _build_vrt(tif_paths: Sequence[Path], vrt_path: Path) -> None:
    ds = gdal.BuildVRT(str(vrt_path), [str(path) for path in tif_paths])
    if ds:
        ds = None


def _write_coverage_csv(
    csv_path: Path,
    rows: Sequence[Tuple[str, int, Optional[float], Optional[float], Optional[float], Optional[float]]],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "station_count", "min", "max", "mean", "std_dev"])
        for row in rows:
            writer.writerow(row)
