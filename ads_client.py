"""
Client utilities for interacting with the NOAA NCEI Access Data Service (ADS).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from qgis.PyQt.QtCore import QStandardPaths, QSettings

from .cdo_client import (
    CDORequestError,
    fetch_stations_in_bbox,
    strip_dataset_prefix,
)


ADS_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


class ADSRequestError(RuntimeError):
    """Raised when the ADS request fails."""


@dataclass
class ADSFetchResult:
    """Metadata describing an ADS data retrieval."""

    dataset: str
    params: Dict[str, str]
    saved_csv: Path
    provenance: Path
    record_count: int
    columns: Sequence[str]
    display_columns: Sequence[str]
    rows: List[Dict[str, str]]
    preview_rows: List[Dict[str, str]]
    elapsed_ms: int
    stations: Sequence[str]


def _get_cache_root() -> Path:
    """Return the plugin-specific cache directory."""
    base_dir = Path(
        QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        or QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation)
    )
    cache_dir = base_dir / "ncei_raster_cube" / "cache" / "ads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _headers_with_token() -> Dict[str, str]:
    """Include the stored CDO token if available."""
    settings = QSettings()
    token = settings.value("ncei_raster_cube/cdo_token", "", str).strip()
    headers: Dict[str, str] = {}
    if token:
        headers["token"] = token
    return headers


def fetch_ads_dataset(
    dataset: str,
    *,
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    data_types: Optional[Sequence[str]] = None,
    units: str = "metric",
    response_format: str = "csv",
    limit_preview_rows: int = 10,
    station_ids: Optional[Sequence[str]] = None,
    keep_columns: Optional[Sequence[str]] = None,
    friendly_names: Optional[Dict[str, str]] = None,
    always_keep: Optional[Sequence[str]] = None,
) -> ADSFetchResult:
    """
    Fetch data from the ADS endpoint and cache the CSV locally.

    Parameters
    ----------
    dataset: str
        ADS dataset identifier (e.g., "daily-summaries").
    start_date, end_date: str
        ISO-formatted date strings (YYYY-MM-DD).
    bbox: tuple
        Bounding box as (north, west, south, east).
    data_types: sequence
        Optional sequence of data type identifiers.
    units: str
        Units parameter forwarded to ADS (`metric`, `standard`, etc.).
    response_format: str
        ADS response format; currently only `csv` is handled fully.
    limit_preview_rows: int
        Number of rows to populate in the preview payload.
    """
    params: Dict[str, str] = {
        "dataset": dataset,
        "startDate": start_date,
        "endDate": end_date,
        "bbox": ",".join(f"{value:.4f}" for value in bbox),
        "units": units,
        "format": response_format,
    }

    if data_types:
        params["dataTypes"] = ",".join(sorted({dt.strip() for dt in data_types if dt}))

    station_id_list = list(station_ids or [])
    if not station_id_list:
        station_id_list = _resolve_stations(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
        )

    if not station_id_list:
        raise ADSRequestError(
            "No stations found in the specified extent/time window. Adjust parameters or verify dataset availability."
        )

    params["stations"] = ",".join(station_id_list)

    headers = _headers_with_token()
    token_used = bool(headers.get("token"))
    response = requests.get(
        ADS_BASE_URL,
        params=params,
        headers=headers,
        timeout=60,
    )
    elapsed_ms = int(response.elapsed.total_seconds() * 1000)

    if response.status_code != 200:
        raise ADSRequestError(
            f"ADS request failed with HTTP {response.status_code}: {response.text[:500]}"
        )

    if response_format.lower() != "csv":
        raise ADSRequestError(
            f"Response format '{response_format}' not yet supported in this workflow."
        )

    text_stream = StringIO(response.text)
    reader = csv.DictReader(text_stream)
    source_columns = reader.fieldnames or []
    rows: List[Dict[str, str]] = list(reader)

    filtered_columns = _determine_columns(
        source_columns,
        rows,
        keep_columns=keep_columns,
        always_keep=always_keep,
    )

    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        filtered_row: Dict[str, str] = {}
        for column in filtered_columns:
            value = row.get(column, "")
            if isinstance(value, str):
                value = value.strip()
            filtered_row[column] = value
        filtered_rows.append(filtered_row)

    display_columns = [
        (friendly_names or {}).get(column.upper(), column)
        for column in filtered_columns
    ]

    record_count = len(filtered_rows)
    preview_rows = filtered_rows[:limit_preview_rows]

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cache_dir = _get_cache_root() / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cache_dir / f"{dataset}_{timestamp}.csv"
    provenance_path = cache_dir / f"{dataset}_{timestamp}_provenance.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=filtered_columns)
        writer.writeheader()
        writer.writerows(filtered_rows)
    provenance_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "params": params,
                "url": response.url,
                "saved_at": timestamp,
                "record_count": record_count,
                "elapsed_ms": elapsed_ms,
                "token_provided": token_used,
                "columns": filtered_columns,
                "friendly_names": {
                    column: (friendly_names or {}).get(column.upper(), column)
                    for column in filtered_columns
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ADSFetchResult(
        dataset=dataset,
        params=params,
        saved_csv=csv_path,
        provenance=provenance_path,
        record_count=record_count,
        columns=filtered_columns,
        display_columns=display_columns,
        rows=filtered_rows,
        preview_rows=preview_rows,
        elapsed_ms=elapsed_ms,
        stations=station_id_list,
    )


def _resolve_stations(
    *,
    dataset: str,
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
) -> List[str]:
    """Lookup station identifiers using the CDO API for the ADS dataset."""
    cdo_dataset = _map_ads_dataset_to_cdo(dataset)
    if not cdo_dataset:
        return []

    try:
        stations = fetch_stations_in_bbox(
            dataset=cdo_dataset,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
        )
    except CDORequestError as err:
        raise ADSRequestError(
            f"Station discovery via CDO failed: {err}"
        ) from err

    prefixed_ids = [station.id for station in stations if station.id]
    return strip_dataset_prefix(prefixed_ids)


def _map_ads_dataset_to_cdo(dataset: str) -> Optional[str]:
    """Map ADS dataset identifiers to their CDO dataset counterparts."""
    mapping = {
        "daily-summaries": "GHCND",
        "global-hourly": "GSOD",
        "global-summary-of-the-day": "GSOD",
    }
    return mapping.get(dataset.lower())


def _determine_columns(
    source_columns: Sequence[str],
    rows: Sequence[Dict[str, str]],
    *,
    keep_columns: Optional[Sequence[str]] = None,
    always_keep: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select the columns that should be retained in the filtered output."""
    if not source_columns:
        return []

    column_lookup = {column.upper(): column for column in source_columns}

    ordered_preferences: List[str] = []
    if always_keep:
        ordered_preferences.extend([col.upper() for col in always_keep])
    if keep_columns:
        ordered_preferences.extend([col.upper() for col in keep_columns])

    seen: set[str] = set()
    preferred_columns: List[str] = []
    for identifier in ordered_preferences:
        if identifier in seen:
            continue
        seen.add(identifier)
        actual = column_lookup.get(identifier)
        if actual:
            preferred_columns.append(actual)

    if preferred_columns:
        return preferred_columns

    # When no explicit keep list is supplied, drop empty columns automatically.
    filtered: List[str] = []
    for column in source_columns:
        if _column_has_data(column, rows):
            filtered.append(column)
    return filtered or list(source_columns)


def _column_has_data(column: str, rows: Sequence[Dict[str, str]]) -> bool:
    """Return True if at least one row contains a non-empty value for the column."""
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return True
        elif value != "":
            return True
    return False


