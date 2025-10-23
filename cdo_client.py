"""
Helper functions for interacting with the NOAA CDO API (v2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import requests
from qgis.PyQt.QtCore import QSettings

CDO_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
MAX_PAGE_SIZE = 1000
DEFAULT_SLEEP = 0.25  # stay under 5 req/sec


class CDORequestError(RuntimeError):
    """Raised when the CDO API request fails."""


@dataclass
class StationRecord:
    """Parsed station metadata from the CDO API."""

    id: str
    name: str
    latitude: float
    longitude: float
    elevation: Optional[float]


def _get_token() -> str:
    """Return the stored CDO token."""
    settings = QSettings()
    token = settings.value("ncei_raster_cube/cdo_token", "", str).strip()
    if not token:
        raise CDORequestError(
            "CDO token missing. Set it via the plugin settings before querying stations."
        )
    return token


def _headers() -> Dict[str, str]:
    """Build authentication headers for CDO requests."""
    return {"token": _get_token()}


def fetch_stations_in_bbox(
    *,
    dataset: str,
    start_date: str,
    end_date: str,
    bbox: Sequence[float],
    include_elevation: bool = False,
) -> List[StationRecord]:
    """
    Retrieve station metadata for a dataset within a bounding box and time window.

    Parameters
    ----------
    dataset: str
        Dataset identifier (e.g., ``GHCND`` for daily summaries, ``GHCND``/``GSOD`` etc.).
    start_date, end_date: str
        ISO 8601 `YYYY-MM-DD` strings delineating the period of interest.
    bbox: Sequence[float]
        North, West, South, East coordinates in decimal degrees (EPSG:4326).
    include_elevation: bool
        Whether to request station elevation metadata (may increase response size).
    """
    north, west, south, east = bbox
    if north < south:
        raise ValueError("North latitude must be >= south latitude")
    if east < west:
        raise ValueError("East longitude must be >= west longitude")

    params = {
        "datasetid": dataset,
        "startdate": start_date,
        "enddate": end_date,
        "extent": f"{south},{west},{north},{east}",
        "limit": MAX_PAGE_SIZE,
        "offset": 1,
    }
    if include_elevation:
        params["includemetadata"] = "false"

    stations: List[StationRecord] = []

    while True:
        response = requests.get(
            f"{CDO_BASE_URL}/stations",
            params=params,
            headers=_headers(),
            timeout=60,
        )

        if response.status_code != 200:
            raise CDORequestError(
                f"CDO station request failed (HTTP {response.status_code}): {response.text[:500]}"
            )

        payload = response.json()
        results = payload.get("results", [])
        if not results:
            break

        for item in results:
            stations.append(
                StationRecord(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    latitude=float(item.get("latitude") or 0.0),
                    longitude=float(item.get("longitude") or 0.0),
                    elevation=item.get("elevation"),
                )
            )

        metadata = payload.get("metadata", {})
        resultset = metadata.get("resultset", {})
        total = resultset.get("count", len(stations))
        offset = resultset.get("offset", params["offset"])
        if offset + MAX_PAGE_SIZE > total:
            break

        params["offset"] = offset + MAX_PAGE_SIZE
        time.sleep(DEFAULT_SLEEP)

    return stations


def strip_dataset_prefix(station_ids: Iterable[str]) -> List[str]:
    """
    Convert CDO-style station identifiers (``DATASET:STATION``) to bare station IDs.
    """
    cleaned: List[str] = []
    for station_id in station_ids:
        if ":" in station_id:
            cleaned.append(station_id.split(":", 1)[1])
        else:
            cleaned.append(station_id)
    return cleaned
