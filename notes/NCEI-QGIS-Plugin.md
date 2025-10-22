---
title: NCEI QGIS Plugin Build Log
project: qgis-ncei-raster-cubes
tags: [qgis, ncei, cdo, ads, weather, plugin, obsidian]
created: 2025-10-22
updated: 2025-10-22
status: in-progress
---

# NCEI QGIS Plugin - Build Log

## Overview
Goal: QGIS plugin that generates time-enabled raster cubes from **NCEI** data (station via CDO/ADS; optional gridded MRMS/NDFD).

## Current Capabilities
<!-- CAPABILITIES:BEGIN -->
- Plugin skeleton registered in QGIS with toolbar/menu action and placeholder dialog.
- Settings dialog persists CDO token in QSettings and validates against CDO `/datasets`.
<!-- CAPABILITIES:END -->

## Next Steps
<!-- NEXT_STEPS:BEGIN -->
- [x] Implement settings panel for CDO token (store in QSettings and validate).
- [ ] Implement ADS `daily-summaries` bbox fetch (CSV -> preview table).
- [ ] Implement CDO `GHCND` fetch with chunked dates and pagination.
- [ ] Grid station data (IDW) to target raster; write CF-NetCDF.
- [ ] Load cube into QGIS and enable temporal controls.
<!-- NEXT_STEPS:END -->

## Decisions
<!-- DECISIONS:BEGIN -->
- Output formats: CF-NetCDF (primary) and stacked GeoTIFF/COG + VRT (secondary).
- Interpolation: IDW first; kriging later.
<!-- DECISIONS:END -->

## Blockers
<!-- BLOCKERS:BEGIN -->
- None yet.
<!-- BLOCKERS:END -->

## Progress Log
<!-- LOG:BEGIN -->
- 2025-10-22 (PDT): Added token settings dialog with validation call and connected plugin actions.
- 2025-10-22 (PDT): Scaffolded QGIS plugin package, metadata, and icon; established Obsidian build log.
- 2025-10-22 (PDT): Initialized Obsidian note and set milestones.
<!-- LOG:END -->

## References
- CDO API docs (base, endpoints, token, limits; pagination; per-request date span limits). :contentReference[oaicite:22]{index=22}  
- NCEI Access Data Service (ADS): endpoint, parameters, **NetCDF** support. :contentReference[oaicite:23]{index=23}  
- MRMS (AWS Open Data), optional gridded precip. :contentReference[oaicite:24]{index=24}  
- NDFD archive via NCEI + AWS (optional retrospective forecasts). :contentReference[oaicite:25]{index=25}
