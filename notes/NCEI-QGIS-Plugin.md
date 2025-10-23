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
- ADS `daily-summaries` bbox fetch dialog downloads CSV, caches raw output, and previews records.
- ADS fetch auto-discovers stations via CDO and can seed bbox from current map extent.
- ADS field picker exposes plain-language variable names and filters cached CSVs to selected columns.
- ADS preview now reports summary stats, generates quick plots, and supports custom CSV export paths.
- ADS UI enforces sane date bounds, persists user preferences, and updates labels to match selected units.
- Station gridding supports multiple methods (IDW, nearest neighbor, linear/TIN, moving average) with optional log transform and station burn-in controls.
- ADS requests chunk station lists >100 IDs and publishes a point layer of fetched stations in QGIS.
<!-- CAPABILITIES:END -->

## Next Steps
<!-- NEXT_STEPS:BEGIN -->
- [x] Implement settings panel for CDO token (store in QSettings and validate).
- [x] Implement ADS `daily-summaries` bbox fetch (CSV -> preview table).
- [ ] Implement CDO `GHCND` fetch with chunked dates and pagination.
- [ ] Interactive Plots 
- [ ] Aggregate to months/seasons/years option
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
- 2025-10-22 (PDT): Added pre-flight validation, compact tabbed layout, unit-aware labels, and persisted UI preferences.
- 2025-10-22 (PDT): Added ADS station chunking plus station layer injection for spatial QA.
- 2025-10-22 (PDT): Introduced multi-method gridding (IDW/nearest/linear/moving average) with precip-friendly log transform and station burn-in options.
- 2025-10-22 (PDT): Added ADS summary stats, plotting, trim-to-selection CSV export with persistent save path.
- 2025-10-22 (PDT): Added friendly-name field picker, column filtering, and station-count tooltips for ADS preview.
- 2025-10-22 (PDT): Integrated CDO station discovery into ADS fetch, added map extent helper, and improved status feedback.
- 2025-10-22 (PDT): Added ADS bbox fetch UI with caching, CSV preview, and token-aware requests.
- 2025-10-22 (PDT): Added token settings dialog with validation call and connected plugin actions.
- 2025-10-22 (PDT): Scaffolded QGIS plugin package, metadata, and icon; established Obsidian build log.
- 2025-10-22 (PDT): Initialized Obsidian note and set milestones.
<!-- LOG:END -->

## References
- CDO API docs (base, endpoints, token, limits; pagination; per-request date span limits). :contentReference[oaicite:22]{index=22}  
- NCEI Access Data Service (ADS): endpoint, parameters, **NetCDF** support. :contentReference[oaicite:23]{index=23}  
- MRMS (AWS Open Data), optional gridded precip. :contentReference[oaicite:24]{index=24}  
- NDFD archive via NCEI + AWS (optional retrospective forecasts). :contentReference[oaicite:25]{index=25}
