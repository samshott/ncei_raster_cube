"""
Dialog for configuring and launching ADS dataset fetches.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qgis.PyQt.QtCore import QDate, QSettings, Qt, QVariant
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsVectorLayer,
)

from .ads_client import ADSFetchResult, ADSRequestError, fetch_ads_dataset
from .settings import tr as tr_settings


DATA_TYPE_CATALOG = {
    "daily-summaries": [
        ("PRCP", "Precipitation (millimeters)"),
        ("SNOW", "Snowfall (millimeters)"),
        ("SNWD", "Snow depth (millimeters)"),
        ("TMAX", "Maximum temperature (deg C)"),
        ("TMIN", "Minimum temperature (deg C)"),
        ("TAVG", "Average temperature (deg C)"),
        ("TOBS", "Temperature at observation time (deg C)"),
        ("AWND", "Average wind speed (m/s)"),
        ("AWDR", "Average wind direction (degrees)"),
        ("WSF2", "Peak wind speed - 2 min average (m/s)"),
        ("WSFG", "Peak wind gust (m/s)"),
        ("WDF2", "Direction of peak wind - 2 min average"),
        ("WESD", "Water equivalent of snow on ground (millimeters)"),
        ("WESF", "Water equivalent of snowfall (millimeters)"),
        ("PSUN", "Percent of possible sunshine (%)"),
        ("TSUN", "Total sunshine (minutes)"),
        ("WT01", "Weather: Fog, ice fog, or freezing fog"),
        ("WT02", "Weather: Heavy fog (<= 1/4 mi visibility)"),
        ("WT03", "Weather: Thunder"),
        ("WT04", "Weather: Ice pellets or sleet"),
        ("WT05", "Weather: Hail"),
        ("WT06", "Weather: Glaze or rime"),
        ("WT07", "Weather: Dust, volcanic ash, or blowing dust"),
        ("WT08", "Weather: Smoke or haze"),
        ("WT09", "Weather: Blowing or drifting snow"),
        ("WT10", "Weather: Tornado, waterspout, or funnel cloud"),
        ("WT11", "Weather: High or damaging winds"),
        ("WT12", "Weather: Blowing spray"),
        ("WT13", "Weather: Mist"),
        ("WT14", "Weather: Drizzle"),
        ("WT15", "Weather: Freezing drizzle or freezing rain"),
        ("WT16", "Weather: Rain or showers"),
        ("WT17", "Weather: Freezing rain"),
        ("WT18", "Weather: Snow or ice pellets"),
        ("WT19", "Weather: Unknown precipitation"),
        ("WT21", "Weather: Ground fog"),
        ("WT22", "Weather: Ice fog or freezing fog"),
    ],
}

UNIT_LABEL_OVERRIDES: Dict[str, Dict[str, Dict[str, str]]] = {
    "daily-summaries": {
        "metric": {
            "PRCP": "Precipitation (mm)",
            "SNOW": "Snowfall (mm)",
            "SNWD": "Snow depth (mm)",
            "WESD": "Water equivalent of snow on ground (mm)",
            "WESF": "Water equivalent of snowfall (mm)",
            "TMAX": "Maximum temperature (deg C)",
            "TMIN": "Minimum temperature (deg C)",
            "TAVG": "Average temperature (deg C)",
            "TOBS": "Temperature at observation time (deg C)",
            "AWND": "Average wind speed (m/s)",
            "WSF2": "Peak wind speed - 2 min avg (m/s)",
            "WSFG": "Peak wind gust (m/s)",
        },
        "standard": {
            "PRCP": "Precipitation (inches)",
            "SNOW": "Snowfall (inches)",
            "SNWD": "Snow depth (inches)",
            "WESD": "Water equivalent of snow on ground (inches)",
            "WESF": "Water equivalent of snowfall (inches)",
            "TMAX": "Maximum temperature (deg F)",
            "TMIN": "Minimum temperature (deg F)",
            "TAVG": "Average temperature (deg F)",
            "TOBS": "Temperature at observation time (deg F)",
            "AWND": "Average wind speed (mph)",
            "WSF2": "Peak wind speed - 2 min avg (mph)",
            "WSFG": "Peak wind gust (mph)",
        },
    }
}

DEFAULT_DATA_TYPES = {
    "daily-summaries": ["PRCP", "TMAX", "TMIN", "TAVG", "SNOW", "SNWD", "AWND"],
}

ALWAYS_KEEP_COLUMNS = ["STATION", "DATE"]

BASE_COLUMN_LABELS = {
    "STATION": "Station ID",
    "DATE": "Date",
}



def tr(message: str) -> str:
    """Local translation helper."""
    return tr_settings(message)


class ADSFetchDialog(QDialog):
    """Provides UI controls to request ADS datasets and preview results."""

    SETTINGS_ROOT = "ncei_raster_cube/ads_dialog"
    SETTINGS_EXPORT_DIR = "ncei_raster_cube/ads_output_dir"

    def __init__(self, iface, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._iface = iface
        self._settings = QSettings()
        self._last_result: ADSFetchResult | None = None
        self._loading_preferences = True
        self._stored_selected_codes: List[str] | None = None
        self._base_type_catalog: Dict[str, str] = {}
        self._station_layer_id: str | None = None

        self._build_widgets()
        self._load_preferences()
        self._connect_signals()

        # Populate data types and finalize initialization.
        self._on_dataset_changed(self._dataset_combo.currentText())
        self._loading_preferences = False
        self._persist_preferences()
        self._clear_plot()
        self._reset_results()

    # ------------------------------------------------------------------ Setup --
    def _build_widgets(self) -> None:
        """Create widgets and assemble the layout."""
        self.setWindowTitle(tr("ADS BBox Fetch"))
        self.resize(900, 640)

        # Controls -------------------------------------------------------------
        self._dataset_combo = QComboBox()
        self._dataset_combo.addItems(["daily-summaries"])

        self._format_combo = QComboBox()
        self._format_combo.addItems(["csv", "netcdf"])
        self._format_combo.setCurrentText("csv")
        self._format_combo.setEnabled(False)

        today = QDate.currentDate()
        default_start = today.addDays(-7)

        self._start_date = QDateEdit()
        self._start_date.setCalendarPopup(True)
        self._start_date.setDate(default_start)
        self._start_date.setMaximumDate(today)

        self._end_date = QDateEdit()
        self._end_date.setCalendarPopup(True)
        self._end_date.setDate(today)
        self._end_date.setMaximumDate(today)
        self._end_date.setMinimumDate(default_start)

        self._north = self._make_coord_spin(90.0, default=49.0)
        self._south = self._make_coord_spin(90.0, default=24.0)
        self._west = self._make_coord_spin(180.0, default=-125.0)
        self._east = self._make_coord_spin(180.0, default=-66.0)

        self._units_combo = QComboBox()
        self._units_combo.addItems(["metric", "standard"])

        self._preview_rows_spin = QSpinBox()
        self._preview_rows_spin.setRange(1, 100)
        self._preview_rows_spin.setValue(10)

        self._data_type_list = QListWidget()
        self._data_type_list.setAlternatingRowColors(True)
        self._data_type_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._data_type_list.setMinimumHeight(220)
        self._data_type_list.setUniformItemSizes(True)

        self._select_all_button = QPushButton(tr("Select All"))
        self._clear_selection_button = QPushButton(tr("Clear"))

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._fetch_button = QPushButton(tr("Fetch Preview"))
        self._use_map_extent_button = QPushButton(tr("Use Map Extent"))
        self._save_button = QPushButton(tr("Save CSV..."))
        self._save_button.setEnabled(False)

        self._record_count_label = QLabel(tr("Records: --"))
        self._station_count_label = QLabel(tr("Stations: --"))

        # Preview/summary/plot widgets ----------------------------------------
        self._preview_table = QTableWidget()
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._preview_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self._stats_table = QTableWidget()
        self._stats_table.setColumnCount(5)
        self._stats_table.setHorizontalHeaderLabels(
            [tr("Field"), tr("Count"), tr("Mean"), tr("Min"), tr("Max")]
        )
        self._stats_table.verticalHeader().setVisible(False)
        self._stats_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)

        self._plot_field_combo = QComboBox()
        self._plot_field_combo.setEnabled(False)

        self._figure = Figure(figsize=(5, 3))
        self._canvas = FigureCanvas(self._figure)
        self._plot_status_label = QLabel()
        self._plot_status_label.setAlignment(Qt.AlignCenter)

        # Layout ---------------------------------------------------------------
        command_row = QHBoxLayout()
        command_row.addWidget(self._fetch_button)
        command_row.addWidget(self._use_map_extent_button)
        command_row.addWidget(self._save_button)
        command_row.addStretch(1)
        command_row.addWidget(self._record_count_label)
        command_row.addWidget(self._station_count_label)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.addLayout(self._build_form_layout())
        controls_layout.addWidget(self._build_fields_group())
        controls_layout.addWidget(self._build_bbox_group())
        controls_layout.addStretch(1)
        controls_widget.setMinimumWidth(280)

        self._tab_widget = QTabWidget()
        self._tab_widget.addTab(self._create_preview_tab(), tr("Preview"))
        self._tab_widget.addTab(self._create_summary_tab(), tr("Summary"))
        self._tab_widget.addTab(self._create_plot_tab(), tr("Plot"))

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self._tab_widget)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(command_row)
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addWidget(self._status_label)

    def _connect_signals(self) -> None:
        """Wire widget signals."""
        self._dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        self._units_combo.currentTextChanged.connect(self._on_units_changed)
        self._select_all_button.clicked.connect(self._select_all_data_types)
        self._clear_selection_button.clicked.connect(self._clear_data_type_selection)
        self._data_type_list.itemChanged.connect(self._on_data_type_item_changed)
        self._preview_rows_spin.valueChanged.connect(self._on_preview_rows_changed)
        self._plot_field_combo.currentIndexChanged.connect(self._update_plot)
        self._fetch_button.clicked.connect(self._on_fetch_clicked)
        self._use_map_extent_button.clicked.connect(
            lambda: self._populate_from_map_extent(initial=False)
        )
        self._save_button.clicked.connect(self._handle_save_csv)
        self._start_date.dateChanged.connect(self._on_start_date_changed)
        self._end_date.dateChanged.connect(self._on_end_date_changed)

    # ----------------------------------------------------------------- Helpers --
    def _make_coord_spin(self, maximum: float, *, default: float) -> QDoubleSpinBox:
        """Create a configured coordinate spin box."""
        spin = QDoubleSpinBox()
        spin.setRange(-abs(maximum), abs(maximum))
        spin.setDecimals(6)
        spin.setSingleStep(0.1)
        spin.setValue(default)
        spin.setAlignment(Qt.AlignCenter)
        return spin

    def _build_form_layout(self) -> QFormLayout:
        """Form layout for core parameters."""
        form = QFormLayout()
        form.addRow(tr("Dataset"), self._dataset_combo)
        form.addRow(tr("Format"), self._format_combo)
        form.addRow(tr("Start date"), self._start_date)
        form.addRow(tr("End date"), self._end_date)
        form.addRow(tr("Units"), self._units_combo)
        form.addRow(tr("Preview rows"), self._preview_rows_spin)
        return form

    def _build_fields_group(self) -> QGroupBox:
        """Group box displaying selectable data fields."""
        group = QGroupBox(tr("Data Fields (ADS dataTypes)"))
        layout = QVBoxLayout()
        layout.addWidget(self._data_type_list)

        button_row = QHBoxLayout()
        button_row.addWidget(self._select_all_button)
        button_row.addWidget(self._clear_selection_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        group.setLayout(layout)
        return group

    def _build_bbox_group(self) -> QGroupBox:
        """Group box containing bounding box controls."""
        group = QGroupBox(tr("Bounding Box (degrees)"))
        grid = QGridLayout()

        grid.addWidget(QLabel(tr("North")), 0, 1)
        grid.addWidget(QLabel(tr("South")), 2, 1)
        grid.addWidget(QLabel(tr("West")), 1, 0)
        grid.addWidget(QLabel(tr("East")), 1, 2)

        grid.addWidget(self._north, 0, 2)
        grid.addWidget(self._south, 2, 2)
        grid.addWidget(self._west, 1, 1)
        grid.addWidget(self._east, 1, 3)

        group.setLayout(grid)
        return group

    def _create_preview_tab(self) -> QWidget:
        """Create the preview tab containing the table."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._preview_table)
        return container

    def _create_summary_tab(self) -> QWidget:
        """Create the summary tab."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._summary_label)
        layout.addWidget(self._stats_table)
        return container

    def _create_plot_tab(self) -> QWidget:
        """Create the plotting tab."""
        container = QWidget()
        layout = QVBoxLayout(container)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel(tr("Field")))
        selector_row.addWidget(self._plot_field_combo, stretch=1)
        layout.addLayout(selector_row)

        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._plot_status_label)
        return container
    # --------------------------------------------------------------- Preferences --
    def _load_preferences(self) -> None:
        """Apply persisted preferences to widgets."""
        root = self.SETTINGS_ROOT

        dataset = self._settings.value(f"{root}/dataset", "daily-summaries", str)
        if dataset in {self._dataset_combo.itemText(i) for i in range(self._dataset_combo.count())}:
            self._dataset_combo.setCurrentText(dataset)

        units = self._settings.value(f"{root}/units", "metric", str)
        if units in {"metric", "standard"}:
            self._units_combo.setCurrentText(units)

        preview = self._settings.value(f"{root}/preview_rows", 10, int)
        if isinstance(preview, int) and 1 <= preview <= 100:
            self._preview_rows_spin.setValue(preview)

        today = QDate.currentDate()
        start_str = self._settings.value(f"{root}/start_date", "", str)
        end_str = self._settings.value(f"{root}/end_date", "", str)

        start_date = QDate.fromString(start_str, Qt.ISODate) if start_str else self._start_date.date()
        end_date = QDate.fromString(end_str, Qt.ISODate) if end_str else self._end_date.date()

        if not start_date.isValid() or start_date > today:
            start_date = today.addDays(-7)
        if not end_date.isValid() or end_date > today:
            end_date = today
        if start_date > end_date:
            start_date = end_date

        self._start_date.blockSignals(True)
        self._end_date.blockSignals(True)
        self._start_date.setDate(start_date)
        self._end_date.setDate(end_date)
        self._start_date.setMaximumDate(end_date)
        self._end_date.setMinimumDate(start_date)
        self._start_date.blockSignals(False)
        self._end_date.blockSignals(False)

        selected_codes_key = f"{root}/selected_codes"
        if self._settings.contains(selected_codes_key):
            saved = self._settings.value(selected_codes_key, "", str)
            if saved:
                self._stored_selected_codes = [
                    code.strip().upper()
                    for code in saved.split(",")
                    if code.strip()
                ]
            else:
                self._stored_selected_codes = []
        else:
            self._stored_selected_codes = None  # Use defaults on first run.

    def _persist_preferences(self) -> None:
        """Persist the current UI preferences."""
        if self._loading_preferences:
            return

        root = self.SETTINGS_ROOT
        self._settings.setValue(f"{root}/dataset", self._dataset_combo.currentText())
        self._settings.setValue(f"{root}/units", self._units_combo.currentText())
        self._settings.setValue(f"{root}/preview_rows", self._preview_rows_spin.value())
        self._settings.setValue(
            f"{root}/start_date",
            self._start_date.date().toString(Qt.ISODate),
        )
        self._settings.setValue(
            f"{root}/end_date",
            self._end_date.date().toString(Qt.ISODate),
        )
        selected_codes = ",".join(self._selected_data_types())
        self._settings.setValue(f"{root}/selected_codes", selected_codes)
        self._stored_selected_codes = self._selected_data_types()

    # ------------------------------------------------------------------ Signals --
    def _on_dataset_changed(self, value: str) -> None:
        self._populate_data_type_options()
        self._update_field_labels()
        if not self._loading_preferences:
            self._reset_results(tr("Dataset changed. Fetch preview again to refresh."), notify=True)
            self._persist_preferences()

    def _on_units_changed(self, value: str) -> None:
        self._update_field_labels()
        if not self._loading_preferences:
            self._reset_results(tr("Units updated. Fetch preview again to refresh."), notify=True)
            self._persist_preferences()

    def _on_preview_rows_changed(self, value: int) -> None:
        if self._last_result:
            self._populate_preview(self._last_result)
        self._persist_preferences()

    def _on_data_type_item_changed(self, item: QListWidgetItem) -> None:
        if self._loading_preferences:
            return
        self._reset_results(tr("Field selection changed. Fetch preview again to refresh."), notify=True)
        self._persist_preferences()

    def _on_start_date_changed(self, qdate: QDate) -> None:
        self._end_date.blockSignals(True)
        if qdate > self._end_date.date():
            self._end_date.setDate(qdate)
        self._end_date.setMinimumDate(qdate)
        self._end_date.blockSignals(False)
        if not self._loading_preferences:
            self._reset_results(tr("Date range updated. Fetch preview again to refresh."), notify=True)
            self._persist_preferences()

    def _on_end_date_changed(self, qdate: QDate) -> None:
        today = QDate.currentDate()
        if qdate > today:
            qdate = today
            self._end_date.blockSignals(True)
            self._end_date.setDate(today)
            self._end_date.blockSignals(False)

        self._start_date.blockSignals(True)
        if qdate < self._start_date.date():
            self._start_date.setDate(qdate)
        self._start_date.setMaximumDate(qdate)
        self._start_date.blockSignals(False)

        if not self._loading_preferences:
            self._reset_results(tr("Date range updated. Fetch preview again to refresh."), notify=True)
            self._persist_preferences()

    # --------------------------------------------------------------- Data setup --
    def _populate_data_type_options(self) -> None:
        """Refresh the data type list for the current dataset and unit selection."""
        dataset_key = self._dataset_combo.currentText().lower()
        options = DATA_TYPE_CATALOG.get(dataset_key, [])
        defaults = [code.upper() for code in DEFAULT_DATA_TYPES.get(dataset_key, [])]

        self._base_type_catalog = {code.upper(): label for code, label in options}

        self._data_type_list.blockSignals(True)
        self._data_type_list.clear()

        if not options:
            placeholder = QListWidgetItem(tr("No curated fields available for this dataset yet."))
            placeholder.setFlags(Qt.NoItemFlags)
            self._data_type_list.addItem(placeholder)
            self._data_type_list.setEnabled(False)
            self._select_all_button.setEnabled(False)
            self._clear_selection_button.setEnabled(False)
            self._data_type_list.blockSignals(False)
            return

        self._data_type_list.setEnabled(True)
        self._select_all_button.setEnabled(True)
        self._clear_selection_button.setEnabled(True)

        stored_codes = self._stored_selected_codes
        applied_codes = (
            [code for code in stored_codes if code in self._base_type_catalog]
            if stored_codes is not None
            else defaults
        )
        if not applied_codes:
            applied_codes = defaults

        for code, _label in options:
            code_upper = code.upper()
            item = QListWidgetItem(tr(self._unit_label(code_upper)))
            item.setData(Qt.UserRole, code_upper)
            item.setToolTip(code_upper)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if code_upper in applied_codes else Qt.Unchecked)
            self._data_type_list.addItem(item)

        self._data_type_list.blockSignals(False)

    def _selected_data_types(self) -> List[str]:
        """Return the checked data type identifiers."""
        selections: List[str] = []
        for index in range(self._data_type_list.count()):
            item = self._data_type_list.item(index)
            if item.flags() & Qt.ItemIsUserCheckable and item.checkState() == Qt.Checked:
                code = item.data(Qt.UserRole)
                if code:
                    normalized = str(code).upper()
                    if normalized not in selections:
                        selections.append(normalized)
        return selections

    def _friendly_name_map(self, selected_codes: Sequence[str]) -> Dict[str, str]:
        """Build a mapping of column ids to user-friendly labels."""
        mapping: Dict[str, str] = {
            key.upper(): tr(label) for key, label in BASE_COLUMN_LABELS.items()
        }
        for code in selected_codes:
            label = self._unit_label(code.upper())
            mapping[code.upper()] = tr(label)
        return mapping

    def _unit_label(self, code: str) -> str:
        """Return the label for a data type given the chosen units."""
        dataset_key = self._dataset_combo.currentText().lower()
        unit_key = self._units_combo.currentText().lower()
        overrides = UNIT_LABEL_OVERRIDES.get(dataset_key, {}).get(unit_key, {})
        if code in overrides:
            return overrides[code]
        return self._base_type_catalog.get(code, code)

    def _update_field_labels(self) -> None:
        """Update list item labels to reflect current units."""
        self._data_type_list.blockSignals(True)
        for index in range(self._data_type_list.count()):
            item = self._data_type_list.item(index)
            if item.flags() & Qt.ItemIsUserCheckable:
                code = item.data(Qt.UserRole)
                if code:
                    item.setText(tr(self._unit_label(str(code))))
        self._data_type_list.blockSignals(False)

    def _select_all_data_types(self) -> None:
        """Mark every available data type as selected."""
        self._data_type_list.blockSignals(True)
        for index in range(self._data_type_list.count()):
            item = self._data_type_list.item(index)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Checked)
        self._data_type_list.blockSignals(False)
        if not self._loading_preferences:
            self._reset_results(tr("Field selection changed. Fetch preview again to refresh."), notify=True)
            self._persist_preferences()

    def _clear_data_type_selection(self) -> None:
        """Uncheck all data types."""
        self._data_type_list.blockSignals(True)
        for index in range(self._data_type_list.count()):
            item = self._data_type_list.item(index)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)
        self._data_type_list.blockSignals(False)
        if not self._loading_preferences:
            self._reset_results(tr("Field selection changed. Fetch preview again to refresh."), notify=True)
            self._persist_preferences()
    # -------------------------------------------------------------- Fetch workflow --
    def _on_fetch_clicked(self) -> None:
        """Trigger the ADS fetch workflow."""
        self._set_status(tr("Requesting data from ADS..."), "info")
        self._fetch_button.setEnabled(False)

        try:
            result = self._perform_fetch()
            self._last_result = result
            self._populate_preview(result)
            self._update_summary_stats(result)
            self._refresh_plot_sources(result)
            self._update_station_layer(result)
            self._save_button.setEnabled(True)
            self._record_count_label.setText(tr("Records: {count}").format(count=result.record_count))
            station_count = len(result.station_records) if result.station_records else len(result.stations)
            self._station_count_label.setText(tr("Stations: {count}").format(count=station_count))
            if station_count:
                preview_ids = ", ".join(result.stations[:8])
                self._station_count_label.setToolTip(preview_ids)
            else:
                self._station_count_label.setToolTip("")
            self._set_status(
                tr("Saved CSV to {path} (elapsed: {elapsed} ms)").format(
                    path=str(result.saved_csv),
                    elapsed=result.elapsed_ms,
                ),
                "success",
            )
            self._persist_preferences()
        except ADSRequestError as exc:
            self._reset_results()
            self._set_status(str(exc), "error")
        except Exception as exc:  # pylint: disable=broad-except
            self._reset_results()
            self._set_status(tr("Unexpected error: {msg}").format(msg=str(exc)), "error")
        finally:
            self._fetch_button.setEnabled(True)

    def _perform_fetch(self) -> ADSFetchResult:
        """Collect inputs and execute the ADS request."""
        dataset = self._dataset_combo.currentText()
        format_ = self._format_combo.currentText()
        start = self._start_date.date().toString("yyyy-MM-dd")
        end = self._end_date.date().toString("yyyy-MM-dd")
        bbox = (
            float(self._north.value()),
            float(self._west.value()),
            float(self._south.value()),
            float(self._east.value()),
        )
        selected_data_types = self._selected_data_types()
        dataset_key = dataset.lower()
        if not selected_data_types:
            selected_data_types = [code.upper() for code in DEFAULT_DATA_TYPES.get(dataset_key, [])]

        if not selected_data_types:
            raise ADSRequestError(tr("Select at least one data field before fetching."))

        units = self._units_combo.currentText()

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        if start_dt > end_dt:
            raise ADSRequestError(tr("Start date must be on or before the end date."))
        if end_dt > datetime.utcnow():
            raise ADSRequestError(tr("End date cannot be in the future."))

        if bbox[0] < bbox[2]:
            raise ADSRequestError(tr("North latitude must be greater than or equal to south latitude."))
        if bbox[3] < bbox[1]:
            raise ADSRequestError(tr("East longitude must be greater than or equal to west longitude."))

        return fetch_ads_dataset(
            dataset,
            start_date=start,
            end_date=end,
            bbox=bbox,
            data_types=selected_data_types,
            units=units,
            response_format=format_,
            limit_preview_rows=self._preview_rows_spin.value(),
            keep_columns=_compose_keep_columns(selected_data_types),
            friendly_names=self._friendly_name_map(selected_data_types),
            always_keep=ALWAYS_KEEP_COLUMNS,
        )

    def _populate_preview(self, result: ADSFetchResult) -> None:
        """Update the preview table with ADS results."""
        columns = list(result.columns)
        display_columns = list(result.display_columns)

        self._preview_table.setColumnCount(len(columns))
        self._preview_table.setHorizontalHeaderLabels(display_columns)

        preview_limit = self._preview_rows_spin.value()
        preview_rows = result.rows[:preview_limit]
        self._preview_table.setRowCount(len(preview_rows))

        for row_idx, row in enumerate(preview_rows):
            for col_idx, column in enumerate(columns):
                value = row.get(column, "")
                item = QTableWidgetItem("" if value is None else str(value))
                self._preview_table.setItem(row_idx, col_idx, item)

        self._preview_table.resizeColumnsToContents()

    # ------------------------------------------------------------ Summary stats --
    def _update_summary_stats(self, result: ADSFetchResult) -> None:
        """Populate the summary stats table."""
        numeric_columns: List[tuple[str, str, List[float]]] = []
        for column, label in zip(result.columns, result.display_columns):
            if column.upper() in ALWAYS_KEEP_COLUMNS:
                continue
            values = self._numeric_values(result, column)
            if values:
                numeric_columns.append((column, label, values))

        self._stats_table.setRowCount(len(numeric_columns))
        for row_idx, (column, label, values) in enumerate(numeric_columns):
            mean_val = sum(values) / len(values)
            stats_values = [
                label,
                str(len(values)),
                f"{mean_val:.2f}",
                f"{min(values):.2f}",
                f"{max(values):.2f}",
            ]
            for col_idx, text in enumerate(stats_values):
                item = QTableWidgetItem(text)
                self._stats_table.setItem(row_idx, col_idx, item)

        if not numeric_columns:
            self._stats_table.setRowCount(0)

        start_date = result.params.get("startDate", "")
        end_date = result.params.get("endDate", "")
        station_count = len(result.station_records) if result.station_records else len(result.stations)
        self._summary_label.setText(
            tr("Dataset: {dataset} | Date range: {start} -> {end} | Stations: {stations} | Records: {records}").format(
                dataset=result.dataset,
                start=start_date,
                end=end_date,
                stations=station_count,
                records=result.record_count,
            )
        )

    def _numeric_values(self, result: ADSFetchResult, column: str) -> List[float]:
        """Extract numeric values for the given column."""
        values: List[float] = []
        for row in result.rows:
            raw = row.get(column)
            if raw is None:
                continue
            raw_str = str(raw).strip()
            if not raw_str:
                continue
            try:
                values.append(float(raw_str))
            except ValueError:
                continue
        return values

    # ---------------------------------------------------------------- Plotting --
    def _refresh_plot_sources(self, result: ADSFetchResult) -> None:
        """Refresh the plot field selector based on available numeric columns."""
        numeric_columns = []
        for column, label in zip(result.columns, result.display_columns):
            if column.upper() in ALWAYS_KEEP_COLUMNS:
                continue
            if self._numeric_values(result, column):
                numeric_columns.append((column, label))

        self._plot_field_combo.blockSignals(True)
        self._plot_field_combo.clear()
        if not numeric_columns:
            self._plot_field_combo.addItem(tr("No numeric fields available"), "")
            self._plot_field_combo.setEnabled(False)
            self._plot_field_combo.blockSignals(False)
            self._clear_plot(tr("Select different fields or adjust the date range to view a plot."))
            return

        self._plot_field_combo.setEnabled(True)
        for column, label in numeric_columns:
            self._plot_field_combo.addItem(label, column)
        self._plot_field_combo.blockSignals(False)
        self._plot_field_combo.setCurrentIndex(0)
        self._update_plot()

    def _update_station_layer(self, result: ADSFetchResult) -> None:
        """Create or refresh a point layer for the fetched stations."""
        project = QgsProject.instance()

        # Remove any previous station layer.
        if self._station_layer_id:
            existing = project.mapLayer(self._station_layer_id)
            if existing:
                project.removeMapLayer(existing.id())
            self._station_layer_id = None

        if not result.station_records:
            return

        layer = QgsVectorLayer("Point?crs=EPSG:4326", tr("ADS Stations"), "memory")
        provider = layer.dataProvider()
        provider.addAttributes(
            [
                QgsField("station_id", QVariant.String),
                QgsField("name", QVariant.String),
                QgsField("latitude", QVariant.Double),
                QgsField("longitude", QVariant.Double),
                QgsField("dataset", QVariant.String),
                QgsField("start", QVariant.String),
                QgsField("end", QVariant.String),
            ]
        )
        layer.updateFields()

        features = []
        for record in result.station_records:
            feature = QgsFeature()
            feature.setGeometry(
                QgsGeometry.fromPointXY(QgsPointXY(record.longitude, record.latitude))
            )
            feature.setAttributes(
                [
                    record.id,
                    record.name,
                    record.latitude,
                    record.longitude,
                    result.dataset,
                    result.params.get("startDate", ""),
                    result.params.get("endDate", ""),
                ]
            )
            features.append(feature)

        provider.addFeatures(features)
        layer.updateExtents()

        added_layer = project.addMapLayer(layer)
        if added_layer:
            self._station_layer_id = added_layer.id()

    def _update_plot(self) -> None:
        """Update the matplotlib plot to match the selected field."""
        if not self._last_result:
            self._clear_plot()
            return

        column = self._plot_field_combo.currentData()
        if not column:
            self._clear_plot()
            return

        dates, values = self._aggregate_by_date(self._last_result, column)
        if not dates:
            self._clear_plot(
                tr("No numeric values found for {field}.").format(
                    field=self._plot_field_combo.currentText()
                )
            )
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.plot(dates, values, marker="o", linestyle="-")
        ax.set_title(self._plot_field_combo.currentText())
        ax.set_xlabel(tr("Date"))
        ax.set_ylabel(tr("Value"))
        ax.grid(True, linestyle="--", alpha=0.5)
        self._figure.autofmt_xdate()
        self._canvas.draw_idle()
        self._plot_status_label.setText("")

    def _aggregate_by_date(
        self,
        result: ADSFetchResult,
        column: str,
    ) -> tuple[List[datetime], List[float]]:
        """Aggregate numeric values by date (averaging across stations)."""
        values_by_date: defaultdict[datetime, List[float]] = defaultdict(list)
        for row in result.rows:
            date_str = row.get("DATE")
            value_str = row.get(column)
            if not date_str or not value_str:
                continue
            try:
                date_obj = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
                value = float(str(value_str).strip())
            except (ValueError, TypeError):
                continue
            values_by_date[date_obj].append(value)

        if not values_by_date:
            return [], []

        dates = sorted(values_by_date.keys())
        averaged = [sum(values_by_date[d]) / len(values_by_date[d]) for d in dates]
        return dates, averaged

    def _clear_plot(self, message: str | None = None) -> None:
        """Reset the plot area with an optional message."""
        self._figure.clear()
        self._canvas.draw_idle()
        self._plot_status_label.setText(message or tr("No plot available."))

    # -------------------------------------------------------------- Save output --
    def _handle_save_csv(self) -> None:
        """Copy the cached CSV to a user-selected location."""
        if not self._last_result:
            QMessageBox.information(
                self,
                tr("Nothing to Save"),
                tr("Run a fetch before saving the CSV output."),
            )
            return

        default_dir = self._settings.value(
            self.SETTINGS_EXPORT_DIR,
            str(Path.home()),
            str,
        )
        suggested_name = _default_export_name(
            self._last_result.dataset,
            self._last_result.params.get("startDate", ""),
            self._last_result.params.get("endDate", ""),
        )
        dialog_path = str(Path(default_dir) / suggested_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save ADS CSV"),
            dialog_path,
            tr("CSV Files (*.csv);;All Files (*.*)"),
        )
        if not file_path:
            return

        target = Path(file_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._last_result.saved_csv, target)
        except OSError as exc:
            QMessageBox.critical(
                self,
                tr("Save Failed"),
                tr("Could not save CSV: {error}").format(error=str(exc)),
            )
            return

        self._settings.setValue(self.SETTINGS_EXPORT_DIR, str(target.parent))
        self._set_status(
            tr("CSV exported to {path}").format(path=str(target)),
            "success",
        )

    # -------------------------------------------------------------- Map extent --
    def _populate_from_map_extent(self, *, initial: bool = False) -> None:
        """Set the bounding box inputs from the current map extent."""
        canvas = getattr(self._iface, "mapCanvas", lambda: None)()
        if not canvas:
            if not initial:
                self._set_status(tr("Map canvas is not available."), "error")
            return

        extent = canvas.extent()
        if extent.isEmpty():
            if not initial:
                self._set_status(tr("Map extent is empty; pan or zoom before using this option."), "error")
            return

        src_crs = canvas.mapSettings().destinationCrs()
        target_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        if not src_crs.isValid():
            src_crs = target_crs

        if src_crs != target_crs:
            try:
                transform = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())
                extent = transform.transformBoundingBox(extent)
            except Exception as exc:  # pylint: disable=broad-except
                if not initial:
                    self._set_status(
                        tr("Failed to transform extent to WGS84: {msg}").format(msg=str(exc)),
                        "error",
                    )
                return

        self._north.setValue(extent.yMaximum())
        self._south.setValue(extent.yMinimum())
        self._west.setValue(extent.xMinimum())
        self._east.setValue(extent.xMaximum())

        if not initial:
            self._set_status(tr("Bounding box populated from current map extent."), "success")

    # -------------------------------------------------------------- Reset state --
    def _reset_results(self, message: str | None = None, *, notify: bool = False) -> None:
        """Clear preview, stats, and plotting state."""
        self._last_result = None
        if self._station_layer_id:
            project = QgsProject.instance()
            existing = project.mapLayer(self._station_layer_id)
            if existing:
                project.removeMapLayer(existing.id())
            self._station_layer_id = None
        self._preview_table.setRowCount(0)
        self._preview_table.setColumnCount(0)
        self._preview_table.setHorizontalHeaderLabels([])
        self._stats_table.setRowCount(0)
        self._summary_label.clear()
        self._plot_field_combo.blockSignals(True)
        self._plot_field_combo.clear()
        self._plot_field_combo.addItem(tr("No numeric fields available"), "")
        self._plot_field_combo.setEnabled(False)
        self._plot_field_combo.blockSignals(False)
        self._clear_plot()
        self._record_count_label.setText(tr("Records: --"))
        self._station_count_label.setText(tr("Stations: --"))
        self._station_count_label.setToolTip("")
        self._save_button.setEnabled(False)
        if notify and not self._loading_preferences:
            self._set_status(message or tr("Parameters changed. Fetch preview again to refresh."), "info")

    # ----------------------------------------------------------------- Status --
    def _set_status(self, message: str, level: str) -> None:
        """Set user-facing status text with color-coding."""
        colors = {
            "info": "#2c5282",
            "success": "#2f855a",
            "error": "#c53030",
        }
        self._status_label.setStyleSheet(f"color: {colors.get(level, '#2d3748')};")
        self._status_label.setText(message)


def _compose_keep_columns(selected_codes: Sequence[str]) -> List[str]:
    """Return ordered list of columns to keep in ADS results."""
    ordered: List[str] = []
    for column in ALWAYS_KEEP_COLUMNS:
        column_upper = column.upper()
        if column_upper not in ordered:
            ordered.append(column_upper)
    for code in selected_codes:
        code_upper = code.upper()
        if code_upper not in ordered:
            ordered.append(code_upper)
    return ordered


def _default_export_name(dataset: str, start: str, end: str) -> str:
    """Generate a default filename for exported CSVs."""
    start_token = start.replace("-", "")
    end_token = end.replace("-", "")
    if start_token and end_token:
        return f"{dataset}_{start_token}-{end_token}.csv"
    return f"{dataset}_ads_export.csv"
