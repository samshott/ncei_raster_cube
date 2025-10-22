"""
Core plugin module for the NCEI Raster Cube QGIS plugin.
"""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .settings import TokenSettingsDialog


class NCEIRasterCubePlugin:
    """Primary plugin controller registered with QGIS."""

    def __init__(self, iface) -> None:
        """Store a reference to the QGIS application interface."""
        self.iface = iface
        self._action: QAction | None = None
        self._settings_action: QAction | None = None
        self._plugin_dir = Path(__file__).resolve().parent

    # --- QGIS lifecycle hooks -------------------------------------------------
    def initGui(self) -> None:
        """Create menu entries and toolbar icons."""
        self._action = QAction(
            self._icon(),
            self.tr("NCEI Raster Cube"),
            self.iface.mainWindow(),
        )
        self._action.triggered.connect(self.run)

        self._settings_action = QAction(
            self.tr("Settings..."),
            self.iface.mainWindow(),
        )
        self._settings_action.triggered.connect(self.open_settings_dialog)

        self.iface.addPluginToMenu(self.tr("&NCEI Raster Cube"), self._action)
        self.iface.addPluginToMenu(
            self.tr("&NCEI Raster Cube"), self._settings_action
        )
        self.iface.addToolBarIcon(self._action)

    def unload(self) -> None:
        """Remove menu entries and toolbar icons."""
        if self._action:
            self.iface.removePluginMenu(self.tr("&NCEI Raster Cube"), self._action)
            self.iface.removeToolBarIcon(self._action)
            self._action = None

        if self._settings_action:
            self.iface.removePluginMenu(
                self.tr("&NCEI Raster Cube"),
                self._settings_action,
            )
            self._settings_action = None

    # --- Plugin functionality -------------------------------------------------
    def run(self) -> None:
        """Entry point when the plugin action is triggered."""
        self.open_settings_dialog()

    def open_settings_dialog(self) -> None:
        """Display the settings dialog to manage API credentials."""
        dialog = TokenSettingsDialog(self.iface.mainWindow())
        dialog.exec_()

    # --- Helpers --------------------------------------------------------------
    def tr(self, message: str) -> str:
        """Return the localized string for the given message."""
        return QCoreApplication.translate("NCEIRasterCubePlugin", message)

    def _icon(self) -> QIcon:
        """Load the plugin icon with graceful fallback."""
        icon_path = self._plugin_dir / "resources" / "icon.svg"
        if icon_path.exists():
            return QIcon(str(icon_path))
        return QIcon()
