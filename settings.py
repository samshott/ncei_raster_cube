"""
Settings dialog for managing the NCEI Raster Cube plugin configuration.
"""

from __future__ import annotations

from typing import Tuple

from qgis.PyQt.QtCore import QCoreApplication, QEventLoop, QSettings, QTimer, Qt, QUrl
from qgis.PyQt.QtNetwork import QNetworkReply, QNetworkRequest
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsNetworkAccessManager


def tr(message: str) -> str:
    """Translate helper for the settings dialog."""
    return QCoreApplication.translate("NCEIRasterCubeSettings", message)


class TokenSettingsDialog(QDialog):
    """Dialog for storing and validating the NOAA NCEI CDO token."""

    SETTINGS_KEY = "ncei_raster_cube/cdo_token"
    VALIDATION_URL = QUrl("https://www.ncei.noaa.gov/cdo-web/api/v2/datasets?limit=1")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("NCEI Token Settings"))
        self.setModal(True)

        self._settings = QSettings()

        self._token_input = QLineEdit()
        self._token_input.setEchoMode(QLineEdit.Password)
        self._token_input.setPlaceholderText(tr("Enter NOAA NCEI CDO API token"))
        self._token_input.setText(self._settings.value(self.SETTINGS_KEY, "", str))

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setTextFormat(Qt.PlainText)

        self._validate_button = QPushButton(tr("Validate"))
        self._validate_button.clicked.connect(self._on_validate_clicked)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self._button_box.accepted.connect(self._on_save_clicked)
        self._button_box.rejected.connect(self.reject)

        form_layout = QFormLayout()
        form_layout.addRow(tr("CDO API Token"), self._token_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._validate_button)
        button_layout.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self._status_label)
        layout.addWidget(self._button_box)

        self.setLayout(layout)

    # --- Slots -----------------------------------------------------------------
    def _on_validate_clicked(self) -> None:
        """Validate the token against the NOAA CDO API."""
        token = self._token_input.text().strip()
        success, message = self._validate_token(token)
        self._update_status(success, message)

    def _on_save_clicked(self) -> None:
        """Persist token to QSettings and close the dialog."""
        token = self._token_input.text().strip()
        self._settings.setValue(self.SETTINGS_KEY, token)
        self._update_status(True, tr("Token saved to QGIS user settings."))
        self.accept()

    # --- Internals -------------------------------------------------------------
    def _validate_token(self, token: str) -> Tuple[bool, str]:
        """Attempt an authenticated request using the provided token."""
        if not token:
            return False, tr("Token is required for validation.")

        nam = QgsNetworkAccessManager.instance()
        request = QNetworkRequest(self.VALIDATION_URL)
        request.setRawHeader(b"token", token.encode("ascii", "ignore"))

        reply = nam.get(request)

        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        reply.finished.connect(loop.quit)

        timeout_ms = 15000
        timer.start(timeout_ms)
        loop.exec_()

        if timer.isActive():
            timer.stop()
            status_code = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
            if reply.error() == QNetworkReply.NoError and status_code == 200:
                reply.deleteLater()
                return True, tr("Token is valid.")

            if status_code in (401, 403):
                message = tr("Token rejected (HTTP {code}). Verify and try again.").format(
                    code=int(status_code)
                )
            elif status_code == 429:
                message = tr("Rate limit exceeded (HTTP 429). Wait before retrying.")
            else:
                message = tr(
                    "Validation failed with HTTP {code}: {error}"
                ).format(code=int(status_code or 0), error=reply.errorString())

            reply.deleteLater()
            return False, message

        # Timeout triggered
        reply.abort()
        reply.deleteLater()
        return False, tr("Validation timed out after {seconds} seconds.").format(
            seconds=timeout_ms / 1000
        )

    def _update_status(self, success: bool, message: str) -> None:
        """Display feedback to the user."""
        color = "#2f855a" if success else "#c53030"
        self._status_label.setStyleSheet(f"color: {color};")
        self._status_label.setText(message)
