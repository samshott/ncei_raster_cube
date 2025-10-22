"""
QGIS plugin entry point for the NCEI Raster Cube project.
"""


def classFactory(iface):  # pylint: disable=invalid-name
    """Load NCEIRasterCubePlugin from this package."""
    from .ncei_raster_cube import NCEIRasterCubePlugin

    return NCEIRasterCubePlugin(iface)
