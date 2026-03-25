from .src.spectrum_node import SpectrumSDXL
from .src.calibrated_spectrum_node import SpectrumSDXLCalibrated

NODE_CLASS_MAPPINGS = {
    "SpectrumSDXL": SpectrumSDXL,
    "SpectrumSDXLCalibrated": SpectrumSDXLCalibrated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumSDXL": "Spectrum Adaptive Forecaster (Agnostic)",
    "SpectrumSDXLCalibrated": "Spectrum Adaptive Forecaster with Calibration (Agnostic)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]