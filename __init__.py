from .src.spectrum_node import SpectrumSDXL
from .src.calibrated_spectrum_node import SpectrumSDXLCalibrated
from .src.old_spectrum_node import SpectrumSDXLOld
from .src.old_calibrated_spectrum_node import SpectrumSDXLCalibratedOld

NODE_CLASS_MAPPINGS = {
    "SpectrumSDXL": SpectrumSDXL,
    "SpectrumSDXLCalibrated": SpectrumSDXLCalibrated,
    "SpectrumSDXLOld": SpectrumSDXLOld,
    "SpectrumSDXLCalibratedOld": SpectrumSDXLCalibratedOld,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumSDXL": "Spectrum Adaptive Forecaster (Agnostic)",
    "SpectrumSDXLCalibrated": "Spectrum Adaptive Forecaster with Calibration (Agnostic)",
    "SpectrumSDXLOld": "Spectrum Adaptive Forecaster (Old/Stability)",
    "SpectrumSDXLCalibratedOld": "Spectrum Adaptive Forecaster with Calibration (Old/Stability)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]