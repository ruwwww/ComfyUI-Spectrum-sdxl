from .spectrum_node import SpectrumSDXL

NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
