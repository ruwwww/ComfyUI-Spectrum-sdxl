# ComfyUI Spectrum SDXL Node

This repository contains a Unnoficial ComfyUI custom node implementing the **Spectrum** sampling acceleration technique, tailored specifically for SDXL models. Spectrum is a training-free method that forecasts spectral features using Chebyshev polynomials and ridge regression to skip redundant UNet computations, achieving significant speed-ups with less quality loss.
> **Technical Note:** This implementation has been corrected to be  mathematically faithful to the core Chebyshev regression described in the official Spectrum paper. The node now should work natively across most architectural families (including SDXL, DiTs like Flux, Cosmos, or Anima), it correctly forecasts the **raw spatial model predictions / features / latents** (prior to any sampler ODE integration)

> **New**: Calibrated Spectrum mode recovers washed-out details by applying **residual correction** after each real forward pass and blending it into forecasts.
>
> **Performance comparison** — **Download the images below and drag them into ComfyUI to instantly load optimized workflows!**
>
>> **SDXL Calibrated Comparisons (30-step Euler)**
>
> | Normal | Spectrum | Calibrated (strength 0.5) | Calibrated (strength 0.8) |
> | :----: | :------: | :----------------------: | :----------------------: |
> | ![Normal](/images/no_cache1.png) | ![Spectrum](/images/spectrum1.png) | ![Cal0.5](/images/calibrated1-s-0.5.png) | ![Cal0.8](/images/calibrated1-s-0.8.png) |
> | **8.8 s** | **4.8 s** | **4.8 s** | **4.8 s** |
> | ![Normal2](/images/no_cache2.png) | ![Spectrum2](/images/spectrum2.png) | ![Cal0.5_2](/images/calibrated2-s-0.5.png) | ![Cal0.8_2](/images/calibrated2-s-0.8.png) |
> | **8.3 s** | **4.6 s** | **4.6 s** | **4.6 s** |
>> **Anima (30-step Euler)**
>
> |                   Default                   |                   Spectrum                    |
> | :-----------------------------------------: | :-------------------------------------------: |
> | ![Default Anima](/images/default-anima.png) | ![Spectrum Anima](/images/spectrum-anima.png) |
> |                 **23.67 s**                 |                  **13.01 s**                  |

---

## Key Features

- **Sampling Acceleration** – Reduce inference time (up to ~2× on SDXL) by skipping UNet evaluations on selected timesteps.
- **Vectorized Batch Processing** – Fully vectorized mathematical operations process conditional and unconditional latents independently. This prevents memory contamination (rainbow artifacts) and sustains ultra-high `it/s` speeds without Python loop bottlenecks.
- **FP8 Tensor-Core Support** – Compatible to run on NVIDIA Tensor Cores in FP8 mode, providing additional speed gains on compatible hardware. Works seamlessly alongside other optimizations.
- **Sage-Attention Friendly** – Orthogonal to Sage Attention; you can enable them together without conflicts.
- **Stability Enhancements** – Incorporates a "Sliding Window" memory limit (prevents mathematical explosions), alongside jitter and anti-NaN safeguards to keep low-precision runs (FP16/FP8) perfectly stable.
- **Step-Based Quality Guard** – A dedicated parameter allows you to seamlessly turn off the forecaster during the final critical steps, preserving high-frequency micro-details (like skin texture and eyes) and eliminating VAE-like compression blur.

## Installation

1. Navigate to the `custom_nodes` directory of your ComfyUI installation.
2. Clone this repository:

```bash
git clone https://github.com/ruwwww/comfyui-spectrum-sdxl

```

3. Restart ComfyUI.

## Parameters

| Parameter               | Description                                                                                                                                                                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`w`**                 | Blending weight between predicted and last true features. Lower values (0.4–0.5) rely more on local momentum, preserving sharpness, while higher values rely on global spectral smoothing. setting `w` to 0 means using Local Taylor based approximation effectively ignoring the global smoothing parameters (`m` and `lam`).                                                                                                    |
| **`m`**                 | Number of Chebyshev polynomial basis functions (forecast complexity). Lower values (3) are generally more stable for short SDXL runs.                                                                                                                                                               |
| **`lam`**               | Ridge regularization strength ($\lambda$).                                                                                                                                                 |
| **`window_size`**       | Initial forecasting window size (number of skipped steps).                                                                                                                                                                                                                                          |
| **`flex_window`**       | Increment added to the window after each actual UNet pass. Higher values result in aggressive acceleration.                                                                                                                                                                                         |
| **`warmup_steps`**      | Number of initial full-model steps before forecasting begins. Gives the model time to establish composition.                                                                                                                                                                                        |
| **`stop_caching_step`** | The exact step count where Spectrum stops accelerating and hands rendering back to the native UNet. Essential for recovering fine details. (e.g., If rendering 25 total steps, set to `22` to let the original UNet render the final 3 steps). Set to `100` to disable the guard for maximum speed. |
| **`steps`** | Additional manual passthrough of KSampler step count into the model forecaster (`t_max` / `_taus` normalization). Set to match your KSampler total steps for stable forecast accuracy and drift reduction. |

## Recommended Settings

For the best balance of **extreme speed** and **high-definition sharpness** (no blur/artifacts), use the following configuration (assuming a standard 25-step generation):

- **`w`**: `0.30`
- **`m`**: `3`
- **`lam`**: `0.1`
- **`window_size`**: `2`
- **`flex_window`**: `0.25`  
- **`warmup_steps`**: `8` _(DiT models need higher warmup steps eg 8-10)_
- **`stop_caching_step`**: `22` _(Always set this to Total Steps minus ~3)_

Adjust `flex_window` higher if you want to push speeds further, or lower if you notice structural degradation. 

## Credits & References

This node implements ideas from the paper:

> **Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration**
> Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon
> Stanford University & ByteDance

- **Paper:** [https://arxiv.org/abs/2603.01623](https://arxiv.org/abs/2603.01623)
- **Project Page:** [https://hanjq17.github.io/Spectrum/](https://hanjq17.github.io/Spectrum/)
- **Official Code:** [https://github.com/hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)

### Citation

If you use this node in your research, please cite the original paper:

```bibtex
@article{han2026adaptive,
  title={Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration},
  author={Han, Jiaqi and Shi, Juntong and Li, Puheng and Ye, Haotian and Guo, Qiushan and Ermon, Stefano},
  journal={arXiv preprint arXiv:2603.01623},
  year={2026}
}

```




