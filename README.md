\# ComfyUI Spectrum SDXL Node



This repository contains a ComfyUI custom node implementing the \*\*Spectrum\*\* sampling acceleration technique, tailored specifically for SDXL models. Spectrum is a training-free method that forecasts spectral features using Chebyshev polynomials and ridge regression to skip redundant UNet computations, achieving significant speed-ups with minimal quality loss.



---



\## Key Features



\* \*\*Sampling Acceleration\*\* – Reduce inference time (up to ~2× on SDXL) by skipping UNet evaluations on selected timesteps.

\* \*\*Batch Compatibility \& Mega‑Speed\*\* – Full batch support with latent‑shape detection; when used with large batches the node can deliver up to \*\*4× overall throughput\*\* by amortizing the cost of forecasting over multiple images.

\* \*\*FP8 Tensor‑Core Support\*\* – Compatible to run on NVIDIA Tensor Cores in FP8 mode, providing additional speed gains on compatible hardware. Works seamlessly alongside other optimizations.

\* \*\*Sage‑Attention Friendly\*\* – Orthogonal to Sage Attention; you can enable them together without conflicts.

\* \*\*Stability Enhancements\*\* – Includes jitter and anti‑NaN safeguards to keep low‑precision runs (FP16/FP8) stable.

\* \*\*Final Quality Guard\*\* – Disables forecasting during the final ~15 % of steps to preserve fine details in the output.



\## Installation



1\. Navigate to the `custom\_nodes` directory of your ComfyUI installation.

2\. Clone this repository:

&nbsp;  ```bash

&nbsp;  git clone https://github.com/ruwwww/comfyui-spectrum-sdxl

&nbsp;  ```

3\. Restart ComfyUI.



\## Parameters



| Parameter      | Description |

|----------------|-------------|

| `w`            | Blending weight between predicted and last true features. High values (≈0.8) help maintain contrast. |

| `m`            | Number of Chebyshev polynomial basis functions (forecast complexity). |

| `lam`          | Ridge regularization strength. Prevents latent explosion/black outputs in low‑precision modes. |

| `window\_size`  | Initial forecasting window size (number of skipped steps). |

| `flex\_window`  | Increment added to the window after each actual UNet pass. |

| `warmup\_steps` | Number of initial full‑model steps before forecasting begins. |



\## Recommended Settings



Based on empirical tests with SDXL:



\* `w` = 0.80

\* `m` = 4

\* `lam` = 0.50

\* `window\_size` = 2

\* `flex\_window` = 0.25

\* `warmup\_steps` = 5



Adjust these values as needed for your workload and model precision.



\## Credits \& References



This node implements ideas from the paper:



> \*\*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration\*\*

> Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon

> Stanford University \& ByteDance



\* Paper: https://arxiv.org/abs/2603.01623

\* Project page: https://hanjq17.github.io/Spectrum/

\* Official code: https://github.com/hanjq17/Spectrum



\### Citation



If you use this node in your research, please cite the original paper:



```bibtex

@article{han2026adaptive,

&nbsp; title={Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration},

&nbsp; author={Han, Jiaqi and Shi, Juntong and Li, Puheng and Ye, Haotian and Guo, Qiushan and Ermon, Stefano},

&nbsp; journal={arXiv preprint arXiv:2603.01623},

&nbsp; year={2026}

}

```





