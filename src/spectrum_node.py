import math
import torch

# ====================== Core Spectrum Logic (Hybrid Chebyshev + Optional Taylor) ======================
class FastChebyshevForecaster:
    def __init__(self, m: int = 4, lam: float = 0.1):
        self.M = m
        self.K = max(m + 2, 8)  # Larger sliding window for better fits in short passes
        self.lam = lam
        self.H_buf = []
        self.T_buf = []
        self.shape = None
        self.dtype = None
        self.t_max = None  # Dynamic for varying workflows

    def _taus(self, t: float) -> float:
        total = getattr(self, 'total_steps', None) \
            or getattr(self, 't_max', None) \
            or getattr(self, 'estimated_total_steps', None) \
            or 30

        if total <= 0:
            total = 30
        return 2.0 * (t / total) - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        T = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if self.M > 0:
            T.append(taus)
            for _ in range(2, self.M + 1):
                T.append(2 * taus * T[-1] - T[-2])
        return torch.cat(T[:self.M + 1], dim=1)

    def update(self, cnt: int, h: torch.Tensor):
        if self.shape and h.shape != self.shape:
            self.reset_buffers()

        self.shape = h.shape
        self.dtype = h.dtype

        self.H_buf.append(h.view(-1))
        self.T_buf.append(self._taus(cnt))
        if len(self.H_buf) > self.K:
            self.H_buf.pop(0)
            self.T_buf.pop(0)

    def predict(self, cnt: int, w: float) -> torch.Tensor:
        if not self.H_buf: return torch.zeros(self.shape, device=torch.device("cpu")).to(self.dtype)
        device = self.H_buf[-1].device

        H = torch.stack(self.H_buf, dim=0).to(torch.float32)
        T = torch.tensor(self.T_buf, dtype=torch.float32, device=device)
        
        P = self.M + 1
        X = self._build_design(T)
        lamI = self.lam * torch.eye(P, device=device)
        XtX = X.T @ X + lamI

        try:
            L = torch.linalg.cholesky(XtX)
        except RuntimeError:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(P, device=device))

        XtH = X.T @ H
        coef = torch.cholesky_solve(XtH, L)

        tau_star = torch.tensor([self._taus(cnt)], device=device)
        pred_cheb = (self._build_design(tau_star) @ coef).squeeze(0)

        h_i = self.H_buf[-1]
        h_taylor = h_i + (h_i - self.H_buf[-2]) if len(self.H_buf) >= 2 else h_i

        # EXACT Official Logic: Always blend using `w`, even if history is incomplete.
        # No dynamic degree clamping, no value clamping.
        res = (1 - w) * h_taylor + w * pred_cheb
            
        return res.to(self.dtype).view(self.shape)

    def reset_buffers(self):
        self.H_buf.clear()
        self.T_buf.clear()
        self.t_max = None


# ====================== ComfyUI Node Wrapper (Batched with Masked/Vectorized/Streams) ======================
class SpectrumSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05}),  # 1.0 = pure Chebyshev
                "m": ("INT", {"default": 4, "min": 1, "max": 8}),
                "lam": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "window_size": ("INT", {"default": 3, "min": 1, "max": 10}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 20}),
                "stop_caching_step": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                "steps": ("INT", {"default": 30, "min": 10, "max": 500, "step": 1, "tooltip": "Temporary workaround: controls step count used for chebyshev. Match this value with your KSampler for consistent results."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling"

    def patch(self, model, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step, steps=30):
        self.total_steps = steps
        if hasattr(self, 'forecaster') and self.forecaster is not None:
            self.forecaster.t_max = steps
            self.forecaster.estimated_total_steps = steps

        state = getattr(model, 'spectrum_state', {})
        state['total_steps'] = steps
        model.spectrum_state = state

        state = {
            "forecasters": None,  
            "cnt": 0,
            "num_cached": [0],  
            "curr_ws": float(window_size),
            "last_t": -1,
            "total_runs": 0,
            "estimated_total_steps": steps,
        }
        
        # Remove any lingering hooks from previously bypassed models to clear global memory leaks
        diffusion_model = model.model.diffusion_model
        if hasattr(diffusion_model, "_sp_hooks"):
            for h in diffusion_model._sp_hooks: h.remove()
            diffusion_model._sp_hooks = []
        if hasattr(diffusion_model, "spectrum_hook_handles"):
            for h in diffusion_model.spectrum_hook_handles: h.remove()
            diffusion_model.spectrum_hook_handles = []

        forecast_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def spectrum_unet_wrapper(model_function, kwargs):
            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
            batch_size = x.shape[0]
            t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) and timestep.numel() > 0 else float(timestep)

            if t_scalar > state["last_t"]:
                state["forecasters"] = None
                state["cnt"] = 0
                state["num_cached"] = [0] * batch_size
                state["curr_ws"] = float(window_size)
                state["total_runs"] += 1
                print(f"[Spectrum] Detected new pass ({state['total_runs']}) - Reset state")

            state["last_t"] = t_scalar

            if state["forecasters"] is None:
                state["forecasters"] = [FastChebyshevForecaster(m=m, lam=lam) for _ in range(batch_size)]
                for f in state["forecasters"]:
                    f.t_max = steps
                    f.estimated_total_steps = steps
            
            if len(state["num_cached"]) != batch_size:
                state["num_cached"] = [0] * batch_size

            do_actual = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            for i in range(batch_size):
                is_micro_final = False
                if stop_caching_step == -1:
                    auto_stop = int(state["estimated_total_steps"] * 0.8)
                    if state["cnt"] >= auto_stop:
                        is_micro_final = True
                elif stop_caching_step > 0 and state["cnt"] >= stop_caching_step:
                    is_micro_final = True
                if state["cnt"] >= warmup_steps and not is_micro_final:
                    do_actual[i] = (state["num_cached"][i] + 1) % math.floor(state["curr_ws"]) == 0

            real_mask = do_actual
            forecast_mask = ~do_actual

            out = torch.empty_like(x)

            # ====================== REAL STEP: Run full diffusion_model → capture RAW 4D tensor (post final_layer/unpatchify) ======================
            if real_mask.any():
                x_real = x[real_mask]
                timestep_real = timestep[real_mask.to(timestep.device)] if isinstance(timestep, torch.Tensor) and timestep.shape[0] == batch_size else timestep
                c_real = {k: v[real_mask.to(v.device)] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v for k, v in c.items()}

                with torch.cuda.stream(torch.cuda.default_stream()):
                    raw_real = model_function(x_real, timestep_real, **c_real)  # ← RAW 4D tensor from diffusion_model

                # ComfyUI's Sampler automatically applies calculate_denoised externally on the UNet's return value
                out[real_mask] = raw_real

                # Update forecaster with the RAW tensor (Spatial Feature)
                real_indices = real_mask.nonzero().squeeze()
                if real_indices.dim() == 0:
                    real_indices = [real_indices.item()]
                else:
                    real_indices = real_indices.tolist()

                for i, idx in enumerate(real_indices):
                    state["forecasters"][idx].update(state["cnt"], raw_real[i])
                    state["num_cached"][idx] = 0

                print(f"[Spectrum] Step {state['cnt']}: Real forward (RAW captured) for {real_mask.sum().item()} items")

            # ====================== SKIP STEP: Forecast RAW tensor ======================
            if forecast_mask.any():
                forecast_indices = forecast_mask.nonzero().squeeze()
                if forecast_indices.dim() == 0:
                    forecast_indices = [forecast_indices.item()]
                else:
                    forecast_indices = forecast_indices.tolist()

                out_forecast = torch.empty((len(forecast_indices), *x.shape[1:]), device=x.device, dtype=x.dtype)

                if forecast_stream:
                    with torch.cuda.stream(forecast_stream):
                        for j, i in enumerate(forecast_indices):
                            raw_pred = state["forecasters"][i].predict(state["cnt"], w)  # forecasted RAW 4D tensor
                            out_forecast[j] = raw_pred

                        out[forecast_mask] = out_forecast
                        for i in forecast_indices:
                            state["num_cached"][i] += 1
                    torch.cuda.current_stream().wait_stream(forecast_stream)
                else:
                    # CPU fallback
                    for j, i in enumerate(forecast_indices):
                        raw_pred = state["forecasters"][i].predict(state["cnt"], w)
                        out_forecast[j] = raw_pred

                    out[forecast_mask] = out_forecast
                    for i in forecast_indices:
                        state["num_cached"][i] += 1

                print(f"[Spectrum] Step {state['cnt']}: Forecast (RAW predicted) for {forecast_mask.sum().item()} items")

            if state["cnt"] >= warmup_steps:
                state["curr_ws"] += flex_window

            state["cnt"] += 1
            return out

        new_model = model.clone()
        
        # SAFEGUARD: Deepcopy model_options to prevent the wrapper from permanently
        # mutating the globally cached CheckpointLoader model in memory.
        import copy
        if hasattr(model, 'model_options'):
            new_model.model_options = copy.deepcopy(model.model_options)
            
        new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)
        return (new_model,)

NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL }
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (Agnostic)"}