import math
import torch

def _tensor_debug_stats(tensor: torch.Tensor) -> str:
    if tensor is None or not torch.is_tensor(tensor):
        return "none"
    t = tensor.detach()
    flat = t.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return f"shape={tuple(t.shape)} finite=0"
    return (
        f"shape={tuple(t.shape)} mean={float(finite.mean()):.4f} "
        f"std={float(finite.std(unbiased=False)):.4f} min={float(finite.min()):.4f} "
        f"max={float(finite.max()):.4f} abs_max={float(finite.abs().max()):.4f}"
    )


# ====================== Spectrum with Residual Calibration ======================
class CaliberatedFastChebyshevForecaster2:
    def __init__(self, m: int = 4, lam: float = 0.1):
        self.M = m
        self.K = max(m + 2, 8)
        self.lam = lam
        self.H_buf = []
        self.T_buf = []
        self.shape = None
        self.dtype = None
        self.t_max = None
        self.residual = None
        self.last_raw_guess = None

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
        return torch.cat(T[: self.M + 1], dim=1)

    def update(self, cnt: int, h: torch.Tensor):
        if self.shape and h.shape != self.shape:
            self.reset_buffers()

        self.shape = h.shape
        self.dtype = h.dtype

        self.H_buf.append(h.detach().view(-1))
        self.T_buf.append(self._taus(cnt))
        if len(self.H_buf) > self.K:
            self.H_buf.pop(0)
            self.T_buf.pop(0)

    def predict(self, cnt: int, w: float, enable_calibration: bool = False, calibration_strength: float = 0.5, use_calibration: bool | None = None) -> torch.Tensor:
        if use_calibration is not None:
            enable_calibration = use_calibration

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
        raw_guess = (1 - w) * h_taylor + w * pred_cheb
            
        self.last_raw_guess = raw_guess.detach().clone()

        if enable_calibration and self.residual is not None:
            effective_residual = self.residual.to(device=device, dtype=torch.float32) * calibration_strength
            final_pred = raw_guess + effective_residual
        else:
            final_pred = raw_guess

        return final_pred.to(self.dtype).view(self.shape)

    def reset_buffers(self):
        self.H_buf.clear()
        self.T_buf.clear()
        self.shape = None
        self.dtype = None
        self.t_max = None
        self.residual = None
        self.last_raw_guess = None


# ====================== ComfyUI Node Wrapper ======================
class SpectrumSDXLCalibrated:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05}),
                "m": ("INT", {"default": 4, "min": 1, "max": 8}),
                "lam": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "window_size": ("INT", {"default": 3, "min": 1, "max": 10}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 20}),
                "stop_caching_step": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                "steps": ("INT", {"default": 30, "min": 10, "max": 500, "step": 1, "tooltip": "Temporary workaround: controls step count used for chebyshev. Match this value with your KSampler for consistent results."}),
                "enable_calibration": ("BOOLEAN", {"default": True}),
                "calibration_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling"

    def patch(self, model, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step, enable_calibration, calibration_strength, debug, steps=30):
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
            "debug": bool(debug),
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

        def _batch_index_tensor(mask: torch.Tensor) -> torch.Tensor:
            return mask.nonzero(as_tuple=False).flatten()

        def _slice_if_batch(value, index_tensor, batch_size):
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                return value[index_tensor.to(value.device)]
            return value

        def spectrum_unet_wrapper(model_function, kwargs):
            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
            batch_size = x.shape[0]
            if isinstance(timestep, torch.Tensor):
                t_scalar = timestep.flatten()[0].item()
            else:
                t_scalar = float(timestep)

            if t_scalar > state["last_t"]:
                state["forecasters"] = None
                state["cnt"] = 0
                state["num_cached"] = [0] * batch_size
                state["curr_ws"] = float(window_size)
                state["total_runs"] += 1
                print(f"[Spectrum Calibrated] Detected new pass ({state['total_runs']}) - Reset state")

            state["last_t"] = t_scalar

            if state["forecasters"] is None:
                state["forecasters"] = [CaliberatedFastChebyshevForecaster2(m=m, lam=lam) for _ in range(batch_size)]
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
                    do_actual[i] = (state["num_cached"][i] + 1) % max(1, math.floor(state["curr_ws"])) == 0

            real_mask = do_actual
            forecast_mask = ~do_actual

            out = torch.empty_like(x)

            if real_mask.any():
                real_indices = _batch_index_tensor(real_mask)
                x_real = x[real_mask]
                timestep_real = _slice_if_batch(timestep, real_indices, batch_size)
                c_real = {k: _slice_if_batch(v, real_indices, batch_size) for k, v in c.items()}

                out_real = model_function(x_real, timestep_real, **c_real)
                out[real_mask] = out_real

                real_indices_list = real_indices.tolist()
                for j, idx in enumerate(real_indices_list):
                    forecaster = state["forecasters"][idx]
                    if enable_calibration and forecaster.last_raw_guess is not None:
                        forecaster.residual = out_real[j].detach().view(-1).to(torch.float32) - forecaster.last_raw_guess
                    forecaster.update(state["cnt"], out_real[j])
                    state["num_cached"][idx] = 0

                if state["debug"]:
                    print(
                        f"[Spectrum Calibrated debug][actual] cnt={state['cnt']} step={state['total_runs']} "
                        f"real_items={real_mask.sum().item()} out_stats={_tensor_debug_stats(out_real.mean(dim=0))}"
                    )
                print(f"[Spectrum Calibrated] Step {state['cnt']}: Real forward for {real_mask.sum().item()} items")

            if forecast_mask.any():
                forecast_indices = _batch_index_tensor(forecast_mask).tolist()
                out_forecast = torch.empty((len(forecast_indices), *x.shape[1:]), device=x.device, dtype=x.dtype)

                if forecast_stream:
                    with torch.cuda.stream(forecast_stream):
                        for j, idx in enumerate(forecast_indices):
                            out_forecast[j] = state["forecasters"][idx].predict(state["cnt"], w, enable_calibration=enable_calibration, calibration_strength=calibration_strength)
                        out[forecast_mask] = out_forecast
                        for idx in forecast_indices:
                            state["num_cached"][idx] += 1
                    torch.cuda.current_stream().wait_stream(forecast_stream)
                else:
                    for j, idx in enumerate(forecast_indices):
                        out_forecast[j] = state["forecasters"][idx].predict(state["cnt"], w, enable_calibration=enable_calibration, calibration_strength=calibration_strength)
                    out[forecast_mask] = out_forecast
                    for idx in forecast_indices:
                        state["num_cached"][idx] += 1

                if state["debug"]:
                    print(
                        f"[Spectrum Calibrated debug][forecast] cnt={state['cnt']} forecast_items={forecast_mask.sum().item()} "
                        f"pred_stats={_tensor_debug_stats(out_forecast.mean(dim=0))}"
                    )
                print(f"[Spectrum Calibrated] Step {state['cnt']}: Forecast for {forecast_mask.sum().item()} items")

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

NODE_CLASS_MAPPINGS = {"SpectrumSDXLCalibrated": SpectrumSDXLCalibrated}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXLCalibrated": "Spectrum Adaptive Forecaster with Calibration (Agnostic)"}
