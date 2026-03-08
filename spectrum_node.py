import torch
import math

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
        return (t / (self.t_max or 50.0)) * 2.0 - 1.0  # Auto-scale based on observed max

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
        device = self.H_buf[-1].device

        H = torch.stack(self.H_buf, dim=0).to(torch.float32)
        T = torch.tensor(self.T_buf, dtype=torch.float32, device=device)

        X = self._build_design(T)
        lamI = self.lam * torch.eye(self.M + 1, device=device)
        XtX = X.T @ X + lamI

        try:
            L = torch.linalg.cholesky(XtX)
        except RuntimeError:
            jitter = 1e-5 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(self.M + 1, device=device))

        XtH = X.T @ H
        coef = torch.cholesky_solve(XtH, L)

        tau_star = torch.tensor([self._taus(cnt)], device=device)
        x_star = self._build_design(tau_star)

        pred_cheb = (x_star @ coef).squeeze(0)

        if len(self.H_buf) >= 2:
            h_i = self.H_buf[-1].to(torch.float32)
            h_im1 = self.H_buf[-2].to(torch.float32)
            h_taylor = h_i + 0.5 * (h_i - h_im1)
        else:
            h_taylor = self.H_buf[-1].to(torch.float32)

        res = (1 - w) * h_taylor + w * pred_cheb
        return torch.clamp(res, -10.0, 10.0).to(self.dtype).view(self.shape)  # Milder for SDXL

    def reset_buffers(self):
        self.H_buf.clear()
        self.T_buf.clear()
        self.t_max = None


# ====================== ComfyUI Node Wrapper (Improved for Hiresfix/Multi-Pass) ======================
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
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling"

    def patch(self, model, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step):
        state = {
            "forecaster": None,
            "cnt": 0,
            "num_cached": 0,
            "curr_ws": float(window_size),
            "last_t": -1,  # Init low to force first reset
            "total_runs": 0,  # Debug multi-pass
            "estimated_total_steps": 50  # Default, updated dynamically
        }

        def spectrum_unet_wrapper(model_function, kwargs):
            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
            t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)

            # Improved reset for hiresfix: Reset if t_scalar > last_t (new pass starts high)
            if t_scalar > state["last_t"]:
                if state["forecaster"]:
                    state["forecaster"].reset_buffers()
                state["cnt"] = 0
                state["num_cached"] = 0
                state["curr_ws"] = float(window_size)
                state["forecaster"] = None  # Full re-init
                state["total_runs"] += 1
                print(f"[Spectrum] Detected new pass ({state['total_runs']}) - Reset state")

            state["last_t"] = t_scalar

            # Update estimated total steps from t_max (for auto stop)
            if state["forecaster"] and state["forecaster"].t_max:
                state["estimated_total_steps"] = int(state["forecaster"].t_max) + 10  # Safe buffer

            is_micro_final = False
            if stop_caching_step == -1:
                # Auto: stop at ~80% steps
                auto_stop = int(state["estimated_total_steps"] * 0.8)
                if state["cnt"] >= auto_stop:
                    is_micro_final = True
            elif stop_caching_step > 0 and state["cnt"] >= stop_caching_step:
                is_micro_final = True

            do_actual = True
            if state["cnt"] >= warmup_steps and not is_micro_final:
                do_actual = (state["num_cached"] + 1) % math.floor(state["curr_ws"]) == 0

            if do_actual:
                out = model_function(x, timestep, **c)
                if state["forecaster"] is None:
                    state["forecaster"] = FastChebyshevForecaster(m=m, lam=lam)

                state["forecaster"].update(state["cnt"], out)
                if state["cnt"] >= warmup_steps:
                    state["curr_ws"] += flex_window
                state["num_cached"] = 0
                print(f"[Spectrum] Step {state['cnt']}: Real forward")
            else:
                out = state["forecaster"].predict(state["cnt"], w=w).to(x.dtype)
                state["num_cached"] += 1
                print(f"[Spectrum] Step {state['cnt']}: Forecast (cached {state['num_cached']})")

            state["cnt"] += 1
            return out

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)
        return (new_model,)

NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)"}
