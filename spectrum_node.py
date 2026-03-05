
import torch
import math

# ====================== Core Spectrum Logic (Uncapped Speed & Sliding Window) ======================
class FastChebyshevForecaster:
    def __init__(self, m: int = 4, lam: float = 1.0):
        self.M = m
        self.K = max(m + 1, 6) 
        self.lam = lam
        self.H_buf = []  
        self.T_buf = []
        self.shape = None
        self.dtype = None

    def _taus(self, t: float) -> float:
        return (t / 50.0) * 2.0 - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        T = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if self.M > 0:
            T.append(taus)
            for _ in range(2, self.M + 1):
                T.append(2 * taus * T[-1] - T[-2])
        return torch.cat(T[:self.M + 1], dim=1)

    def update(self, cnt: int, h: torch.Tensor):
        if self.shape is not None and h.shape != self.shape:
            self.H_buf.clear()
            self.T_buf.clear()
        
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
            jitter = 1e-4 * XtX.diag().mean()
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
        return torch.clamp(res, -15.0, 15.0).to(self.dtype).view(self.shape)

# ====================== ComfyUI Node Wrapper ======================
class SpectrumSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "m": ("INT", {"default": 3, "min": 1, "max": 8}),
                "lam": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 10}),
                "flex_window": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 2.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 4, "min": 0, "max": 20}),
                # TODO: USE SIGMA BASED STOP CACHING
                "stop_caching_step": ("INT", {"default": 22, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling"

    def patch(self, model, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step):
        state = {
            "forecaster": None, "cnt": 0, "num_cached": 0, "curr_ws": float(window_size),
            "last_t": None
        }

        def spectrum_unet_wrapper(model_function, kwargs):
            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
            t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)
            
            # Deteksi start run baru tetap menggunakan t_scalar karena sangat handal
            if state["last_t"] is None or t_scalar > state["last_t"] + 10:
                state.update({
                    "cnt": 0, "num_cached": 0, "curr_ws": float(window_size), 
                    "forecaster": None
                })
            state["last_t"] = t_scalar
            
            is_micro_final = False
            if stop_caching_step > 0 and state["cnt"] >= stop_caching_step:
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
            else:
                out = state["forecaster"].predict(state["cnt"], w=w).to(x.dtype)
                state["num_cached"] += 1

            state["cnt"] += 1
            return out

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)
        return (new_model,)

NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)"}
