import torch
import math
import torch
import math

# ====================== Core Spectrum Logic (Stable & Batch Fix) ======================
class ChebyshevForecaster(torch.nn.Module):
    def __init__(self, M: int = 4, K: int = 100, lam: float = 0.1):
        super().__init__()
        self.M = M
        self.K = K
        self.lam = lam
        self.register_buffer("t_buf", torch.empty(0, dtype=torch.float32))
        self.register_buffer("H_buf", torch.empty(0, dtype=torch.float32))
        self._coef = None
        self._shape = None

    def _taus(self, t: torch.Tensor, total_steps: int) -> torch.Tensor:
        # Normalisasi ke [-1, 1] agar polinomial stabil
        return 2.0 * (t.float() / float(total_steps)) - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        K = taus.shape[0]
        T = [torch.ones((K, 1), device=taus.device, dtype=torch.float32)]
        if self.M == 0: return T[0]
        T.append(taus.unsqueeze(1))
        for m in range(2, self.M + 1):
            T.append(2 * taus.unsqueeze(1) * T[-1] - T[-2])
        return torch.cat(T[:self.M + 1], dim=1)

    def update(self, t: torch.Tensor, h: torch.Tensor, total_steps: int):
        # h: [B, C, H, W]
        # simpan bentuk asli untuk rekonstruksi
        self._shape = h.shape
        # Flatten tapi simpan dimensi Batch (PENTING!)
        h_flat = h.view(1, -1).float() 
        t_tau = self._taus(t, total_steps)
        
        # FIX: Reset jika total elemen berubah (Batch Size atau Resolusi berubah)
        if self.H_buf.numel() > 0 and h_flat.shape[1] != self.H_buf.shape[1]:
            self.t_buf = torch.empty(0, dtype=torch.float32, device=h.device)
            self.H_buf = torch.empty(0, dtype=torch.float32, device=h.device)

        if self.t_buf.numel() == 0:
            self.t_buf = t_tau
            self.H_buf = h_flat
        else:
            self.t_buf = torch.cat([self.t_buf, t_tau])
            self.H_buf = torch.cat([self.H_buf, h_flat])
            if self.t_buf.shape[0] > self.K:
                self.t_buf, self.H_buf = self.t_buf[-self.K:], self.H_buf[-self.K:]
        self._coef = None

    def predict(self, t_star: torch.Tensor, total_steps: int) -> torch.Tensor:
        if self._coef is None:
            taus = self.t_buf.flatten()
            X = self._build_design(taus)
            lamI = self.lam * torch.eye(X.shape[1], device=X.device)
            # Proteksi Ridge Regression (Jitter) agar tidak meledak ke HITAM
            XtX = X.T @ X + lamI
            try:
                L = torch.linalg.cholesky(XtX)
            except RuntimeError:
                jitter = 1e-6 * XtX.diag().mean()
                L = torch.linalg.cholesky(XtX + jitter * torch.eye(X.shape[1], device=X.device))
            
            self._coef = torch.cholesky_solve(X.T @ self.H_buf, L)

        tau_star = self._taus(t_star, total_steps).flatten()
        x_star = self._build_design(tau_star)
        return (x_star @ self._coef).view(self._shape)

class SpectrumForecaster(torch.nn.Module):
    def __init__(self, M=4, K=100, lam=0.1, w=0.5):
        super().__init__()
        self.cheb = ChebyshevForecaster(M, K, lam)
        self.w = w

    def predict(self, t_star: torch.Tensor, total_steps: int, last_h: torch.Tensor) -> torch.Tensor:
        h_cheb = self.cheb.predict(t_star, total_steps)
        # Gunakan 'last_h' untuk stabilitas (Anti-NaN) tapi dengan blending Spectrum
        # Res direkomendasikan memakai w yang tinggi (0.8) agar tidak 'memudar'
        return (1 - self.w) * last_h + self.w * h_cheb

# ====================== ComfyUI Node ======================
class SpectrumSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "m": ("INT", {"default": 4, "min": 1, "max": 8}),
                "lam": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 10}),
                "flex_window": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 20}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling"

    def patch(self, model, w, m, lam, window_size, flex_window, warmup_steps):
        state = {
            "forecaster": None,
            "cnt": 0,
            "num_cached": 0,
            "curr_ws": float(window_size),
            "last_t": None,
            "start_t": None,
            "warmup": warmup_steps,
            "w": w, "m": m, "lam": lam,
            "flex": flex_window,
            "window_size": window_size,
        }

        def spectrum_unet_wrapper(model_function, kwargs):
            x = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]

            t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)
            
            # Reset jika sampling baru
            if state["last_t"] is None or t_scalar > state["last_t"] + 100:
                state["cnt"] = 0
                state["num_cached"] = 0
                state["curr_ws"] = float(state["window_size"])
                state["forecaster"] = None
                state["start_t"] = t_scalar

            state["last_t"] = t_scalar

            # 10% Final Quality Guard (Berhenti caching di akhir biar detail tajam)
            is_final_phase = False
            if state["start_t"] is not None:
                is_final_phase = t_scalar < (state["start_t"] * 0.15) # 15% terakhir paksa real UNet

            do_actual = True
            if state["cnt"] >= state["warmup"] and not is_final_phase:
                do_actual = (state["num_cached"] + 1) % math.floor(state["curr_ws"]) == 0

            if do_actual:
                out = model_function(x, timestep, **c)
                if state["forecaster"] is None:
                    state["forecaster"] = SpectrumForecaster(M=state["m"], lam=state["lam"], w=state["w"])
                
                state["forecaster"].cheb.update(
                    torch.tensor([state["cnt"]], device=x.device),
                    out,
                    60 # Normalisasi index
                )
                
                if state["cnt"] >= state["warmup"]:
                    state["curr_ws"] += state["flex"]
                state["num_cached"] = 0
            else:
                # Ambil laten terakhir yang asli untuk blending stabil
                last_h = state["forecaster"].cheb.H_buf[-1].view(state["forecaster"].cheb._shape)
                out = state["forecaster"].predict(
                    torch.tensor([state["cnt"]], device=x.device),
                    60,
                    last_h
                ).to(x.dtype)
                state["num_cached"] += 1

            state["cnt"] += 1
            return out

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)
        return (new_model,)

NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)"}
NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)"}
