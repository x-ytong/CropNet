import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset



class CropGlobeDataset:
    def __init__(
        self,
        dayItv: int = 5,
        staDOY: int = 121,
        endDOY: int = 334,
        aug_mode="none",                   # "none" or "augm"
        re__peat=2,
        aug_prob: tuple = (1, 0.5, 0.25),  # (timeshift, timescale, mg_warping)
        modality: str = 'S2',
        # augmentation config
        shift_days: int = 50,              # TimeShift
        scale_days: int = 50,              # TimeScale
        warp_sigma: float = 0.10,          # MagnitudeWarping
        warp_knots: int = 5,               # MagnitudeWarping
    ):
        super().__init__()

        self.dayItv = int(dayItv)
        self.staDOY = int(staDOY)
        self.endDOY = int(endDOY)

        self.shift_days = shift_days
        self.scale_days = scale_days
        self.warp_sigma = warp_sigma
        self.warp_knots = warp_knots

        self.base_sta_idx = (staDOY - 1) // self.dayItv
        self.base_end_idx = (endDOY - 1) // self.dayItv + 1
        self.target_len = self.base_end_idx - self.base_sta_idx

        self.modality = modality.upper()
        if self.modality not in ["S2", "S12"]:
            raise ValueError(f"Unsupported modality: {self.modality}")

        if aug_mode not in ["none", "augm"]:
            raise ValueError(f"Unsupported aug_mode: {aug_mode}")
        self.aug_mode = aug_mode

        self.p_timeshift = float(aug_prob[0])
        self.p_timescale = float(aug_prob[1])
        self.p_mgwarping = float(aug_prob[2])

        self.re__peat = re__peat
        self.rng = np.random.default_rng(seed=None)

    def __call__(self, dataset):
        x, y = dataset.tensors
        x = x.squeeze(1)

        if self.aug_mode == "none":
            fea = x[:, :, self.base_sta_idx:self.base_end_idx]
            return TensorDataset(fea.unsqueeze(1), y)
        elif self.aug_mode == "augm":
            out_x = []
            out_y = []
            for _ in range(self.re__peat):
                for i in tqdm(range(len(y)), total=len(y), desc="Augmentation"):
                    fea = x[i]  # [band, time]
                    lbl = y[i]

                    full_fea = fea
                    fea = full_fea[:, self.base_sta_idx:self.base_end_idx]

                    if self.rng.random() < self.p_timeshift:
                        fea = self._timeshift(full_fea)
                    if self.rng.random() < self.p_timescale:
                        fea = self._timescale(full_fea)
                    if self.rng.random() < self.p_mgwarping:
                        fea = self._mg_warping(fea)

                    out_x.append(fea)
                    out_y.append(lbl)

            base_x = x[:, :, self.base_sta_idx:self.base_end_idx]
            base_y = y
            out_x = torch.cat([base_x, torch.stack(out_x, dim=0)], dim=0)  # [N, band, time]
            out_y = torch.cat([base_y, torch.stack(out_y, dim=0)], dim=0)

            return TensorDataset(out_x.unsqueeze(1), out_y)
        else:
            raise ValueError(f"Unsupported aug_mode: {self.aug_mode}")

        # ---------- augmentations ----------

    def _timeshift(self, fea: torch.Tensor) -> torch.Tensor:
        overall_shift = self.rng.integers(-self.shift_days, self.shift_days + 1)

        new_sta = self.staDOY + overall_shift
        new_end = self.endDOY + overall_shift

        new_sta = max(1, min(365, new_sta))
        new_end = max(1, min(365, new_end))

        sta_idx = (new_sta - 1) // self.dayItv
        end_idx = (new_end - 1) // self.dayItv + 1
        fea = fea[:, sta_idx:end_idx]

        fea = fea.unsqueeze(0)  # [band, time] -> [1, band, time]
        fea = F.interpolate(fea, size=self.target_len, mode="linear", align_corners=False)
        fea = fea.squeeze(0)
        return fea

    def _timescale(self, fea: torch.Tensor) -> torch.Tensor:
        sta_shift = self.rng.integers(-self.scale_days, self.scale_days + 1)
        end_shift = self.rng.integers(-self.scale_days, self.scale_days + 1)

        new_sta = self.staDOY + sta_shift
        new_end = self.endDOY + end_shift

        new_sta = max(1, min(365, new_sta))
        new_end = max(1, min(365, new_end))
        if new_end <= new_sta:
            new_end = min(365, new_sta + self.dayItv)

        sta_idx = (new_sta - 1) // self.dayItv
        end_idx = (new_end - 1) // self.dayItv + 1
        fea = fea[:, sta_idx:end_idx]

        fea = fea.unsqueeze(0)  # [band, time] -> [1, band, time]
        fea = F.interpolate(fea, size=self.target_len, mode="linear", align_corners=False)
        fea = fea.squeeze(0)
        return fea

    def _mg_warping(self, fea: torch.Tensor) -> torch.Tensor:
        if self.modality == "S2":
            warps = self.rng.normal(1.0, self.warp_sigma, size=self.warp_knots)

            orig_steps = np.arange(fea.shape[-1])
            warp_steps = np.linspace(0, fea.shape[-1] - 1, num=self.warp_knots)

            warp_curve = CubicSpline(warp_steps, warps)(orig_steps).astype(np.float32)
            warp_curve = torch.from_numpy(warp_curve).to(fea.device)  # [T]

            fea = fea * warp_curve.unsqueeze(0)  # [Band, T]
            return fea

        orig_steps = np.arange(fea.shape[-1])
        warp_steps = np.linspace(0, fea.shape[-1] - 1, num=self.warp_knots)

        # S2 multiplicative curve (mean around 1)
        s2_warps = self.rng.normal(1.0, self.warp_sigma, size=self.warp_knots)
        s2_curve = CubicSpline(warp_steps, s2_warps)(orig_steps).astype(np.float32)
        s2_curve = np.clip(s2_curve, 1 - 0.3, 1 + 0.3)
        s2_curve = torch.from_numpy(s2_curve).to(fea.device)

        # S1 additive curve in dB (mean around 0)
        s1_warps = self.rng.normal(0.0, self.warp_sigma, size=self.warp_knots)
        s1_curve = CubicSpline(warp_steps, s1_warps)(orig_steps).astype(np.float32)
        s1_curve = np.clip(s1_curve, -0.3, 0.3)
        s1_curve = torch.from_numpy(s1_curve).to(fea.device)

        fea[:10] = fea[:10] * s2_curve.unsqueeze(0)
        fea[10:12] = fea[10:12] + s1_curve.unsqueeze(0)
        return fea


