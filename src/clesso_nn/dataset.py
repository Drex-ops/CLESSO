"""
dataset.py -- PyTorch Dataset and data loading for CLESSO NN.

Loads the feather files exported by export_for_nn.R and builds
tensors for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------
# Haversine distance
# --------------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in km between points given in degrees."""
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = (np.radians(x) for x in (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------
# Fourier positional encoding
# --------------------------------------------------------------------------

def compute_fourier_features(lon_deg, lat_deg, n_frequencies=5,
                             max_wavelength=40.0):
    """Multi-scale Fourier features from geographic coordinates.

    Wavelengths = max_wavelength / 2^k  for k = 0 .. n_frequencies-1
    e.g. [40, 20, 10, 5, 2.5] degrees (continental → regional).

    For each wavelength λ, produces:
        sin(2π·lon/λ), cos(2π·lon/λ), sin(2π·lat/λ), cos(2π·lat/λ)

    Returns:
        features: (n, 4*n_frequencies) float32 array
        names:    list of feature names
    """
    if n_frequencies <= 0:
        return np.empty((len(lon_deg), 0), dtype=np.float32), []

    cols = []
    names = []
    for k in range(n_frequencies):
        wavelength = max_wavelength / (2 ** k)
        freq = 2.0 * np.pi / wavelength
        cols.append(np.sin(freq * lon_deg).astype(np.float32))
        names.append(f"fourier_lon_sin_{k}")
        cols.append(np.cos(freq * lon_deg).astype(np.float32))
        names.append(f"fourier_lon_cos_{k}")
        cols.append(np.sin(freq * lat_deg).astype(np.float32))
        names.append(f"fourier_lat_sin_{k}")
        cols.append(np.cos(freq * lat_deg).astype(np.float32))
        names.append(f"fourier_lat_cos_{k}")

    return np.column_stack(cols), names


# --------------------------------------------------------------------------
# Load exported data
# --------------------------------------------------------------------------

def load_export(export_dir: str | Path) -> dict:
    """Load all feather files from the R export directory.

    Returns a dict with keys:
        pairs            - DataFrame of observation pairs
        site_covariates  - DataFrame of per-site covariates
        env_site_table   - DataFrame of per-site env values (or None)
        site_obs_richness - DataFrame with site_id, S_obs (or None)
        metadata         - dict from metadata.json
    """
    export_dir = Path(export_dir)

    pairs = feather.read_feather(export_dir / "pairs.feather")
    site_covariates = feather.read_feather(export_dir / "site_covariates.feather")

    env_path = export_dir / "env_site_table.feather"
    env_site_table = feather.read_feather(env_path) if env_path.exists() else None

    # Prefer corrected S_obs (true species count from raw ALA) over pipeline-subsampled values
    rich_path_corrected = export_dir / "site_obs_richness_CORRECTED.feather"
    rich_path_original = export_dir / "site_obs_richness.feather"
    if rich_path_corrected.exists():
        site_obs_richness = feather.read_feather(rich_path_corrected)
        print(f"  [S_obs] Using CORRECTED file: {rich_path_corrected.name}")
    elif rich_path_original.exists():
        site_obs_richness = feather.read_feather(rich_path_original)
        print(f"  [S_obs] Using original file: {rich_path_original.name}")
    else:
        site_obs_richness = None

    with open(export_dir / "metadata.json") as f:
        metadata = json.load(f)

    return dict(
        pairs=pairs,
        site_covariates=site_covariates,
        env_site_table=env_site_table,
        site_obs_richness=site_obs_richness,
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# Build site index and covariate tensors
# --------------------------------------------------------------------------

class SiteData:
    """Holds site-level information as tensors and lookup mappings."""

    def __init__(
        self,
        site_covariates: pd.DataFrame,
        env_site_table: Optional[pd.DataFrame],
        site_obs_richness: Optional[pd.DataFrame],
        metadata: dict,
        effort_cov_names: Optional[list[str]] = None,
    ):
        # -- Site ID → contiguous 0-based index mapping --
        site_ids = site_covariates["site_id"].values
        self.site_id_to_idx = {sid: i for i, sid in enumerate(site_ids)}
        self.n_sites = len(site_ids)

        # -- Effort / detectability covariates (separate from alpha) --
        # These are kept in a separate W matrix and fed to EffortNet.
        self.effort_cov_names: list[str] = []
        self.W: Optional[torch.Tensor] = None
        self.w_mean: Optional[np.ndarray] = None
        self.w_std: Optional[np.ndarray] = None

        _effort_names = effort_cov_names or []
        # Keep only names that actually exist in the data
        _effort_names = [c for c in _effort_names if c in site_covariates.columns]
        if _effort_names:
            self.effort_cov_names = _effort_names
            W_raw = site_covariates[_effort_names].values.astype(np.float32)
            self.w_mean = np.nanmean(W_raw, axis=0)
            self.w_std = np.nanstd(W_raw, axis=0)
            self.w_std[self.w_std == 0] = 1.0
            W = (W_raw - self.w_mean) / self.w_std
            np.nan_to_num(W, copy=False, nan=0.0)
            self.W = torch.from_numpy(W)  # (n_sites, K_effort)
            print(f"  [Effort] {len(_effort_names)} effort covariates: "
                  f"{', '.join(_effort_names)}")

        # -- Alpha covariates (site-level features for richness model) --
        # Use all numeric columns except site_id as alpha covariates.
        # Exclude effort columns (they go through EffortNet separately).
        # Optionally exclude raw lon/lat (when using Fourier encoding instead).
        exclude_coords = metadata.get("exclude_coords_from_alpha", False)
        _effort_set = set(self.effort_cov_names)
        alpha_cols = [c for c in site_covariates.columns
                      if c != "site_id"
                      and c not in _effort_set
                      and pd.api.types.is_numeric_dtype(site_covariates[c])]
        if exclude_coords:
            alpha_cols = [c for c in alpha_cols if c not in ("lon", "lat")]
        Z_raw = site_covariates[alpha_cols].values.astype(np.float32)

        # -- Fourier positional encoding of coordinates (for alpha model) --
        fourier_n_freq = metadata.get("fourier_n_frequencies", 0)
        fourier_max_wl = metadata.get("fourier_max_wavelength", 40.0)
        self.fourier_n_frequencies = fourier_n_freq
        self.fourier_max_wavelength = fourier_max_wl

        if fourier_n_freq > 0 and "lon" in site_covariates.columns:
            lon_deg = site_covariates["lon"].values.astype(np.float32)
            lat_deg = site_covariates["lat"].values.astype(np.float32)
            fourier_feats, fourier_names = compute_fourier_features(
                lon_deg, lat_deg, fourier_n_freq, fourier_max_wl)
            Z_raw = np.column_stack([Z_raw, fourier_feats])
            alpha_cols = alpha_cols + fourier_names
            print(f"  [Fourier] Added {len(fourier_names)} Fourier features "
                  f"({fourier_n_freq} frequencies, max λ={fourier_max_wl}°)")

        self.alpha_cov_names = alpha_cols

        # Standardise (centre + scale)
        self.z_mean = np.nanmean(Z_raw, axis=0)
        self.z_std = np.nanstd(Z_raw, axis=0)
        self.z_std[self.z_std == 0] = 1.0
        Z = (Z_raw - self.z_mean) / self.z_std
        np.nan_to_num(Z, copy=False, nan=0.0)
        self.Z = torch.from_numpy(Z)  # (n_sites, K_alpha)

        # -- Beta covariates (env values for turnover pairwise distances) --
        # These are per-site env values; pairwise |diff| is computed on-the-fly.
        if env_site_table is not None:
            env_cols = [c for c in env_site_table.columns
                        if c != "site_id" and pd.api.types.is_numeric_dtype(env_site_table[c])]
            self.env_cov_names = env_cols

            # Align env table to same site order
            env_df = env_site_table.set_index("site_id").reindex(site_ids)
            E_raw = env_df[env_cols].values.astype(np.float32)

            # Standardise env covariates (important for balanced gradients)
            self.e_mean = np.nanmean(E_raw, axis=0)
            self.e_std = np.nanstd(E_raw, axis=0)
            self.e_std[self.e_std == 0] = 1.0
            E = (E_raw - self.e_mean) / self.e_std
            np.nan_to_num(E, copy=False, nan=0.0)
            self.E = torch.from_numpy(E)  # (n_sites, K_env)
        else:
            self.env_cov_names = []
            self.E = None

        # Geographic lon/lat diffs in beta (OLD approach, controlled by include_geo_in_beta)
        geo_use = metadata.get("include_geo_in_beta", False)
        if geo_use and "lon" in site_covariates.columns:
            geo_raw = site_covariates[["lon", "lat"]].values.astype(np.float32)
            self.geo_mean = np.nanmean(geo_raw, axis=0)
            self.geo_std = np.nanstd(geo_raw, axis=0)
            self.geo_std[self.geo_std == 0] = 1.0
            self.geo = torch.from_numpy((geo_raw - self.geo_mean) / self.geo_std)
        else:
            self.geo = None

        # Geographic distance in beta (NEW approach: haversine km)
        self.include_geo_dist_in_beta = metadata.get("include_geo_dist_in_beta", False)
        self.geo_dist_scale = None
        if self.include_geo_dist_in_beta and "lon" in site_covariates.columns:
            # Store raw lon/lat in degrees for haversine computation
            self.lon_deg = site_covariates["lon"].values.astype(np.float32)
            self.lat_deg = site_covariates["lat"].values.astype(np.float32)

            # Estimate distance scale from random site pairs
            rng = np.random.default_rng(42)
            n_sample = min(200_000, self.n_sites * (self.n_sites - 1) // 2)
            idx_a = rng.integers(0, self.n_sites, size=n_sample)
            idx_b = rng.integers(0, self.n_sites, size=n_sample)
            different = idx_a != idx_b
            idx_a, idx_b = idx_a[different], idx_b[different]
            d_sample = haversine_km(
                self.lon_deg[idx_a], self.lat_deg[idx_a],
                self.lon_deg[idx_b], self.lat_deg[idx_b])
            self.geo_dist_scale = float(np.std(d_sample))
            if self.geo_dist_scale < 1.0:
                self.geo_dist_scale = 1000.0  # fallback
            print(f"  [GeoDist] scale={self.geo_dist_scale:.1f} km "
                  f"(mean={np.mean(d_sample):.0f} km, "
                  f"p95={np.percentile(d_sample, 95):.0f} km)")
        else:
            self.lon_deg = None
            self.lat_deg = None

        # -- Observed richness per site (for lower-bound penalty) --
        self.S_obs = torch.zeros(self.n_sites, dtype=torch.float32)
        if site_obs_richness is not None:
            for _, row in site_obs_richness.iterrows():
                idx = self.site_id_to_idx.get(row["site_id"])
                if idx is not None:
                    self.S_obs[idx] = float(row["S_obs"])

    @property
    def K_alpha(self) -> int:
        return self.Z.shape[1]

    @property
    def K_effort(self) -> int:
        """Number of effort / detectability covariates for EffortNet."""
        return self.W.shape[1] if self.W is not None else 0

    @property
    def K_env(self) -> int:
        """Number of pairwise-distance input features for beta network."""
        k = 0
        if self.E is not None:
            k += self.E.shape[1]
        if self.geo is not None:
            k += self.geo.shape[1]
        if self.include_geo_dist_in_beta:
            k += 1  # haversine distance (single scalar)
        return k

    def get_env_at_site(self, idx: torch.Tensor) -> torch.Tensor:
        """Return concatenated env features [env_covs | geo] for site indices."""
        parts = []
        if self.E is not None:
            parts.append(self.E[idx])
        if self.geo is not None:
            parts.append(self.geo[idx])
        if not parts:
            raise ValueError("No environmental covariates available for beta model")
        return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------
# Pair dataset
# --------------------------------------------------------------------------

class CLESSOPairDataset(Dataset):
    """Dataset of observation pairs for CLESSO NN training.

    Each item returns:
        site_i_idx  : int   - 0-based site index for observation i
        site_j_idx  : int   - 0-based site index for observation j
        y           : float - 0 (match) or 1 (mismatch)
        is_within   : float - 1 if within-site pair, 0 if between-site
        weight      : float - pair weight
        env_diff    : Tensor (K_env,) - |env_i - env_j| for beta model
    """

    def __init__(self, pairs: pd.DataFrame, site_data: SiteData,
                 use_unit_weights: bool = False):
        self.n = len(pairs)
        self.site_data = site_data

        # Map site IDs to indices
        self.site_i = np.array([site_data.site_id_to_idx[s] for s in pairs["site_i"]],
                               dtype=np.int64)
        self.site_j = np.array([site_data.site_id_to_idx[s] for s in pairs["site_j"]],
                               dtype=np.int64)
        self.y = pairs["y"].values.astype(np.float32)
        self.is_within = pairs["is_within"].values.astype(np.float32)
        if use_unit_weights:
            self.weight = np.ones(self.n, dtype=np.float32)
        else:
            self.weight = pairs["w"].values.astype(np.float32) if "w" in pairs.columns \
                else np.ones(self.n, dtype=np.float32)

        # Pre-compute pairwise env differences |env_i - env_j|
        # This is the neural analog of the I-spline transformed X matrix.
        env_i = site_data.get_env_at_site(
            torch.from_numpy(self.site_i))
        env_j = site_data.get_env_at_site(
            torch.from_numpy(self.site_j))
        self.env_diff = torch.abs(env_i - env_j).numpy()

        # Optionally append haversine geographic distance (single dim)
        if site_data.include_geo_dist_in_beta and site_data.lon_deg is not None:
            geo_dist = haversine_km(
                site_data.lon_deg[self.site_i], site_data.lat_deg[self.site_i],
                site_data.lon_deg[self.site_j], site_data.lat_deg[self.site_j],
            )
            geo_dist_norm = (geo_dist / site_data.geo_dist_scale).astype(np.float32)
            self.env_diff = np.column_stack([self.env_diff, geo_dist_norm])

        # Zero out env_diff for within-site pairs (same site → zero distance)
        within_mask = self.is_within == 1
        self.env_diff[within_mask] = 0.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "site_i": self.site_i[idx],
            "site_j": self.site_j[idx],
            "y": self.y[idx],
            "is_within": self.is_within[idx],
            "weight": self.weight[idx],
            "env_diff": self.env_diff[idx],
        }


# --------------------------------------------------------------------------
# Collate + DataLoader helpers
# --------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of sample dicts into batched tensors."""
    return {
        "site_i": torch.tensor([b["site_i"] for b in batch], dtype=torch.long),
        "site_j": torch.tensor([b["site_j"] for b in batch], dtype=torch.long),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float32),
        "is_within": torch.tensor([b["is_within"] for b in batch], dtype=torch.float32),
        "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
        "env_diff": torch.tensor(
            np.stack([b["env_diff"] for b in batch]),
            dtype=torch.float32,
        ),
    }


def make_dataloaders(
    pairs: pd.DataFrame,
    site_data: SiteData,
    val_fraction: float = 0.1,
    batch_size: int = 8192,
    seed: int = 42,
    num_workers: int = 0,
    use_unit_weights: bool = False,
) -> tuple[DataLoader, DataLoader, CLESSOPairDataset, CLESSOPairDataset]:
    """Split pairs into train/val and return DataLoaders."""

    rng = np.random.default_rng(seed)
    n = len(pairs)
    n_val = int(n * val_fraction)
    indices = rng.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_pairs = pairs.iloc[train_idx].reset_index(drop=True)
    val_pairs = pairs.iloc[val_idx].reset_index(drop=True)

    train_ds = CLESSOPairDataset(train_pairs, site_data, use_unit_weights)
    val_ds = CLESSOPairDataset(val_pairs, site_data, use_unit_weights)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, train_ds, val_ds

# --------------------------------------------------------------------------
# Hard-pair mining for between-site pairs
# --------------------------------------------------------------------------

class HardPairMiner:
    """Importance-corrected hard-pair mining for between-site beta training.

    Scores every training pair by its current per-pair NLL (hardness), assigns
    difficulty bins within each coarse cell (between-match / between-mismatch),
    and builds a weighted sampler whose proposal distribution q(p) over-samples
    hard pairs.  The importance correction ratio pi(p)/q(p) is stored per-pair
    so the training loop can multiply it into the loss weights, preserving the
    original probabilistic target.

    Usage in the training loop::

        miner = HardPairMiner(train_ds, config)
        # At start of each beta phase (or every N epochs):
        miner.refresh_scores(model, site_data, device)
        loader = miner.make_dataloader(batch_size)
        for batch in loader:
            # batch["mining_iw"] contains the importance correction ratio
            imp_correction = batch["mining_iw"]
            ...

    When hard mining is disabled (lambda_hm=0), the miner degrades to uniform
    sampling with mining_iw=1 for every pair.
    """

    def __init__(self, dataset: CLESSOPairDataset, config):
        self.dataset = dataset
        self.n = len(dataset)

        # Config
        self.lambda_hm: float = config.hard_mining_lambda
        self.n_bins: int = config.hard_mining_n_bins
        self.bin_weights: list[float] = list(config.hard_mining_bin_weights)
        self.a_max: float = config.hard_mining_a_max

        assert len(self.bin_weights) == self.n_bins, (
            f"hard_mining_bin_weights length ({len(self.bin_weights)}) "
            f"must equal hard_mining_n_bins ({self.n_bins})"
        )

        # Coarse cell membership: between-match (y=0) vs between-mismatch (y=1)
        # (All pairs in dataset are between-site when used from cyclic/two-stage)
        self.y = dataset.y  # float32 array, 0 or 1
        self.cell_match = self.y < 0.5   # boolean mask: True = match
        self.cell_mismatch = ~self.cell_match

        # Per-pair importance correction (initialised to 1 = no correction)
        self.iw = np.ones(self.n, dtype=np.float32)

        # Per-pair sampling probabilities (uniform initially)
        self.q = np.ones(self.n, dtype=np.float64) / self.n

    @torch.no_grad()
    def refresh_scores(
        self,
        model,
        site_data,
        device: str,
        batch_size: int = 16384,
    ) -> None:
        """Forward-pass over all training pairs to compute hardness scores.

        Updates self.q (proposal distribution) and self.iw (importance
        correction ratio pi/q, truncated by a_max).
        """
        model.eval()
        Z = site_data.Z.to(device)
        W = site_data.W.to(device) if site_data.W is not None else None
        eps = 1e-7

        # ---- Compute per-pair NLL ----
        nll = np.empty(self.n, dtype=np.float64)

        for start in range(0, self.n, batch_size):
            end = min(start + batch_size, self.n)
            idx = np.arange(start, end)
            batch = {
                "site_i": torch.tensor(self.dataset.site_i[idx], dtype=torch.long, device=device),
                "site_j": torch.tensor(self.dataset.site_j[idx], dtype=torch.long, device=device),
                "env_diff": torch.tensor(self.dataset.env_diff[idx], dtype=torch.float32, device=device),
                "is_within": torch.tensor(self.dataset.is_within[idx], dtype=torch.float32, device=device),
            }
            z_i = Z[batch["site_i"]]
            z_j = Z[batch["site_j"]]
            w_i = W[batch["site_i"]] if W is not None else None
            w_j = W[batch["site_j"]] if W is not None else None
            fwd = model.forward(z_i, z_j, batch["env_diff"], batch["is_within"],
                                w_i=w_i, w_j=w_j)
            p = fwd["p_match"].clamp(eps, 1.0 - eps)
            y = torch.tensor(self.dataset.y[idx], dtype=torch.float32, device=device)
            h = -(1.0 - y) * torch.log(p) - y * torch.log(1.0 - p)
            nll[start:end] = h.cpu().numpy()

        # ---- Build proposal distribution q(p) ----
        if self.lambda_hm <= 0.0:
            # No mining: uniform proposal, iw=1
            self.q[:] = 1.0 / self.n
            self.iw[:] = 1.0
            return

        # Target distribution pi(p): equal mass across the two coarse cells,
        # uniform within each cell.
        pi = np.empty(self.n, dtype=np.float64)
        n_match = self.cell_match.sum()
        n_mismatch = self.cell_mismatch.sum()
        if n_match > 0:
            pi[self.cell_match] = 0.5 / n_match
        if n_mismatch > 0:
            pi[self.cell_mismatch] = 0.5 / n_mismatch

        # Hard-pair proposal h*(p): extra mass on difficult bins within
        # each coarse cell.
        h_star = np.zeros(self.n, dtype=np.float64)
        bin_w = np.array(self.bin_weights, dtype=np.float64)
        bin_w /= bin_w.sum()  # normalise

        for cell_mask, cell_prob in [(self.cell_match, 0.5),
                                     (self.cell_mismatch, 0.5)]:
            cell_idx = np.where(cell_mask)[0]
            if len(cell_idx) == 0:
                continue

            cell_nll = nll[cell_idx]

            # Quantile-based difficulty bins (avoids eta-clamp boundary issue:
            # pairs are ranked *within* each cell, so saturated-eta pairs
            # don't automatically dominate).
            bin_edges = np.quantile(
                cell_nll,
                np.linspace(0, 1, self.n_bins + 1),
            )
            # np.searchsorted: bin 0 = easiest, bin n_bins-1 = hardest
            bin_idx = np.searchsorted(bin_edges[1:-1], cell_nll, side="right")
            # Clamp to valid range
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

            for d in range(self.n_bins):
                in_bin = cell_idx[bin_idx == d]
                n_in_bin = len(in_bin)
                if n_in_bin == 0:
                    continue
                # h*(p) = Pr(cell) * Pr(bin|cell) / N_{cell,bin}
                h_star[in_bin] = cell_prob * bin_w[d] / n_in_bin

        # Mixture proposal: q(p) = (1-λ)·π(p) + λ·h*(p)
        q = (1.0 - self.lambda_hm) * pi + self.lambda_hm * h_star

        # Avoid division by zero
        q = np.maximum(q, 1e-12)
        q /= q.sum()  # renormalise for numerical safety
        self.q = q

        # Importance correction: pi(p) / q(p), truncated at a_max
        ratio = pi / q
        ratio = np.minimum(ratio, self.a_max)
        self.iw = ratio.astype(np.float32)

    def make_dataloader(
        self,
        batch_size: int = 8192,
        num_workers: int = 0,
    ) -> DataLoader:
        """Build a DataLoader that samples pairs according to q(p).

        Each batch dict includes an extra key ``mining_iw`` with the
        per-pair importance correction ratio (float32).
        """
        from torch.utils.data import WeightedRandomSampler

        # WeightedRandomSampler draws `num_samples` indices with replacement,
        # proportional to the given weights.  One "epoch" = N draws so every
        # pair is seen ~once on average.
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(self.q.astype(np.float64)),
            num_samples=self.n,
            replacement=True,
        )

        # Wrap dataset to inject mining_iw
        wrapped = _MinedDatasetWrapper(self.dataset, self.iw)

        return DataLoader(
            wrapped,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=_mined_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )


class _MinedDatasetWrapper(Dataset):
    """Thin wrapper that attaches mining importance weights to each sample."""

    def __init__(self, dataset: CLESSOPairDataset, iw: np.ndarray):
        self.dataset = dataset
        self.iw = iw

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        item["mining_iw"] = self.iw[idx]
        return item


def _mined_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate that includes the mining_iw field."""
    base = collate_fn(batch)
    base["mining_iw"] = torch.tensor(
        [b["mining_iw"] for b in batch], dtype=torch.float32,
    )
    return base