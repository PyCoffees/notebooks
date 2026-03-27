from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from astropy.timeseries import LombScargle

import importlib
import MLP
importlib.reload(MLP)
from MLP import NeuralNetwork


# =========================================================
# Configuration
# =========================================================

@dataclass
class RVNNConfig:
    # Sampling / dataset sizes
    n_phase_points: int = 160
    known_samples: int = 1200
    new_samples: int = 200
    validation_fraction: float = 0.25
    random_seed: int = 42

    # Neural network
    hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32)
    activation: str = "tanh"
    learning_rate_init: float = 3e-3
    max_epochs: int = 120
    init_scale: float = 0.05
    optimizer: str = "adam"
    batch_size: Optional[int] = 32

    # Input error handling
    errors_mode: str = "full"   # "none", "mean", "full"

    # Synthetic orbit priors
    K_min: float = 10.0
    K_max: float = 120.0

    gamma_min: float = -20.0
    gamma_max: float = 20.0

    e_min: float = 0.0
    e_max: float = 0.6

    omega_min: float = 0.0
    omega_max: float = 2.0 * np.pi

    stellar_mass_min: float = 0.5   # Msun
    stellar_mass_max: float = 0.8   # Msun

    # Period prior
    P_min: float = 1.5
    P_max: float = 300.0

    # Noise / jitter / activity
    rv_err_min: float = 3.0
    rv_err_max: float = 15.0

    jitter_scale: float = 0.6
    trend_scale: float = 2.0
    activity_amp_max: float = 8.0

    outlier_fraction: float = 0.03
    outlier_scale: float = 4.0

    activity_period_min_factor: float = 0.7
    activity_period_max_factor: float = 2.0

    period_sampling: str = "mixture"   # "uniform", "loguniform", "mixture"
    short_period_fraction: float = 0.45
    medium_period_fraction: float = 0.35

    # Irregular observation campaign
    time_span_min: float = 80.0
    time_span_max: float = 250.0
    n_obs_min: int = 35
    n_obs_max: int = 90
    samples_per_peak: int = 10
    period_error_frac_medium: float = 0.003
    period_error_frac_full: float = 0.01

    # Plot / display
    rv_unit: str = r"m\,s^{-1}"
    print_every_epoch: bool = True
    smooth_phase_width: float = 0.06

    @property
    def target_names(self) -> List[str]:
        return ["logP", "K", "gamma", "h_esinw", "k_ecosw", "phi0"]


# =========================================================
# Labels / plot style
# =========================================================

PLOT_LABELS = {
    "phase": r"Phase",
    "rv": r"RV [$\mathrm{m\,s^{-1}}$]",
    "mse": r"MSE loss",
    "r2": r"$R^2$",

    "K": r"$K$ [$\mathrm{m\,s^{-1}}$]",
    "gamma": r"$\gamma$ [$\mathrm{m\,s^{-1}}$]",
    "h_esinw": r"$h = e\sin\omega$",
    "k_ecosw": r"$k = e\cos\omega$",

    "true_K": r"$K_{\rm true}$ [$\mathrm{m\,s^{-1}}$]",
    "pred_K": r"$K_{\rm pred}$ [$\mathrm{m\,s^{-1}}$]",

    "true_gamma": r"$\gamma_{\rm true}$ [$\mathrm{m\,s^{-1}}$]",
    "pred_gamma": r"$\gamma_{\rm pred}$ [$\mathrm{m\,s^{-1}}$]",

    "true_h_esinw": r"$h_{\rm true}$",
    "pred_h_esinw": r"$h_{\rm pred}$",

    "true_k_ecosw": r"$k_{\rm true}$",
    "pred_k_ecosw": r"$k_{\rm pred}$",
}

PLOT_LABELS.update({
    "phi0": r"$\phi_0$",
    "true_phi0": r"$\phi_{0,\rm true}$",
    "pred_phi0": r"$\phi_{0,\rm pred}$",
})

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"


# =========================================================
# Keplerian utilities
# =========================================================

def solve_kepler(
    M: np.ndarray,
    e: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """Solve Kepler equation E - e sin E = M."""
    M = np.asarray(M, dtype=float)
    E = M.copy()

    if e > 0.8:
        E[:] = np.pi

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if np.max(np.abs(dE)) < tol:
            break

    return E


def true_anomaly_from_mean_anomaly(M: np.ndarray, e: float) -> np.ndarray:
    E = solve_kepler(M, e)
    cosf = (np.cos(E) - e) / (1.0 - e * np.cos(E))
    sinf = (np.sqrt(1.0 - e**2) * np.sin(E)) / (1.0 - e * np.cos(E))
    return np.arctan2(sinf, cosf)


def keplerian_rv_from_phase(
    phase: np.ndarray,
    K: float,
    e: float,
    omega: float,
    gamma: float,
    phi0: float,
) -> np.ndarray:
    """
    Keplerian RV model in folded phase including a horizontal phase offset phi0.
    """
    phase = np.asarray(phase, dtype=float)
    phase_shifted = np.mod(phase - phi0, 1.0)
    M = 2.0 * np.pi * phase_shifted
    nu = true_anomaly_from_mean_anomaly(M, e)
    return gamma + K * (np.cos(nu + omega) + e * np.cos(omega))


def keplerian_rv_from_time(
    t: np.ndarray,
    period: float,
    t0: float,
    K: float,
    e: float,
    omega: float,
    gamma: float,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    M = 2.0 * np.pi * ((t - t0) / period)
    M = np.mod(M, 2.0 * np.pi)
    nu = true_anomaly_from_mean_anomaly(M, e)
    return gamma + K * (np.cos(nu + omega) + e * np.cos(omega))


def phase_fold(t: np.ndarray, period: float, t0: Optional[float] = None) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if t0 is None:
        t0 = np.min(t)
    return ((t - t0) / period) % 1.0


# =========================================================
# Synthetic generation helpers
# =========================================================
def compute_K(P_days, Mp_mjup, Mstar_msun, e):
    G = 6.67430e-11
    M_sun = 1.98847e30
    M_jup = 1.89813e27
    day = 86400.0

    P_sec = np.asarray(P_days, dtype=float) * day
    Mp = np.asarray(Mp_mjup, dtype=float) * M_jup
    Mstar = np.asarray(Mstar_msun, dtype=float) * M_sun
    e = np.asarray(e, dtype=float)

    K = (
        (2.0 * np.pi * G / P_sec) ** (1.0 / 3.0)
        * Mp
        / (Mstar + Mp) ** (2.0 / 3.0)
        / np.sqrt(1.0 - e**2)
    )
    return K

def msini_from_params(P_days, K, e, Mstar_msun):
    G = 6.67430e-11
    M_sun = 1.98847e30
    M_jup = 1.89813e27
    day = 86400.0

    P_sec = np.asarray(P_days, dtype=float) * day
    K = np.asarray(K, dtype=float)
    e = np.asarray(e, dtype=float)
    Mstar = np.asarray(Mstar_msun, dtype=float) * M_sun

    msini = (
        K
        * np.sqrt(1.0 - e**2)
        * (P_sec / (2.0 * np.pi * G)) ** (1.0 / 3.0)
        * Mstar ** (2.0 / 3.0)
    )
    return msini / M_jup

def sample_orbit_parameters(rng: np.random.Generator, cfg: RVNNConfig) -> Dict[str, float]:
    Mp = 10 ** rng.uniform(-1.0, 0.0)   # 0.1--100 Mjup
    Mstar = rng.uniform(cfg.stellar_mass_min, cfg.stellar_mass_max)
    P = sample_period(rng, cfg)

    gamma = rng.uniform(cfg.gamma_min, cfg.gamma_max)
    e = rng.uniform(cfg.e_min, cfg.e_max)
    omega = rng.uniform(cfg.omega_min, cfg.omega_max)

    K = compute_K(P, Mp, Mstar, e)

    h = e * np.sin(omega)
    k = e * np.cos(omega)

    return {
        "P": P,
        "Mp_msini": Mp,
        "Mstar": Mstar,
        "K": K,
        "gamma": gamma,
        "e": e,
        "omega": omega,
        "h": h,
        "k": k,
    }

def sample_period(rng: np.random.Generator, cfg: RVNNConfig) -> float:
    if cfg.period_sampling == "uniform":
        return rng.uniform(cfg.P_min, cfg.P_max)

    if cfg.period_sampling == "loguniform":
        return 10 ** rng.uniform(np.log10(cfg.P_min), np.log10(cfg.P_max))

    if cfg.period_sampling == "mixture":
        u = rng.random()

        P1 = cfg.P_min
        P2 = min(10.0, cfg.P_max)
        P3 = min(30.0, cfg.P_max)
        P4 = cfg.P_max

        if u < cfg.short_period_fraction:
            return 10 ** rng.uniform(np.log10(P1), np.log10(P2))
        elif u < cfg.short_period_fraction + cfg.medium_period_fraction:
            return 10 ** rng.uniform(np.log10(P2), np.log10(P3))
        else:
            return 10 ** rng.uniform(np.log10(P3), np.log10(P4))

    raise ValueError("Invalid period_sampling")

def sample_irregular_times(
    rng: np.random.Generator,
    time_span: float,
    n_obs: int,
    t_start: float = 0.0,
) -> np.ndarray:
    return np.sort(rng.uniform(t_start, t_start + time_span, size=n_obs))


def add_systematics_from_phase(
    phase: np.ndarray,
    rv: np.ndarray,
    rv_err: np.ndarray,
    rng: np.random.Generator,
    cfg: RVNNConfig,
) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    rv = np.asarray(rv, dtype=float)
    rv_err = np.asarray(rv_err, dtype=float)

    out = rv.copy()

    out += rng.normal(0.0, rv_err, size=rv.size)

    jitter_sigma = cfg.jitter_scale * float(np.median(rv_err))
    out += rng.normal(0.0, jitter_sigma, size=rv.size)

    x = phase - np.mean(phase)
    drift_coeff = rng.uniform(-cfg.trend_scale, cfg.trend_scale)
    out += drift_coeff * x

    act_amp = rng.uniform(0.0, cfg.activity_amp_max)
    act_freq = rng.uniform(cfg.activity_period_min_factor, cfg.activity_period_max_factor)
    act_phi = rng.uniform(0.0, 2.0 * np.pi)
    out += act_amp * np.sin(2.0 * np.pi * act_freq * phase + act_phi)

    mask_out = rng.random(rv.size) < cfg.outlier_fraction
    if np.any(mask_out):
        sigma_out = cfg.outlier_scale * np.median(rv_err)
        out[mask_out] += rng.normal(0.0, sigma_out, size=np.sum(mask_out))

    return out


def add_rv_noise_and_errors(
    rv: np.ndarray,
    rng: np.random.Generator,
    cfg: RVNNConfig,
    phase: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    sigma_meas = rng.uniform(cfg.rv_err_min, cfg.rv_err_max)
    rv_err_meas = sigma_meas * rng.uniform(0.8, 1.2, size=rv.size)

    jitter_sigma = cfg.jitter_scale * float(np.median(rv_err_meas))
    rv_err_eff = np.sqrt(rv_err_meas**2 + jitter_sigma**2)

    if phase is None:
        phase = np.linspace(0.0, 1.0, rv.size, endpoint=False)
    rv_noisy = add_systematics_from_phase(phase, rv, rv_err_meas, rng, cfg)

    return rv_noisy, rv_err_eff


def build_rv_input(
    rv_curve: np.ndarray,
    rv_err: np.ndarray,
    cfg: RVNNConfig,
) -> np.ndarray:
    if cfg.errors_mode == "none":
        return rv_curve
    if cfg.errors_mode == "mean":
        err_mean = np.array([np.mean(rv_err)], dtype=float)
        return np.concatenate([rv_curve, err_mean], axis=0)
    if cfg.errors_mode == "full":
        return np.concatenate([rv_curve, rv_err], axis=0)
    raise ValueError("errors_mode must be 'none', 'mean', or 'full'")


# =========================================================
# Periodogram / binning helpers
# =========================================================

def compute_rv_periodogram(
    t: np.ndarray,
    rv: np.ndarray,
    rv_err: Optional[np.ndarray] = None,
    P_min: float = 1.0,
    P_max: Optional[float] = None,
    samples_per_peak: int = 10,
) -> Dict[str, Any]:
    t = np.asarray(t, dtype=float)
    rv = np.asarray(rv, dtype=float)
    if rv_err is not None:
        rv_err = np.asarray(rv_err, dtype=float)

    baseline = t.max() - t.min()
    if P_max is None:
        P_max = 0.8 * baseline

    P_max = max(P_max, P_min * 1.5)

    min_frequency = 1.0 / P_max
    max_frequency = 1.0 / P_min

    ls = LombScargle(t, rv, dy=rv_err, center_data=True, fit_mean=True)
    frequency, power = ls.autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=samples_per_peak,
    )

    period = 1.0 / frequency
    best_idx = int(np.argmax(power))

    return {
        "ls": ls,
        "frequency": frequency,
        "period": period,
        "power": power,
        "best_period": period[best_idx],
        "best_frequency": frequency[best_idx],
        "best_power": power[best_idx],
    }


def bin_folded_rv_to_grid(
    phase: np.ndarray,
    rv: np.ndarray,
    rv_err: np.ndarray,
    phase_grid: np.ndarray,
    width_factor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted circular kernel binning from irregular folded data to a fixed grid.
    More stable than hard nearest-bin assignment.
    """
    phase = np.asarray(phase, dtype=float)
    rv = np.asarray(rv, dtype=float)
    rv_err = np.asarray(rv_err, dtype=float)
    phase_grid = np.asarray(phase_grid, dtype=float)

    dx = np.median(np.diff(phase_grid))
    width = width_factor * dx

    rv_grid = np.zeros_like(phase_grid, dtype=float)
    err_grid = np.zeros_like(phase_grid, dtype=float)

    for i, ph0 in enumerate(phase_grid):
        d = np.abs(phase - ph0)
        d = np.minimum(d, 1.0 - d)

        w_phase = np.exp(-0.5 * (d / width) ** 2)
        w_err = 1.0 / np.maximum(rv_err**2, 1e-10)
        w = w_phase * w_err

        if np.sum(w) > 0:
            rv_grid[i] = np.sum(w * rv) / np.sum(w)

            mask_core = w_phase > np.exp(-0.5)
            if np.any(mask_core):
                err_grid[i] = np.sqrt(1.0 / np.sum(w_err[mask_core]))
            else:
                err_grid[i] = np.sqrt(1.0 / np.sum(w_err))
        else:
            j = np.argmin(d)
            rv_grid[i] = rv[j]
            err_grid[i] = rv_err[j]

    return rv_grid, err_grid


def smooth_folded_rv(
    phase: np.ndarray,
    rv: np.ndarray,
    rv_err: np.ndarray,
    width: float = 0.06,
) -> np.ndarray:
    """
    Smooth curve for visualisation on the folded phase grid.
    """
    phase = np.asarray(phase, dtype=float)
    rv = np.asarray(rv, dtype=float)
    rv_err = np.asarray(rv_err, dtype=float)

    out = np.zeros_like(rv)

    for i, ph0 in enumerate(phase):
        d = np.abs(phase - ph0)
        d = np.minimum(d, 1.0 - d)

        w_phase = np.exp(-0.5 * (d / width) ** 2)
        w_err = 1.0 / np.maximum(rv_err**2, 1e-10)
        w = w_phase * w_err

        out[i] = np.sum(w * rv) / np.sum(w)

    return out


def smooth_series(y: np.ndarray, window: int = 9) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    y_pad = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


# =========================================================
# Synthetic dataset generation
# =========================================================
def generate_phased_rv_dataset(
    cfg: RVNNConfig,
    n_samples: int,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    phase_grid = np.linspace(0.0, 1.0, cfg.n_phase_points, endpoint=False)

    X_list = []
    X_rv_list = []
    X_err_list = []
    y_list = []
    true_params = []

    Mstar_list = []
    phase_offset_list = []
    best_period_feature_list = []

    irregular_t_list = []
    irregular_rv_list = []
    irregular_err_list = []
    periodogram_list = []
    best_period_list = []
    true_period_list = []

    for _ in range(n_samples):
        pars = sample_orbit_parameters(rng, cfg)
        Mstar_list.append(pars["Mstar"])

        period_true = pars["P"]
        period_used = period_true * rng.normal(1.0, cfg.period_error_frac_medium)

        phi0 = rng.uniform(0.0, 1.0)
        shifted_phase = np.mod(phase_grid + phi0, 1.0)

        rv_clean = keplerian_rv_from_phase(
            phase_grid,
            K=pars["K"],
            e=pars["e"],
            omega=pars["omega"],
            gamma=pars["gamma"],
            phi0=phi0,
        )
        rv_noisy, rv_err = add_rv_noise_and_errors(rv_clean, rng, cfg, phase=shifted_phase)

        t_pseudo = phase_grid * period_true
        phase_est = phase_fold(t_pseudo, period_used, t0=0.0)
        order = np.argsort(phase_est)

        rv_curve = np.interp(phase_grid, phase_est[order], rv_noisy[order], period=1.0)
        err_curve = np.interp(phase_grid, phase_est[order], rv_err[order], period=1.0)

        # Input = RV + ERR + log10(best_period)
        x_core = build_rv_input(rv_curve, err_curve, cfg)
        x = np.concatenate([x_core, np.array([np.log10(period_used)], dtype=float)])

        y = np.array([
            np.log10(pars["P"]),
            pars["K"],
            pars["gamma"],
            pars["h"],
            pars["k"],
            phi0,
        ], dtype=float)

        X_list.append(x)
        X_rv_list.append(rv_curve)
        X_err_list.append(err_curve)
        y_list.append(y)

        true_params.append([
            pars["P"],
            pars["Mp_msini"],
            pars["Mstar"],
            pars["K"],
            pars["gamma"],
            pars["e"],
            pars["omega"],
            pars["h"],
            pars["k"],
            phi0,
        ])

        phase_offset_list.append(phi0)
        best_period_feature_list.append(np.log10(period_used))

        irregular_t_list.append(np.full(1, np.nan))
        irregular_rv_list.append(np.full(1, np.nan))
        irregular_err_list.append(np.full(1, np.nan))
        periodogram_list.append(None)
        best_period_list.append(period_used)
        true_period_list.append(period_true)

    return {
        "phase": phase_grid,
        "X": np.asarray(X_list, dtype=float),
        "X_rv": np.asarray(X_rv_list, dtype=float),
        "X_err": np.asarray(X_err_list, dtype=float),
        "y": np.asarray(y_list, dtype=float),
        "true_params": np.asarray(true_params, dtype=float),
        "irregular_t": irregular_t_list,
        "irregular_rv": irregular_rv_list,
        "irregular_err": irregular_err_list,
        "periodograms": periodogram_list,
        "best_periods": np.asarray(best_period_list, dtype=float),
        "true_periods": np.asarray(true_period_list, dtype=float),
        "best_period_feature": np.asarray(best_period_feature_list, dtype=float),
        "phi0": np.asarray(phase_offset_list, dtype=float),
        "Mstar": np.asarray(Mstar_list, dtype=float),
    }


# =========================================================
# Model wrapper
# =========================================================

class RVOrbitRegressor:
    def __init__(self, cfg: RVNNConfig):
        self.cfg = cfg

        if cfg.errors_mode == "none":
            input_dim = cfg.n_phase_points + 1
        elif cfg.errors_mode == "mean":
            input_dim = cfg.n_phase_points + 2
        elif cfg.errors_mode == "full":
            input_dim = 2 * cfg.n_phase_points + 1
        else:
            raise ValueError("errors_mode must be 'none', 'mean', or 'full'")

        self.model = NeuralNetwork(
            layer_sizes=[input_dim, *cfg.hidden_layer_sizes, 6],
            task="regression",
            hidden_activation=cfg.activation,
            normalize_X=True,
            normalize_y=True,
            optimizer=cfg.optimizer,
            batch_size=cfg.batch_size,
            seed=cfg.random_seed,
            init_scale=cfg.init_scale,
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_score": [],
            "val_score": [],
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model.fit(
            X_train,
            y_train,
            epochs=self.cfg.max_epochs,
            lr=self.cfg.learning_rate_init,
            verbose_every=0,
            X_val=X_val,
            y_val=y_val,
        )

        self.history["train_loss"] = list(self.model.loss_history)
        self.history["val_loss"] = list(self.model.val_loss_history)
        self.history["train_score"] = list(self.model.train_score_history)
        self.history["val_score"] = list(self.model.val_score_history)

        if self.cfg.print_every_epoch:
            print("\nEpoch history")
            for i in range(len(self.history["train_loss"])):
                trl = self.history["train_loss"][i]
                vall = self.history["val_loss"][i] if i < len(self.history["val_loss"]) else np.nan
                trs = self.history["train_score"][i] if i < len(self.history["train_score"]) else np.nan
                vas = self.history["val_score"][i] if i < len(self.history["val_score"]) else np.nan
                print(
                    f"Epoch {i + 1:03d} | "
                    f"train_loss={trl:.6f} | val_loss={vall:.6f} | "
                    f"train_R2={trs:.6f} | val_R2={vas:.6f}"
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)


# =========================================================
# Dataset split
# =========================================================

def prepare_rv_datasets(cfg: RVNNConfig) -> Dict[str, Any]:
    known = generate_phased_rv_dataset(cfg, n_samples=cfg.known_samples, seed_offset=0)
    new = generate_phased_rv_dataset(cfg, n_samples=cfg.new_samples, seed_offset=10_000)

    indices = np.arange(cfg.known_samples)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg.validation_fraction,
        random_state=cfg.random_seed,
    )

    train = {
        "phase": known["phase"],
        "X": known["X"][train_idx],
        "X_rv": known["X_rv"][train_idx],
        "X_err": known["X_err"][train_idx],
        "y": known["y"][train_idx],
        "Mstar": known["Mstar"][train_idx],
    }

    val = {
        "phase": known["phase"],
        "X": known["X"][val_idx],
        "X_rv": known["X_rv"][val_idx],
        "X_err": known["X_err"][val_idx],
        "y": known["y"][val_idx],
        "Mstar": known["Mstar"][val_idx],
    }

    return {
        "phase": known["phase"],
        "train": train,
        "val": val,
        "known_full": known,
        "new_unseen": new,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


# =========================================================
# Plotting
# =========================================================

def plot_one_example(
    dataset: Dict[str, Any],
    idx: int = 0,
    smooth_width: float = 0.06,
) -> None:
    phase = dataset["phase"]
    rv = dataset["X_rv"][idx]
    err = dataset["X_err"][idx]

    rv_smooth = smooth_folded_rv(phase, rv, err, width=smooth_width)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(
        phase,
        rv,
        yerr=err,
        fmt="o",
        ms=3.2,
        alpha=0.65,
        elinewidth=1.0,
        capsize=0,
        label="Folded RV data",
    )
    ax.plot(phase, rv_smooth, lw=2.2, label="Weighted smooth")

    ax.set_xlabel(PLOT_LABELS["phase"])
    ax.set_ylabel(PLOT_LABELS["rv"])
    ax.set_title("Example phased RV curve")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()



def plot_periodogram(result: Dict[str, Any], xscale: str = "log") -> None:
    period = result["period"]
    power = result["power"]
    best_period = result["best_period"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(period, power, lw=1.6)
    ax.axvline(best_period, ls="--", lw=1.4, label=rf"Best $P={best_period:.3f}$")

    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Lomb-Scargle power")
    ax.set_title("RV periodogram")
    ax.set_xscale(xscale)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_learning_curves(history: Dict[str, List[float]], smooth_window: int = 9) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    tr_loss = np.asarray(history["train_loss"], dtype=float)
    va_loss = np.asarray(history["val_loss"], dtype=float)
    tr_r2 = np.asarray(history["train_score"], dtype=float)
    va_r2 = np.asarray(history["val_score"], dtype=float)

    best_epoch = int(np.argmin(va_loss)) + 1 if len(va_loss) else None

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))

    axes[0].plot(epochs, tr_loss, alpha=0.25)
    axes[0].plot(epochs, smooth_series(tr_loss, smooth_window), lw=2.0, label="Train")

    if len(va_loss) == len(epochs):
        axes[0].plot(epochs, va_loss, alpha=0.25)
        axes[0].plot(epochs, smooth_series(va_loss, smooth_window), lw=2.0, label="Val.")

    if best_epoch is not None:
        axes[0].axvline(best_epoch, ls="--", lw=1.2, label=f"Best epoch = {best_epoch}")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(PLOT_LABELS["mse"])
    axes[0].set_title("Training history")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, tr_r2, alpha=0.25)
    axes[1].plot(epochs, smooth_series(tr_r2, smooth_window), lw=2.0, label="Train")

    if len(va_r2) == len(epochs):
        axes[1].plot(epochs, va_r2, alpha=0.25)
        axes[1].plot(epochs, smooth_series(va_r2, smooth_window), lw=2.0, label="Val.")

    if best_epoch is not None:
        axes[1].axvline(best_epoch, ls="--", lw=1.2, label=f"Best epoch = {best_epoch}")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(PLOT_LABELS["r2"])
    axes[1].set_title("Score history")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


PLOT_LABELS.update({
    "logP": r"$\log_{10} P$ [d]",
    "K": r"$K$ [$\mathrm{m\,s^{-1}}$]",

    "true_logP": r"$\log_{10} P_{\rm true}$",
    "pred_logP": r"$\log_{10} P_{\rm pred}$",

    "true_K": r"$K_{\rm true}$ [$\mathrm{m\,s^{-1}}$]",
    "pred_K": r"$K_{\rm pred}$ [$\mathrm{m\,s^{-1}}$]",
})

def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> None:
    n_params = y_true.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(4.2 * n_params, 4.0))
    axes = np.atleast_1d(axes)

    for j, ax in enumerate(axes):
        name = names[j]

        ax.scatter(y_true[:, j], y_pred[:, j], s=12, alpha=0.7)

        mn = min(y_true[:, j].min(), y_pred[:, j].min())
        mx = max(y_true[:, j].max(), y_pred[:, j].max())
        ax.plot([mn, mx], [mn, mx], "--", lw=1.2)

        xlabel = PLOT_LABELS.get(f"true_{name}", f"True {name}")
        ylabel = PLOT_LABELS.get(f"pred_{name}", f"Pred {name}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# =========================================================
# Inference on irregular data
# =========================================================
def reconstruct_orbital_params(
    y_pred: np.ndarray,
    Mstar_msun: np.ndarray,
) -> Dict[str, np.ndarray]:
    logP = y_pred[:, 0]
    K = y_pred[:, 1]
    gamma = y_pred[:, 2]
    h = y_pred[:, 3]
    k = y_pred[:, 4]
    phi0 = np.mod(y_pred[:, 5], 1.0)

    P = 10.0 ** logP
    e = np.sqrt(h**2 + k**2)
    e = np.clip(e, 0.0, 0.99)
    omega = np.arctan2(h, k)

    Msini = msini_from_params(P, K, e, Mstar_msun)

    return {
        "logP": logP,
        "K": K,
        "P": P,
        "Msini": Msini,
        "gamma": gamma,
        "h": h,
        "k": k,
        "e": e,
        "omega": omega,
        "phi0": phi0,
        "Mstar": np.asarray(Mstar_msun, dtype=float),
    }


def predict_from_irregular_rv(
    model: RVOrbitRegressor,
    cfg: RVNNConfig,
    t: np.ndarray,
    rv: np.ndarray,
    rv_err: np.ndarray,
    Mstar_msun: float,
    P_min: float = 1.0,
    P_max: Optional[float] = None,
    samples_per_peak: int = 10,
) -> Dict[str, Any]:
    periodogram = compute_rv_periodogram(
        t=t,
        rv=rv,
        rv_err=rv_err,
        P_min=P_min,
        P_max=P_max,
        samples_per_peak=samples_per_peak,
    )

    best_period = periodogram["best_period"]
    phase = phase_fold(t, best_period)
    phase_grid = np.linspace(0.0, 1.0, cfg.n_phase_points, endpoint=False)

    rv_grid, err_grid = bin_folded_rv_to_grid(phase, rv, rv_err, phase_grid)
    x_core = build_rv_input(rv_grid, err_grid, cfg)
    x = np.concatenate([x_core, np.array([np.log10(best_period)], dtype=float)]).reshape(1, -1)

    y_pred = model.predict(x)
    pars = reconstruct_orbital_params(y_pred, np.array([Mstar_msun]))

    return {
        "best_period": best_period,
        "periodogram": periodogram,
        "phase": phase,
        "phase_grid": phase_grid,
        "rv_grid": rv_grid,
        "err_grid": err_grid,
        "y_pred": y_pred,
        "predicted_params": {k: v[0] for k, v in pars.items()},
    }

def plot_folded_rv_from_irregular(
    phase: np.ndarray,
    rv: np.ndarray,
    rv_err: np.ndarray,
    phase_grid: np.ndarray,
    rv_grid: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(phase, rv, yerr=rv_err, fmt="o", ms=4, alpha=0.7, label="Folded data")
    ax.plot(phase_grid, rv_grid, lw=2, label="Binned / gridded")
    ax.set_xlabel(PLOT_LABELS["phase"])
    ax.set_ylabel(PLOT_LABELS["rv"])
    ax.set_title("Folded irregular RV data")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def _best_text_box_corner(
    phase: np.ndarray,
    rv: np.ndarray,
) -> Tuple[float, float, str, str]:
    """
    Choose among the 4 corners by minimizing how many points fall
    into the corresponding corner box region.
    """
    phase = np.asarray(phase, dtype=float)
    rv = np.asarray(rv, dtype=float)

    rv_min = np.min(rv)
    rv_max = np.max(rv)
    rv_span = max(rv_max - rv_min, 1e-8)

    candidates = [
        (0.03, 0.97, "left",  "top",    (0.00, 0.38, 0.62, 1.00)),  # upper-left
        (0.97, 0.97, "right", "top",    (0.62, 1.00, 0.62, 1.00)),  # upper-right
        (0.03, 0.03, "left",  "bottom", (0.00, 0.38, 0.00, 0.38)),  # lower-left
        (0.97, 0.03, "right", "bottom", (0.62, 1.00, 0.00, 0.38)),  # lower-right
    ]

    best = None
    best_score = np.inf

    for x, y, ha, va, (px0, px1, py0, py1) in candidates:
        y_norm = (rv - rv_min) / rv_span
        mask = (
            (phase >= px0) & (phase <= px1) &
            (y_norm >= py0) & (y_norm <= py1)
        )
        score = np.sum(mask)

        if score < best_score:
            best_score = score
            best = (x, y, ha, va)

    return best

def plot_example_grid(
    model: RVOrbitRegressor,
    dataset: Dict[str, Any],
    n_examples: int = 9,
    seed: int = 123,
) -> None:
    rng = np.random.default_rng(seed)
    n_available = len(dataset["X"])
    idxs = rng.choice(n_available, size=min(n_examples, n_available), replace=False)

    y_pred = model.predict(dataset["X"][idxs])
    y_true = dataset["y"][idxs]
    Mstar_sel = dataset["Mstar"][idxs]

    pred_pars = reconstruct_orbital_params(y_pred, Mstar_sel)

    ncols = 3
    nrows = int(np.ceil(len(idxs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.6 * ncols, 4.3 * nrows),
        squeeze=False
    )

    phase = np.asarray(dataset["phase"], dtype=float)
    phase_grid = np.linspace(0.0, 1.0, 600)

    for i_ax, (ax, idx, yt, i_loc) in enumerate(zip(axes.ravel(), idxs, y_true, range(len(idxs)))):
        rv = dataset["X_rv"][idx].copy()
        err = dataset["X_err"][idx].copy()

        ax.errorbar(phase, rv, yerr=err, fmt="o", ms=2.2, alpha=0.55)

        logPt, Kt, gt, ht, kt, phi0t = yt
        Pt = 10.0 ** logPt
        et = np.clip(np.sqrt(ht**2 + kt**2), 0.0, 0.99)
        omegat = np.arctan2(ht, kt)
        Mt = msini_from_params(Pt, Kt, et, Mstar_sel[i_loc])

        Pp = pred_pars["P"][i_loc]
        Kp = pred_pars["K"][i_loc]
        gp = pred_pars["gamma"][i_loc]
        hp = pred_pars["h"][i_loc]
        kp = pred_pars["k"][i_loc]
        ep = pred_pars["e"][i_loc]
        omegap = pred_pars["omega"][i_loc]
        phi0p = pred_pars["phi0"][i_loc]
        Mp = pred_pars["Msini"][i_loc]

        rv_true_model = keplerian_rv_from_phase(
            phase_grid,
            K=Kt,
            e=et,
            omega=omegat,
            gamma=gt,
            phi0=phi0t,
        )
        rv_pred_model = keplerian_rv_from_phase(
            phase_grid,
            K=Kp,
            e=ep,
            omega=omegap,
            gamma=gp,
            phi0=phi0p,
        )

        ax.plot(phase_grid, rv_true_model, lw=2.0, ls="--", label="True")
        ax.plot(phase_grid, rv_pred_model, lw=2.0, label="Pred")

        row = i_ax // ncols
        if row == nrows - 1:
            ax.set_xlabel(PLOT_LABELS["phase"])
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if i_ax % ncols == 0:
            ax.set_ylabel(PLOT_LABELS["rv"])
        else:
            ax.set_ylabel("")

        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.25)

        txt = (
            rf"$P_t={Pt:.2f}$ d   "
            rf"$P_p={Pp:.2f}$ d   "
            rf"$K_t={Kt:.1f}$   "
            rf"$K_p={Kp:.1f}$" "\n"
            rf"$(M\sin i)_t={Mt:.2f}\,M_J$   "
            rf"$(M\sin i)_p={Mp:.2f}\,M_J$   "
            rf"$\gamma_t={gt:.1f}$   "
            rf"$\gamma_p={gp:.1f}$" "\n"
            rf"$e_t={et:.2f}$   "
            rf"$e_p={ep:.2f}$   "
            r"$\phi_{0,t}$"+rf"$={phi0t:.2f}$   "
            r"$\phi_{0,p}$"+rf"$={phi0p:.2f}$"
        )

        ax.text(
            0.5, 1.03, txt,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=11
        )

    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Training examples with true and predicted parameters", y=1.04, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# =========================================================
# Main experiment
# =========================================================

def run_rv_orbit_regression_experiment(
    cfg: Optional[RVNNConfig] = None,
    make_plots: bool = True,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = RVNNConfig()

    data = prepare_rv_datasets(cfg)
    model = RVOrbitRegressor(cfg)

    model.fit(
        data["train"]["X"],
        data["train"]["y"],
        data["val"]["X"],
        data["val"]["y"],
    )

    y_train_pred = model.predict(data["train"]["X"])
    y_val_pred = model.predict(data["val"]["X"])
    y_new_pred = model.predict(data["new_unseen"]["X"])

    train_r2 = model.score(data["train"]["X"], data["train"]["y"])
    val_r2 = model.score(data["val"]["X"], data["val"]["y"])
    new_r2 = model.score(data["new_unseen"]["X"], data["new_unseen"]["y"])

    final_train_loss = model.history["train_loss"][-1]
    final_val_loss = model.history["val_loss"][-1]
    final_train_score = model.history["train_score"][-1]
    final_val_score = model.history["val_score"][-1]
    best_epoch = int(np.argmin(model.history["val_loss"])) + 1

    if make_plots:
        plot_one_example(
            data["known_full"],
            idx=0,
            smooth_width=cfg.smooth_phase_width,
        )
        plot_learning_curves(model.history)
        plot_true_vs_pred(data["val"]["y"], y_val_pred, cfg.target_names)

        plot_example_grid(model, data["train"], n_examples=9)

    print("\n========== SUMMARY ==========")
    print(f"Final train loss    : {final_train_loss:.6f}")
    print(f"Final val loss      : {final_val_loss:.6f}")
    print(f"Final train R^2     : {final_train_score:.6f}")
    print(f"Final val R^2       : {final_val_score:.6f}")
    print(f"Best epoch (val)    : {best_epoch}")
    print(f"Global train R^2    : {train_r2:.6f}")
    print(f"Global val   R^2    : {val_r2:.6f}")
    print(f"Global new   R^2    : {new_r2:.6f}")

    return {
        "config": cfg,
        "data": data,
        "model": model,
        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "y_new_pred": y_new_pred,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "new_r2": new_r2,
    }


if __name__ == "__main__":
    cfg = RVNNConfig(
        n_phase_points=160,
        known_samples=1500,
        new_samples=300,
        hidden_layer_sizes=(128, 64, 32),
        activation="tanh",
        learning_rate_init=3e-3,
        max_epochs=100,
        init_scale=0.05,
        optimizer="adam",
        batch_size=32,
        errors_mode="full",
        rv_err_min=1.5,
        rv_err_max=8.0,
        jitter_scale=0.35,
        trend_scale=1.0,
        activity_amp_max=4.0,
        random_seed=42,
    )
    run_rv_orbit_regression_experiment(cfg, make_plots=True)