from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import copy

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = [
    "Transit",
    "Pulsator",
    "Eclipsing binary",
    "Starspot modulation",
]


@dataclass
class LightCurveNNConfig:
    """Configuration for the synthetic light-curve morphology experiment."""

    n_phase_points: int = 240
    known_per_class: int = 40
    new_per_class: int = 15

    random_seed: int = 42

    # Neural network
    hidden_layer_sizes: Tuple[int, int] = (96, 48)
    activation: str = "relu"
    learning_rate_init: float = 8e-4
    alpha: float = 3e-4
    batch_size: int = 16
    max_epochs: int = 250
    patience: int = 25
    validation_fraction: float = 0.25

    # Data generation
    baseline_flux: float = 1.0
    base_noise_min: float = 0.002
    base_noise_max: float = 0.02

    @property
    def class_names(self) -> List[str]:
        return list(CLASS_NAMES)

    @property
    def n_classes(self) -> int:
        return len(CLASS_NAMES)

def wrap_phase_diff(phase: np.ndarray, center: float) -> np.ndarray:
    """Signed minimum phase difference on a circular domain [0, 1)."""
    return ((phase - center + 0.5) % 1.0) - 0.5


def add_correlated_noise(
    flux: np.ndarray,
    rng: np.random.Generator,
    white_std: float,
    corr_std: float = 0.0,
    kernel_size: int = 9,
) -> np.ndarray:
    """Add white + weak correlated noise."""
    noisy = flux + rng.normal(0.0, white_std, size=flux.size)

    if corr_std > 0:
        kernel_size = max(3, int(kernel_size))
        kernel = np.ones(kernel_size, dtype=float)
        kernel /= kernel.sum()
        corr = rng.normal(0.0, corr_std, size=flux.size)
        corr = np.convolve(corr, kernel, mode="same")
        noisy += corr

    return noisy


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    eye = np.eye(n_classes, dtype=int)
    return eye[y]


def smooth_box(x: np.ndarray, center: float, width: float, edge: float) -> np.ndarray:
    """Smooth top-hat profile based on two logistic edges."""
    left = 1.0 / (1.0 + np.exp(-(x - (center - width / 2.0)) / edge))
    right = 1.0 / (1.0 + np.exp(-(x - (center + width / 2.0)) / edge))
    return left - right


def generate_transit_curve(
    phase: np.ndarray,
    rng: np.random.Generator,
    cfg: LightCurveNNConfig,
) -> np.ndarray:
    """Synthetic transit-like phased light curve."""
    depth = rng.uniform(0.008, 0.06)
    width = rng.uniform(0.025, 0.10)
    center = rng.uniform(0.35, 0.65)
    edge = rng.uniform(0.0025, 0.010)

    profile = smooth_box(phase, center=center, width=width, edge=edge)

    # Slight baseline curvature / trend
    slope = rng.uniform(-0.01, 0.01)
    quad = rng.uniform(-0.01, 0.01)
    x = phase - 0.5
    baseline = cfg.baseline_flux + slope * x + quad * x**2

    flux = baseline - depth * profile

    # Optional weak in-transit anomaly (e.g. spot-crossing-like small bump)
    if rng.random() < 0.35:
        bump_amp = rng.uniform(0.0, 0.20 * depth)
        bump_center = center + rng.uniform(-0.25 * width, 0.25 * width)
        bump_width = rng.uniform(0.08 * width, 0.25 * width)
        dphi = wrap_phase_diff(phase, bump_center)
        flux += bump_amp * np.exp(-0.5 * (dphi / bump_width) ** 2)

    return flux

def generate_pulsator_curve(
    phase: np.ndarray,
    rng: np.random.Generator,
    cfg: LightCurveNNConfig,
) -> np.ndarray:
    """Synthetic pulsating-star light curve with asymmetric morphology."""
    amp = rng.uniform(0.04, 0.14)

    # Phase shift
    phi = rng.uniform(0.0, 1.0)
    ph = (phase + phi) % 1.0

    # Build an asymmetric waveform:
    # steeper rise + slower decline using harmonics
    a2 = rng.uniform(0.25, 0.55)
    a3 = rng.uniform(0.05, 0.20)

    base = (
        np.sin(2 * np.pi * ph)
        + a2 * np.sin(4 * np.pi * ph)
        + a3 * np.sin(6 * np.pi * ph)
    )

    # Normalize to roughly [-1, 1]
    base /= np.max(np.abs(base)) + 1e-8

    flux = cfg.baseline_flux + amp * base

    # Small cycle-to-cycle style distortion
    flux += rng.uniform(-0.01, 0.01) * (phase - 0.5)

    return flux

def generate_eclipsing_binary_curve(
    phase: np.ndarray,
    rng: np.random.Generator,
    cfg: LightCurveNNConfig,
) -> np.ndarray:
    """Synthetic eclipsing-binary light curve."""
    primary_depth = rng.uniform(0.10, 0.35)
    secondary_depth = primary_depth * rng.uniform(0.10, 0.75)

    primary_width = rng.uniform(0.035, 0.12)
    secondary_width = rng.uniform(0.03, 0.11)

    primary_center = rng.uniform(0.42, 0.58)
    secondary_center = (primary_center + 0.5 + rng.normal(0.0, 0.03)) % 1.0

    edge1 = rng.uniform(0.003, 0.012)
    edge2 = rng.uniform(0.003, 0.012)

    primary = smooth_box(phase, center=primary_center, width=primary_width, edge=edge1)
    secondary = smooth_box(phase, center=secondary_center, width=secondary_width, edge=edge2)

    ellipsoidal_amp = rng.uniform(0.0, 0.025)
    oconnell = rng.uniform(-0.01, 0.01)

    flux = (
        cfg.baseline_flux
        - primary_depth * primary
        - secondary_depth * secondary
        + ellipsoidal_amp * np.cos(4 * np.pi * (phase - primary_center))
        + oconnell * np.sin(2 * np.pi * (phase - primary_center))
    )

    return flux

def generate_spotted_star_curve(
    phase: np.ndarray,
    rng: np.random.Generator,
    cfg: LightCurveNNConfig,
) -> np.ndarray:
    """Synthetic rotational starspot modulation."""
    phi1 = rng.uniform(0, 2 * np.pi)
    phi2 = rng.uniform(0, 2 * np.pi)

    amp1 = rng.uniform(0.02, 0.10)
    amp2 = rng.uniform(0.0, 0.80 * amp1)

    flux = (
        cfg.baseline_flux
        + amp1 * np.sin(2 * np.pi * phase + phi1)
        + amp2 * np.sin(4 * np.pi * phase + phi2)
    )

    # Broad spot-induced depression / enhancement
    bump_amp = rng.uniform(-0.04, 0.02)
    bump_center = rng.uniform(0.1, 0.9)
    bump_width = rng.uniform(0.06, 0.18)
    dphi = wrap_phase_diff(phase, bump_center)
    flux += bump_amp * np.exp(-0.5 * (dphi / bump_width) ** 2)

    return flux

_GENERATORS = {
    0: generate_transit_curve,
    1: generate_pulsator_curve,
    2: generate_eclipsing_binary_curve,
    3: generate_spotted_star_curve,
}


def add_noise_and_normalize(flux: np.ndarray, rng: np.random.Generator, cfg: LightCurveNNConfig) -> np.ndarray:
    noisy = flux + rng.normal(0.0, rng.uniform(cfg.base_noise_min, cfg.base_noise_max), size=flux.size)
    median = np.median(noisy)
    if median != 0:
        noisy = noisy / median
    return noisy


def generate_dataset(
    cfg: LightCurveNNConfig,
    n_per_class: int,
    seed_offset: int = 0,
) -> Dict[str, np.ndarray]:
    """Generate a labeled synthetic dataset of phased light curves."""
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    phase = np.linspace(0.0, 1.0, cfg.n_phase_points, endpoint=False)

    curves: List[np.ndarray] = []
    y_int: List[int] = []

    for class_id in range(cfg.n_classes):
        generator = _GENERATORS[class_id]
        for _ in range(n_per_class):
            base_flux = generator(phase, rng, cfg)
            curve = add_noise_and_normalize(base_flux, rng, cfg)
            curves.append(curve)
            y_int.append(class_id)

    X = np.asarray(curves, dtype=float)
    y = np.asarray(y_int, dtype=int)
    y_onehot = one_hot_encode(y, cfg.n_classes)

    return {
        "phase": phase,
        "X": X,
        "y": y,
        "y_onehot": y_onehot,
        "class_names": np.array(cfg.class_names),
    }


class LightCurveMorphologyNN:
    """Small MLP classifier for synthetic phased light-curve morphology."""

    def __init__(self, cfg: LightCurveNNConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            activation=cfg.activation,
            solver="adam",
            alpha=cfg.alpha,
            learning_rate_init=cfg.learning_rate_init,
            batch_size=max(1, cfg.batch_size),
            max_iter=1,
            warm_start=True,
            shuffle=True,
            random_state=cfg.random_seed,
        )
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self.best_state: Optional[Dict[str, Sequence[np.ndarray]]] = None
        self.best_val_loss = np.inf

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        Xtr = self.scaler.fit_transform(X_train)
        Xva = self.scaler.transform(X_val)
        classes = np.arange(self.cfg.n_classes)
        patience_counter = 0

        for epoch in range(self.cfg.max_epochs):
            if epoch == 0:
                self.model.partial_fit(Xtr, y_train, classes=classes)
            else:
                self.model.partial_fit(Xtr, y_train)

            p_tr = self.model.predict_proba(Xtr)
            p_va = self.model.predict_proba(Xva)

            train_loss = log_loss(y_train, p_tr, labels=classes)
            val_loss = log_loss(y_val, p_va, labels=classes)
            train_acc = accuracy_score(y_train, np.argmax(p_tr, axis=1))
            val_acc = accuracy_score(y_val, np.argmax(p_va, axis=1))

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {
                    "coefs_": copy.deepcopy(self.model.coefs_),
                    "intercepts_": copy.deepcopy(self.model.intercepts_),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.patience:
                break

        if self.best_state is not None:
            self.model.coefs_ = self.best_state["coefs_"]
            self.model.intercepts_ = self.best_state["intercepts_"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)


def prepare_datasets(cfg: LightCurveNNConfig) -> Dict[str, Dict[str, np.ndarray]]:
    known = generate_dataset(cfg, n_per_class=cfg.known_per_class, seed_offset=0)
    new = generate_dataset(cfg, n_per_class=cfg.new_per_class, seed_offset=10_000)

    X_train, X_val, y_train, y_val = train_test_split(
        known["X"],
        known["y"],
        test_size=cfg.validation_fraction,
        stratify=known["y"],
        random_state=cfg.random_seed,
    )

    split = {
        "phase": known["phase"],
        "train": {"X": X_train, "y": y_train},
        "val": {"X": X_val, "y": y_val},
        "known_full": known,
        "new_unseen": new,
    }
    return split


def plot_one_example_per_class(dataset: Dict[str, np.ndarray], cfg: LightCurveNNConfig) -> None:
    phase = dataset["phase"]
    X = dataset["X"]
    y = dataset["y"]

    fig, axes = plt.subplots(cfg.n_classes, 1, figsize=(10, 2.5 * cfg.n_classes), sharex=True)
    if cfg.n_classes == 1:
        axes = [axes]

    for class_id, ax in enumerate(axes):
        idx = np.where(y == class_id)[0][0]
        ax.plot(phase, X[idx], lw=1.5)
        ax.set_ylabel("Normalized flux")
        ax.set_title(cfg.class_names[class_id])
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Phase")
    plt.tight_layout()
    plt.show()


def plot_learning_curves(history: Dict[str, List[float]]) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training history")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy history")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    _, ax = plt.subplots(figsize=(6.2, 5.4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")

    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_new_examples_with_predictions(
    phase: np.ndarray,
    X_new: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    n_show: int = 12,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    n_show = min(n_show, len(X_new))
    idx = rng.choice(len(X_new), size=n_show, replace=False)

    ncols = 3
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for panel, i in enumerate(idx):
        ax = axes[panel]
        ax.plot(phase, X_new[i], lw=1.2)
        ax.set_title(
            f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}",
            fontsize=10,
        )
        ax.grid(alpha=0.25)

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel("Phase")
    for k in range(0, len(axes), ncols):
        axes[k].set_ylabel("Flux")

    plt.tight_layout()
    plt.show()

def plot_new_examples_by_class(
    phase: np.ndarray,
    X_new: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    per_class: int = 3,
) -> None:
    n_classes = len(class_names)
    n_show = per_class * n_classes
    ncols = per_class
    nrows = n_classes

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 2.6 * nrows),
        sharex=True
    )
    axes = np.atleast_2d(axes)

    for c in range(n_classes):
        idx_class = np.where(y_true == c)[0][:per_class]

        for j in range(per_class):
            ax = axes[c, j]

            if j < len(idx_class):
                i = idx_class[j]
                ax.plot(phase, X_new[i], lw=1.2)
                ax.set_title(
                    f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}",
                    fontsize=10,
                )
                ax.grid(alpha=0.25)
            else:
                ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Phase")

    for r in range(nrows):
        axes[r, 0].set_ylabel("Flux")

    plt.tight_layout()
    plt.show()


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> None:
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}")
    for class_id, class_name in enumerate(class_names):
        count = np.sum(y_true == class_id)
        correct = np.sum((y_true == class_id) & (y_pred == class_id))
        print(f"  {class_name:<20s} {correct:>3d} / {count:<3d} correct")


def run_lightcurve_morphology_experiment(
    cfg: Optional[LightCurveNNConfig] = None,
    make_plots: bool = True,
) -> Dict[str, object]:
    """Run the full synthetic-astrophysics classification experiment.

    Returns a dictionary with datasets, fitted model, metrics, and predictions so the module
    can be imported and used cleanly from Jupyter.
    """
    if cfg is None:
        cfg = LightCurveNNConfig()

    data = prepare_datasets(cfg)
    model = LightCurveMorphologyNN(cfg)
    model.fit(
        data["train"]["X"],
        data["train"]["y"],
        data["val"]["X"],
        data["val"]["y"],
    )

    train_pred = model.predict(data["train"]["X"])
    val_pred = model.predict(data["val"]["X"])
    known_pred = model.predict(data["known_full"]["X"])
    new_pred = model.predict(data["new_unseen"]["X"])

    train_acc = accuracy_score(data["train"]["y"], train_pred)
    val_acc = accuracy_score(data["val"]["y"], val_pred)
    known_acc = accuracy_score(data["known_full"]["y"], known_pred)
    new_acc = accuracy_score(data["new_unseen"]["y"], new_pred)

    if make_plots:
        plot_one_example_per_class(data["known_full"], cfg)
        plot_learning_curves(model.history)
        plot_confusion(data["known_full"]["y"], known_pred, cfg.class_names, "Confusion matrix: known labeled set")
        plot_confusion(data["new_unseen"]["y"], new_pred, cfg.class_names, "Confusion matrix: new unseen set")
        plot_new_examples_with_predictions(
            data["phase"],
            data["new_unseen"]["X"],
            data["new_unseen"]["y"],
            new_pred,
            cfg.class_names,
            n_show=min(12, len(data["new_unseen"]["X"])),
        )
        plot_new_examples_by_class(
            data["phase"],
            data["new_unseen"]["X"],
            data["new_unseen"]["y"],
            new_pred,
            cfg.class_names,
            per_class=3,
        )

    print("=== Known labeled set ===")
    summarize_predictions(data["known_full"]["y"], known_pred, cfg.class_names)
    print("\n=== New unseen set ===")
    summarize_predictions(data["new_unseen"]["y"], new_pred, cfg.class_names)

    return {
        "config": cfg,
        "data": data,
        "model": model,
        "known_predictions": known_pred,
        "new_predictions": new_pred,
        "known_accuracy": known_acc,
        "new_accuracy": new_acc,
    }


if __name__ == "__main__":
    run_lightcurve_morphology_experiment(LightCurveNNConfig(), make_plots=True)
