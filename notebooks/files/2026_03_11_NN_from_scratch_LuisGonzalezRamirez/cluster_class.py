# ---------------------------------------------------------
# A tiny 2→2→1 neural net + geometry plots + “loss-surface model”
# for two proper-motion clusters (mu_alpha*, mu_delta) in mas/yr.
#
# Usage:
#   from cluster_class import ProperMotionNNmodel
#   model = ProperMotionNNmodel(X_raw, y)   # X_raw: (N,2), y: (N,1) in {0,1}
#   model.plot_clusters_two_panels(title_pad=18)
#   model.plot_geometry_raw(title="Initial geometry", title_pad=18)
#   model.train(steps=800, lr=0.2)
#   model.plot_geometry_raw(title="After training", title_pad=18)
#   model.model_two_planes(w0_range=(-8,8), w1_range=(-8,8), grid=80, steps=40, lr=0.12)
# ---------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator

try:
    # Only needed for notebook-style animation
    from IPython.display import clear_output
except Exception:  # pragma: no cover
    clear_output = None


Array = np.ndarray


@dataclass
class NNParams:
    W: Array  # (2,2)
    b: Array  # (2,)
    v: Array  # (2,1)
    c: Array  # (1,)

class DataSelection:
    """
    This class is a placeholder for any data selection logic you might want to implement.
    """

    def __init__(self, ax):
        self.ax = ax
        pass

    def make_two_clusters(self, N1=200, N2=200, seed=0):
        rng = np.random.default_rng(seed)

        # Cluster 1
        mean1 = np.array([5.3, -1.0])  # mas/yr
        cov1  = np.array([[0.15, 0.02],
                        [0.02, 0.12]])

        # Cluster 2
        mean2 = np.array([4.0, -1.7])  # mas/yr
        cov2  = np.array([[0.20, -0.03],
                        [-0.03, 0.18]])

        X1 = rng.multivariate_normal(mean1, cov1, size=N1)
        X2 = rng.multivariate_normal(mean2, cov2, size=N2)

        X = np.vstack([X1, X2])                 # (N,2)
        y = np.vstack([np.zeros((N1,1)), np.ones((N2,1))])  # (N,1) -> 0:A, 1:B

        # Normalize for tanh
        mu = X.mean(axis=0)
        sig = X.std(axis=0)
        Xn = (X - mu) / sig

        return X1, X2, X, Xn, y, mu, sig

    def plot_cov_ellipse(self, data, n_std=1.0, **kwargs):
        cov = np.cov(data.T)
        mean = data.mean(axis=0)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))

        width, height = 2 * n_std * np.sqrt(eigvals)

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            **kwargs,
        )
        self.ax.add_patch(ellipse)


class ProperMotionNNmodel:
    """
    Geometric interpretation (key idea):
      Each hidden neuron i defines a line in *normalized* input space:
          z_i = w_i^T x_norm + b_i = 0
      With tanh, the neuron transitions smoothly across that line.

    In the geometry plot (RAW mas/yr):
      We draw those two hidden-neuron lines mapped back into raw coordinates.
      Those are the two “cuts” the hidden layer is making in the proper-motion plane.
      The final decision boundary is y_hat = 0.5 (a nonlinear combination of both cuts).
    """

    def __init__(
        self,
        X_raw: Array,
        y: Array,
        seed: int = 42,
        init_scale: float = 0.7,
    ):
        """
        Parameters
        ----------
        X_raw : (N,2) array
            Proper motions in mas/yr: [mu_alpha*, mu_delta]
        y : (N,1) array
            Class labels {0,1}
        """
        X_raw = np.asarray(X_raw, dtype=float)
        y = np.asarray(y, dtype=float)

        if X_raw.ndim != 2 or X_raw.shape[1] != 2:
            raise ValueError("X_raw must have shape (N,2).")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("y must have shape (N,1).")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("y must contain only 0/1 labels.")

        self.X_raw = X_raw
        self.y = y

        # Normalization for the NN (tanh is sensitive to scale)
        self.mu = self.X_raw.mean(axis=0)
        self.sig = self.X_raw.std(axis=0)
        self.X_norm = (self.X_raw - self.mu) / self.sig

        self.rng = np.random.default_rng(seed)
        self.params = self._init_params(scale=init_scale)

    # ----------------------------
    # NN core
    # ----------------------------
    @staticmethod
    def tanh(z: Array) -> Array:
        return np.tanh(z)

    @staticmethod
    def dtanh(z: Array) -> Array:
        return 1.0 - np.tanh(z) ** 2

    @staticmethod
    def sigmoid(z: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def bce(y_hat: Array, y: Array, eps: float = 1e-9) -> float:
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def _init_params(self, scale: float = 0.7) -> NNParams:
        return NNParams(
            W=scale * self.rng.standard_normal((2, 2)),
            b=np.zeros(2, dtype=float),
            v=scale * self.rng.standard_normal((2, 1)),
            c=np.zeros(1, dtype=float),
        )

    def forward(self, X_norm: Array, params: Optional[NNParams] = None) -> Tuple[Array, Dict[str, Array]]:
        p = params or self.params
        Z = X_norm @ p.W + p.b  # (N,2)
        A = self.tanh(Z)        # (N,2)
        logits = A @ p.v + p.c  # (N,1)
        y_hat = self.sigmoid(logits)
        cache = {"X": X_norm, "Z": Z, "A": A, "y_hat": y_hat}
        return y_hat, cache

    def backward(self, y: Array, cache: Dict[str, Array], params: Optional[NNParams] = None) -> NNParams:
        p = params or self.params
        X, Z, A, y_hat = cache["X"], cache["Z"], cache["A"], cache["y_hat"]
        N = X.shape[0]

        # BCE with sigmoid: dL/dlogits = (y_hat - y)/N
        dlogits = (y_hat - y) / N  # (N,1)

        dv = A.T @ dlogits                 # (2,1)
        dc = np.sum(dlogits, axis=0)       # (1,)

        dA = dlogits @ p.v.T               # (N,2)
        dZ = dA * self.dtanh(Z)            # (N,2)

        dW = X.T @ dZ                      # (2,2)
        db = np.sum(dZ, axis=0)            # (2,)

        return NNParams(W=dW, b=db, v=dv, c=dc)

    def step(self, grads: NNParams, lr: float = 0.2) -> None:
        self.params.W -= lr * grads.W
        self.params.b -= lr * grads.b
        self.params.v -= lr * grads.v
        self.params.c -= lr * grads.c

    def train(self, steps: int = 800, lr: float = 0.2, verbose_every: int = 200) -> None:
        for i in range(steps):
            y_hat, cache = self.forward(self.X_norm)
            grads = self.backward(self.y, cache)
            self.step(grads, lr=lr)
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"Iter {i+1:4d} | BCE = {self.bce(y_hat, self.y):.4f}")

    def predict_proba(self, X_raw_new: Array) -> Array:
        X_raw_new = np.asarray(X_raw_new, dtype=float)

        if X_raw_new.ndim == 1:
            X_raw_new = X_raw_new.reshape(1, -1)

        if X_raw_new.ndim != 2 or X_raw_new.shape[1] != 2:
            raise ValueError("X_raw_new must have shape (N,2) or (2,).")

        Xn = (X_raw_new - self.mu) / self.sig
        y_hat, _ = self.forward(Xn)
        return y_hat[:, 0]


    def predict(self, X_raw_new: Array, threshold: float = 0.5) -> Array:
        prob = self.predict_proba(X_raw_new)
        return (prob >= threshold).astype(int)


    def predict_with_reject(
        self,
        X_raw_new: Array,
        threshold: float = 0.5,
        confidence_threshold: float = 0.10,
    ):
        """
        Predict with optional rejection for the MLP model.

        Rejection rule:
        if |P(class 1) - 0.5| < confidence_threshold
        -> label = -1

        Returns
        -------
        labels : (N,) array in {-1,0,1}
        prob_B : (N,) probability of class 1
        confidence : (N,) distance to 0.5
        """
        prob_B = self.predict_proba(X_raw_new)
        confidence = np.abs(prob_B - 0.5)

        labels = (prob_B >= threshold).astype(int)
        labels[confidence < confidence_threshold] = -1

        return labels, prob_B, confidence

    # ----------------------------
    # Data plots (RAW mas/yr)
    # ----------------------------
    def plot_clusters_two_panels(
        self,
        title_pad: int = 18,
        dark_face: str = "navy",
        light_face: str = "gold",
    ) -> None:
        """
        Two horizontal subplots:
          left: class 0 only
          right: class 1 only

        Styling requirement:
          - dark points => white edge
          - light points => black edge
        """
        X0 = self.X_raw[self.y[:, 0] == 0]
        X1 = self.X_raw[self.y[:, 0] == 1]

        _, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Class 0 (dark) with white edge
        ax[0].scatter(
            X0[:, 0], X0[:, 1],
            s=22, alpha=0.85,
            facecolor=dark_face,
            edgecolor="white",
            linewidth=0.7,
            label="Cluster A (class 0)"
        )
        ax[0].set_title("Proper motion: Cluster A", pad=title_pad)
        ax[0].set_xlabel(r"$\mu_{\alpha*}$  [mas yr$^{-1}$]")
        ax[0].set_ylabel(r"$\mu_{\delta}$   [mas yr$^{-1}$]")
        ax[0].grid(alpha=0.25)
        ax[0].legend(framealpha=1.0, facecolor="white", fontsize=12)

        # Class 1 (light) with black edge
        ax[1].scatter(
            X1[:, 0], X1[:, 1],
            s=22, alpha=0.85,
            facecolor=light_face,
            edgecolor="black",
            linewidth=0.7,
            label="Cluster B (class 1)"
        )
        ax[1].set_title("Proper motion: Cluster B", pad=title_pad)
        ax[1].set_xlabel(r"$\mu_{\alpha*}$  [mas yr$^{-1}$]")
        ax[1].set_ylabel(r"$\mu_{\delta}$   [mas yr$^{-1}$]")
        ax[1].grid(alpha=0.25)
        ax[1].legend(framealpha=1.0, facecolor="white", fontsize=12)

        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Geometry plot (RAW mas/yr)
    # ----------------------------
    def _hidden_line_raw_coeffs(self, neuron_index: int) -> Tuple[float, float, float]:
        """
        Hidden neuron i line in normalized space:
            w^T x_norm + b = 0

        Map to RAW coordinates:
            x_norm = (x_raw - mu)/sig
        => (w/sig)^T x_raw + (b - sum(w*mu/sig)) = 0
           a*x + b*y + d = 0
        """
        w = self.params.W[:, neuron_index]
        bi = self.params.b[neuron_index]

        a = w[0] / self.sig[0]
        b = w[1] / self.sig[1]
        d = bi - (w[0] * self.mu[0] / self.sig[0] + w[1] * self.mu[1] / self.sig[1])
        return float(a), float(b), float(d)

    def plot_geometry_raw(
        self,
        title: str = "NN geometry in proper-motion space",
        title_pad: int = 18,
        dark_face: str = "navy",
        light_face: str = "gold",
    ) -> None:
        """
        Shows:
          - background: P(class 1) from the network (evaluated on a RAW grid)
          - decision boundary: y_hat = 0.5
          - points: class 0 (dark+white edge), class 1 (light+black edge)
          - two straight lines: hidden neuron boundaries (z_i = 0)
        """
        X0 = self.X_raw[self.y[:, 0] == 0]
        X1 = self.X_raw[self.y[:, 0] == 1]

        x0min, x0max = self.X_raw[:, 0].min() - 1, self.X_raw[:, 0].max() + 1
        x1min, x1max = self.X_raw[:, 1].min() - 1, self.X_raw[:, 1].max() + 1

        gx0 = np.linspace(x0min, x0max, 360)
        gx1 = np.linspace(x1min, x1max, 360)
        XX0, XX1 = np.meshgrid(gx0, gx1)
        G_raw = np.c_[XX0.ravel(), XX1.ravel()]
        G_norm = (G_raw - self.mu) / self.sig

        y_hat, _ = self.forward(G_norm)
        Z = y_hat.reshape(XX0.shape)

        plt.figure(figsize=(8.0, 6.3))
        plt.contourf(XX0, XX1, Z, levels=45)
        cbar = plt.colorbar(label=r"$P({\rm Cluster\, B }| \mu_{\alpha*}, \mu_{\delta})$")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.update_ticks()
        plt.contour(XX0, XX1, Z, levels=[0.5], linewidths=2)

        # scatter styling per requirement
        plt.scatter(
            X0[:, 0], X0[:, 1],
            s=22, alpha=0.9,
            facecolor=dark_face,
            edgecolor="white",
            linewidth=0.7,
            label="Cluster A (class 0)"
        )
        plt.scatter(
            X1[:, 0], X1[:, 1],
            s=22, alpha=0.9,
            facecolor=light_face,
            edgecolor="black",
            linewidth=0.7,
            label="Cluster B (class 1)"
        )

        # hidden neuron lines (z_i = 0)
        # These are the two linear cuts the hidden layer applies in proper-motion space.
        line_handles = []
        for i in range(2):
            a, b, d = self._hidden_line_raw_coeffs(i)
            if abs(b) > 1e-12:
                ys = -(a * gx0 + d) / b
                h, = plt.plot(gx0, ys, linewidth=3, label=f"Neuron {i} line ($z=0$)")
            else:
                # vertical
                if abs(a) > 1e-12:
                    xconst = -d / a
                    h = plt.axvline(xconst, linewidth=3, label=f"Neuron {i} line ($z=0$)")
                else:
                    continue
            line_handles.append(h)

        plt.xlim(x0min, x0max)
        plt.ylim(x1min, x1max)
        plt.title(title, pad=title_pad)
        plt.xlabel(r"$\mu_{\alpha*}$  [mas yr$^{-1}$]")
        plt.ylabel(r"$\mu_{\delta}$   [mas yr$^{-1}$]")
        plt.grid(alpha=0.2)

        # Legend: solid white box, fully opaque
        plt.legend(framealpha=1.0, facecolor="white", fontsize=12)
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Loss-surface “model” helpers
    # ----------------------------
    def _loss_surface_neuron_plane(
        self,
        neuron: int,
        w0_range: Tuple[float, float],
        w1_range: Tuple[float, float],
        grid: int,
    ) -> Tuple[Array, Array, Array]:
        """
        Loss surface in plane (W[0,neuron], W[1,neuron]), holding everything else fixed.
        Returns:
          a_vals, b_vals, L (shape: [grid, grid]) where L[j,i] corresponds to (a_vals[i], b_vals[j])
        """
        W_backup = self.params.W.copy()

        a_vals = np.linspace(w0_range[0], w0_range[1], grid)
        b_vals = np.linspace(w1_range[0], w1_range[1], grid)
        L = np.zeros((grid, grid), dtype=float)

        for i, a in enumerate(a_vals):
            for j, b in enumerate(b_vals):
                self.params.W[0, neuron] = a
                self.params.W[1, neuron] = b
                y_hat, _ = self.forward(self.X_norm)
                L[j, i] = self.bce(y_hat, self.y)

        self.params.W = W_backup
        return a_vals, b_vals, L

    @staticmethod
    def _start_point_at_loss_quantile_interior(
        a_vals: np.ndarray,
        b_vals: np.ndarray,
        L: np.ndarray,
        q: float = 0.80,             # 0.80 = percentil 80 (alto pero no máximo)
        margin_cells: int = 6,
        band: float = 0.03,          # tolerancia ±3% alrededor del quantil
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Pick a start point (a,b) such that the loss is around a chosen quantile q,
        restricted to an interior window (avoids border pinning).

        Example:
        q=0.80 starts in the top-20% region, but not necessarily at the absolute max.
        """
        rng = np.random.default_rng(seed)

        ny, nx = L.shape
        i0, i1 = margin_cells, max(margin_cells + 1, nx - margin_cells)
        j0, j1 = margin_cells, max(margin_cells + 1, ny - margin_cells)

        # fallback if grid too small
        if i1 <= i0 + 1 or j1 <= j0 + 1:
            margin_cells = max(1, min(nx, ny) // 10)
            i0, i1 = margin_cells, nx - margin_cells
            j0, j1 = margin_cells, ny - margin_cells

        Lin = L[j0:j1, i0:i1]
        flat = Lin.ravel()

        target = np.quantile(flat, q)
        lo = np.quantile(flat, max(0.0, q - band))
        hi = np.quantile(flat, min(1.0, q + band))

        candidates = np.where((Lin >= lo) & (Lin <= hi))
        if candidates[0].size == 0:
            # if band too tight, just pick closest-to-target
            idx = int(np.argmin(np.abs(flat - target)))
            jj, ii = np.unravel_index(idx, Lin.shape)
        else:
            k = rng.integers(0, candidates[0].size)
            jj, ii = int(candidates[0][k]), int(candidates[1][k])

        ii += i0
        jj += j0

        a0 = float(a_vals[ii])
        b0 = float(b_vals[jj])
        return a0, b0

    @staticmethod
    def _clip(val: float, lo: float, hi: float) -> float:
        return float(np.clip(val, lo, hi))

    # ----------------------------
    # models / animations
    # ----------------------------
    def model_two_planes(
        self,
        w0_range: Tuple[float, float] = (-8, 8),
        w1_range: Tuple[float, float] = (-8, 8),
        grid: int = 80,
        steps: int = 40,
        lr: float = 0.12,
        pause: float = 0.25,
        clip_to_grid: bool = True,
        title_pad: int = 18,
        margin_cells: int = 6,
        top_frac: float = 0.02,
        restore_at_end: bool = True,
    ) -> None:
        """
        Two simultaneous subplots (double width):
        left  plane: (W[0,0], W[1,0])  = (mu_alpha*→neuron0, mu_delta→neuron0)
        right plane: (W[0,1], W[1,1])  = (mu_alpha*→neuron1, mu_delta→neuron1)

        Start points:
        Chosen near HIGH loss but strictly INSIDE the grid (avoids boundary pinning).

        Plot:
        Point + path are drawn AFTER the parameter update (so motion is visible).
        """
        if clear_output is None:
            raise RuntimeError("This animation requires IPython (e.g., Jupyter) for clear_output.")

        W_backup = self.params.W.copy()

        # Compute both boards once
        a0, b0, L0 = self._loss_surface_neuron_plane(neuron=0, w0_range=w0_range, w1_range=w1_range, grid=grid)
        a1, b1, L1 = self._loss_surface_neuron_plane(neuron=1, w0_range=w0_range, w1_range=w1_range, grid=grid)

        A0, B0 = np.meshgrid(a0, b0)
        A1, B1 = np.meshgrid(a1, b1)

        # Start near high-loss interior (NOT argmax-at-border)
        s00, s10 = self._start_point_at_loss_quantile_interior(
            a0, b0, L0,
            q=0.75,  # start in top 25% loss region (but not necessarily at absolute max)
            margin_cells=margin_cells,
            band=0.03
        )
        s01, s11 = self._start_point_at_loss_quantile_interior(
            a1, b1, L1,
            q=0.75,
            margin_cells=margin_cells,
            band=0.03
        )

        self.params.W[0, 0], self.params.W[1, 0] = s00, s10
        self.params.W[0, 1], self.params.W[1, 1] = s01, s11

        path00, path10 = [], []
        path01, path11 = [], []

        for t in range(steps):
            clear_output(wait=True)

            # Gradient with full network
            y_hat, cache = self.forward(self.X_norm)
            grads = self.backward(self.y, cache)

            # Update the 4 shown weights
            self.params.W[0, 0] -= lr * grads.W[0, 0]
            self.params.W[1, 0] -= lr * grads.W[1, 0]
            self.params.W[0, 1] -= lr * grads.W[0, 1]
            self.params.W[1, 1] -= lr * grads.W[1, 1]

            if clip_to_grid:
                self.params.W[0, 0] = self._clip(self.params.W[0, 0], a0[0], a0[-1])
                self.params.W[1, 0] = self._clip(self.params.W[1, 0], b0[0], b0[-1])
                self.params.W[0, 1] = self._clip(self.params.W[0, 1], a1[0], a1[-1])
                self.params.W[1, 1] = self._clip(self.params.W[1, 1], b1[0], b1[-1])

            # Record AFTER update (so path moves)
            w00, w10 = float(self.params.W[0, 0]), float(self.params.W[1, 0])
            w01, w11 = float(self.params.W[0, 1]), float(self.params.W[1, 1])

            path00.append(w00); path10.append(w10)
            path01.append(w01); path11.append(w11)

            # Plot
            fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))

            cf0 = ax[0].contourf(A0, B0, L0, levels=70)
            cbar0 = fig.colorbar(cf0, ax=ax[0], label="BCE loss")
            cbar0.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            cbar0.update_ticks()
            ax[0].plot(path00, path10, "r.-")
            ax[0].scatter([w00], [w10], s=160, edgecolor="k")
            ax[0].set_title(f"Neuron 0 weight plane (step {t+1}/{steps})", pad=title_pad)
            ax[0].set_xlabel(r"$W_{[0,0]}$  ($\mu_{\alpha*}\rightarrow$ neuron 0)")
            ax[0].set_ylabel(r"$W_{[1,0]}$  ($\mu_{\delta}\rightarrow$ neuron 0)")
            ax[0].grid(alpha=0.15)

            cf1 = ax[1].contourf(A1, B1, L1, levels=70)
            cbar1 = fig.colorbar(cf1, ax=ax[1], label="BCE loss")
            cbar1.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            cbar1.update_ticks()
            ax[1].plot(path01, path11, "r.-")
            ax[1].scatter([w01], [w11], s=160, edgecolor="k")
            ax[1].set_title(f"Neuron 1 weight plane (step {t+1}/{steps})", pad=title_pad)
            ax[1].set_xlabel(r"$W_{[0,1]}$  ($\mu_{\alpha*}\rightarrow$ neuron 1)")
            ax[1].set_ylabel(r"$W_{[1,1]}$  ($\mu_{\delta}\rightarrow$ neuron 1)")
            ax[1].grid(alpha=0.15)

            plt.tight_layout()
            plt.show()

            print(f"Current BCE: {self.bce(y_hat, self.y):.4f}")
            time.sleep(pause)

        if restore_at_end:
            self.params.W = W_backup
    
    # Add this to proper_motion_nn_model.py inside the ProperMotionNNmodel class
    def model_single_plane(
        self,
        neuron: int = 0,
        w0_range: Tuple[float, float] = (-8, 8),
        w1_range: Tuple[float, float] = (-8, 8),
        grid: int = 90,
        steps: int = 45,
        lr: float = 0.12,
        pause: float = 0.25,
        clip_to_grid: bool = True,
        title_pad: int = 18,
        margin_cells: int = 6,
        top_frac: float = 0.02,
    ) -> None:
        """
        Single-board game for ONE hidden neuron.

        Board axes:
        x-axis: W[0, neuron]  (mu_alpha* -> hidden neuron)
        y-axis: W[1, neuron]  (mu_delta  -> hidden neuron)

        Key improvements:
        - Start point is chosen near high loss but strictly inside the grid.
        - We plot the point AFTER the update, so motion is visible.
        """
        if clear_output is None:
            raise RuntimeError("This animation requires IPython (e.g., Jupyter) for clear_output.")
        if neuron not in (0, 1):
            raise ValueError("neuron must be 0 or 1.")

        W_backup = self.params.W.copy()

        # Compute surface once (board)
        a_vals, b_vals, L = self._loss_surface_neuron_plane(
            neuron=neuron, w0_range=w0_range, w1_range=w1_range, grid=grid
        )
        A, B = np.meshgrid(a_vals, b_vals)

        # Start near high-loss interior (NOT on border)
        a0, b0 = self._start_point_at_loss_quantile_interior(
            a_vals, b_vals, L,
            q=0.75,
            margin_cells=margin_cells,
            band=0.03
        )
        self.params.W[0, neuron] = a0
        self.params.W[1, neuron] = b0

        path_a, path_b = [], []

        for t in range(steps):
            clear_output(wait=True)

            # Forward/grad
            y_hat, cache = self.forward(self.X_norm)
            grads = self.backward(self.y, cache)

            # Update ONLY these two
            self.params.W[0, neuron] -= lr * grads.W[0, neuron]
            self.params.W[1, neuron] -= lr * grads.W[1, neuron]

            if clip_to_grid:
                self.params.W[0, neuron] = self._clip(self.params.W[0, neuron], a_vals[0], a_vals[-1])
                self.params.W[1, neuron] = self._clip(self.params.W[1, neuron], b_vals[0], b_vals[-1])

            # Record AFTER update (so you see movement)
            a = float(self.params.W[0, neuron])
            b = float(self.params.W[1, neuron])
            path_a.append(a)
            path_b.append(b)

            # Plot
            plt.figure(figsize=(7.2, 6.0))
            cf = plt.contourf(A, B, L, levels=75)
            cbar = plt.colorbar(cf, label="BCE loss")
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            cbar.update_ticks()

            plt.plot(path_a, path_b, "r.-")
            plt.scatter([a], [b], s=170, edgecolor="k")

            plt.title(f"Hidden neuron {neuron} weight plane (step {t+1}/{steps})", pad=title_pad)
            plt.xlabel(rf"$W_{{[0,{neuron}]}}$  ($\mu_{{\alpha*}}\rightarrow$ neuron {neuron})")
            plt.ylabel(rf"$W_{{[1,{neuron}]}}$  ($\mu_{{\delta}}\rightarrow$ neuron {neuron})")
            plt.grid(alpha=0.15)
            plt.tight_layout()
            plt.show()

            print(f"Current BCE: {self.bce(y_hat, self.y):.4f}")
            time.sleep(pause)

        self.params.W = W_backup


def _get_Xnorm_and_y(model):
    # Try common attribute names (no questions, just robust)
    if hasattr(model, "X_norm"):
        Xn = model.X_norm
    elif hasattr(model, "Xn"):
        Xn = model.Xn
    elif hasattr(model, "X_normalized"):
        Xn = model.X_normalized
    else:
        # compute from X_raw if needed
        Xraw = model.X_raw if hasattr(model, "X_raw") else model.X
        Xn = (Xraw - model.mu) / model.sig

    y = model.y
    return Xn, y

def evaluate_training_set(model, threshold=0.5):
    Xn, y_true = _get_Xnorm_and_y(model)
    y_prob, _ = model.forward(Xn)  # (N,1) = P(class 1)

    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))

    conf = np.array([[TN, FP],
                     [FN, TP]])

    acc = float(np.mean(y_pred == y_true))

    # selection matrix: [true_label, pred_label, P(class1)]
    selection = np.hstack([y_true, y_pred, y_prob])

    return selection, conf, acc, y_prob, y_pred

def make_fake_points(seed=123, nA=10, nB=10, n_bridge=5):
    rng = np.random.default_rng(seed)

    # Cluster centers (same style as your training example)
    meanA = np.array([5.0, -2.0])
    meanB = np.array([4.0, -3.0])

    covA = np.array([[0.20, 0.02],
                     [0.02, 0.18]])

    covB = np.array([[0.22,-0.02],
                     [-0.02,0.20]])

    # Points around each cluster
    XA = rng.multivariate_normal(meanA, covA, size=nA)
    XB = rng.multivariate_normal(meanB, covB, size=nB)

    # ---- Bridge points between clusters ----
    # Interpolate between centroids
    t = rng.uniform(0.05, 0.85, size=(n_bridge, 1))
    base = meanA + t * meanB

    # Small noise around interpolation
    bridge_noise = rng.normal(0.0, 0.5, size=base.shape)
    X_bridge = base + bridge_noise

    # Combine everything
    X_new = np.vstack([XA, XB, X_bridge])

    return X_new

def plot_confusion_matrix(
    conf,
    class_names=None,
    title="Confusion matrix",
    pad=18,
    cmap="Blues",
    annotate=True,
    value_fmt="d",
):
    conf = np.asarray(conf)

    if conf.ndim != 2 or conf.shape[0] != conf.shape[1]:
        raise ValueError("conf must be a square 2D array.")

    K = conf.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(K)]

    if len(class_names) != K:
        raise ValueError("len(class_names) must match conf.shape[0].")

    fig_w = max(5.5, 1.15 * K + 2.5)
    fig_h = max(5.0, 1.00 * K + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(conf, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, pad=pad, fontsize=18)
    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("True label", fontsize=16)

    ticks = np.arange(K)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16)

    if annotate:
        max_val = np.max(conf) if conf.size > 0 else 0.0

        for i in range(K):
            for j in range(K):
                val = conf[i, j]

                if value_fmt == "d":
                    txt = f"{int(val)}"
                else:
                    txt = format(val, value_fmt)

                color = "white" if val > 0.5 * max_val else "black"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=14,
                    color=color,
                )

    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(K - 0.5, -0.5)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def confusion_matrix_reject_columns(y_true, y_pred):
    """
    Rows    = true labels      : [0, 1]
    Columns = predicted labels : [-1, 0, 1]

    Here:
      -1 = Cluster A
       0 = Reject
       1 = Cluster B
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)

    true_labels = [0, 1]
    pred_labels = [-1, 0, 1]

    true_to_i = {lab: i for i, lab in enumerate(true_labels)}
    pred_to_j = {lab: j for j, lab in enumerate(pred_labels)}

    conf = np.zeros((2, 3), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in true_to_i and yp in pred_to_j:
            conf[true_to_i[yt], pred_to_j[yp]] += 1

    return conf

def plot_confusion_matrix_rect(
    conf,
    x_names,
    y_names,
    title="Confusion matrix",
    pad=18,
    cmap="Blues",
    annotate=True,
    value_fmt="d",
    fontsize=14,
    cbar_label="Count",
):
    conf = np.asarray(conf)

    if conf.ndim != 2:
        raise ValueError("conf must be a 2D array.")

    n_rows, n_cols = conf.shape

    if len(x_names) != n_cols:
        raise ValueError("len(x_names) must match number of columns.")
    if len(y_names) != n_rows:
        raise ValueError("len(y_names) must match number of rows.")

    fig_w = max(5.8, 1.15 * n_cols + 2.8)
    fig_h = max(4.6, 1.10 * n_rows + 2.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(conf, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.set_title(title, pad=pad, fontsize=18)
    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("True label", fontsize=16)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(x_names, rotation=20, ha="right", fontsize=16)
    ax.set_yticklabels(y_names, fontsize=16)

    if annotate:
        max_val = np.max(conf) if conf.size > 0 else 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = conf[i, j]
                txt = f"{int(val)}" if value_fmt == "d" else format(val, value_fmt)
                color = "white" if val > 0.5 * max_val else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize, color=color)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def evaluate_training_set_with_reject_nn(
    model,
    threshold=0.5,
    confidence_threshold=0.10,
):
    y_true = model.y[:, 0].astype(int)

    y_pred, y_prob, confidence = model.predict_with_reject(
        model.X_raw,
        confidence_threshold=confidence_threshold,
    )

    conf = confusion_matrix_reject_columns(y_true, y_pred)

    valid = (y_pred != -1)
    acc_no_reject = float(np.mean(y_true[valid] == y_pred[valid])) if np.any(valid) else np.nan
    reject_fraction = float(np.mean(y_pred == -1))

    return conf, acc_no_reject, reject_fraction, y_prob, y_pred, confidence

def evaluate_rbf_training_set_with_reject(
    model,
    score_threshold=0.15,
    activation_threshold=0.20,
):
    y_true = model.y[:, 0].astype(int)

    y_pred, score, max_phi = model.predict_with_reject(
        model.X_raw,
        score_threshold=score_threshold,
        activation_threshold=activation_threshold,
    )

    conf = confusion_matrix_reject_columns(y_true, y_pred)

    valid = (y_pred != 0)
    # among non-rejected points:
    # predicted -1 means A -> compare with y_true==0
    # predicted +1 means B -> compare with y_true==1
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    acc_total = float(np.mean(y_true == y_pred_binary))
    acc_no_reject = float(np.mean(y_true[valid] == y_pred_binary[valid])) if np.any(valid) else np.nan
    reject_fraction = float(np.mean(y_pred == 0))

    return conf, acc_no_reject, acc_total, reject_fraction, score, y_pred, max_phi

def plot_new_points(model, X_new, X, y):
    # Normalize with training statistics
    X_new_norm = (X_new - model.mu) / model.sig

    # Forward pass
    prob_B, _ = model.forward(X_new_norm)   # P(class 1 = Cluster B)
    prob_B = prob_B[:,0]
    prob_A = 1 - prob_B

    # Predicted label
    y_new_pred = (prob_B >= 0.5).astype(int)

    for i in range(len(X_new)):
        print(
            f"{i:02d}  mu_a*={X_new[i,0]:6.3f}  mu_d={X_new[i,1]:6.3f}  "
            f"-> Predicted: {'B' if y_new_pred[i]==1 else 'A'}  "
            f"P(A)={prob_A[i]:.3f}  P(B)={prob_B[i]:.3f}"
        )

    plt.figure(figsize=(7,6))

    # Training faded
    plt.scatter(X[:,0], X[:,1],
                c=y[:,0],
                s=18,
                edgecolor="k",
                linewidth=0.2,
                alpha=0.25,
                label="Training")

    # Color by predicted class
    colors = np.where(y_new_pred==1, "gold", "purple")

    # Size by confidence
    sizes = 80 + 300*np.abs(prob_B - 0.5)

    plt.scatter(X_new[:,0], X_new[:,1],
                c=colors,
                s=sizes,
                marker="*",
                edgecolor="black",
                linewidth=1.2,
                label="New (classified)")

    plt.xlabel(r"$\mu_{\alpha*}$  [mas yr$^{-1}$]")
    plt.ylabel(r"$\mu_{\delta}$   [mas yr$^{-1}$]")
    plt.title("New points classified by the NN", pad=18)
    plt.legend(framealpha=1.0, facecolor="white", fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# A small RBF network for two proper-motion clusters:
#
#   input (2D raw proper motion) -> normalize
#   hidden RBF layer (H local basis functions)
#   output sigmoid for binary classification
#
# Key idea:
#   phi_j(x) = exp( - ||x - c_j||^2 / (2 sigma_j^2) )
#
# where c_j are RBF centers in NORMALIZED space.
#
# This is much more local than a tiny tanh MLP, so it behaves
# better for "far away" stars. You can also enable a rejection
# rule based on low RBF activation.
# ---------------------------------------------------------

@dataclass
class RBFParams:
    centers: Array   # (H, 2)
    sigmas: Array    # (H,)
    v: Array         # (H, 1)
    c: Array         # (1,)   kept for compatibility, but not used


class ProperMotionRBFModel:
    """
    Symmetric RBF classifier for two proper-motion clusters.

    Main idea
    ---------
    The output is now in [-1, +1]:

        score(x) = tanh(Phi(x) @ v)

    where
        -1  -> Cluster A
        +1  -> Cluster B
         0  -> neutral / ambiguous / reject candidate

    Training targets
    ----------------
    Input labels remain {0,1}, but internally we map them to {-1,+1}:
        0 -> -1
        1 -> +1

    Rejection rule
    --------------
    A point is rejected if either:
      1) |score| < score_threshold
      2) max_j phi_j(x) < activation_threshold

    This gives both:
      - output ambiguity rejection
      - geometric support rejection
    """

    def __init__(
        self,
        X_raw: Array,
        y: Array,
        hidden_units: int = 6,
        seed: int = 42,
        init_sigma: float = 0.9,
        init_output_scale: float = 0.3,
        sigma_min: float = 0.15,
        sigma_max: float = 3.0,
    ):
        X_raw = np.asarray(X_raw, dtype=float)
        y = np.asarray(y, dtype=float)

        if X_raw.ndim != 2 or X_raw.shape[1] != 2:
            raise ValueError("X_raw must have shape (N,2).")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("y must have shape (N,1).")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("y must contain only 0/1 labels.")
        if hidden_units < 2:
            raise ValueError("hidden_units must be >= 2.")
        if init_sigma <= 0:
            raise ValueError("init_sigma must be positive.")

        self.X_raw = X_raw
        self.y = y                        # original labels in {0,1}
        self.y_pm = 2.0 * y - 1.0         # internal labels in {-1,+1}

        self.hidden_units = hidden_units
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # normalization
        self.mu = self.X_raw.mean(axis=0)
        self.sig = self.X_raw.std(axis=0)
        self.sig = np.where(self.sig < 1e-12, 1.0, self.sig)
        self.X_norm = (self.X_raw - self.mu) / self.sig

        self.rng = np.random.default_rng(seed)
        self.params = self._init_params(
            init_sigma=init_sigma,
            init_output_scale=init_output_scale,
        )

        self.loss_history = []

    def _ensure_2d_input(self, X_raw_new: Array) -> Array:
        X_raw_new = np.asarray(X_raw_new, dtype=float)

        if X_raw_new.ndim == 1:
            if X_raw_new.shape[0] != 2:
                raise ValueError("A single point must have shape (2,).")
            X_raw_new = X_raw_new.reshape(1, 2)

        if X_raw_new.ndim != 2 or X_raw_new.shape[1] != 2:
            raise ValueError("Input must have shape (N,2) or (2,).")

        return X_raw_new

    # --------------------------------------------------
    # activations / loss
    # --------------------------------------------------
    @staticmethod
    def tanh(z: Array) -> Array:
        return np.tanh(z)

    @staticmethod
    def dtanh_from_output(a: Array) -> Array:
        return 1.0 - a**2

    @staticmethod
    def mse(y_hat: Array, y: Array) -> float:
        return float(np.mean((y_hat - y) ** 2))

    # --------------------------------------------------
    # init
    # --------------------------------------------------
    def _init_params(
        self,
        init_sigma: float,
        init_output_scale: float,
    ) -> RBFParams:
        """
        Initialize centers by sampling training points in normalized space.
        """
        N = self.X_norm.shape[0]
        idx = self.rng.choice(N, size=self.hidden_units, replace=False)
        centers = self.X_norm[idx].copy()

        sigmas = np.full(self.hidden_units, init_sigma, dtype=float)

        v = init_output_scale * self.rng.standard_normal((self.hidden_units, 1))
        c = np.zeros(1, dtype=float)  # kept for compatibility, not used

        return RBFParams(
            centers=centers,
            sigmas=sigmas,
            v=v,
            c=c,
        )

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def _rbf_features(
        self,
        X_norm: Array,
        params: Optional[RBFParams] = None,
    ) -> Tuple[Array, Array]:
        """
        Compute:
          d2   : squared distances to centers, shape (N,H)
          Phi  : RBF activations,            shape (N,H)
        """
        p = params or self.params

        diff = X_norm[:, None, :] - p.centers[None, :, :]   # (N,H,2)
        d2 = np.sum(diff**2, axis=2)                        # (N,H)

        sigma2 = p.sigmas[None, :] ** 2                     # (1,H)
        Phi = np.exp(-0.5 * d2 / sigma2)                    # (N,H)

        return d2, Phi

    def forward(
        self,
        X_norm: Array,
        params: Optional[RBFParams] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Output score in [-1, +1]:
            score = tanh(Phi @ v)

        Far from all centers:
            Phi ~ 0  -> score ~ tanh(0) = 0
        """
        p = params or self.params

        d2, Phi = self._rbf_features(X_norm, params=p)
        logits = Phi @ p.v           # NO output bias
        score = self.tanh(logits)    # in [-1,+1]

        cache = {
            "X": X_norm,
            "d2": d2,
            "Phi": Phi,
            "logits": logits,
            "score": score,
        }
        return score, cache

    # --------------------------------------------------
    # backward
    # --------------------------------------------------
    def backward(
        self,
        y_pm: Array,
        cache: Dict[str, Array],
        params: Optional[RBFParams] = None,
    ) -> RBFParams:
        """
        Gradients for:
          centers, sigmas, v

        Loss:
            MSE(score, y_pm)

        where y_pm in {-1, +1}.
        """
        p = params or self.params

        X = cache["X"]             # (N,2)
        d2 = cache["d2"]           # (N,H)
        Phi = cache["Phi"]         # (N,H)
        score = cache["score"]     # (N,1)

        N = X.shape[0]
        H = p.centers.shape[0]

        # MSE = mean((score - y)^2)
        # dL/dscore = 2*(score - y)/N
        dscore = 2.0 * (score - y_pm) / N                 # (N,1)

        # score = tanh(logits)
        dlogits = dscore * self.dtanh_from_output(score)  # (N,1)

        # output layer
        dv = Phi.T @ dlogits                               # (H,1)
        dc = np.zeros_like(p.c)                            # bias not used

        # hidden feature sensitivity
        dPhi = dlogits @ p.v.T                             # (N,H)

        dcenters = np.zeros_like(p.centers)                # (H,2)
        dsigmas = np.zeros_like(p.sigmas)                  # (H,)

        for j in range(H):
            sigma_j = p.sigmas[j]
            sigma2_j = sigma_j ** 2
            sigma3_j = sigma_j ** 3

            diff_j = X - p.centers[j]                      # (N,2)
            phi_j = Phi[:, j:j+1]                          # (N,1)
            dphi_j = dPhi[:, j:j+1]                        # (N,1)

            # d phi_j / d center_j = phi_j * (X - c_j) / sigma_j^2
            dcenters[j] = np.sum(dphi_j * phi_j * diff_j / sigma2_j, axis=0)

            # d phi_j / d sigma_j = phi_j * d2_j / sigma_j^3
            dsigmas[j] = np.sum(dphi_j[:, 0] * Phi[:, j] * d2[:, j] / sigma3_j)

        return RBFParams(
            centers=dcenters,
            sigmas=dsigmas,
            v=dv,
            c=dc,
        )

    # --------------------------------------------------
    # optimization
    # --------------------------------------------------
    def step(
        self,
        grads: RBFParams,
        lr_centers: float = 0.05,
        lr_sigmas: float = 0.01,
        lr_out: float = 0.10,
    ) -> None:
        self.params.centers -= lr_centers * grads.centers
        self.params.sigmas  -= lr_sigmas  * grads.sigmas
        self.params.v       -= lr_out     * grads.v
        self.params.c       -= lr_out     * grads.c  # stays unused

        self.params.sigmas = np.clip(self.params.sigmas, self.sigma_min, self.sigma_max)

    def train(
        self,
        steps: int = 1000,
        lr_centers: float = 0.05,
        lr_sigmas: float = 0.01,
        lr_out: float = 0.10,
        verbose_every: int = 100,
    ) -> None:
        self.loss_history = []

        for i in range(steps):
            score, cache = self.forward(self.X_norm)
            loss = self.mse(score, self.y_pm)
            grads = self.backward(self.y_pm, cache)
            self.step(grads, lr_centers=lr_centers, lr_sigmas=lr_sigmas, lr_out=lr_out)

            self.loss_history.append(loss)

            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"Iter {i+1:4d} | MSE = {loss:.4f}")

    # --------------------------------------------------
    # inference
    # --------------------------------------------------
    def predict_score(self, X_raw_new: Array) -> Array:
        """
        Return symmetric score in [-1,+1]:
          -1 -> Cluster A
          +1 -> Cluster B
           0 -> neutral / ambiguous
        """
        X_raw_new = self._ensure_2d_input(X_raw_new)
        Xn = (X_raw_new - self.mu) / self.sig
        score, _ = self.forward(Xn)
        return score[:, 0]

    def predict_proba(self, X_raw_new: Array) -> Array:
        """
        Convert symmetric score in [-1,+1] to a pseudo-probability in [0,1]:
            P(B) = (score + 1)/2
        """
        score = self.predict_score(X_raw_new)
        return 0.5 * (score + 1.0)

    def predict(self, X_raw_new: Array, threshold: float = 0.0) -> Array:
        """
        Hard prediction without rejection:
          score >= 0 -> class 1
          score <  0 -> class 0
        """
        score = self.predict_score(X_raw_new)
        return (score >= threshold).astype(int)

    def predict_with_reject(
        self,
        X_raw_new: Array,
        score_threshold: float = 0.15,
        activation_threshold: float = 0.20,
    ) -> Tuple[Array, Array, Array]:
        """
        Predict with optional rejection.

        Labels:
        -1 = Cluster A
        0 = Reject
        +1 = Cluster B

        Decision rule:
        if |score| < score_threshold -> reject
        if max_j phi_j(x) < activation_threshold -> reject
        else sign(score) decides between A and B

        Returns
        -------
        labels : (N,) array in {-1,0,1}
        score : (N,) output in [-1,+1]
        max_phi : (N,) maximum hidden activation
        """
        X_raw_new = self._ensure_2d_input(X_raw_new)
        Xn = (X_raw_new - self.mu) / self.sig

        score, _ = self.forward(Xn)
        score = score[:, 0]

        _, Phi = self._rbf_features(Xn)
        max_phi = np.max(Phi, axis=1)

        # Default: sign-based classification
        labels = np.where(score >= 0.0, 1, -1)

        # Reject if too close to zero
        labels[np.abs(score) < score_threshold] = 0

        # Reject if too far from all centers
        labels[max_phi < activation_threshold] = 0

        return labels, score, max_phi

    # --------------------------------------------------
    # plots
    # --------------------------------------------------
    def plot_losses(self) -> None:
        if len(self.loss_history) == 0:
            raise RuntimeError("No loss history found. Train the model first.")

        plt.figure(figsize=(7, 4.5))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel("Training step")
        plt.ylabel("MSE loss")
        plt.title("RBF training loss", pad=14)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    def _centers_raw(self) -> Array:
        return self.params.centers * self.sig + self.mu

    def plot_geometry_raw(
        self,
        title: str = "RBF geometry in proper-motion space",
        score_threshold: float = 0.15,
        activation_threshold: Optional[float] = None,
        dark_face: str = "navy",
        light_face: str = "gold",
    ) -> None:
        """
        Shows:
        - background continuous score in [-1, +1]
        - decision boundary: score = 0
        - optional reject contour from max(phi)
        - training points
        - RBF centers

        Interpretation of the score:
        score ~ -1  -> Cluster A
        score ~  0  -> neutral / ambiguous
        score ~ +1  -> Cluster B
        """
        X0 = self.X_raw[self.y[:, 0] == 0]
        X1 = self.X_raw[self.y[:, 0] == 1]

        x0min, x0max = self.X_raw[:, 0].min() - 1.0, self.X_raw[:, 0].max() + 1.0
        x1min, x1max = self.X_raw[:, 1].min() - 1.0, self.X_raw[:, 1].max() + 1.0

        gx0 = np.linspace(x0min, x0max, 320)
        gx1 = np.linspace(x1min, x1max, 320)
        XX0, XX1 = np.meshgrid(gx0, gx1)

        G_raw = np.c_[XX0.ravel(), XX1.ravel()]
        G_norm = (G_raw - self.mu) / self.sig

        score_grid, _ = self.forward(G_norm)
        score_grid = score_grid.reshape(XX0.shape)

        _, Phi = self._rbf_features(G_norm)
        M = np.max(Phi, axis=1).reshape(XX0.shape)

        # Mask unsupported region
        if activation_threshold is not None:
            S_plot = np.ma.masked_where(M < activation_threshold, score_grid)
        else:
            S_plot = score_grid

        centers_raw = self._centers_raw()

        plt.figure(figsize=(8.3, 6.6))
        cf = plt.contourf(
            XX0, XX1, S_plot,
            levels=np.linspace(-1.0, 1.0, 45),
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        cbar = plt.colorbar(
            cf,
            label=r"Continuous score $s(\mu_{\alpha*},\mu_{\delta}) \in [-1,1]$"
        )
        cbar.set_ticks([-1, 0, 1])

        # Main decision boundary: score = 0
        plt.contour(
            XX0, XX1, score_grid,
            levels=[0.0],
            colors="black",
            linewidths=2,
        )

        # Optional reject support boundary
        if activation_threshold is not None:
            plt.contour(
                XX0, XX1, M,
                levels=[activation_threshold],
                colors="black",
                linewidths=2,
                linestyles="--",
            )

        plt.scatter(
            X0[:, 0], X0[:, 1],
            s=22, alpha=0.9,
            facecolor=dark_face,
            edgecolor="white",
            linewidth=0.7,
            label="Cluster A (class 0)",
        )
        plt.scatter(
            X1[:, 0], X1[:, 1],
            s=22, alpha=0.9,
            facecolor=light_face,
            edgecolor="black",
            linewidth=0.7,
            label="Cluster B (class 1)",
        )

        plt.scatter(
            centers_raw[:, 0], centers_raw[:, 1],
            s=120,
            marker="X",
            facecolor="red",
            edgecolor="black",
            linewidth=1.0,
            label="RBF centers",
        )

        plt.xlim(x0min, x0max)
        plt.ylim(x1min, x1max)
        plt.xlabel(r"$\mu_{\alpha*}$  [mas yr$^{-1}$]")
        plt.ylabel(r"$\mu_{\delta}$   [mas yr$^{-1}$]")
        plt.title(title, pad=16)
        plt.grid(alpha=0.2)
        plt.legend(framealpha=1.0, facecolor="white", fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_new_points(
        self,
        X_new: Array,
        score_threshold: float = 0.15,
        activation_threshold: Optional[float] = 0.20,
        title: str = "New points classified by the RBF model",
        dark_face: str = "navy",
        light_face: str = "gold",
        reject_face: str = "lightgray",
    ) -> None:
        """
        Plot new points over the continuous RBF score field.

        Score interpretation:
        score ~ -1  -> Cluster A
        score ~  0  -> neutral / ambiguous
        score ~ +1  -> Cluster B
        """
        X_new = self._ensure_2d_input(X_new)

        # -------------------------
        # Predict new points
        # -------------------------
        if activation_threshold is None:
            score = self.predict_score(X_new)
            max_phi = np.full(X_new.shape[0], np.nan)
            labels = np.where(score >= 0.0, 1, 0)
            labels[np.abs(score) < score_threshold] = -1
        else:
            labels, score, max_phi = self.predict_with_reject(
                X_new,
                score_threshold=score_threshold,
                activation_threshold=activation_threshold,
            )

        for i in range(len(X_new)):
            pred_txt = "Reject" if labels[i] == -1 else ("B" if labels[i] == 1 else "A")
            extra = "" if activation_threshold is None else f"  max_phi={max_phi[i]:.3f}"
            print(
                f"{i:02d}  mu_a*={X_new[i,0]:7.3f}  mu_d={X_new[i,1]:7.3f}"
                f"  -> Predicted: {pred_txt}"
                f"  score={score[i]:+.3f}"
                f"{extra}"
            )

        # -------------------------
        # Background grid
        # -------------------------
        x0min = min(self.X_raw[:, 0].min(), X_new[:, 0].min()) - 1.0
        x0max = max(self.X_raw[:, 0].max(), X_new[:, 0].max()) + 1.0
        x1min = min(self.X_raw[:, 1].min(), X_new[:, 1].min()) - 1.0
        x1max = max(self.X_raw[:, 1].max(), X_new[:, 1].max()) + 1.0

        gx0 = np.linspace(x0min, x0max, 320)
        gx1 = np.linspace(x1min, x1max, 320)
        XX0, XX1 = np.meshgrid(gx0, gx1)

        G_raw = np.c_[XX0.ravel(), XX1.ravel()]
        G_norm = (G_raw - self.mu) / self.sig

        score_grid, _ = self.forward(G_norm)
        score_grid = score_grid.reshape(XX0.shape)

        _, Phi = self._rbf_features(G_norm)
        M = np.max(Phi, axis=1).reshape(XX0.shape)

        if activation_threshold is not None:
            S_plot = np.ma.masked_where(M < activation_threshold, score_grid)
        else:
            S_plot = score_grid

        # -------------------------
        # Training split
        # -------------------------
        X0 = self.X_raw[self.y[:, 0] == 0]
        X1 = self.X_raw[self.y[:, 0] == 1]

        # -------------------------
        # New-point styling
        # -------------------------
        sizes = 90 + 320 * np.abs(score)

        mask_rej = labels == -1
        mask_A = labels == 0
        mask_B = labels == 1

        # -------------------------
        # Plot
        # -------------------------
        plt.figure(figsize=(8.2, 6.6))
        cf = plt.contourf(
            XX0, XX1, S_plot,
            levels=np.linspace(-1.0, 1.0, 45),
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            alpha=0.85,
        )
        cbar = plt.colorbar(
            cf,
            label=r"Continuous score $s(\mu_{\alpha*},\mu_{\delta}) \in [-1,1]$"
        )
        cbar.set_ticks([-1, 0, 1])

        # Decision boundary: score = 0
        plt.contour(
            XX0, XX1, score_grid,
            levels=[0.0],
            colors="black",
            linewidths=2,
        )

        # Reject support boundary
        if activation_threshold is not None:
            plt.contour(
                XX0, XX1, M,
                levels=[activation_threshold],
                colors="black",
                linewidths=2,
                linestyles="--",
            )

        # Training A
        plt.scatter(
            X0[:, 0], X0[:, 1],
            s=20, alpha=0.75,
            facecolor=dark_face,
            edgecolor="white",
            linewidth=0.5,
            label="Cluster A",
        )

        # Training B
        plt.scatter(
            X1[:, 0], X1[:, 1],
            s=20, alpha=0.75,
            facecolor=light_face,
            edgecolor="black",
            linewidth=0.5,
            label="Cluster B",
        )

        # New accepted as A
        plt.scatter(
            X_new[mask_A, 0], X_new[mask_A, 1],
            s=sizes[mask_A],
            marker="*",
            facecolor=dark_face,
            edgecolor="black",
            linewidth=1.1,
            zorder=6,
            label="New points → A",
        )

        # New accepted as B
        plt.scatter(
            X_new[mask_B, 0], X_new[mask_B, 1],
            s=sizes[mask_B],
            marker="*",
            facecolor=light_face,
            edgecolor="black",
            linewidth=1.1,
            zorder=6,
            label="New points → B",
        )

        # Rejected
        plt.scatter(
            X_new[mask_rej, 0], X_new[mask_rej, 1],
            s=sizes[mask_rej] + 40,
            marker="X",
            facecolor=reject_face,
            edgecolor="black",
            linewidth=1.4,
            zorder=7,
            label="Rejected points",
        )

        # Centers
        centers_raw = self._centers_raw()
        plt.scatter(
            centers_raw[:, 0], centers_raw[:, 1],
            s=130,
            marker="X",
            facecolor="red",
            edgecolor="black",
            linewidth=1.0,
            label="RBF centers",
        )

        plt.xlabel(r"$\mu_{\alpha*}$  [mas yr$^{-1}$]")
        plt.ylabel(r"$\mu_{\delta}$   [mas yr$^{-1}$]")
        plt.title(title, pad=16)
        plt.grid(alpha=0.2)
        plt.legend(framealpha=1.0, facecolor="white", fontsize=10, ncols=3)
        plt.tight_layout()
        plt.show()

def compare_models_on_new_points(nn_model, rbf_model, X_new):

    prob_nn = nn_model.predict_proba(X_new)
    pred_nn = (prob_nn >= 0.5).astype(int)

    labels_rbf, score, max_phi = rbf_model.predict_with_reject(X_new)

    for i in range(len(X_new)):
        print(
            f"{i:02d} "
            f"mu_a*={X_new[i,0]:6.3f} "
            f"mu_d={X_new[i,1]:6.3f} | "
            f"MLP: P(B)={prob_nn[i]:.3f} -> {pred_nn[i]} | "
            f"RBF: s(B)={score[i]:.3f}, "
            f"max_phi={max_phi[i]:.3f} -> {labels_rbf[i]}"
        )