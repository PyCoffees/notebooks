import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# -----------------------
# Data
# -----------------------
def make_xor(n=400, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(float)  # 1 if same sign
    X = X + noise * rng.normal(size=X.shape)
    return X, y.reshape(-1, 1)

# -----------------------
# Activations + Loss
# -----------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(x):
    return np.tanh(x)

def dtanh(z):
    t = np.tanh(z)
    return 1.0 - t**2

def bce(y_hat, y, eps=1e-12):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

# -----------------------
# Base class
# -----------------------
class _XORBase:
    """
    Base utilities shared by all XOR MLP variants:
    - fit / predict_proba
    - plot_boundary
    - loss_surface_last2 + descent_game_last2:
      Explore 2D loss landscape varying ONLY two weights of the final layer.
    """
    def __init__(self, seed=42, scale=1.5):
        self.seed = seed
        self.scale = scale
        self.params = None
        self.history = None

    # --- mandatory in subclasses ---
    def _init_params(self):
        raise NotImplementedError

    def _forward(self, X, params):
        raise NotImplementedError

    def _backward(self, y, params, cache):
        raise NotImplementedError

    def _final_weight_matrix_key(self):
        """
        Return the key name of the final weight matrix in params.
        Example: 'W2' for 2-2-1 and 2-4-1, 'W3' for 2-2-2-1.
        """
        raise NotImplementedError

    # --- public API ---
    def init(self):
        self.params = self._init_params()
        return self.params

    def predict_proba(self, X, params=None):
        if params is None:
            params = self.params
        yhat, _ = self._forward(X, params)
        return yhat

    def loss(self, X, y, params=None):
        yhat = self.predict_proba(X, params=params)
        return bce(yhat, y)

    def fit(self, X, y, lr=0.3, epochs=2000, print_every=200):
        if self.params is None:
            self.init()

        start = time.time()

        hist = {"loss": []}
        for ep in range(1, epochs + 1):
            yhat, cache = self._forward(X, self.params)
            L = bce(yhat, y)
            grads = self._backward(y, self.params, cache)

            for k in self.params:
                self.params[k] = self.params[k] - lr * grads[k]

            hist["loss"].append(L)

            if ep == 1 or ep % print_every == 0:
                print(f"Epoch {ep:4d} | BCE = {L:.4f}")

        end = time.time()

        self.history = hist
        print(f"Training completed in {end - start:.2f} seconds.")
        return self.params, hist

    def plot_loss(self, title="Training loss (BCE)"):
        if self.history is None:
            raise ValueError("No training history. Call fit() first.")
        plt.figure(figsize=(6,4))
        plt.plot(self.history["loss"])
        plt.title(title)
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.grid(True)
        plt.show()

    def plot_boundary(self, X, y, title="Decision boundary", levels=30):
        x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 240)
        x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 240)
        X1, X2m = np.meshgrid(x1, x2)
        grid = np.c_[X1.ravel(), X2m.ravel()]
        yhat = self.predict_proba(grid).reshape(X1.shape)

        plt.figure(figsize=(5.8,4.8))
        plt.contourf(X1, X2m, yhat, levels=levels)
        plt.scatter(X[:,0], X[:,1], c=y[:,0], s=14, edgecolor="white", linewidth=0.3)
        plt.title(title)
        plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
        plt.grid(True)
        plt.show()

    # -----------------------
    # Loss surface + game (vary last-layer weights only)
    # We always vary two coordinates: [0,0] and [1,0] of the final weight matrix
    # (works for last layer size >= 2, which is true for our three models).
    # -----------------------
    def loss_surface_last2(self, X, y, t1_range=(-8, 8), t2_range=(-8, 8), grid=120):
        if self.params is None:
            raise ValueError("Call init() (or fit()) first.")
        key = self._final_weight_matrix_key()
        W_orig = self.params[key].copy()

        t1_vals = np.linspace(*t1_range, grid)
        t2_vals = np.linspace(*t2_range, grid)
        L = np.zeros((grid, grid))

        for i, t1 in enumerate(t1_vals):
            for j, t2 in enumerate(t2_vals):
                self.params[key][0,0] = t1
                self.params[key][1,0] = t2
                L[j, i] = self.loss(X, y)

        self.params[key] = W_orig
        return t1_vals, t2_vals, L

    def descent_game_last2(
        self, X, y,
        t1_vals, t2_vals, Lgrid,
        lr=0.6, steps=45, pause=0.12,
        start_far=True, start_point=None,
        show_contours=True,
        title_prefix="Step"
    ):
        if self.params is None:
            raise ValueError("Call init() (or fit()) first.")
        key = self._final_weight_matrix_key()
        W_orig = self.params[key].copy()
        T1, T2 = np.meshgrid(t1_vals, t2_vals)

        # start position
        if start_point is not None:
            self.params[key][0,0] = float(start_point[0])
            self.params[key][1,0] = float(start_point[1])
        elif start_far:
            self.params[key][0,0] = float(t1_vals[-1]) * 0.9
            self.params[key][1,0] = float(t2_vals[0])  * 0.9

        path_t1, path_t2 = [], []

        for s in range(steps):
            clear_output(wait=True)

            yhat, cache = self._forward(X, self.params)
            grads = self._backward(y, self.params, cache)

            t1 = float(self.params[key][0,0])
            t2 = float(self.params[key][1,0])
            path_t1.append(t1); path_t2.append(t2)

            # update ONLY the two selected weights
            self.params[key][0,0] -= lr * grads[key][0,0]
            self.params[key][1,0] -= lr * grads[key][1,0]

            plt.figure(figsize=(6,5))
            if show_contours:
                plt.contourf(T1, T2, Lgrid, levels=60)
                plt.colorbar(label="BCE loss")

            plt.plot(path_t1, path_t2, "r.-", linewidth=2)
            plt.scatter(t1, t2, s=120, facecolor="white", edgecolor="red", linewidth=2)

            # key is like "W2" or "W3" -> split into "W" and "2/3"
            knum = key[1:]  # "2" or "3"

            plt.title(
                f"{title_prefix} {s+1} | "
                + rf"$W_{{{knum},[0,0]}}=$" + f"{t1:.3f}, "
                + rf"$W_{{{knum},[1,0]}}=$" + f"{t2:.3f}",
                pad=18
            )
            plt.xlabel(rf"$W_{{{knum},[0,0]}}$")
            plt.ylabel(rf"$W_{{{knum},[1,0]}}$")
            plt.grid(True)
            plt.show()

            time.sleep(pause)

        print("🎉 Neural Network Descent Complete!")
        self.params[key] = W_orig
        return np.array(path_t1), np.array(path_t2)

# -----------------------
# Model 1: 2-2-1 (one hidden layer of 2)
# -----------------------
class XOR_MLP_2_2_1(_XORBase):
    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        H = 2
        return {
            "W1": self.scale * rng.standard_normal((2, H)),
            "b1": np.zeros((1, H)),
            "W2": self.scale * rng.standard_normal((H, 1)),
            "b2": np.zeros((1, 1)),
        }

    def _forward(self, X, params):
        Z1 = X @ params["W1"] + params["b1"]
        A1 = tanh(Z1)
        Z2 = A1 @ params["W2"] + params["b2"]
        Yhat = sigmoid(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "Yhat": Yhat}
        return Yhat, cache

    def _backward(self, y, params, cache):
        X, Z1, A1, Yhat = cache["X"], cache["Z1"], cache["A1"], cache["Yhat"]
        N = X.shape[0]

        dZ2 = (Yhat - y) / N
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ params["W2"].T
        dZ1 = dA1 * dtanh(Z1)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def _final_weight_matrix_key(self):
        return "W2"

# -----------------------
# Model 2: 2-2-2-1 (two hidden layers of 2)
# -----------------------
class XOR_MLP_2_2_2_1(_XORBase):
    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        H1 = 2
        H2 = 2
        return {
            "W1": self.scale * rng.standard_normal((2, H1)),
            "b1": np.zeros((1, H1)),
            "W2": self.scale * rng.standard_normal((H1, H2)),
            "b2": np.zeros((1, H2)),
            "W3": self.scale * rng.standard_normal((H2, 1)),
            "b3": np.zeros((1, 1)),
        }

    def _forward(self, X, params):
        Z1 = X @ params["W1"] + params["b1"]
        A1 = tanh(Z1)

        Z2 = A1 @ params["W2"] + params["b2"]
        A2 = tanh(Z2)

        Z3 = A2 @ params["W3"] + params["b3"]
        Yhat = sigmoid(Z3)

        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "Yhat": Yhat}
        return Yhat, cache

    def _backward(self, y, params, cache):
        X, Z1, A1 = cache["X"], cache["Z1"], cache["A1"]
        Z2, A2 = cache["Z2"], cache["A2"]
        Yhat = cache["Yhat"]
        N = X.shape[0]

        dZ3 = (Yhat - y) / N
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ params["W3"].T
        dZ2 = dA2 * dtanh(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ params["W2"].T
        dZ1 = dA1 * dtanh(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def _final_weight_matrix_key(self):
        return "W3"

# -----------------------
# Model 3: 2-4-1 (one hidden layer of 4)
# -----------------------
class XOR_MLP_2_4_1(_XORBase):
    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        H = 4
        return {
            "W1": self.scale * rng.standard_normal((2, H)),
            "b1": np.zeros((1, H)),
            "W2": self.scale * rng.standard_normal((H, 1)),
            "b2": np.zeros((1, 1)),
        }

    def _forward(self, X, params):
        Z1 = X @ params["W1"] + params["b1"]
        A1 = tanh(Z1)
        Z2 = A1 @ params["W2"] + params["b2"]
        Yhat = sigmoid(Z2)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "Yhat": Yhat}
        return Yhat, cache

    def _backward(self, y, params, cache):
        X, Z1, A1, Yhat = cache["X"], cache["Z1"], cache["A1"], cache["Yhat"]
        N = X.shape[0]

        dZ2 = (Yhat - y) / N
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ params["W2"].T
        dZ1 = dA1 * dtanh(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def _final_weight_matrix_key(self):
        return "W2"

# -----------------------
# Model 4: 2-64-64-1 (deep & wide)
# -----------------------
class XOR_MLP_2_64_64_1(_XORBase):
    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        H1 = 64
        H2 = 64

        return {
            "W1": self.scale * rng.standard_normal((2, H1)),
            "b1": np.zeros((1, H1)),

            "W2": self.scale * rng.standard_normal((H1, H2)),
            "b2": np.zeros((1, H2)),

            "W3": self.scale * rng.standard_normal((H2, 1)),
            "b3": np.zeros((1, 1)),
        }

    def _forward(self, X, params):
        Z1 = X @ params["W1"] + params["b1"]
        A1 = tanh(Z1)

        Z2 = A1 @ params["W2"] + params["b2"]
        A2 = tanh(Z2)

        Z3 = A2 @ params["W3"] + params["b3"]
        Yhat = sigmoid(Z3)

        cache = {
            "X": X,
            "Z1": Z1, "A1": A1,
            "Z2": Z2, "A2": A2,
            "Z3": Z3,
            "Yhat": Yhat
        }
        return Yhat, cache

    def _backward(self, y, params, cache):
        X = cache["X"]
        Z1, A1 = cache["Z1"], cache["A1"]
        Z2, A2 = cache["Z2"], cache["A2"]
        Yhat = cache["Yhat"]

        N = X.shape[0]

        # Output layer
        dZ3 = (Yhat - y) / N
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        # Hidden layer 2
        dA2 = dZ3 @ params["W3"].T
        dZ2 = dA2 * dtanh(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer 1
        dA1 = dZ2 @ params["W2"].T
        dZ1 = dA1 * dtanh(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3
        }

    def _final_weight_matrix_key(self):
        return "W3"
    
# -----------------------
# Model 5: 2-1024-1024-1 (ridiculously wide)
# -----------------------
class XOR_MLP_2_1024_1024_1(_XORBase):
    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        H1 = 1024
        H2 = 1024
        return {
            "W1": self.scale * rng.standard_normal((2, H1)),
            "b1": np.zeros((1, H1)),

            "W2": self.scale * rng.standard_normal((H1, H2)),
            "b2": np.zeros((1, H2)),

            "W3": self.scale * rng.standard_normal((H2, 1)),
            "b3": np.zeros((1, 1)),
        }

    def _forward(self, X, params):
        Z1 = X @ params["W1"] + params["b1"]
        A1 = tanh(Z1)

        Z2 = A1 @ params["W2"] + params["b2"]
        A2 = tanh(Z2)

        Z3 = A2 @ params["W3"] + params["b3"]
        Yhat = sigmoid(Z3)

        cache = {
            "X": X,
            "Z1": Z1, "A1": A1,
            "Z2": Z2, "A2": A2,
            "Z3": Z3,
            "Yhat": Yhat
        }
        return Yhat, cache

    def _backward(self, y, params, cache):
        X = cache["X"]
        Z1, A1 = cache["Z1"], cache["A1"]
        Z2, A2 = cache["Z2"], cache["A2"]
        Yhat = cache["Yhat"]
        N = X.shape[0]

        dZ3 = (Yhat - y) / N
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ params["W3"].T
        dZ2 = dA2 * dtanh(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ params["W2"].T
        dZ1 = dA1 * dtanh(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3
        }

    def _final_weight_matrix_key(self):
        return "W3"