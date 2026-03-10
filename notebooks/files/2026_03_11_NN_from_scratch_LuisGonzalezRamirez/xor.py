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

def dtanh(x):
    t = np.tanh(x)
    return 1.0 - t**2

def bce(y_hat, y, eps=1e-12):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

# -----------------------
# Model (2-2-1)
# -----------------------
def init_params_xor(seed=42, hidden=2, scale=1.5):
    rng = np.random.default_rng(seed)
    params = {
        "W1": scale * rng.standard_normal((2, hidden)),
        "b1": np.zeros((1, hidden)),
        "W2": scale * rng.standard_normal((hidden, 1)),
        "b2": np.zeros((1, 1)),
    }
    return params

def forward_xor(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = X @ W1 + b1           # (N,2)
    A1 = tanh(Z1)              # (N,2)
    Z2 = A1 @ W2 + b2          # (N,1)
    Yhat = sigmoid(Z2)         # (N,1)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "Yhat": Yhat}
    return Yhat, cache

def backward_xor(y, params, cache):
    Z1, A1, Yhat = cache["Z1"], cache["A1"], cache["Yhat"]
    W2 = params["W2"]
    N = cache["X"].shape[0]

    # BCE + sigmoid => dL/dZ2 = (Yhat - y)/N
    dZ2 = (Yhat - y) / N       # (N,1)

    dW2 = A1.T @ dZ2           # (2,1)
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1,1)

    dA1 = dZ2 @ W2.T           # (N,2)
    dZ1 = dA1 * dtanh(Z1)      # (N,2)

    dW1 = cache["X"].T @ dZ1   # (2,2)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1,2)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return grads

def step(params, grads, lr=0.3):
    for k in params:
        params[k] = params[k] - lr * grads[k]
    return params

def train_xor(X, y, params, lr=0.3, epochs=2000, print_every=200):
    hist = {"loss": []}
    for ep in range(1, epochs + 1):
        yhat, cache = forward_xor(X, params)
        L = bce(yhat, y)
        grads = backward_xor(y, params, cache)
        params = step(params, grads, lr=lr)
        hist["loss"].append(L)
        if ep == 1 or ep % print_every == 0:
            print(f"Epoch {ep:4d} | BCE = {L:.4f}")
    return params, hist

# -----------------------
# Plots
# -----------------------
def plot_boundary(params, X, y, title="Decision boundary", levels=30):
    x1 = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 220)
    x2 = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 220)
    X1, X2m = np.meshgrid(x1, x2)
    grid = np.c_[X1.ravel(), X2m.ravel()]
    yhat, _ = forward_xor(grid, params)
    Z = yhat.reshape(X1.shape)

    plt.figure(figsize=(5.6,4.6))
    plt.contourf(X1, X2m, Z, levels=levels)
    plt.scatter(X[:,0], X[:,1], c=y[:,0], s=14, edgecolor="white", linewidth=0.3)
    plt.title(title)
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.grid(True)
    plt.show()

# -----------------------
# Loss surface for the "game" (vary only W2[0,0], W2[1,0])
# -----------------------
def compute_loss_surface_W2(X, y, params, t1_range=(-6, 6), t2_range=(-6, 6), grid=100):
    W2_orig = params["W2"].copy()

    t1_vals = np.linspace(*t1_range, grid)
    t2_vals = np.linspace(*t2_range, grid)
    L = np.zeros((grid, grid))

    for i, t1 in enumerate(t1_vals):
        for j, t2 in enumerate(t2_vals):
            params["W2"][0,0] = t1
            params["W2"][1,0] = t2
            yhat, _ = forward_xor(X, params)
            L[j, i] = bce(yhat, y)

    params["W2"] = W2_orig
    return t1_vals, t2_vals, L

# -----------------------
# The game: animate descent on the real NN loss surface
# -----------------------
def nn_descent_game_W2(
    X, y, params,
    t1_vals, t2_vals, Lgrid,
    lr=0.6,
    steps=45,
    pause=0.12,
    start_far=True,
    start_point=None,   # optional tuple (t1, t2)
    show_contours=True
):
    """
    We freeze all parameters except W2[0,0] and W2[1,0].
    We animate gradient descent using the real backprop gradients.

    start_far:
      If True, we start from a point far from the center (near the edges).
    start_point:
      If provided, overrides start_far, explicit (t1, t2).
    """
    T1, T2 = np.meshgrid(t1_vals, t2_vals)
    W2_orig = params["W2"].copy()

    # Choose a far starting point
    if start_point is not None:
        params["W2"][0,0] = float(start_point[0])
        params["W2"][1,0] = float(start_point[1])
    elif start_far:
        # pick opposite corners to make the trajectory visible
        params["W2"][0,0] = float(t1_vals[-1]) * 0.9
        params["W2"][1,0] = float(t2_vals[0])  * 0.9

    path_t1, path_t2 = [], []

    for step_idx in range(steps):
        clear_output(wait=True)

        yhat, cache = forward_xor(X, params)
        grads = backward_xor(y, params, cache)

        t1 = float(params["W2"][0,0])
        t2 = float(params["W2"][1,0])
        path_t1.append(t1); path_t2.append(t2)

        # Update ONLY these two parameters
        params["W2"][0,0] -= lr * grads["W2"][0,0]
        params["W2"][1,0] -= lr * grads["W2"][1,0]

        plt.figure(figsize=(6,5))
        if show_contours:
            plt.contourf(T1, T2, Lgrid, levels=60)
            plt.colorbar(label="BCE loss")

        plt.plot(path_t1, path_t2, "r.-", linewidth=2)
        plt.scatter(t1, t2, s=120, facecolor="white", edgecolor="red", linewidth=2)

        plt.title(
            f"Step {step_idx+1} | "
            + r"$W_{2,[0,0]}=$" + f"{t1:.3f}, "
            + r"$W_{2,[1,0]}=$" + f"{t2:.3f}",
            pad=18
        )
        plt.xlabel(r"$W_{2,[0,0]}$")
        plt.ylabel(r"$W_{2,[1,0]}$")
        plt.grid(True)
        plt.show()

        time.sleep(pause)

    print("🎉 Neural Network Descent Complete!")

    # restore
    params["W2"] = W2_orig
    return np.array(path_t1), np.array(path_t2)

def plot_loss_surface_2d(t1_vals, t2_vals, Lgrid, title="Loss surface"):
    T1, T2 = np.meshgrid(t1_vals, t2_vals)
    plt.figure(figsize=(6,5))
    plt.contourf(T1, T2, Lgrid, levels=60)
    plt.colorbar(label="BCE loss")
    plt.xlabel("W2[0,0]")
    plt.ylabel("W2[1,0]")
    plt.title(title)
    plt.grid(True)
    plt.show()