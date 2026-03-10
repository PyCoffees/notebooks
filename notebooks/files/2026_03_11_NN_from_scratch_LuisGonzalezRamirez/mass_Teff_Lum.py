import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# =========================================================
# Basic cleaning
# =========================================================
def prepare_btsettl_dataframe_lum2(df, use_Li=False):
    """
    Clean dataframe for luminosity regression with inputs:
        - Teff(K)
        - M/Ms
    optional:
        - Li

    Target:
        - L/Ls
    """
    cols = ["Teff(K)", "M/Ms", "L/Ls"]
    if use_Li:
        cols.append("Li")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=cols)

    mask = (
        (out["Teff(K)"] > 0) &
        (out["M/Ms"] > 0) &
        (out["L/Ls"] > 0)
    )

    if use_Li:
        mask &= np.isfinite(out["Li"])
        mask &= (out["Li"] >= 0)

    out = out[mask].copy()
    out = out.sort_values(["Teff(K)", "M/Ms"]).reset_index(drop=True)
    return out


# =========================================================
# Standardization
# =========================================================
def fit_standardizer(y):
    mu = np.mean(y, axis=0)
    sig = np.std(y, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return mu, sig


def transform_standard(y, mu, sig):
    return (y - mu) / sig


def inverse_standard(y_n, mu, sig):
    return y_n * sig + mu


# =========================================================
# Feature builders
# =========================================================
def build_X_lum_from_teff_mass(df, use_logs=True):
    """
    Build feature matrix X for:
        luminosity = NN(Teff, mass)

    If use_logs=True:
        X = [log10(Teff), log10(M/Ms)]
    else:
        X = [Teff, M/Ms]
    """
    teff = df["Teff(K)"].to_numpy(dtype=float)
    mass = df["M/Ms"].to_numpy(dtype=float)

    if use_logs:
        X = np.column_stack([
            np.log10(teff),
            np.log10(mass),
        ])
    else:
        X = np.column_stack([teff, mass])

    return X


def build_X_lum_from_teff_mass_li(df, use_logs=True):
    """
    Build feature matrix X for:
        luminosity = NN(Teff, mass, Li)

    If use_logs=True:
        X = [log10(Teff), log10(M/Ms), Li]
    else:
        X = [Teff, M/Ms, Li]
    """
    teff = df["Teff(K)"].to_numpy(dtype=float)
    mass = df["M/Ms"].to_numpy(dtype=float)
    li = df["Li"].to_numpy(dtype=float)

    if use_logs:
        X = np.column_stack([
            np.log10(teff),
            np.log10(mass),
            li,
        ])
    else:
        X = np.column_stack([teff, mass, li])

    return X


def build_y_loglum(df):
    """
    Target:
        y = log10(L/Ls)
    """
    return np.log10(df["L/Ls"].to_numpy(dtype=float)).reshape(-1, 1)


# =========================================================
# Prediction helpers
# =========================================================
def predict_lum_from_teff_mass(
    model,
    teff,
    mass,
    y_mu,
    y_sig,
    use_logs=True,
):
    """
    Predict luminosity from Teff and mass.

    Returns
    -------
    lum_pred_lsun : ndarray
    loglum_pred : ndarray
    """
    teff = np.asarray(teff, dtype=float).reshape(-1)
    mass = np.asarray(mass, dtype=float).reshape(-1)

    valid = np.isfinite(teff) & np.isfinite(mass)
    if use_logs:
        valid &= (teff > 0) & (mass > 0)

    teff_v = teff[valid]
    mass_v = mass[valid]

    if use_logs:
        X = np.column_stack([
            np.log10(teff_v),
            np.log10(mass_v),
        ])
    else:
        X = np.column_stack([teff_v, mass_v])

    y_pred_n = np.asarray(model.predict(X), dtype=float).reshape(-1, 1)
    y_pred_loglum = inverse_standard(y_pred_n, y_mu, y_sig).reshape(-1)
    lum_pred = 10 ** y_pred_loglum

    out_lum = np.full(teff.shape, np.nan, dtype=float)
    out_loglum = np.full(teff.shape, np.nan, dtype=float)
    out_lum[valid] = lum_pred
    out_loglum[valid] = y_pred_loglum

    return out_lum, out_loglum


def predict_lum_from_teff_mass_li(
    model,
    teff,
    mass,
    Li,
    y_mu,
    y_sig,
    use_logs=True,
):
    """
    Predict luminosity from Teff, mass, and Li.

    Returns
    -------
    lum_pred_lsun : ndarray
    loglum_pred : ndarray
    """
    teff = np.asarray(teff, dtype=float).reshape(-1)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    Li = np.asarray(Li, dtype=float).reshape(-1)

    valid = np.isfinite(teff) & np.isfinite(mass) & np.isfinite(Li)
    if use_logs:
        valid &= (teff > 0) & (mass > 0)

    teff_v = teff[valid]
    mass_v = mass[valid]
    li_v = Li[valid]

    if use_logs:
        X = np.column_stack([
            np.log10(teff_v),
            np.log10(mass_v),
            li_v,
        ])
    else:
        X = np.column_stack([teff_v, mass_v, li_v])

    y_pred_n = np.asarray(model.predict(X), dtype=float).reshape(-1, 1)
    y_pred_loglum = inverse_standard(y_pred_n, y_mu, y_sig).reshape(-1)
    lum_pred = 10 ** y_pred_loglum

    out_lum = np.full(teff.shape, np.nan, dtype=float)
    out_loglum = np.full(teff.shape, np.nan, dtype=float)
    out_lum[valid] = lum_pred
    out_loglum[valid] = y_pred_loglum

    return out_lum, out_loglum


# =========================================================
# Synthetic stars for luminosity estimation
# =========================================================
def make_synthetic_stars_for_lum_estimation_teff_mass(
    df,
    n_samples=300,
    teff_noise_frac=0.015,
    mass_noise_frac=0.05,
    seed=42,
):
    """
    Synthetic observed stars with:
        - Teff_obs
        - mass_obs

    True labels:
        - lum_true_Lsun
    """
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(df), size=n_samples, replace=True)
    sample = df.iloc[idx].copy().reset_index(drop=True)

    teff_true = sample["Teff(K)"].to_numpy(dtype=float)
    mass_true = sample["M/Ms"].to_numpy(dtype=float)
    lum_true = sample["L/Ls"].to_numpy(dtype=float)

    teff_obs = teff_true * (1.0 + rng.normal(0.0, teff_noise_frac, size=n_samples))
    mass_obs = mass_true * (1.0 + rng.normal(0.0, mass_noise_frac, size=n_samples))

    teff_obs = np.clip(teff_obs, df["Teff(K)"].min(), df["Teff(K)"].max())
    mass_obs = np.clip(mass_obs, 1e-8, None)

    syn = pd.DataFrame({
        "Teff_true": teff_true,
        "mass_true": mass_true,
        "lum_true_Lsun": lum_true,
        "Teff_obs": teff_obs,
        "mass_obs": mass_obs,
    })

    return syn


def make_synthetic_stars_for_lum_estimation_teff_mass_li(
    df,
    n_samples=300,
    teff_noise_frac=0.015,
    mass_noise_frac=0.05,
    li_noise_abs=0.03,
    seed=42,
):
    """
    Synthetic observed stars with:
        - Teff_obs
        - mass_obs
        - li_obs

    True labels:
        - lum_true_Lsun
    """
    if "Li" not in df.columns:
        raise ValueError("Column 'Li' not found in dataframe.")

    rng = np.random.default_rng(seed)

    idx = rng.choice(len(df), size=n_samples, replace=True)
    sample = df.iloc[idx].copy().reset_index(drop=True)

    teff_true = sample["Teff(K)"].to_numpy(dtype=float)
    mass_true = sample["M/Ms"].to_numpy(dtype=float)
    li_true = sample["Li"].to_numpy(dtype=float)
    lum_true = sample["L/Ls"].to_numpy(dtype=float)

    teff_obs = teff_true * (1.0 + rng.normal(0.0, teff_noise_frac, size=n_samples))
    mass_obs = mass_true * (1.0 + rng.normal(0.0, mass_noise_frac, size=n_samples))
    li_obs = li_true + rng.normal(0.0, li_noise_abs, size=n_samples)

    teff_obs = np.clip(teff_obs, df["Teff(K)"].min(), df["Teff(K)"].max())
    mass_obs = np.clip(mass_obs, 1e-8, None)
    li_obs = np.clip(li_obs, 0.0, 1.0)

    syn = pd.DataFrame({
        "Teff_true": teff_true,
        "mass_true": mass_true,
        "li_true": li_true,
        "lum_true_Lsun": lum_true,
        "Teff_obs": teff_obs,
        "mass_obs": mass_obs,
        "li_obs": li_obs,
    })

    return syn


# =========================================================
# Mesh builder in (Teff, M)
# =========================================================
def _build_teff_mass_mesh_from_dataframe(
    df_clean,
    n_teff=300,
    n_mass=300,
    teff_range=None,
    mass_range=None,
):
    teff_data = df_clean["Teff(K)"].to_numpy(dtype=float)
    mass_data = df_clean["M/Ms"].to_numpy(dtype=float)

    if teff_range is None:
        teff_min, teff_max = np.min(teff_data), np.max(teff_data)
    else:
        teff_min, teff_max = teff_range

    if mass_range is None:
        mass_min, mass_max = np.min(mass_data), np.max(mass_data)
    else:
        mass_min, mass_max = mass_range

    teff_vals = np.linspace(teff_min, teff_max, n_teff)
    mass_vals = np.linspace(mass_min, mass_max, n_mass)

    Teff_grid, Mass_grid = np.meshgrid(teff_vals, mass_vals)
    return Teff_grid, Mass_grid


# =========================================================
# 2D luminosity maps in (Teff, M)
# =========================================================
def plot_nn_predicted_lum_map_teff_mass(
    model,
    df_clean,
    y_mu,
    y_sig,
    n_teff=300,
    n_mass=300,
    teff_range=None,
    mass_range=None,
    use_logs=True,
    log_lum_color=True,
    cmap="plasma",
    invert_xaxis=True,
    show_model_points=True,
):
    """
    Plot a 2D luminosity map in the (Teff, M) plane:
        luminosity = NN(Teff, mass)
    """
    Teff_grid, Mass_grid = _build_teff_mass_mesh_from_dataframe(
        df_clean=df_clean,
        n_teff=n_teff,
        n_mass=n_mass,
        teff_range=teff_range,
        mass_range=mass_range,
    )

    teff_flat = Teff_grid.ravel()
    mass_flat = Mass_grid.ravel()

    lum_pred, loglum_pred = predict_lum_from_teff_mass(
        model=model,
        teff=teff_flat,
        mass=mass_flat,
        y_mu=y_mu,
        y_sig=y_sig,
        use_logs=use_logs,
    )

    Z = loglum_pred.reshape(Teff_grid.shape) if log_lum_color else lum_pred.reshape(Teff_grid.shape)

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    cf = ax.contourf(
        Teff_grid,
        Mass_grid,
        Z,
        levels=200,
        cmap=cmap,
    )

    if show_model_points:
        ax.scatter(
            df_clean["Teff(K)"],
            df_clean["M/Ms"],
            s=12,
            c="k",
            alpha=0.30,
            linewidths=0,
            label="BT-Settl points",
        )
        ax.legend(fontsize=9, loc="best")

    ax.set_xlabel(r"$T_{\rm eff}\ [{\rm K}]$")
    ax.set_ylabel(r"$M/M_\odot$")
    ax.set_title(r"NN-predicted luminosity in the $(T_{\rm eff}, M)$ plane")

    if invert_xaxis:
        ax.invert_xaxis()

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(
        r"Predicted $\log_{10}(L/L_\odot)$"
        if log_lum_color else
        r"Predicted $L/L_\odot$"
    )

    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_true_vs_nn_lum_maps_teff_mass(
    model,
    df_clean,
    y_mu,
    y_sig,
    n_teff=300,
    n_mass=300,
    teff_range=None,
    mass_range=None,
    use_logs=True,
    log_lum_color=True,
    cmap="plasma",
    invert_xaxis=True,
    show_model_points=True,
):
    """
    Compare true and NN-predicted luminosity maps in the (Teff, M) plane.

    The true panel is built from the irregular BT-Settl grid using triangulation.
    The predicted panel is built from a dense meshgrid evaluated by the NN.
    """
    teff_true = df_clean["Teff(K)"].to_numpy(dtype=float)
    mass_true = df_clean["M/Ms"].to_numpy(dtype=float)
    lum_true = df_clean["L/Ls"].to_numpy(dtype=float)

    if log_lum_color:
        z_true = np.log10(lum_true)
    else:
        z_true = lum_true

    Teff_grid, Mass_grid = _build_teff_mass_mesh_from_dataframe(
        df_clean=df_clean,
        n_teff=n_teff,
        n_mass=n_mass,
        teff_range=teff_range,
        mass_range=mass_range,
    )

    teff_flat = Teff_grid.ravel()
    mass_flat = Mass_grid.ravel()

    lum_pred, loglum_pred = predict_lum_from_teff_mass(
        model=model,
        teff=teff_flat,
        mass=mass_flat,
        y_mu=y_mu,
        y_sig=y_sig,
        use_logs=use_logs,
    )

    Z_pred = loglum_pred.reshape(Teff_grid.shape) if log_lum_color else lum_pred.reshape(Teff_grid.shape)

    vmin = min(np.nanmin(z_true), np.nanmin(Z_pred))
    vmax = max(np.nanmax(z_true), np.nanmax(Z_pred))

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), sharex=True, sharey=True, constrained_layout=True)

    tcf_true = axes[0].tricontourf(
        teff_true,
        mass_true,
        z_true,
        levels=120,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if show_model_points:
        axes[0].scatter(
            teff_true,
            mass_true,
            s=12,
            c="k",
            alpha=0.25,
            linewidths=0,
        )

    axes[0].set_title("True luminosity from grid")
    axes[0].set_xlabel(r"$T_{\rm eff}\ [{\rm K}]$")
    axes[0].set_ylabel(r"$M/M_\odot$")
    axes[0].grid(alpha=0.25)
    if invert_xaxis:
        axes[0].invert_xaxis()

    tcf_pred = axes[1].contourf(
        Teff_grid,
        Mass_grid,
        Z_pred,
        levels=200,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if show_model_points:
        axes[1].scatter(
            teff_true,
            mass_true,
            s=12,
            c="k",
            alpha=0.20,
            linewidths=0,
        )

    axes[1].set_title("NN-predicted luminosity")
    axes[1].set_xlabel(r"$T_{\rm eff}\ [{\rm K}]$")
    axes[1].grid(alpha=0.25)
    if invert_xaxis:
        axes[1].invert_xaxis()

    cbar = fig.colorbar(
        tcf_pred,
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.02,
    )
    cbar.set_label(
        r"$\log_{10}(L/L_\odot)$"
        if log_lum_color else
        r"$L/L_\odot$"
    )

    fig.suptitle(r"Luminosity maps in the $(T_{\rm eff}, M)$ plane", y=1.15, fontsize=20)
    plt.show()


# =========================================================
# Optional slices by age
# =========================================================
def plot_true_lum_slices_teff_mass_by_age(
    df_clean,
    ages_to_plot,
    log_lum_color=True,
    cmap="plasma",
    invert_xaxis=True,
):
    """
    Plot true BT-Settl luminosity slices in the (Teff, M) plane for selected ages.
    """
    ages_to_plot = np.asarray(ages_to_plot, dtype=float)

    ncols = 2
    nrows = int(np.ceil(len(ages_to_plot) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.2 * ncols, 5.0 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    lum_all = df_clean["L/Ls"].to_numpy(dtype=float)
    color_all = np.log10(lum_all) if log_lum_color else lum_all
    vmin = np.nanmin(color_all)
    vmax = np.nanmax(color_all)

    for ax, age in zip(axes.flat, ages_to_plot):
        sub = df_clean[np.isclose(df_clean["age_Myr"], age)].copy()

        if len(sub) < 3:
            ax.set_visible(False)
            continue

        teff = sub["Teff(K)"].to_numpy(dtype=float)
        mass = sub["M/Ms"].to_numpy(dtype=float)
        lum = sub["L/Ls"].to_numpy(dtype=float)

        z = np.log10(lum) if log_lum_color else lum

        tcf = ax.tricontourf(
            teff,
            mass,
            z,
            levels=120,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.scatter(teff, mass, s=12, c="k", alpha=0.25, linewidths=0)
        ax.set_title(f"Age = {age:g} Myr")
        ax.set_xlabel(r"$T_{\rm eff}\ [{\rm K}]$")
        ax.set_ylabel(r"$M/M_\odot$")
        ax.grid(alpha=0.25)

        if invert_xaxis:
            ax.invert_xaxis()

    for ax in axes.flat[len(ages_to_plot):]:
        ax.set_visible(False)

    cbar = fig.colorbar(
        tcf,
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.02,
    )
    cbar.set_label(
        r"$\log_{10}(L/L_\odot)$"
        if log_lum_color else
        r"$L/L_\odot$"
    )

    plt.show()


def plot_nn_lum_slices_teff_mass_by_age(
    model,
    df_clean,
    y_mu,
    y_sig,
    ages_to_plot,
    ncols=2,
    n_teff=260,
    n_mass=260,
    use_logs=True,
    log_lum_color=True,
    cmap="plasma",
    invert_xaxis=True,
):
    """
    Plot several NN-predicted luminosity maps in the (Teff, M) plane,
    one panel per age label in the dataframe title only.

    Note: the prediction itself is age-independent in this model.
    The age is only used to label or compare slices visually.
    """
    ages_to_plot = np.asarray(ages_to_plot, dtype=float)
    nrows = int(np.ceil(len(ages_to_plot) / ncols))

    Teff_grid, Mass_grid = _build_teff_mass_mesh_from_dataframe(
        df_clean=df_clean,
        n_teff=n_teff,
        n_mass=n_mass,
    )

    teff_flat = Teff_grid.ravel()
    mass_flat = Mass_grid.ravel()

    lum_pred, loglum_pred = predict_lum_from_teff_mass(
        model=model,
        teff=teff_flat,
        mass=mass_flat,
        y_mu=y_mu,
        y_sig=y_sig,
        use_logs=use_logs,
    )

    Z = loglum_pred.reshape(Teff_grid.shape) if log_lum_color else lum_pred.reshape(Teff_grid.shape)
    global_min = np.nanmin(Z)
    global_max = np.nanmax(Z)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.2 * ncols, 5.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    last_cf = None
    for ax, age in zip(axes.flat, ages_to_plot):
        last_cf = ax.contourf(
            Teff_grid,
            Mass_grid,
            Z,
            levels=180,
            cmap=cmap,
            vmin=global_min,
            vmax=global_max,
        )

        ax.set_title(fr"NN luminosity map (reference age label = {age:g} Myr)")
        ax.set_xlabel(r"$T_{\rm eff}\ [{\rm K}]$")
        ax.set_ylabel(r"$M/M_\odot$")
        ax.grid(alpha=0.25)

        if invert_xaxis:
            ax.invert_xaxis()

    for ax in axes.flat[len(ages_to_plot):]:
        ax.set_visible(False)

    cbar = fig.colorbar(
        last_cf,
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.02,
    )
    cbar.set_label(
        r"Predicted $\log_{10}(L/L_\odot)$"
        if log_lum_color else
        r"Predicted $L/L_\odot$"
    )

    plt.show()


# =========================================================
# Point-wise evaluation
# =========================================================
def plot_true_vs_predicted_lum(
    loglum_true,
    loglum_pred,
    title="Synthetic luminosity estimation",
):
    loglum_true = np.asarray(loglum_true, dtype=float).reshape(-1)
    loglum_pred = np.asarray(loglum_pred, dtype=float).reshape(-1)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    ax.scatter(loglum_true, loglum_pred, s=24, alpha=0.75)

    xmin = min(loglum_true.min(), loglum_pred.min())
    xmax = max(loglum_true.max(), loglum_pred.max())
    ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1.5)

    ax.set_xlabel(r"True $\log_{10}(L/L_\odot)$")
    ax.set_ylabel(r"Predicted $\log_{10}(L/L_\odot)$")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_lum_error_metrics(loglum_true, loglum_pred):
    loglum_true = np.asarray(loglum_true, dtype=float).reshape(-1)
    loglum_pred = np.asarray(loglum_pred, dtype=float).reshape(-1)

    abs_err = np.abs(loglum_pred - loglum_true)

    print("Mean absolute luminosity error [dex]:", float(np.mean(abs_err)))
    print("Median absolute luminosity error [dex]:", float(np.median(abs_err)))