import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------
def prepare_btsettl_dataframe(df):
    cols = ["Teff(K)", "M/Ms", "age_Myr"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=cols)
    out = out[(out["Teff(K)"] > 0) & (out["M/Ms"] > 0) & (out["age_Myr"] > 0)]
    out = out.sort_values(["age_Myr", "Teff(K)"]).reset_index(drop=True)
    return out

def fit_standardizer(y):
    mu = np.mean(y, axis=0)
    sig = np.std(y, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return mu, sig

def transform_standard(y, mu, sig):
    return (y - mu) / sig

def inverse_standard(y_n, mu, sig):
    return y_n * sig + mu

def predict_mass_from_teff_age(model, teff, age_myr, y_mu, y_sig):
    teff = np.asarray(teff, dtype=float).reshape(-1)
    age_myr = np.asarray(age_myr, dtype=float).reshape(-1)

    X = np.column_stack([
        teff,
        np.log10(age_myr),
    ])

    y_pred_n = model.predict(X)
    y_pred_logm = inverse_standard(y_pred_n, y_mu, y_sig)
    mass_pred = 10 ** y_pred_logm[:, 0]
    return mass_pred

# ---------------------------------------------------------
# Curves: model vs neural network prediction
# ---------------------------------------------------------
def mode_nn_prediction(ages_to_plot, df_clean, model_mass, y_fwd_mu, y_fwd_sig):

    logages = np.log10(ages_to_plot)
    norm = Normalize(vmin=logages.min(), vmax=logages.max())
    cmap = plt.cm.Greys

    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    for age in ages_to_plot:
        sub = df_clean[np.isclose(df_clean["age_Myr"], age)].sort_values("Teff(K)")
        if len(sub) < 3:
            continue

        teff_model = sub["Teff(K)"].to_numpy(dtype=float)
        mass_model = sub["M/Ms"].to_numpy(dtype=float)

        color = cmap(norm(np.log10(age)))

        # Real BT-Settl curve
        ax.plot(
            teff_model,
            mass_model,
            linewidth=1.8,
            color=color,
            alpha=0.95,
        )

        # NN prediction on a Teff grid over the same Teff range
        teff_grid = np.linspace(teff_model.min(), teff_model.max(), 250)
        mass_pred = predict_mass_from_teff_age(
            model_mass,
            teff=teff_grid,
            age_myr=np.full_like(teff_grid, age, dtype=float),
            y_mu=y_fwd_mu,
            y_sig=y_fwd_sig,
        )

        ax.plot(
            teff_grid,
            mass_pred,
            linestyle="--",
            linewidth=1.5,
            color=color,
            alpha=0.95,
        )

    ax.set_xlabel(r"$T_{\rm eff}\ [{\rm K}]$")
    ax.set_ylabel(r"$M/M_\odot$")
    ax.set_title("BT-Settl models (solid) vs neural-network prediction (dashed)")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"$\log_{10}({\rm age\,[Myr]})$")

    ax.invert_xaxis()

    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

def curve_by_curve_error(ages_to_plot, df_clean, model_mass, y_fwd_mu, y_fwd_sig):
    curve_errors = []

    for age in ages_to_plot:
        sub = df_clean[np.isclose(df_clean["age_Myr"], age)].sort_values("Teff(K)")
        if len(sub) < 3:
            continue

        teff_model = sub["Teff(K)"].to_numpy(dtype=float)
        mass_model = sub["M/Ms"].to_numpy(dtype=float)

        mass_pred = predict_mass_from_teff_age(
            model_mass,
            teff=teff_model,
            age_myr=np.full_like(teff_model, age, dtype=float),
            y_mu=y_fwd_mu,
            y_sig=y_fwd_sig,
        )

        rel = np.abs((mass_pred - mass_model) / mass_model)
        curve_errors.append(np.mean(rel))

    print("Mean curve-by-curve relative error:", float(np.mean(curve_errors)))

def predict_age_from_teff_mass(model, teff, mass, y_mu, y_sig):
    teff = np.asarray(teff, dtype=float).reshape(-1)
    mass = np.asarray(mass, dtype=float).reshape(-1)

    X = np.column_stack([
        teff,
        np.log10(mass),
    ])

    y_pred_n = model.predict(X)
    y_pred_logage = inverse_standard(y_pred_n, y_mu, y_sig)
    age_pred = 10 ** y_pred_logage[:, 0]
    return age_pred, y_pred_logage[:, 0]

def make_synthetic_stars_for_age_estimation(
    df,
    n_samples=300,
    teff_noise_frac=0.015,   # 1.5%
    mass_noise_frac=0.05,    # 5%
    seed=42,
):
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(df), size=n_samples, replace=True)
    sample = df.iloc[idx].copy().reset_index(drop=True)

    teff_true = sample["Teff(K)"].to_numpy(dtype=float)
    mass_true = sample["M/Ms"].to_numpy(dtype=float)
    age_true = sample["age_Myr"].to_numpy(dtype=float)

    teff_obs = teff_true * (1.0 + rng.normal(0.0, teff_noise_frac, size=n_samples))
    mass_obs = mass_true * (1.0 + rng.normal(0.0, mass_noise_frac, size=n_samples))

    teff_obs = np.clip(teff_obs, df["Teff(K)"].min(), df["Teff(K)"].max())
    mass_obs = np.clip(mass_obs, 1e-6, None)

    syn = pd.DataFrame({
        "Teff_true": teff_true,
        "mass_true": mass_true,
        "age_true_Myr": age_true,
        "Teff_obs": teff_obs,
        "mass_obs": mass_obs,
    })

    return syn