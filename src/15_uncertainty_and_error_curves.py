import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("data/clean")
FIG = Path("figures")
FIG.mkdir(parents=True, exist_ok=True)

def binned_mean_sem(x, y, bins):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 20:
        return None

    bin_ids = np.digitize(x, bins) - 1
    xb, yb, se = [], [], []
    for k in range(len(bins) - 1):
        idx = bin_ids == k
        if idx.sum() < 10:
            continue
        xv = x[idx]
        yv = y[idx]
        xb.append(np.mean(xv))
        yb.append(np.mean(yv))
        se.append(np.std(yv, ddof=1) / np.sqrt(len(yv)))
    if len(xb) < 3:
        return None
    return np.array(xb), np.array(yb), np.array(se)

def plot_line_with_band(xb, yb, se, label=None, color=None):
    plt.plot(xb, yb, label=label, color=color)
    plt.fill_between(xb, yb - se, yb + se, alpha=0.25, color=color)

def main():
    df = pd.read_parquet(CLEAN / "bioage_estimates.parquet").copy()
    df = df[np.isfinite(df["ChronAge"])].copy()

    df["BioAge_sd"] = np.sqrt(df["BioAge_var"].clip(lower=0))

    bins = np.linspace(df["ChronAge"].min(), df["ChronAge"].max(), 14)

    plt.figure()
    cur = binned_mean_sem(df["ChronAge"].to_numpy(), df["BioAge_sd"].to_numpy(), bins)
    if cur is not None:
        xb, yb, se = cur
        plot_line_with_band(xb, yb, se, label="All", color="black")
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Posterior SD of BioAge (years)")
    plt.title("Uncertainty in biological age increases with age (All)")
    plt.tight_layout()
    plt.savefig(FIG / "11_uncertainty_vs_age_all.png", dpi=200)
    plt.close()

    plt.figure()
    for sex, col in [("F", "tab:blue"), ("M", "tab:orange")]:
        d = df[df["Sex"] == sex]
        cur = binned_mean_sem(d["ChronAge"].to_numpy(), d["BioAge_sd"].to_numpy(), bins)
        if cur is None:
            continue
        xb, yb, se = cur
        plot_line_with_band(xb, yb, se, label=sex, color=col)
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Posterior SD of BioAge (years)")
    plt.title("Uncertainty in biological age by sex")
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.savefig(FIG / "12_uncertainty_vs_age_by_sex.png", dpi=200)
    plt.close()

    ev = pd.read_parquet(CLEAN / "predictive_eval.parquet").copy()

    ev["abs_err_x"] = np.abs(ev["err_x"])
    bins2 = np.linspace(np.nanmin(df["ChronAge"]), np.nanmax(df["ChronAge"]), 14)

    plt.figure()
    cur = binned_mean_sem(ev["gap_true"].to_numpy() * 0 + np.nan, ev["abs_err_x"].to_numpy(), bins2)
    chron_est = (ev["x_true"] - ev["gap_true"]).to_numpy()
    cur = binned_mean_sem(chron_est, ev["abs_err_x"].to_numpy(), bins2)
    if cur is not None:
        xb, yb, se = cur
        plot_line_with_band(xb, yb, se, label="All", color="black")
    plt.xlabel("Chronological age (years)")
    plt.ylabel("|1-step BioAge prediction error| (years)")
    plt.title("1-step BioAge prediction error vs age (All)")
    plt.tight_layout()
    plt.savefig(FIG / "13_abs_pred_error_vs_age_all.png", dpi=200)
    plt.close()

    plt.figure()
    for sex, col in [("F", "tab:blue"), ("M", "tab:orange")]:
        d = ev[ev["Sex"] == sex]
        chron_est = (d["x_true"] - d["gap_true"]).to_numpy()
        cur = binned_mean_sem(chron_est, d["abs_err_x"].to_numpy(), bins2)
        if cur is None:
            continue
        xb, yb, se = cur
        plot_line_with_band(xb, yb, se, label=sex, color=col)
    plt.xlabel("Chronological age (years)")
    plt.ylabel("|1-step BioAge prediction error| (years)")
    plt.title("1-step BioAge prediction error vs age by sex")
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.savefig(FIG / "14_abs_pred_error_vs_age_by_sex.png", dpi=200)
    plt.close()

    print("✅ Saved curves:")
    for name in [
        "11_uncertainty_vs_age_all.png",
        "12_uncertainty_vs_age_by_sex.png",
        "13_abs_pred_error_vs_age_all.png",
        "14_abs_pred_error_vs_age_by_sex.png",
    ]:
        print(" -", FIG / name)

if __name__ == "__main__":
    main()
