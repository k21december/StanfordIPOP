import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("data/clean")
FIG = Path("figures")
FIG.mkdir(parents=True, exist_ok=True)

def binned_curve(df, xcol, ycol, bins=20, min_n=10):
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < min_n:
        return None

    edges = np.quantile(x, np.linspace(0, 1, bins+1))
    edges = np.unique(edges)
    if len(edges) < 4:
        return None

    xb, yb, se = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (x >= lo) & (x <= hi)
        if idx.sum() < min_n:
            continue
        xv = x[idx]
        yv = y[idx]
        xb.append(np.mean(xv))
        yb.append(np.mean(yv))
        se.append(np.std(yv, ddof=1) / np.sqrt(len(yv)))
    if len(xb) < 3:
        return None
    return np.array(xb), np.array(yb), np.array(se)

def plot_group_curves(df, group_cols, title, outprefix, min_group_n=8):
    plt.figure()
    for key, g in df.groupby(group_cols):
        n_sub = g["HashedID"].nunique()
        if n_sub < min_group_n:
            continue
        cur = binned_curve(g, "ChronAge", "BioAge", bins=18, min_n=12)
        if cur is None:
            continue
        xb, yb, se = cur
        label = key if isinstance(key, str) else " | ".join(map(str, key))
        plt.plot(xb, yb, label=f"{label} (n={n_sub})")
        plt.fill_between(xb, yb-1.96*se, yb+1.96*se, alpha=0.15)
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Biological age (years)")
    plt.title(title + " — BioAge vs ChronAge")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / f"{outprefix}_bioage_vs_chronage.png", dpi=200)
    plt.close()

    plt.figure()
    for key, g in df.groupby(group_cols):
        n_sub = g["HashedID"].nunique()
        if n_sub < min_group_n:
            continue
        cur = binned_curve(g, "ChronAge", "BioAge_gap", bins=18, min_n=12)
        if cur is None:
            continue
        xb, yb, se = cur
        label = key if isinstance(key, str) else " | ".join(map(str, key))
        plt.plot(xb, yb, label=f"{label} (n={n_sub})")
        plt.fill_between(xb, yb-1.96*se, yb+1.96*se, alpha=0.15)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Chronological age (years)")
    plt.ylabel("BioAge − ChronAge (years)")
    plt.title(title + " — BioAge Gap vs ChronAge")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / f"{outprefix}_gap_vs_chronage.png", dpi=200)
    plt.close()

def main():
    df = pd.read_parquet(CLEAN / "bioage_estimates.parquet")

    df = df[np.isfinite(df["ChronAge"])].copy()

    df["_all"] = "All"
    plot_group_curves(df, "_all", "General", "01_general", min_group_n=1)

    plot_group_curves(df, "RaceLabel", "By race", "02_race", min_group_n=5)

    plot_group_curves(df, "Sex", "By sex", "03_sex", min_group_n=5)

    df["SexRace"] = df["Sex"].astype(str) + " | " + df["RaceLabel"].astype(str)
    plot_group_curves(df, "SexRace", "By sex × race", "04_sex_race", min_group_n=6)

    print("✅ Curves saved to figures/")
    for p in sorted(FIG.glob("*.png")):
        print(" -", p)

if __name__ == "__main__":
    main()
