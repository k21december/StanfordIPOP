import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("data/clean")
FIG = Path("figures")
FIG.mkdir(parents=True, exist_ok=True)

COLOR_MAP = {
    "F": "tab:blue",
    "M": "tab:orange",
}

def sample_subjects(df, n=25, seed=7, min_visits=5):
    sub_counts = df.groupby("HashedID").size()
    eligible = sub_counts[sub_counts >= min_visits].index.tolist()
    if len(eligible) <= n:
        return eligible
    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=n, replace=False).tolist()

def spaghetti_bioage(df, title, outname, color_by=None, n_sub=25):
    df = df[np.isfinite(df["ChronAge"]) & np.isfinite(df["BioAge"])].copy()
    subs = sample_subjects(df, n=n_sub, min_visits=5)
    d = df[df["HashedID"].isin(subs)].copy()
    d = d.sort_values(["HashedID", "ChronAge", "VisitID"])

    plt.figure()

    if color_by is None:
        for sid, g in d.groupby("HashedID"):
            plt.plot(g["ChronAge"], g["BioAge"], marker="o", linewidth=1, markersize=3, alpha=0.6)
    else:
        # plot per-subject trajectories but color by group (sex)
        for key, gg in d.groupby(color_by):
            col = COLOR_MAP.get(str(key), "gray")
            for sid, g in gg.groupby("HashedID"):
                plt.plot(g["ChronAge"], g["BioAge"], marker="o", linewidth=1, markersize=3, alpha=0.6, color=col)
            # legend proxy (one per group)
            plt.plot([], [], color=col, label=str(key))
        plt.legend(title=color_by, fontsize=9)

    plt.xlabel("Chronological age (years)")
    plt.ylabel("Biological age (years)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / outname, dpi=200)
    plt.close()

def spaghetti_gap(df, title, outname, color_by=None, n_sub=25):
    df = df[np.isfinite(df["ChronAge"]) & np.isfinite(df["BioAge_gap"])].copy()
    subs = sample_subjects(df, n=n_sub, min_visits=5)
    d = df[df["HashedID"].isin(subs)].copy()
    d = d.sort_values(["HashedID", "ChronAge", "VisitID"])

    plt.figure()

    if color_by is None:
        for sid, g in d.groupby("HashedID"):
            plt.plot(g["ChronAge"], g["BioAge_gap"], marker="o", linewidth=1, markersize=3, alpha=0.6)
    else:
        for key, gg in d.groupby(color_by):
            col = COLOR_MAP.get(str(key), "gray")
            for sid, g in gg.groupby("HashedID"):
                plt.plot(g["ChronAge"], g["BioAge_gap"], marker="o", linewidth=1, markersize=3, alpha=0.6, color=col)
            plt.plot([], [], color=col, label=str(key))
        plt.legend(title=color_by, fontsize=9)

    plt.axhline(0, linewidth=1)
    plt.xlabel("Chronological age (years)")
    plt.ylabel("BioAge − ChronAge (years)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / outname, dpi=200)
    plt.close()

def main():
    df = pd.read_parquet(CLEAN / "bioage_estimates.parquet").copy()

    spaghetti_bioage(df, "MIDUS-style: individual trajectories (BioAge vs ChronAge)",
                     "05_spaghetti_bioage_all.png", color_by=None, n_sub=25)
    spaghetti_gap(df, "MIDUS-style: individual trajectories (Gap vs ChronAge)",
                  "06_spaghetti_gap_all.png", color_by=None, n_sub=25)

    spaghetti_bioage(df, "MIDUS-style: individual trajectories by sex (BioAge vs ChronAge)",
                     "07_spaghetti_bioage_by_sex.png", color_by="Sex", n_sub=30)
    spaghetti_gap(df, "MIDUS-style: individual trajectories by sex (Gap vs ChronAge)",
                  "08_spaghetti_gap_by_sex.png", color_by="Sex", n_sub=30)

    print("✅ Spaghetti plots saved to figures/")
    for p in sorted(FIG.glob("0*_spaghetti_*.png")):
        print(" -", p)

if __name__ == "__main__":
    main()
