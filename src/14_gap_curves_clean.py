import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("data/clean")
FIG = Path("figures")
FIG.mkdir(parents=True, exist_ok=True)

def binned_curve(df, x, y, bins):
    df = df[[x, y]].dropna().copy()
    df["bin"] = pd.cut(df[x], bins=bins, include_lowest=True)
    g = df.groupby("bin")
    mid = g[x].mean()
    mean = g[y].mean()
    sem = g[y].std() / np.sqrt(g[y].count())
    return mid, mean, sem

def plot_curve(x, mean, sem, label=None, color=None):
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - sem, mean + sem, alpha=0.25, color=color)

def main():
    df = pd.read_parquet(CLEAN / "bioage_estimates.parquet").copy()
    df["gap"] = df["BioAge"] - df["ChronAge"]

    bins = np.linspace(df["ChronAge"].min(), df["ChronAge"].max(), 12)

    plt.figure()
    x, m, s = binned_curve(df, "ChronAge", "gap", bins)
    plot_curve(x, m, s, label="All subjects", color="black")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Biological − Chronological age (years)")
    plt.title("Biological age gap over time (population)")
    plt.tight_layout()
    plt.savefig(FIG / "09_gap_over_time_all.png", dpi=200)
    plt.close()

    plt.figure()
    for sex, col in [("F", "tab:blue"), ("M", "tab:orange")]:
        d = df[df["Sex"] == sex]
        x, m, s = binned_curve(d, "ChronAge", "gap", bins)
        plot_curve(x, m, s, label=sex, color=col)

    plt.axhline(0, linewidth=1)
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Biological − Chronological age (years)")
    plt.title("Biological age gap over time by sex")
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.savefig(FIG / "10_gap_over_time_by_sex.png", dpi=200)
    plt.close()

    print("✅ Clean gap curves saved:")
    print(" - figures/09_gap_over_time_all.png")
    print(" - figures/10_gap_over_time_by_sex.png")

if __name__ == "__main__":
    main()
