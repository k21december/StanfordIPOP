import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN = Path("data/clean")
OUTDIR = Path("results/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(CLEAN / "bioage_estimates_2state.parquet")

# Ensure sorted
df = df.sort_values(["HashedID", "t_years"])

# ---------- 1. MetAge vs future labs ----------
df["future_labs"] = df.groupby("HashedID")["zmean_labs"].shift(-1)

plt.figure(figsize=(6,5))
plt.scatter(df["MetAge_gap"], df["future_labs"], alpha=0.3)
plt.xlabel("MetAge Gap")
plt.ylabel("Future Labs (zmean)")
plt.title("MetAge predicts future labs")
plt.savefig(OUTDIR / "metage_vs_future_labs.png", dpi=300)
plt.close()

# ---------- 2. Immune state vs infection ----------
inf = df["Event"].astype(str).str.contains("Infection", case=False, na=False)

plt.figure(figsize=(6,5))
plt.boxplot([
    df.loc[~inf, "ImmuneState"].dropna(),
    df.loc[inf, "ImmuneState"].dropna()
])
plt.xticks([1,2], ["Baseline", "Infection"])
plt.ylabel("Immune State")
plt.title("Immune state increases during infection")
plt.savefig(OUTDIR / "immune_infection_boxplot.png", dpi=300)
plt.close()

# ---------- 3. Trajectory example ----------
summary = (
    df.groupby("HashedID")
      .agg(
          n_visits=("VisitID", "count"),
          max_t=("t_years", "max"),
          immune_max=("ImmuneState", "max"),
          immune_std=("ImmuneState", "std"),
      )
      .sort_values(["immune_max", "max_t", "n_visits"], ascending=False)
)

sample_id = summary.index[0]
g = df[df["HashedID"] == sample_id].sort_values("t_years")

fig, ax1 = plt.subplots(figsize=(7,5))
ax1.plot(g["t_years"], g["ChronAge"], label="ChronAge", linestyle="--")
ax1.plot(g["t_years"], g["MetAge"], label="MetAge")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Age / MetAge")

ax2 = ax1.twinx()
ax2.plot(g["t_years"], g["ImmuneState"], label="ImmuneState")
ax2.set_ylabel("Immune State")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title(f"Subject trajectory ({sample_id})")
plt.tight_layout()
plt.savefig(OUTDIR / "example_trajectory.png", dpi=300)
plt.close()

# ---------- 4. Male vs Female comparison ----------
if "Sex" in df.columns:
    male = df[df["Sex"] == "M"]
    female = df[df["Sex"] == "F"]

    plt.figure(figsize=(6,5))
    plt.hist(male["MetAge_gap"], bins=30, alpha=0.5, label="Male")
    plt.hist(female["MetAge_gap"], bins=30, alpha=0.5, label="Female")
    plt.legend()
    plt.xlabel("MetAge Gap")
    plt.title("Male vs Female MetAge distribution")
    plt.savefig(OUTDIR / "sex_metage_hist.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.hist(male["ImmuneState"], bins=30, alpha=0.5, label="Male")
    plt.hist(female["ImmuneState"], bins=30, alpha=0.5, label="Female")
    plt.legend()
    plt.xlabel("Immune State")
    plt.title("Male vs Female immune state")
    plt.savefig(OUTDIR / "sex_immune_hist.png", dpi=300)
    plt.close()

print("Plots saved to:", OUTDIR)
