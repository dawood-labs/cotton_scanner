"""Mean NDVI curve per reference class, and where the model's cotton probability sits.

One panel per AOI. The point of the picture is the separation: if rice sits on top of
cotton through the season then the confusion is in the signal and no threshold fixes it;
if it separates and the model still calls it cotton, the training set is what is short.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/home/jovyan/FAO/cotton")
CURVES = BASE / "validation_data/reference_curves"
OUT = BASE / "validation_runs/reference_curves.png"

COLOR = {"cotton": "#c1440e", "rice": "#2e7d32", "sugarcane": "#6a1b9a",
         "fall_maize": "#e0a800", "orchard": "#1565c0"}


def main(paths):
    tables = {p.stem: pd.read_parquet(p) for p in paths}
    if not tables:
        print("no curve tables yet")
        return

    fig, axes = plt.subplots(len(tables), 2, figsize=(13, 3.1 * len(tables)), squeeze=False)
    for row, (name, table) in enumerate(sorted(tables.items())):
        dates = [c for c in table.columns if c[:2] == "20"]
        stamps = pd.to_datetime(dates)

        curve_ax, prob_ax = axes[row]
        for crop, group in table.groupby("crop"):
            mean = group[dates].mean()
            curve_ax.plot(stamps, mean, label=f"{crop} (n={len(group):,})",
                          color=COLOR.get(crop), lw=1.8)
            curve_ax.fill_between(stamps, group[dates].quantile(0.25),
                                  group[dates].quantile(0.75),
                                  color=COLOR.get(crop), alpha=0.10, lw=0)
            prob_ax.hist(group["cotton_prob"], bins=25, range=(0, 1), histtype="step",
                         density=True, color=COLOR.get(crop), lw=1.6, label=crop)

        curve_ax.set_title(f"{name} -- mean NDVI, IQR shaded", fontsize=10)
        curve_ax.set_ylim(-0.1, 0.95)
        curve_ax.legend(fontsize=7, loc="upper right")
        curve_ax.grid(alpha=0.25)
        prob_ax.axvline(0.5, color="black", ls="--", lw=1)
        prob_ax.set_title(f"{name} -- model cotton probability", fontsize=10)
        prob_ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"written: {OUT}")


if __name__ == "__main__":
    given = [Path(a) for a in sys.argv[1:]]
    main(given or sorted(CURVES.glob("*.parquet")))
