"""Match every training cluster against the real per-crop curves and see what it is.

The training labels were assigned by looking at each k-means cluster's curve and
deciding by eye. Nothing checked those calls against a crop that was actually surveyed.
Now we have surveyed curves -- cotton from the mill ground truth, rice / sugarcane /
fall maize from the national scans, orchards from the exclusion mask -- all smoothed the
same way on the same 35 dates.

So: for each training cluster centroid, find which reference crop's curve it sits
closest to. A cluster labelled cotton whose nearest neighbour is rice is not proof of a
mislabel, but it is the shortlist, and the shortlist is what a human can re-check.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/jovyan/FAO/cotton")
TRAINING = BASE / "training_data_parquet/cotton_2025_training_data_v2.csv"
CURVES = BASE / "validation_data/reference_curves"
OUT = BASE / "validation_runs/label_audit"
META = ["AOI", "Cluster_ID", "Cluster_Label", "Feature_Type", "Pixel_ID", "Confidence_Score"]


def reference_profiles(dates):
    """Median curve per crop, pooled over every validation AOI.

    Median rather than mean: the scans include some mixed and mis-drawn polygons, and a
    median does not let a handful of them drag the profile.
    """
    tables = [pd.read_parquet(p) for p in sorted(CURVES.glob("*.parquet"))]
    if not tables:
        return None
    pooled = pd.concat(tables, ignore_index=True)
    missing = [d for d in dates if d not in pooled.columns]
    if missing:
        raise ValueError(f"reference curves lack {len(missing)} of the training dates, e.g. {missing[:3]}")
    return pooled.groupby("crop")[dates].median(), pooled.groupby("crop").size()


def main():
    df = pd.read_csv(TRAINING)
    dates = [c for c in df.columns if c not in META]

    built = reference_profiles(dates)
    if built is None:
        print(f"no reference curves under {CURVES} yet -- run sample_reference_curves.py first")
        return 0
    profiles, counts = built
    print("reference profiles built from surveyed polygons:")
    print(counts.to_string(), "\n")

    centroids = df[df.Feature_Type == "Centroid"].copy()
    curves = centroids[dates].values
    reference = profiles.values
    crops = list(profiles.index)

    # Mean absolute difference over the season. Correlation would call a rice curve and a
    # cotton curve the same thing whenever they merely rise and fall together, and that is
    # exactly the pair we are trying to separate.
    distance = np.abs(curves[:, None, :] - reference[None, :, :]).mean(axis=2)
    nearest = distance.argmin(axis=1)
    centroids["nearest_crop"] = [crops[i] for i in nearest]
    centroids["distance"] = distance.min(axis=1).round(4)
    second = np.partition(distance, 1, axis=1)[:, 1]
    centroids["margin"] = (second - distance.min(axis=1)).round(4)

    print("what each training label actually looks like (cluster counts):")
    print(pd.crosstab(centroids.Cluster_Label, centroids.nearest_crop).to_string())

    suspect = centroids[(centroids.Cluster_Label == "cotton") &
                        (centroids.nearest_crop != "cotton")].sort_values("margin", ascending=False)
    print(f"\ncotton-labelled clusters whose nearest surveyed crop is NOT cotton: "
          f"{len(suspect)} of {(centroids.Cluster_Label == 'cotton').sum()}")
    print("highest-confidence cases first (margin = how far ahead the winner was):")
    print(suspect[["AOI", "Cluster_ID", "nearest_crop", "distance", "margin"]]
          .head(25).to_string(index=False))

    missed = centroids[(centroids.Cluster_Label == "non-cotton") &
                       (centroids.nearest_crop == "cotton")].sort_values("margin", ascending=False)
    print(f"\nnon-cotton clusters that look like surveyed cotton: {len(missed)}")
    print(missed[["AOI", "Cluster_ID", "distance", "margin"]].head(15).to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    centroids[["AOI", "Cluster_ID", "Cluster_Label", "nearest_crop", "distance", "margin"]] \
        .to_csv(OUT / "cluster_vs_reference.csv", index=False)
    print(f"\nwritten: {OUT / 'cluster_vs_reference.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
