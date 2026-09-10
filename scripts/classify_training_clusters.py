"""Ask a classifier trained on surveyed crops what each training cluster really is.

`label_vs_reference.py` matches a cluster to the nearest median crop profile, and that
turned out to be too blunt: cotton and rice medians sit close together, so the pair we
most need separated is the pair the metric cannot separate. Worse, the orchard profile
is flat-ish and behaves as an attractor for anything unremarkable.

A classifier does not have that problem. It is fitted on the surveyed pixels themselves
-- cotton from the mill survey, rice / sugarcane / fall maize from the national scans,
orchards from the exclusion mask -- so it learns where the boundary between rice and
cotton actually runs, not merely which median is nearer. Applying it to the 1,150
training cluster centroids says what each cluster looks like to evidence that was
collected on the ground.

This is a second opinion on the labels, not a verdict. A cluster it calls rice is a
cluster to look at.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

BASE = Path("/home/jovyan/FAO/cotton")
TRAINING = BASE / "training_data_parquet/cotton_2025_training_data_v2.csv"
CURVES = BASE / "validation_data/reference_curves"
OUT = BASE / "validation_runs/label_audit"
META = ["AOI", "Cluster_ID", "Cluster_Label", "Feature_Type", "Pixel_ID", "Confidence_Score"]
PER_CLASS = 12000     # keeps the fit balanced and finishes on two cores


def main():
    df = pd.read_csv(TRAINING)
    dates = [c for c in df.columns if c not in META]

    tables = [pd.read_parquet(p) for p in sorted(CURVES.glob("*.parquet"))]
    if not tables:
        print(f"no surveyed curves under {CURVES}")
        return 1
    pooled = pd.concat(tables, ignore_index=True)
    pooled = pooled.groupby("crop", group_keys=False).apply(
        lambda g: g.sample(min(len(g), PER_CLASS), random_state=0), include_groups=False)
    pooled["crop"] = pd.concat(tables, ignore_index=True).groupby("crop", group_keys=False).apply(
        lambda g: g.sample(min(len(g), PER_CLASS), random_state=0))["crop"].values

    X = pooled[dates].values
    y = pooled["crop"].values
    print("fitted on surveyed pixels:")
    print(pd.Series(y).value_counts().to_string())

    model = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_leaf=2,
                                   class_weight="balanced_subsample", random_state=42, n_jobs=2)
    # How well it can tell these crops apart at all. If cotton and rice are not separable
    # in this signal, no amount of relabelling helps and the answer is a different sensor.
    scores = cross_val_score(model, X, y, cv=3, scoring="f1_macro", n_jobs=1)
    print(f"\n3-fold macro F1 on the surveyed pixels: {scores.mean():.3f} "
          f"(fold spread {scores.min():.3f}-{scores.max():.3f})")
    model.fit(X, y)

    centroids = df[df.Feature_Type == "Centroid"].copy()
    probabilities = model.predict_proba(centroids[dates].values)
    classes = list(model.classes_)
    centroids["looks_like"] = [classes[i] for i in probabilities.argmax(axis=1)]
    centroids["confidence"] = probabilities.max(axis=1).round(3)
    centroids["p_cotton"] = probabilities[:, classes.index("cotton")].round(3)

    print("\nwhat the surveyed-crop classifier calls each training cluster:")
    print(pd.crosstab(centroids.Cluster_Label, centroids.looks_like).to_string())

    for label, other in [("cotton", True), ("non-cotton", False)]:
        subset = centroids[centroids.Cluster_Label == label]
        wrong = subset[(subset.looks_like != "cotton") if other else (subset.looks_like == "cotton")]
        confident = wrong[wrong.confidence >= 0.6]
        print(f"\n{label} clusters the classifier disagrees with: {len(wrong)} of {len(subset)}"
              f", of which {len(confident)} at confidence >= 0.6")
        if len(confident):
            print(confident.groupby("looks_like").size().sort_values(ascending=False).to_string())
            print(confident.sort_values("confidence", ascending=False)
                  [["AOI", "Cluster_ID", "looks_like", "confidence", "p_cotton"]]
                  .head(15).to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cluster_classified.csv"
    centroids[["AOI", "Cluster_ID", "Cluster_Label", "looks_like", "confidence", "p_cotton"]] \
        .to_csv(path, index=False)
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
