"""Audit the training labels without needing the cluster rasters, which are gone.

Three things the parquet can be asked on its own:

  1. Phenology. Cotton in Pakistan greens up through July and peaks August-September.
     A cluster labelled cotton whose curve peaks in May, or in December, is either a
     different crop or a mislabel. Same in reverse for non-cotton clusters that peak
     exactly where cotton does.
  2. Leave-one-AOI-out agreement. Train on 22 AOIs, predict the 23rd. An AOI whose own
     labels the other 22 cannot reproduce is the AOI to look at first: either it was
     labelled to a different standard, or it holds a crop the others do not.
  3. Contradiction. Curves that are near-identical but carry opposite labels. Those
     pairs cannot both be right.

Nothing here decides that a label IS wrong. It ranks what to re-check by eye.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path("/home/jovyan/FAO/cotton")
TRAINING = BASE / "training_data_parquet/cotton_2025_training_data_v2.csv"
OUT = BASE / "validation_runs/label_audit"
META = ["AOI", "Cluster_ID", "Cluster_Label", "Feature_Type", "Pixel_ID", "Confidence_Score"]
# Cotton's own peak window. Anything labelled cotton peaking outside it is worth an eye.
COTTON_PEAK = ("2025-07-01", "2025-10-15")


def load():
    df = pd.read_csv(TRAINING)
    dates = [c for c in df.columns if c not in META]
    return df, dates


def phenology(df, dates):
    stamps = pd.to_datetime(dates)
    curves = df[dates].values
    peak_at = stamps[np.nanargmax(curves, axis=1)]
    df = df.assign(peak_date=peak_at, peak_ndvi=np.nanmax(curves, axis=1),
                   ndvi_range=np.nanmax(curves, axis=1) - np.nanmin(curves, axis=1))

    inside = (df.peak_date >= pd.Timestamp(COTTON_PEAK[0])) & (df.peak_date <= pd.Timestamp(COTTON_PEAK[1]))
    df = df.assign(peaks_in_cotton_window=inside)

    print("=== 1. Phenology ===")
    print("peak month by label (% of rows):")
    table = pd.crosstab(df.Cluster_Label, df.peak_date.dt.strftime("%b"), normalize="index")
    order = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print((table.reindex(columns=[m for m in order if m in table.columns]) * 100).round(1).to_string())

    odd = df[(df.Cluster_Label == "cotton") & ~df.peaks_in_cotton_window]
    print(f"\ncotton rows peaking outside {COTTON_PEAK[0]}..{COTTON_PEAK[1]}: "
          f"{len(odd):,} of {(df.Cluster_Label == 'cotton').sum():,} "
          f"({100*len(odd)/max(1,(df.Cluster_Label=='cotton').sum()):.1f}%)")
    if len(odd):
        worst = (odd.groupby(["AOI", "Cluster_ID"]).size()
                    .sort_values(ascending=False).head(12))
        print("worst cotton clusters (rows peaking outside the window):")
        print(worst.to_string())

    flat = df[(df.Cluster_Label == "cotton") & (df.ndvi_range < 0.15)]
    print(f"\ncotton rows with almost no seasonal swing (range < 0.15): {len(flat):,}")
    if len(flat):
        print(flat.groupby(["AOI", "Cluster_ID"]).size().sort_values(ascending=False).head(8).to_string())
    return df


def leave_one_aoi_out(df, dates):
    print("\n=== 2. Leave-one-AOI-out agreement ===")
    X, y, groups = df[dates].values, (df.Cluster_Label == "cotton").astype(int).values, df.AOI.values
    rows = []
    for aoi in sorted(df.AOI.unique()):
        held = groups == aoi
        if held.sum() == 0 or len(np.unique(y[~held])) < 2:
            continue
        model = RandomForestClassifier(n_estimators=120, max_depth=30, random_state=42, n_jobs=1)
        model.fit(X[~held], y[~held])
        pred = model.predict(X[held])
        truth = y[held]
        rows.append({
            "AOI": aoi,
            "rows": int(held.sum()),
            "cotton_rows": int(truth.sum()),
            "agreement_pct": round(100 * (pred == truth).mean(), 1),
            "cotton_called_non": int(((truth == 1) & (pred == 0)).sum()),
            "non_called_cotton": int(((truth == 0) & (pred == 1)).sum()),
        })
    table = pd.DataFrame(rows).sort_values("agreement_pct")
    print(table.to_string(index=False))
    return table


def contradictions(df, dates, radius=0.04):
    """Near-identical curves carrying opposite labels."""
    print("\n=== 3. Contradictory clusters ===")
    centroids = df[df.Feature_Type == "Centroid"]
    if centroids.empty:
        print("no centroid rows")
        return pd.DataFrame()
    curves = centroids[dates].values
    labels = (centroids.Cluster_Label == "cotton").values
    # Mean absolute distance between every pair of cluster centroids: 1,150 x 1,150 is
    # small enough to do outright.
    distance = np.abs(curves[:, None, :] - curves[None, :, :]).mean(axis=2)
    np.fill_diagonal(distance, np.inf)
    opposite = labels[:, None] != labels[None, :]
    distance = np.where(opposite, distance, np.inf)

    nearest = distance.argmin(axis=1)
    best = distance.min(axis=1)
    hits = np.where(best < radius)[0]
    print(f"cluster centroids within {radius} mean-NDVI of an opposite-labelled cluster: "
          f"{len(hits)} of {len(centroids)}")
    if len(hits) == 0:
        return pd.DataFrame()
    left = centroids.iloc[hits]
    right = centroids.iloc[nearest[hits]]
    pairs = pd.DataFrame({
        "aoi_a": left.AOI.values, "cluster_a": left.Cluster_ID.values,
        "label_a": left.Cluster_Label.values,
        "aoi_b": right.AOI.values, "cluster_b": right.Cluster_ID.values,
        "label_b": right.Cluster_Label.values,
        "mean_ndvi_gap": best[hits].round(4),
    }).sort_values("mean_ndvi_gap")
    print(pairs.head(20).to_string(index=False))
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, dates = load()
    df = phenology(df, dates)
    agreement = leave_one_aoi_out(df, dates)
    pairs = contradictions(df, dates)

    df[["AOI", "Cluster_ID", "Cluster_Label", "Feature_Type", "peak_date",
        "peak_ndvi", "ndvi_range", "peaks_in_cotton_window"]].to_csv(OUT / "row_phenology.csv", index=False)
    agreement.to_csv(OUT / "leave_one_aoi_out.csv", index=False)
    if len(pairs):
        pairs.to_csv(OUT / "contradictory_clusters.csv", index=False)
    print(f"\nwritten under {OUT}")


if __name__ == "__main__":
    main()
