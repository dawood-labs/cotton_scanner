"""Rebuild the "other" class out of evidence instead of leftovers.

"other" is the old training set's non-cotton rows. Its job at inference is to catch
generic land -- fallow, wheat stubble, settlement, water -- that no crop scan covers and
that would otherwise have to be called some crop. But those rows were labelled by eye,
and the audit says the label is dirty: on held-out data the model puts 1,315 of 5,157
"other" rows into rice, which means real rice is sitting in there.

The cleaner is a classifier fitted only on surveyed polygons, so it owes nothing to the
old labels. Where it is confident an "other" row is one of the five surveyed crops, that
row moves to that class. Where it is not confident, the row stays "other" -- which is
the right home for genuinely ambiguous ground.

The confidence cut is a judgement, so it is a flag with a stated default, not a constant
buried in the code.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path("/home/jovyan/FAO/cotton")
LABELLED = BASE / "training_v2/labelled_pixels.parquet"
OLD_CSV = BASE / "training_data_parquet/cotton_2025_training_data_v2.csv"
OUT = BASE / "training_v2/other_class.parquet"
DATES = [str(d.date()) for d in pd.date_range("2025-04-01", "2025-12-29", freq="8D")]
CODES = {"cotton": 1, "rice": 2, "sugarcane": 3, "fall_maize": 5, "orchard": 6}
OLD_META = ["AOI", "Cluster_ID", "Cluster_Label", "Feature_Type", "Pixel_ID", "Confidence_Score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="how sure the surveyed-crop classifier must be before a row "
                             "is moved out of 'other'")
    parser.add_argument("--per-class", type=int, default=30000)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    surveyed = pd.read_parquet(LABELLED)
    surveyed = (surveyed.groupby("crop", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.per_class), random_state=0),
                       include_groups=False)
                .reset_index(drop=True))
    surveyed["crop"] = (pd.read_parquet(LABELLED).groupby("crop", group_keys=False)
                        .apply(lambda g: g.sample(min(len(g), args.per_class), random_state=0))
                        ["crop"].values)

    cleaner = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_leaf=2,
                                     class_weight="balanced_subsample", random_state=42, n_jobs=2)
    cleaner.fit(surveyed[DATES].values, surveyed["crop"].values)

    old = pd.read_csv(OLD_CSV)
    old = old[old["Cluster_Label"] == "non-cotton"].reset_index(drop=True)
    probabilities = cleaner.predict_proba(old[DATES].values)
    classes = list(cleaner.classes_)
    verdict = np.array(classes)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)

    moved = confidence >= args.confidence
    print(f"old non-cotton rows: {len(old):,}")
    print(f"confidently one of the surveyed crops (>= {args.confidence}): {int(moved.sum()):,}")
    print(pd.Series(verdict[moved]).value_counts().to_string())
    print(f"staying as 'other': {int((~moved).sum()):,}")

    frame = old.loc[~moved, DATES].copy()
    frame["label"] = 4
    frame["crop"] = "other"
    frame["tile"] = "old_" + old.loc[~moved, "AOI"].astype(str)
    frame["lon"] = np.nan
    frame["lat"] = np.nan

    # The rows that moved are not discarded: they join the class the surveyed evidence
    # says they are, which is extra data for exactly the crops we are short of.
    reassigned = old.loc[moved, DATES].copy()
    reassigned["crop"] = verdict[moved]
    reassigned["label"] = reassigned["crop"].map(CODES)
    reassigned["tile"] = "old_" + old.loc[moved, "AOI"].astype(str)
    reassigned["lon"] = np.nan
    reassigned["lat"] = np.nan

    table = pd.concat([frame, reassigned], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out)
    print(f"\nwritten {len(table):,} rows -> {args.out}")
    print(table.groupby("crop").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
