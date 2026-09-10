"""Fill each class's training quota with the rows the model gets wrong, not random ones.

v2 caps every class at 45,000 rows because sugarcane outnumbered cotton eight to one.
Drawing those 45,000 at random takes mostly easy rows -- healthy, unambiguous cane -- and
leaves the boundary thinly represented. The validation shows exactly that: the cane v2
calls cotton peaks at 0.67 where healthy cane peaks at 0.74, sitting between the two
classes, and there are 3,976 such pixels v1 got right and v2 does not.

So the quota is spent deliberately: a share of it on rows the current model already
misclassifies, the rest drawn at random so the easy majority does not disappear
altogether. Standard hard-negative mining, and the mistakes it targets are measured on
the training set rather than guessed at.

The output keeps the labelled_pixels schema, so it drops straight into train_model_v2.py.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path("/home/jovyan/FAO/cotton")
LABELLED = BASE / "training_v2/labelled_pixels.parquet"
MODEL = BASE / "timeseries_model/model_v2/cotton_rf_v2.joblib"
OUT = BASE / "training_v2/labelled_pixels_mined.parquet"
DATES = [str(d.date()) for d in pd.date_range("2025-04-01", "2025-12-29", freq="8D")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--labelled", type=Path, default=LABELLED)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--per-class", type=int, default=45000)
    parser.add_argument("--hard-share", type=float, default=0.5,
                        help="fraction of each class's quota reserved for rows the model "
                             "currently gets wrong")
    parser.add_argument("--batch", type=int, default=100000)
    args = parser.parse_args()

    table = pd.read_parquet(args.labelled)
    model = joblib.load(args.model)
    model.n_jobs = 2

    # Predicted in batches: a million rows through a 250-tree forest at once is a needless
    # memory spike on a box with two cores.
    predictions = np.empty(len(table), dtype=np.int64)
    values = table[DATES].values
    for start in range(0, len(table), args.batch):
        stop = min(start + args.batch, len(table))
        predictions[start:stop] = model.predict(values[start:stop])
    table = table.assign(predicted=predictions)
    wrong = table["predicted"] != table["label"]

    print("how often each class is already got wrong, in the training set itself:")
    summary = table.groupby("crop").apply(
        lambda g: pd.Series({"rows": len(g), "wrong": int((g.predicted != g.label).sum()),
                             "wrong_pct": round(100 * (g.predicted != g.label).mean(), 1)}),
        include_groups=False)
    print(summary.to_string())

    hard_quota = int(args.per_class * args.hard_share)
    parts = []
    for crop, group in table.groupby("crop"):
        hard = group[wrong.loc[group.index]]
        easy = group[~wrong.loc[group.index]]
        take_hard = min(len(hard), hard_quota)
        # Whatever the hard pool cannot fill goes back to the random draw, so a class the
        # model already handles well is not silently short of rows.
        take_easy = min(len(easy), args.per_class - take_hard)
        picked = pd.concat([
            hard.sample(take_hard, random_state=0) if take_hard else hard.iloc[:0],
            easy.sample(take_easy, random_state=0) if take_easy else easy.iloc[:0],
        ])
        print(f"  {crop:11s} {take_hard:6,} hard + {take_easy:6,} easy = {len(picked):6,}")
        parts.append(picked)

    mined = pd.concat(parts, ignore_index=True).drop(columns="predicted")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mined.to_parquet(args.out)
    print(f"\nwritten {len(mined):,} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
