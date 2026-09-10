"""Compare models at the same recall, so the answer is not just "one is shyer".

Hard-negative mining lowered every commission number and lowered recall with them. That
is what any more conservative model does, and moving the probability cut on the old model
does it too. The only question worth asking is whether the trade-off *curve* moved: at
the recall the new model achieves, does the old model make more false positives or fewer?

So each model is swept across cotton probability cuts, the cut that lands nearest the
target recall is picked, and the commissions are read off there. Same recall, same
evidence, and the comparison is about the model rather than its shyness.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path("/home/jovyan/FAO/cotton")
CURVES = BASE / "validation_data/reference_curves"
DATES = [str(d.date()) for d in pd.date_range("2025-04-01", "2025-12-29", freq="8D")]
CROPS = ["rice", "sugarcane", "fall_maize", "orchard"]


def sweep(model_path: Path, table: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(model_path)
    model.n_jobs = 2
    cotton_at = list(model.classes_).index(1)
    probability = model.predict_proba(table[DATES].values)[:, cotton_at]

    rows = []
    for cut in np.round(np.arange(0.05, 0.96, 0.025), 3):
        called = probability >= cut
        entry = {"cut": cut,
                 "recall": 100 * called[(table.crop == "cotton").values].mean()}
        for crop in CROPS:
            at = (table.crop == crop).values
            if at.any():
                entry[crop] = 100 * called[at].mean()
        rows.append(entry)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=path pairs, e.g. v2=/path/a.joblib mined=/path/b.joblib")
    parser.add_argument("--recalls", nargs="*", type=float, default=[75, 78, 80, 83, 85])
    parser.add_argument("--curves", type=Path, default=CURVES,
                        help="directory of sampled per-crop curve parquets")
    parser.add_argument("--csv", type=Path, default=BASE / "validation_runs/matched_recall.csv")
    args = parser.parse_args()

    tables = [pd.read_parquet(p) for p in sorted(args.curves.glob("*.parquet"))]
    if not tables:
        print(f"no curve parquets under {args.curves}")
        return 1
    pooled = pd.concat(tables, ignore_index=True)

    curves = {}
    for pair in args.models:
        name, _, path = pair.partition("=")
        curves[name] = sweep(Path(path), pooled)

    rows = []
    for target in args.recalls:
        for name, curve in curves.items():
            hit = curve.iloc[(curve["recall"] - target).abs().argmin()]
            entry = {"target_recall": target, "model": name,
                     "cut": hit["cut"], "recall": round(hit["recall"], 1)}
            for crop in CROPS:
                if crop in curve.columns:
                    entry[crop] = round(hit[crop], 1)
            entry["mean_commission"] = round(
                np.mean([hit[c] for c in CROPS if c in curve.columns]), 2)
            rows.append(entry)

    frame = pd.DataFrame(rows)
    for target, part in frame.groupby("target_recall"):
        print(f"\nat about {target:.0f}% cotton recall:")
        print(part.drop(columns="target_recall").to_string(index=False))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv, index=False)
    print(f"\nwritten: {args.csv}")
    print("\nlower mean_commission at the same recall = the better model, not the shyer one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
