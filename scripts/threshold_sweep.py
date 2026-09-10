"""Show the recall / commission trade-off across cotton probability cuts, per AOI.

v2 lifted Faran-1's recall from 57% to 79% and pushed its rice commission from 10% to
22% doing it. Which of those two matters more is not a question the data can answer --
it depends on what the acreage is used for -- so this does not pick a cut. It lays out
what each cut costs and buys, so the person who owns that decision can see the whole
curve rather than the one point a default threshold happens to land on.

Read as: at cut c, a pixel is cotton when the model's cotton probability is at least c.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path("/home/jovyan/FAO/cotton")
CURVES = BASE / "validation_data/reference_curves"
MODEL = BASE / "timeseries_model/model_v2/cotton_rf_v2.joblib"
CUTS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--curves", type=Path, default=CURVES)
    parser.add_argument("--csv", type=Path, default=BASE / "validation_runs/threshold_sweep.csv")
    args = parser.parse_args()

    tables = {p.stem: pd.read_parquet(p) for p in sorted(args.curves.glob("*.parquet"))}
    if not tables:
        print(f"no reference curves under {args.curves}")
        return 1

    model = joblib.load(args.model)
    model.n_jobs = 2
    cotton_at = list(model.classes_).index(1)
    dates = [str(d.date()) for d in pd.date_range("2025-04-01", "2025-12-29", freq="8D")]

    rows = []
    for aoi, table in tables.items():
        probability = model.predict_proba(table[dates].values)[:, cotton_at]
        for cut in CUTS:
            called_cotton = probability >= cut
            entry = {"aoi": aoi, "cut": cut}
            for crop, index in table.groupby("crop").groups.items():
                at = table.index.get_indexer(index)
                entry[crop] = round(100 * called_cotton[at].mean(), 1)
            rows.append(entry)

    frame = pd.DataFrame(rows)
    order = ["cotton", "rice", "sugarcane", "fall_maize", "orchard"]
    columns = ["aoi", "cut"] + [c for c in order if c in frame.columns]
    frame = frame[columns]

    for aoi, part in frame.groupby("aoi"):
        print(f"\n{aoi}   (cotton = recall, the rest = commission)")
        print(part.drop(columns="aoi").to_string(index=False))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv, index=False)
    print(f"\nwritten: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
