"""Score model v1 and model v2 on the same evidence, so the comparison means something.

The evidence is the reference curves: smoothed NDVI sampled inside polygons somebody
actually surveyed, in the validation AOIs. Same 35 dates, same smoothing the inference
worker applies, so both models see identical input and any difference is the model.

What comes out, per reference class:

  * cotton   -- percent predicted cotton is recall. The survey is positives only, so this
    is the most it can tell us.
  * everything else -- percent predicted cotton is commission. Those pixels are cotton in
    nobody's account, so anything called cotton there is a false positive with a crop name
    attached, which is the whole reason v2 is multiclass.

Comparing on a model's own held-out split would be unfair in both directions: the two
models were split differently and v1's split was random rows. This is the only shared
ground.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model_v2 import EXPECTED_DATES, check_feature_columns  # noqa: E402

BASE = Path("/home/jovyan/FAO/cotton")
CURVES = BASE / "validation_data/reference_curves"
V1 = BASE / "timeseries_model/model_v1/best_rf_classifier.joblib"
V2 = BASE / "timeseries_model/model_v2/cotton_rf_v2.joblib"
COTTON = 1
CLASS_ORDER = ["cotton", "rice", "sugarcane", "fall_maize", "orchard"]


def load_curves(curve_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(curve_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        check_feature_columns(df.columns, str(path))
        if "crop" not in df.columns:
            print(f"skipping {path.name}: no `crop` column")
            continue
        if "aoi" not in df.columns:
            df = df.assign(aoi=path.stem)
        frames.append(df[EXPECTED_DATES + ["crop", "aoi"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def percent_cotton(model, X: np.ndarray) -> np.ndarray:
    """True where the model says cotton. v1 is binary [1, 4], v2 multiclass; class 1 is
    cotton in both, so the same test works for either."""
    return model.predict(X) == COTTON


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves", type=Path, default=CURVES)
    parser.add_argument("--v1", type=Path, default=V1)
    parser.add_argument("--v2", type=Path, default=V2)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table here")
    args = parser.parse_args()

    missing = []
    if not args.curves.is_dir() or not list(args.curves.glob("*.parquet")):
        missing.append(f"reference curves under {args.curves} "
                       f"(build them with sample_reference_curves.py)")
    if not args.v1.exists():
        missing.append(f"model v1 at {args.v1}")
    if not args.v2.exists():
        missing.append(f"model v2 at {args.v2} (train it with train_model_v2.py)")
    if missing:
        print("nothing to compare yet, still missing:")
        for item in missing:
            print(f"  - {item}")
        return 0

    curves = load_curves(args.curves)
    if curves.empty:
        print(f"no usable curve parquets under {args.curves}")
        return 0

    models = {"v1": joblib.load(args.v1), "v2": joblib.load(args.v2)}
    for name, model in models.items():
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and n_in != len(EXPECTED_DATES):
            raise SystemExit(f"{name} expects {n_in} features, the curves carry "
                             f"{len(EXPECTED_DATES)}; these are not the same feature space")

    X = curves[EXPECTED_DATES].to_numpy(dtype="float32")
    for name, model in models.items():
        curves[f"cotton_{name}"] = percent_cotton(model, X)

    def table(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for crop in [c for c in CLASS_ORDER if c in set(frame["crop"])]:
            part = frame[frame["crop"] == crop]
            v1 = 100 * part["cotton_v1"].mean()
            v2 = 100 * part["cotton_v2"].mean()
            rows.append({"crop": crop, "pixels": len(part),
                         "meaning": "recall" if crop == "cotton" else "commission",
                         "v1_pct": round(v1, 1), "v2_pct": round(v2, 1),
                         "delta": round(v2 - v1, 1)})
        return pd.DataFrame(rows)

    out = []
    for aoi in sorted(curves["aoi"].unique()):
        part = table(curves[curves["aoi"] == aoi]).assign(aoi=aoi)
        out.append(part)
        print(f"\n{aoi}")
        print(part.drop(columns="aoi").to_string(index=False))

    pooled = table(curves).assign(aoi="POOLED")
    out.append(pooled)
    print("\npooled over all AOIs")
    print(pooled.drop(columns="aoi").to_string(index=False))
    print("\ncotton delta positive = v2 finds more real cotton; "
          "other-crop delta negative = v2 makes fewer false positives there")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(out, ignore_index=True).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
