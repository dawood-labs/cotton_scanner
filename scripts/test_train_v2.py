"""End-to-end check on train_model_v2.py, run before trusting a real training run.

The point is not the accuracy on made-up curves, it is that the machinery holds: the
script runs to completion, writes the four artefacts, the model card records the 35 dates
in the order the model was actually fed, and a feature-column mismatch is refused rather
than absorbed. That last one is the check the whole rewrite exists for, so it needs a test
that fails loudly if someone ever softens it into a warning.

Runs in well under a minute on synthetic data, single core, so it can be run while the
real jobs are using the box.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parent / "train_model_v2.py"
DATES = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-04-01", periods=35, freq="8D")]
# Peak composite index per class, roughly where each crop greens up, so the classes are
# separable enough for the run to produce a real confusion matrix.
PEAKS = {1: 22, 2: 14, 3: 12, 5: 27, 6: 8}
CROPS = {1: "cotton", 2: "rice", 3: "sugarcane", 5: "fall_maize", 6: "orchard"}
TILES = ["t_alpha", "t_beta", "t_gamma", "t_delta"]


def synthetic(rows_per_class_per_tile: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.arange(len(DATES))
    records = []
    for tile in TILES:
        for label, peak in PEAKS.items():
            width = 5.0 + rng.normal(0, 0.3)
            base = 0.9 * np.exp(-0.5 * ((steps - peak) / width) ** 2) - 0.05
            for _ in range(rows_per_class_per_tile):
                curve = base + rng.normal(0, 0.03, len(DATES)) + rng.normal(0, 0.02)
                record = dict(zip(DATES, np.clip(curve, -1, 1)))
                record.update(label=label, crop=CROPS[label], tile=tile,
                              lon=71.0 + rng.random(), lat=30.0 + rng.random())
                records.append(record)
    return pd.DataFrame(records)


def run_training(parquet: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--labelled", str(parquet), "--out-dir", str(out_dir),
         "--n-trials", "2", "--cv-folds", "2", "--n-jobs", "1", "--no-old-negatives"],
        capture_output=True, text=True)


def check_good_run(work: Path) -> None:
    parquet = work / "labelled_pixels.parquet"
    synthetic().to_parquet(parquet, index=False)
    out_dir = work / "model_v2"
    result = run_training(parquet, out_dir)
    assert result.returncode == 0, f"training failed:\n{result.stdout}\n{result.stderr}"

    model_path = out_dir / "cotton_rf_v2.joblib"
    assert model_path.exists(), "model joblib was not written"
    for artefact in ("metrics.json", "model_card.json", "confusion_matrix.png",
                     "feature_importance.png", "training.log"):
        assert (out_dir / artefact).exists(), f"{artefact} was not written"

    card = json.loads((out_dir / "model_card.json").read_text())
    assert card["feature_dates"] == DATES, "model card dates are not the 35 expected, in order"
    assert card["n_features"] == 35
    assert card["class_codes"]["1"] == "cotton", "cotton must stay class 1"
    assert set(card["training_tiles"]) and not (
        set(card["training_tiles"]) & set(card["held_out_tiles"])), \
        "a tile appears in both train and held-out; the split is not group-aware"
    assert card["sklearn_version"]

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["per_class"], "metrics carry no per-class breakdown"
    for name, scores in metrics["per_class"].items():
        for field in ("precision", "recall", "f1-score", "support"):
            assert field in scores, f"per-class {name} has no {field}"
    assert len(metrics["confusion_matrix"]["counts"]) == len(metrics["confusion_matrix"]["labels"])
    print(f"  training run ok, macro F1 {metrics['macro_f1']:.3f}, "
          f"held out {card['held_out_tiles']}")


def check_column_mismatch(work: Path) -> None:
    df = synthetic(rows_per_class_per_tile=6)

    reordered = df[[DATES[1], DATES[0]] + DATES[2:] + ["label", "crop", "tile", "lon", "lat"]]
    path = work / "reordered.parquet"
    reordered.to_parquet(path, index=False)
    result = run_training(path, work / "model_reordered")
    assert result.returncode != 0, "a reordered date column was accepted"
    assert "out of order" in (result.stdout + result.stderr), \
        f"the rejection did not say what was wrong:\n{result.stderr}"

    dropped = df.drop(columns=[DATES[17]])
    path = work / "dropped.parquet"
    dropped.to_parquet(path, index=False)
    result = run_training(path, work / "model_dropped")
    assert result.returncode != 0, "a missing date column was accepted"
    assert "absent" in (result.stdout + result.stderr), \
        f"the rejection did not say what was wrong:\n{result.stderr}"
    print("  reordered and missing feature columns both refused")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="train_v2_test_") as tmp:
        work = Path(tmp)
        print("checking a clean end-to-end run")
        check_good_run(work)
        print("checking the feature-column guard")
        check_column_mismatch(work)
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
