"""Train the cotton 2025 NDVI time-series classifier, v2.

Three things v1 got wrong, and this script exists to stop each of them happening again:

  1. Feature order was never checked. v1 read its 35 NDVI dates positionally, so when the
     inference window slipped one 8-day step the model still ran, still produced a map, and
     over-called cotton area by 63%. Here the date columns are validated against the
     expected list, in order, and a mismatch stops the run instead of being absorbed.
  2. It was binary, cotton vs "non-cotton". A false positive told us nothing about what was
     actually there. v2 is multiclass on purpose: rice, sugarcane, fall maize and orchard
     each keep their own code, so the confusion matrix names the crop we are losing to and
     the next fix has somewhere to aim.
  3. It was scored on a random row split. Pixels in one field are near-copies of each
     other, so a random split puts the same field on both sides and reports an accuracy the
     model does not have in a district it has never seen. v2 holds out whole tiles.

Class weighting is balanced_subsample because the classes are very unbalanced, and the
headline number is macro F1, not accuracy: with this class mix accuracy is mostly a
measure of how common the majority class is.

Alongside the model it writes a model_card.json — the 35 dates in order, the class codes,
the tiles trained on, rows per class, sklearn version. That card is what the inference
pipeline checks its input against, which is the durable version of fix (1).
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path("/home/jovyan/FAO/cotton")
LABELLED = BASE / "training_v2/labelled_pixels.parquet"
OLD_CSV = BASE / "training_data_parquet/cotton_2025_training_data_v2.csv"
REFERENCE_CURVES = BASE / "validation_data/reference_curves"
OUT_DIR = BASE / "timeseries_model/model_v2"

# The training window, fixed. 35 composites, 8-day step, 2025-04-01 .. 2025-12-29.
EXPECTED_DATES = [d.strftime("%Y-%m-%d")
                  for d in pd.date_range("2025-04-01", periods=35, freq="8D")]

# Codes are frozen: cotton is always 1, so downstream rasters and thresholds keep meaning.
# 4 is "other" (generic land) and stays 4 because that is what v1 used for non-cotton.
CLASS_NAMES = {1: "cotton", 2: "rice", 3: "sugarcane", 4: "other", 5: "fall_maize", 6: "orchard"}
OLD_META = ["AOI", "Cluster_ID", "Cluster_Label", "Feature_Type", "Pixel_ID", "Confidence_Score"]

log = logging.getLogger("train_v2")


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(out_dir / "training.log", mode="w")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def check_feature_columns(columns, source: str) -> None:
    """Reject anything but the 35 expected dates, in order.

    Not a warning. A silently reordered or missing feature column is exactly the bug class
    that cost this project a 63% area error, and it is invisible in every downstream number.
    """
    present = [c for c in columns if c in set(EXPECTED_DATES)]
    missing = [d for d in EXPECTED_DATES if d not in set(columns)]
    if missing:
        raise SystemExit(
            f"feature column check failed for {source}: {len(missing)} of the 35 expected "
            f"date columns are absent, first few {missing[:5]}. Expected "
            f"{EXPECTED_DATES[0]} .. {EXPECTED_DATES[-1]} on an 8-day step."
        )
    if present != EXPECTED_DATES:
        first_bad = next(i for i, (a, b) in enumerate(zip(present, EXPECTED_DATES)) if a != b)
        raise SystemExit(
            f"feature column check failed for {source}: date columns are out of order at "
            f"position {first_bad} (found {present[first_bad]}, expected "
            f"{EXPECTED_DATES[first_bad]}). The model reads features positionally, so a "
            f"reordered column silently shifts every date."
        )


def load_labelled(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"labelled pixels not found: {path}")
    df = pd.read_parquet(path)
    check_feature_columns(df.columns, str(path))
    for column in ("label", "tile"):
        if column not in df.columns:
            raise SystemExit(f"{path} has no `{column}` column; it is required")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["tile"] = df["tile"].astype(str)
    unknown = sorted(set(df["label"]) - set(CLASS_NAMES))
    if unknown:
        raise SystemExit(f"{path} carries label codes {unknown} which are not in {CLASS_NAMES}")
    return df


def drop_rows_that_look_like_cotton(frame: pd.DataFrame, curves_dir: Path) -> pd.DataFrame:
    """Remove old non-cotton rows that sit closest to the surveyed cotton profile.

    The old labels were assigned by eye, one k-means cluster at a time. The audit found
    roughly half the cotton-labelled clusters peaking in September, where rice peaks --
    so the non-cotton side is not above suspicion either. Feeding a row that is really
    cotton in as "other" teaches the model to reject the crop it is meant to find, which
    is the more expensive of the two mistakes here.

    Nearest surveyed profile by mean absolute difference over the season. Correlation
    would call rice and cotton the same thing whenever they merely rise and fall
    together, which is the pair that has to stay separated.
    """
    tables = [pd.read_parquet(p) for p in sorted(curves_dir.glob("*.parquet"))]
    if not tables:
        log.info("no surveyed curves under %s, keeping every old negative", curves_dir)
        return frame
    pooled = pd.concat(tables, ignore_index=True)
    if any(d not in pooled.columns for d in EXPECTED_DATES):
        log.warning("surveyed curves do not carry the 35 training dates, keeping every old negative")
        return frame

    profiles = pooled.groupby("crop")[EXPECTED_DATES].median()
    crops = list(profiles.index)
    if "cotton" not in crops:
        log.warning("surveyed curves carry no cotton profile, keeping every old negative")
        return frame

    distance = np.abs(frame[EXPECTED_DATES].values[:, None, :] - profiles.values[None, :, :]).mean(axis=2)
    looks_like_cotton = np.array(crops)[distance.argmin(axis=1)] == "cotton"
    log.info("dropping %d of %d old negatives that sit nearest the surveyed cotton profile",
             int(looks_like_cotton.sum()), len(frame))
    return frame[~looks_like_cotton].reset_index(drop=True)


def load_old_negatives(path: Path, cap: int | None, seed: int,
                       curves_dir: Path | None = None) -> pd.DataFrame:
    """The old training set's non-cotton rows, remapped to label 4.

    The polygon scans only cover crops somebody went out and surveyed. Fallow, wheat
    stubble, settlement and water appear in no scan, and a model that has never seen them
    has to place them somewhere — historically, in cotton. These rows are the only source
    of that generic land we have.
    """
    if not path.exists():
        log.info("old training csv not found at %s, continuing without generic negatives", path)
        return pd.DataFrame()
    old = pd.read_csv(path)
    check_feature_columns(old.columns, str(path))
    old = old[old["Cluster_Label"] == "non-cotton"]
    frame = old[EXPECTED_DATES].copy()
    frame["label"] = 4
    frame["crop"] = "other"
    # Prefixed so an old AOI can never collide with a new tile id: these rows are their own
    # groups, and holding one out must hold out the whole AOI.
    frame["tile"] = "old_" + old["AOI"].astype(str)
    frame["lon"] = np.nan
    frame["lat"] = np.nan
    frame = frame.reset_index(drop=True)

    # Filter before capping, so the cap is spent on rows that survived the check.
    if curves_dir is not None:
        frame = drop_rows_that_look_like_cotton(frame, curves_dir)
    if cap is not None and len(frame) > cap:
        frame = frame.sample(n=cap, random_state=seed).reset_index(drop=True)
    return frame


def split_by_tile(df: pd.DataFrame, test_size: float, seed: int):
    """Hold out whole tiles, keeping the class mix roughly intact on both sides.

    Whole tiles because neighbouring pixels in one field are near-copies: a random row
    split puts the same field on both sides and reports a score the model does not have in
    a district it has never seen.

    Stratified rather than a plain group shuffle because several groups are single-class --
    every old-set AOI is nothing but label 4 -- and an unstratified draw can hand back a
    held-out set containing one class, which scores a perfect macro F1 and means nothing.
    Fold size is therefore approximately, not exactly, test_size.
    """
    y = df["label"].to_numpy()
    groups = df["tile"].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise SystemExit(f"only {n_groups} distinct tile(s) in the training data; a "
                         f"tile-held-out split needs at least 2")
    smallest_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(int(round(1 / test_size)), n_groups, smallest_class))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, y, groups))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    absent = sorted(set(train["label"]) - set(test["label"]))
    if len(set(test["label"])) < 2:
        raise SystemExit(
            "the held-out tiles carry fewer than 2 classes, so no useful score can come "
            "out of them. Either the tiles are single-crop, or --test-size is too small "
            "for this class mix."
        )
    if absent:
        log.warning("held-out tiles contain no %s; those classes are unscored here",
                    ", ".join(CLASS_NAMES[c] for c in absent))
    return train, test


def cv_folds(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """As many folds as the data can actually carry."""
    n_groups = len(np.unique(groups))
    smallest_class = int(pd.Series(y).value_counts().min())
    return max(2, min(requested, n_groups, smallest_class))


# What v1's own search settled on. Used when the search is skipped, so a first model
# lands in minutes instead of an hour on two cores and tuning can come after there is
# something to tune against.
DEFAULT_PARAMS = {"n_estimators": 250, "max_depth": 30, "min_samples_leaf": 1,
                  "min_samples_split": 2, "max_features": "sqrt"}


class FixedStudy:
    """Stands in for an optuna study when --n-trials is 0, so the rest of the run and
    everything it writes stays identical."""

    def __init__(self, params):
        self.best_params = dict(params)
        self.best_value = float("nan")
        self.trials = []


def tune(X, y, groups, n_trials: int, n_jobs: int, seed: int, folds: int):
    if n_trials <= 0:
        log.info("search skipped, using %s", DEFAULT_PARAMS)
        return FixedStudy(DEFAULT_PARAMS)

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_index = list(splitter.split(X, y, groups))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        }
        scores = []
        for step, (fit_idx, val_idx) in enumerate(fold_index):
            model = RandomForestClassifier(
                class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs, **params)
            model.fit(X[fit_idx], y[fit_idx])
            pred = model.predict(X[val_idx])
            # Macro over the classes actually in this fold's validation part; a class the
            # fold does not contain should neither help nor punish the trial.
            scores.append(f1_score(y[val_idx], pred, average="macro",
                                   labels=np.unique(y[val_idx]), zero_division=0))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def plot_confusion(matrix: np.ndarray, labels: list[int], path: Path) -> None:
    # Row-normalised, because raw counts of a 30:1 class mix only show the majority class.
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = matrix / matrix.sum(axis=1, keepdims=True)
    shares = np.nan_to_num(shares)
    names = [CLASS_NAMES[c] for c in labels]
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 3, 1.2 * len(labels) + 2.5))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference")
    ax.set_title("v2 confusion on held-out tiles (row %, count)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{100 * shares[i, j]:.1f}%\n{matrix[i, j]:,}", ha="center",
                    va="center", fontsize=8,
                    color="white" if shares[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importances: np.ndarray, path: Path) -> None:
    """Importance against the calendar, not against a feature index.

    Plotted by date so the peaks can be read as phenology: if the model is leaning on
    dates outside cotton's own July-October rise, that is worth knowing.
    """
    stamps = pd.to_datetime(EXPECTED_DATES)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(stamps, importances, width=6, color="#4c72b0")
    ax.axvspan(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-15"),
               color="#dd8452", alpha=0.15, label="cotton peak window")
    ax.set_ylabel("gini importance")
    ax.set_title("v2 feature importance by composite date")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labelled", type=Path, default=LABELLED)
    parser.add_argument("--old-csv", type=Path, default=OLD_CSV,
                        help="old training set; its non-cotton rows become label 4")
    parser.add_argument("--no-old-negatives", action="store_true",
                        help="train on the polygon-scanned classes only")
    parser.add_argument("--extra-labelled", type=Path, default=None,
                        help="another table in the labelled_pixels schema to merge, e.g. "
                             "the rebuilt 'other' class. Replaces --old-csv when given.")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="cap rows per class before splitting. Sugarcane outnumbers "
                             "cotton eight to one, and the extra rows buy nothing but "
                             "hours of fitting on two cores.")
    parser.add_argument("--max-other", type=int, default=None,
                        help="cap on label-4 rows drawn from the old set")
    parser.add_argument("--reference-curves", type=Path, default=REFERENCE_CURVES,
                        help="surveyed per-crop curves; old negatives nearest the cotton "
                             "profile are dropped. Pass a missing path to skip the check.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="approximate share of rows to hold out, taken as whole tiles")
    parser.add_argument("--n-jobs", type=int, default=2, help="this box has 2 cores")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(args.out_dir)
    log.info("labelled pixels: %s", args.labelled)

    df = load_labelled(args.labelled)
    log.info("loaded %d rows over %d tiles", len(df), df["tile"].nunique())
    if args.extra_labelled:
        # Already in the labelled_pixels schema, already cleaned: merge it as it is rather
        # than putting it back through the old-CSV path, which would filter on a column
        # this table does not have.
        extra = pd.read_parquet(args.extra_labelled)
        check_feature_columns(extra.columns, str(args.extra_labelled))
        if args.max_other:
            extra = (extra.groupby("label", group_keys=False)
                     .apply(lambda g: g.sample(min(len(g), args.max_other), random_state=args.seed))
                     .reset_index(drop=True))
        log.info("merging %d rows from %s", len(extra), args.extra_labelled)
        df = pd.concat([df, extra], ignore_index=True)
    elif not args.no_old_negatives:
        curves = args.reference_curves if args.reference_curves.exists() else None
        other = load_old_negatives(args.old_csv, args.max_other, args.seed, curves)
        if len(other):
            log.info("merging %d generic non-crop rows from the old set as label 4", len(other))
            df = pd.concat([df, other], ignore_index=True)

    if args.max_per_class:
        # Capped per class. The group split happens after this, so a whole tile can still
        # land in the holdout; what the cap removes is redundancy, not independence.
        before = len(df)
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.max_per_class), random_state=args.seed))
                .reset_index(drop=True))
        log.info("capped at %d rows per class: %d rows -> %d", args.max_per_class, before, len(df))

    counts = df["label"].value_counts().sort_index()
    for code, count in counts.items():
        log.info("  class %d %-10s %8d rows", code, CLASS_NAMES[code], count)
    if len(counts) < 2:
        raise SystemExit("need at least 2 classes to train")

    train, test = split_by_tile(df, args.test_size, args.seed)
    train_tiles = sorted(train["tile"].unique())
    test_tiles = sorted(test["tile"].unique())
    log.info("train %d rows / %d tiles, held-out %d rows / %d tiles",
             len(train), len(train_tiles), len(test), len(test_tiles))
    log.info("held-out tiles: %s", ", ".join(test_tiles))

    # Fit on plain arrays: the inference worker hands the model a numpy block, and a model
    # that remembers column names warns on every call. The model card carries the names.
    X_train = train[EXPECTED_DATES].to_numpy(dtype="float32")
    y_train = train["label"].to_numpy()
    X_test = test[EXPECTED_DATES].to_numpy(dtype="float32")
    y_test = test["label"].to_numpy()
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise SystemExit("feature matrix holds NaN or inf; the smoother should have filled these")

    folds = cv_folds(y_train, train["tile"].values, args.cv_folds)
    log.info("optuna: %d trials, %d-fold grouped CV, optimising macro F1", args.n_trials, folds)
    study = tune(X_train, y_train, train["tile"].values, args.n_trials, args.n_jobs,
                 args.seed, folds)
    log.info("best cv macro F1 %.4f with %s", study.best_value, study.best_params)

    model = RandomForestClassifier(class_weight="balanced_subsample", random_state=args.seed,
                                   n_jobs=args.n_jobs, **study.best_params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    labels = sorted(set(y_test.tolist()) | set(pred.tolist()))
    report = classification_report(y_test, pred, labels=labels,
                                   target_names=[CLASS_NAMES[c] for c in labels],
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, pred, labels=labels)
    macro_f1 = f1_score(y_test, pred, average="macro", labels=labels, zero_division=0)

    log.info("held-out tiles, per class:")
    log.info("  %-11s %9s %9s %9s %9s", "class", "precision", "recall", "f1", "support")
    for code in labels:
        row = report[CLASS_NAMES[code]]
        log.info("  %-11s %9.3f %9.3f %9.3f %9d", CLASS_NAMES[code], row["precision"],
                 row["recall"], row["f1-score"], int(row["support"]))
    log.info("macro F1 %.4f | balanced accuracy %.4f | (accuracy %.4f, not the headline)",
             macro_f1, balanced_accuracy_score(y_test, pred), report["accuracy"])
    log.info("confusion (rows reference, cols predicted, order %s):",
             [CLASS_NAMES[c] for c in labels])
    for code, row in zip(labels, matrix):
        log.info("  %-11s %s", CLASS_NAMES[code], " ".join(f"{v:8d}" for v in row))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "cotton_rf_v2.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "accuracy": float(report["accuracy"]),
        "per_class": {CLASS_NAMES[c]: {"code": int(c), **{k: float(v) for k, v in report[CLASS_NAMES[c]].items()}}
                      for c in labels},
        "confusion_matrix": {"labels": [int(c) for c in labels],
                             "class_names": [CLASS_NAMES[c] for c in labels],
                             "counts": matrix.tolist()},
        "cv": {"folds": folds, "best_macro_f1": float(study.best_value),
               "n_trials": args.n_trials,
               "completed_trials": len([t for t in study.trials
                                        if t.state == optuna.trial.TrialState.COMPLETE])},
        "best_params": study.best_params,
        "held_out_tiles": test_tiles,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    card = {
        "model_file": model_path.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_dates": EXPECTED_DATES,
        "n_features": len(EXPECTED_DATES),
        "class_codes": {str(c): CLASS_NAMES[c] for c in sorted(CLASS_NAMES)},
        "classes_present": [int(c) for c in model.classes_],
        "training_tiles": train_tiles,
        "held_out_tiles": test_tiles,
        "rows_per_class": {CLASS_NAMES[int(c)]: int(n) for c, n in counts.items()},
        "training_rows_per_class": {CLASS_NAMES[int(c)]: int(n)
                                    for c, n in train["label"].value_counts().sort_index().items()},
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "sources": {"labelled_pixels": str(args.labelled),
                    "old_negatives": None if args.no_old_negatives else str(args.old_csv)},
        "best_params": study.best_params,
    }
    (args.out_dir / "model_card.json").write_text(json.dumps(card, indent=2))

    plot_confusion(matrix, labels, args.out_dir / "confusion_matrix.png")
    plot_importance(model.feature_importances_, args.out_dir / "feature_importance.png")
    log.info("written under %s", args.out_dir)


if __name__ == "__main__":
    main()
