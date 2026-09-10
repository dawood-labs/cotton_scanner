"""Pull smoothed NDVI curves for every reference class straight out of the saved tiles.

Why not read the classification map: the map only says cotton / not-cotton. To see *why*
rice or an orchard gets called cotton we need the curve the model actually saw, which
means the same Whittaker smoothing the inference worker applies, on the same 35 dates.

Output: one parquet per AOI, columns = the 35 training dates, plus `crop` and the model's
cotton probability for that pixel. That table answers the diagnosis and, if we decide to
retrain, is already the hard-negative set.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/jovyan/FAO/optimized_code_testing/cropstack")

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from band_utils import parse_band_stack            # noqa: E402
from inference_workers import get_penalty_matrix, smooth_ndvi_block  # noqa: E402

BASE = Path("/home/jovyan/FAO/cotton")
RUNS = BASE / "validation_runs"
REFS = BASE / "validation_data/reference_crops"
OUT = BASE / "validation_data/reference_curves"
MODEL = BASE / "timeseries_model/model_v1/best_rf_classifier.joblib"
WINDOW = ("2025-04-01", "2025-12-29")
MAX_PER_CLASS = 20000      # enough for a stable mean curve without holding a district in RAM


def sample_one(slug: str, rng: np.random.Generator) -> pd.DataFrame:
    ndvi_dir = sorted((RUNS / slug).glob("1_ndvi_run_*"))[-1]
    tiles = sorted((ndvi_dir / "raw_ndvi_tiles").glob("sentinel_*m_tile_*.tif"))
    refs = gpd.read_file(REFS / f"{slug}.gpkg")
    labels = sorted(refs["crop"].unique())
    code_of = {label: i + 1 for i, label in enumerate(labels)}

    frames = []
    for tile in tiles:
        with rasterio.open(tile) as src:
            red_idx, nir_idx, dates = parse_band_stack(src.descriptions)
            stamps = pd.to_datetime(dates)
            keep = [i for i, d in enumerate(stamps)
                    if pd.Timestamp(WINDOW[0]) <= d <= pd.Timestamp(WINDOW[1])]
            penalty = get_penalty_matrix(len(dates), 0.5, 2)

            burned = np.zeros((src.height, src.width), dtype="uint8")
            for label in labels:
                part = refs[refs["crop"] == label].to_crs(src.crs)
                shapes = [(g, code_of[label]) for g in part.geometry if g is not None and not g.is_empty]
                if not shapes:
                    continue
                # Later classes do not overwrite earlier ones: a pixel claimed by two
                # scans is ambiguous ground truth and is dropped below instead.
                layer = rasterize(shapes, out_shape=burned.shape, transform=src.transform,
                                  fill=0, dtype="uint8", all_touched=False)
                clash = (burned > 0) & (layer > 0)
                burned = np.where(layer > 0, layer, burned)
                burned[clash] = 255                    # 255 = claimed twice, unusable

            for _, window in src.block_windows(1):
                block_labels = burned[window.row_off:window.row_off + window.height,
                                      window.col_off:window.col_off + window.width]
                if not (block_labels > 0).any():
                    continue
                raw = src.read(window=window)
                red = raw[red_idx].astype(np.float32)
                nir = raw[nir_idx].astype(np.float32)
                denom = nir + red
                ndvi = np.full_like(red, np.nan)
                np.divide(nir - red, denom, out=ndvi, where=denom > 0)

                smoothed = smooth_ndvi_block(ndvi, penalty, (-1.0, 1.0), missing_value=np.nan)
                flat = smoothed[keep].transpose(1, 2, 0).reshape(-1, len(keep))
                flat_labels = block_labels.reshape(-1)

                usable = (flat_labels > 0) & (flat_labels != 255) & ~np.all(np.isnan(flat), axis=1)
                if not usable.any():
                    continue
                frame = pd.DataFrame(np.nan_to_num(flat[usable], nan=0.0),
                                     columns=[str(stamps[i].date()) for i in keep])
                frame["crop"] = [labels[c - 1] for c in flat_labels[usable]]
                frames.append(frame)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    # Cap each class so one big scan does not drown the others in the mean curve.
    table = table.groupby("crop", group_keys=False).apply(
        lambda g: g.sample(min(len(g), MAX_PER_CLASS), random_state=0))
    date_cols = [c for c in table.columns if c != "crop"]

    model = joblib.load(MODEL)
    model.n_jobs = 2
    cotton_col = list(model.classes_).index(1)
    table["cotton_prob"] = model.predict_proba(table[date_cols].values)[:, cotton_col]
    table["aoi"] = slug
    return table.reset_index(drop=True)


def main(slugs, runs_dir=None, refs_dir=None, out_dir=None):
    global RUNS, REFS, OUT
    # Parameterised so the same sampler serves a validation AOI and a whole district;
    # the work is identical, only the folders differ.
    if runs_dir: RUNS = Path(runs_dir)
    if refs_dir: REFS = Path(refs_dir)
    if out_dir: OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for slug in slugs:
        if not (RUNS / slug).exists() or not (REFS / f"{slug}.gpkg").exists():
            continue
        table = sample_one(slug, rng)
        if table.empty:
            print(f"{slug}: nothing sampled", flush=True)
            continue
        table.to_parquet(OUT / f"{slug}.parquet")
        summary = table.groupby("crop")["cotton_prob"].agg(["count", "mean"]).round(3)
        print(f"{slug}\n{summary.to_string()}\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slugs", nargs="*")
    parser.add_argument("--runs-dir")
    parser.add_argument("--refs-dir")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    main(args.slugs or ["al_moiz_2_1", "baba_fareed_1", "baba_fareed_2",
                        "faran_1", "layyah_1", "layyah_orchards"],
         args.runs_dir, args.refs_dir, args.out_dir)
