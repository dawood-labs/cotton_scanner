"""Re-predict the cached validation tiles with a different model, and score the maps.

The raw Sentinel tiles for all six validation AOIs are already on disk, so swapping the
model costs compute and nothing else -- no re-acquisition, and the imagery is
bit-identical between the two runs, which is what makes the comparison a comparison.

Writes into validation_runs/<aoi>__<tag>/1_ndvi_run_1/ so the original v1 maps stay
where they are and both can be scored side by side.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/jovyan/FAO/optimized_code_testing/cropstack")

from inference_workers import mosaic_prediction_tiles, worker_process_tile  # noqa: E402
from postprocess import apply_strict_directional_sieve  # noqa: E402

BASE = Path("/home/jovyan/FAO/cotton")
RUNS = BASE / "validation_runs"
WINDOW = ("2025-04-01", "2025-12-29")
AOIS = ["al_moiz_2_1", "baba_fareed_1", "baba_fareed_2", "faran_1", "layyah_1", "layyah_orchards"]


def rerun(slug: str, model_path: str, tag: str) -> Path | None:
    source = sorted((RUNS / slug).glob("1_ndvi_run_*"))
    if not source:
        print(f"{slug}: no NDVI run to reuse", flush=True)
        return None
    tiles = sorted((source[-1] / "raw_ndvi_tiles").glob("sentinel_*m_tile_*.tif"))
    if not tiles:
        print(f"{slug}: raw tiles are gone; pull them back with gcs_cache.py", flush=True)
        return None

    out_dir = RUNS / f"{slug}__{tag}" / "1_ndvi_run_1"
    chunks = out_dir / "tile_predictions"
    chunks.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{slug}_rf_classification_map.tif"

    existing = list(out_dir.glob("*_sieved_*.tif"))
    if existing:
        print(f"{slug}: already built -> {existing[0].name}", flush=True)
        return existing[0]

    predictions = []
    for tile in tiles:
        print(f"{slug}: {tile.name}", flush=True)
        predictions.append(worker_process_tile(tile, chunks, model_path, *WINDOW)["prediction"])
    mosaic_prediction_tiles(predictions, final)
    # Same sieve settings the pipeline applies, or the difference measured would include
    # the post-processing rather than the model.
    sieved = apply_strict_directional_sieve(final, target_classes=[1], min_pixel_size=20,
                                            connectivity=4, nodata_val=255)
    print(f"{slug}: -> {sieved}", flush=True)
    return Path(sieved)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True, help="suffix for the output run, e.g. v2")
    parser.add_argument("--aois", nargs="*", default=AOIS)
    args = parser.parse_args()

    for slug in args.aois:
        rerun(slug, args.model, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
