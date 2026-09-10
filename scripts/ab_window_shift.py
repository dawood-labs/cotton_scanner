"""Re-predict the saved NDVI tiles through the old, shifted inference window.

The aligned maps already exist under validation_runs/. This reuses their raw tiles --
same imagery, same smoothing, same model -- and changes one thing: the date window. The
old notebook asked for 2025-04-02..2025-12-31, which drops the 04-01 composite and hands
the model 04-09..12-31, one 8-day step later than the dates it was trained on.

Holding everything else fixed is the point. Any difference in the scores below is the
window and nothing else.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/jovyan/FAO/optimized_code_testing/cropstack")

from inference_workers import mosaic_prediction_tiles, worker_process_tile  # noqa: E402
from postprocess import apply_strict_directional_sieve  # noqa: E402

BASE = Path("/home/jovyan/FAO/cotton")
RUNS = BASE / "validation_runs"
MODEL = str(BASE / "timeseries_model/model_v1/best_rf_classifier.joblib")
SHIFTED = ("2025-04-02", "2025-12-31")


def rebuild(slug: str) -> Path:
    ndvi_dir = sorted((RUNS / slug).glob("1_ndvi_run_*"))[-1]
    tiles = sorted((ndvi_dir / "raw_ndvi_tiles").glob("sentinel_*m_tile_*.tif"))
    if not tiles:
        raise FileNotFoundError(f"{slug}: raw tiles were deleted; re-run with delete_raw_ndvi_tiles=false")

    out_dir = RUNS / f"{slug}_shifted" / "1_ndvi_run_1"
    chunks = out_dir / "tile_predictions"
    chunks.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{slug}_rf_classification_map.tif"
    if list(out_dir.glob("*_sieved_*.tif")):
        print(f"{slug}: already built", flush=True)
        return final

    predictions = []
    for tile in tiles:
        print(f"{slug}: {tile.name}", flush=True)
        result = worker_process_tile(tile, chunks, MODEL, SHIFTED[0], SHIFTED[1])
        predictions.append(result["prediction"])
    mosaic_prediction_tiles(predictions, final)
    # The aligned maps are scored after sieving, so this one has to be too, or the
    # comparison measures the sieve as well as the window.
    apply_strict_directional_sieve(final, target_classes=[1], min_pixel_size=20,
                                   connectivity=4, nodata_val=255)
    print(f"{slug}: -> {final}", flush=True)
    return final


if __name__ == "__main__":
    for slug in sys.argv[1:] or ["baba_fareed_1"]:
        rebuild(slug)
