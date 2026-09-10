"""Fetch the Sentinel-2 red/NIR stack for the selected training cells.

The calling convention is copied from cropstack's `ndvi_pipeline._acquire_tiles_from_stac`
and must stay copied. Training features have to be produced by the same code path as
inference features, or the retrain reintroduces BUG-1 (CONTEXT.md) from the other end:
a training set built on a slightly different window still trains, still scores well on
its own held-out split, and is wrong everywhere else.

The dates are the load-bearing part. farmdar names each composite by the window's END
date, so start=2025-03-24 / end=2025-12-29 / step=8 yields exactly the 35 composites
2025-04-01 .. 2025-12-29 -- the model's feature vector. Any other start shifts every
feature by a multiple of 8 days.

One fetch call per cell rather than one call over all of them: farmdar snaps its tile
grid to the AOI's own bbox and numbers tiles from 1 per call, so a single call over
sixteen scattered cells would grid the whole Sindh-to-Punjab bounding box. Per-cell also
gives free resumability -- a cell whose GeoTIFF already opens is skipped.
"""
import argparse
import sys
import time
from pathlib import Path

FARMDAR = "/home/jovyan/shared/git/standard-libraries/.worktrees/824850c677f49ef5b23af6040e9d2b165e586996"
CROPSTACK = "/home/jovyan/FAO/optimized_code_testing/cropstack"
sys.path.insert(0, FARMDAR)
sys.path.insert(0, CROPSTACK)

import geopandas as gpd                              # noqa: E402
import rasterio                                       # noqa: E402

from band_utils import parse_band_stack               # noqa: E402

BASE = Path("/home/jovyan/FAO/cotton")
TILES = BASE / "training_v2/tiles/training_tiles.gpkg"
RAW = BASE / "training_v2/tiles/raw"

START, END = "2025-03-24", "2025-12-29"
STEP_DAYS = 8
EXPECTED_DATES = 35
WINDOW = ("2025-04-01", "2025-12-29")


def existing_tile(tile_dir: Path):
    """A tile counts as done only if it opens, parses, and carries the 35 dates.

    A half-written GeoTIFF from a killed run opens fine at the header and fails on read,
    which would otherwise be discovered days later during sampling.
    """
    for path in sorted(tile_dir.glob("sentinel_*m_tile_*.tif")):
        try:
            with rasterio.open(path) as src:
                _, _, dates = parse_band_stack(src.descriptions)
                inside = [d for d in dates if WINDOW[0] <= d <= WINDOW[1]]
                if len(inside) != EXPECTED_DATES:
                    continue
                src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
            return path
        except Exception:
            continue
    return None


def acquire(row, workers: int):
    from farmdar.sentinel import fetch_sentinel_imagery   # never modified, only called

    tile_dir = RAW / row.tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)

    done = existing_tile(tile_dir)
    if done is not None:
        print(f"{row.tile_id}: already present ({done.name})", flush=True)
        return done

    aoi_path = tile_dir / "aoi.gpkg"
    gpd.GeoDataFrame([{"tile_id": row.tile_id}], geometry=[row.geometry], crs=4326).to_file(
        aoi_path, driver="GPKG")

    started_at = time.time()
    result = fetch_sentinel_imagery(
        aoi=str(aoi_path),
        start=START,
        end=END,
        bands=["red", "nir"],
        out_dir=str(tile_dir),
        step=STEP_DAYS,
        res_m=10,
        tile_deg=0.1,
        cloud_lt=97,
        workers=workers,
        build_vrt_mosaic=False,   # tiles are consumed individually, no mosaic needed
        clip_to_aoi=False,
    )
    outcomes = result.get("results") or []
    failed = [r for r in outcomes if str(r.get("status", "")).startswith("failed")]
    minutes = (time.time() - started_at) / 60

    if failed:
        print(f"{row.tile_id}: FAILED {[(r.get('tile_id'), r.get('status')) for r in failed]} "
              f"after {minutes:.1f} min", flush=True)
        return None

    produced = existing_tile(tile_dir)
    if produced is None:
        # Reported success but the stack does not carry the 35 dates: either the window
        # changed upstream or the write was short. Either way it must not be sampled.
        print(f"{row.tile_id}: acquired but does not carry {EXPECTED_DATES} dates in "
              f"{WINDOW[0]}..{WINDOW[1]}; not usable", flush=True)
        return None

    print(f"{row.tile_id}: {produced.name} in {minutes:.1f} min", flush=True)
    return produced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("pilot", "extension", "all"), default="pilot")
    parser.add_argument("--limit", type=int, default=None,
                        help="acquire only the first N cells of the batch")
    # One tile per call, so threads over tiles buy nothing; raising this only raises peak
    # memory (cropstack budgets ~1.5 GiB per tile in flight).
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cells = gpd.read_file(TILES).to_crs(4326)
    if args.batch != "all":
        cells = cells[cells.batch == args.batch]
    cells = cells.sort_values("score", ascending=False)
    if args.limit:
        cells = cells.head(args.limit)
    if cells.empty:
        print(f"no cells for batch {args.batch!r} in {TILES}")
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"acquiring {len(cells)} cell(s), batch={args.batch}, {START}..{END} step {STEP_DAYS}d",
          flush=True)

    produced = [acquire(row, args.workers) for row in cells.itertuples()]
    ok = [p for p in produced if p is not None]
    print(f"\n{len(ok)} of {len(cells)} cell(s) usable, under {RAW}")
    if len(ok) < len(cells):
        print("re-run the same command to retry the rest; finished cells are skipped")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
