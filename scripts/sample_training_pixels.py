"""Turn the acquired tiles plus the labelled polygons into the training pixel table.

Every step here exists to make a training row identical to what the inference worker
would compute for the same pixel: parse the band stack with cropstack's own parser,
NDVI from red/NIR, Whittaker smoothing over the FULL series with lambda 0.5 / order 2,
and only then cut to the 35 dates 2025-04-01..2025-12-29. Smoothing before the cut
matters -- the smoother borrows from neighbouring dates, so a series trimmed first has
different endpoints than the same series trimmed last, and the model would be trained on
curves nobody ever infers on.

Label rules, both of which cost samples on purpose:

  * A pixel claimed by two classes is dropped, not resolved. The other-crop layers are
    model outputs from separate scans, not survey; where two of them overlap, at least
    one is wrong, and there is no basis for choosing which.
  * Each class mask is eroded by one pixel. A 10 m pixel on a field boundary is a mix of
    the field and whatever is next to it -- a road, a drain, the neighbouring crop. Those
    pixels carry a blended curve with a confident label, which is the single most
    effective way to teach a classifier the wrong thing. One pixel of erosion is the
    cheapest fix; it costs the field's outline and keeps its interior.

Memory: one tile at a time, one raster block at a time. A 0.1 deg tile at 10 m is
~1100x1100 x 70 bands, which is 340 MB as float32 before smoothing allocates its own
copies -- so the stack is never read whole.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/jovyan/FAO/optimized_code_testing/cropstack")

import geopandas as gpd                                   # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import rasterio                                            # noqa: E402
from rasterio.features import rasterize                    # noqa: E402
from scipy.ndimage import binary_erosion                   # noqa: E402

from band_utils import parse_band_stack                    # noqa: E402
from inference_workers import get_penalty_matrix, smooth_ndvi_block  # noqa: E402

BASE = Path("/home/jovyan/FAO/cotton")
TILES = BASE / "training_v2/tiles/training_tiles.gpkg"
RAW = BASE / "training_v2/tiles/raw"
OUT = BASE / "training_v2/labelled_pixels.parquet"

# Codes match the validation scripts' CLASS_ORDER and the v1 model's cotton=1. 4 is left
# free because v1 used it for the lumped "non-cotton" class.
CLASS_CODES = {"cotton": 1, "rice": 2, "sugarcane": 3, "fall_maize": 5, "orchard": 6}
AMBIGUOUS = 255

SOURCES = {
    "cotton": [
        "validation_data/Al-Moiz-2-Cotton-2025/Al-Moiz-2-Cotton-2025.shp",
        "validation_data/Baba-Fareed-Cotton-2025/Baba-Fareed-Cotton-2025.shp",
        "validation_data/Faran-Cotton-2025/Faran-Cotton-2025.shp",
        "validation_data/Layyah-Cotton-2025/Layyah-Cotton-2025.shp",
    ],
    "rice": ["validation_data/Corteva_Rice_2025_2nd_Scan/Corteva_Rice_2025_2nd_Scan.shp"],
    "sugarcane": ["validation_data/Sugarcane_3m-10m_Pakistan-Scan_2025/Sugarcane_3m-10m_Pakistan-Scan_2025.shp"],
    "fall_maize": ["validation_data/Corteva-Fall-Maize-2025/Corteva-Fall-Maize-2025.shp"],
    "orchard": ["validation_data/orchard_exclusion_mask/orchard_exclusion_mask.gpkg"],
}

WINDOW = ("2025-04-01", "2025-12-29")
SMOOTH_LAMBDA, SMOOTH_ORDER = 0.5, 2
CLIP_BOUNDS = (-1.0, 1.0)
MAX_PER_CLASS_PER_TILE = 20000


def label_raster(sources, shape, transform, crs, bounds):
    """Burn every class onto the tile grid, erode each, and mark double-claims unusable.

    Erosion is done per class before the classes are merged, so a class only loses its
    own boundary and not the boundary of whatever it happens to sit next to.
    """
    burned = np.zeros(shape, dtype="uint8")
    for crop, paths in sources.items():
        mask = np.zeros(shape, dtype=bool)
        for path in paths:
            part = gpd.read_file(path, bbox=bounds)
            if part.empty:
                continue
            part = part.to_crs(crs)
            bad = ~part.geometry.is_valid
            if bad.any():
                part.loc[bad, "geometry"] = part.loc[bad, "geometry"].make_valid()
            shapes = [(g, 1) for g in part.geometry
                      if g is not None and not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon")]
            if not shapes:
                continue
            # all_touched=False: a polygon owns the pixels whose centres it contains, so a
            # thin field cannot claim the track beside it.
            mask |= rasterize(shapes, out_shape=shape, transform=transform,
                              fill=0, dtype="uint8", all_touched=False).astype(bool)
        if not mask.any():
            continue
        mask = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        if not mask.any():
            continue
        clash = (burned > 0) & mask
        burned = np.where(mask, CLASS_CODES[crop], burned).astype("uint8")
        burned[clash] = AMBIGUOUS
    return burned


def sample_tile(tif: Path, sources, rng, max_per_class: int) -> pd.DataFrame:
    with rasterio.open(tif) as src:
        red_idx, nir_idx, dates = parse_band_stack(src.descriptions)
        keep = [i for i, d in enumerate(dates) if WINDOW[0] <= d <= WINDOW[1]]
        columns = [dates[i] for i in keep]
        penalty = get_penalty_matrix(len(dates), SMOOTH_LAMBDA, SMOOTH_ORDER)

        labels = label_raster(sources, (src.height, src.width), src.transform, src.crs,
                              tuple(gpd.GeoSeries([], crs=src.crs).total_bounds) if False
                              else _bounds_4326(src))
        if not (labels > 0).any():
            return pd.DataFrame()

        code_to_crop = {v: k for k, v in CLASS_CODES.items()}
        frames = []
        for _, window in src.block_windows(1):
            rows = slice(window.row_off, window.row_off + window.height)
            cols = slice(window.col_off, window.col_off + window.width)
            block_labels = labels[rows, cols]
            usable = (block_labels > 0) & (block_labels != AMBIGUOUS)
            if not usable.any():
                continue

            raw = src.read(window=window)
            red = raw[red_idx].astype(np.float32)
            nir = raw[nir_idx].astype(np.float32)
            denom = nir + red
            ndvi = np.full_like(red, np.nan)
            np.divide(nir - red, denom, out=ndvi, where=denom > 0)
            del raw, denom

            smoothed = smooth_ndvi_block(ndvi, penalty, CLIP_BOUNDS, missing_value=np.nan)
            del ndvi
            flat = smoothed[keep].transpose(1, 2, 0).reshape(-1, len(keep))
            del smoothed

            flat_labels = block_labels.reshape(-1)
            pick = usable.reshape(-1) & ~np.all(np.isnan(flat), axis=1)
            if not pick.any():
                continue

            xs, ys = rasterio.transform.xy(
                src.window_transform(window),
                *np.nonzero(usable), offset="center")
            keep_xy = ~np.all(np.isnan(flat[usable.reshape(-1)]), axis=1)
            lon, lat = _to_4326(np.asarray(xs)[keep_xy], np.asarray(ys)[keep_xy], src.crs)

            frame = pd.DataFrame(np.nan_to_num(flat[pick], nan=0.0), columns=columns)
            frame["label"] = flat_labels[pick].astype("int16")
            frame["crop"] = [code_to_crop[c] for c in frame["label"]]
            frame["lon"], frame["lat"] = lon, lat
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table["tile"] = tif.parent.name
    # Cap per class per tile: one national scan can cover most of a cell, and without a
    # cap that class alone would outnumber the surveyed cotton by an order of magnitude.
    table = table.groupby("crop", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max_per_class), random_state=0))
    return table.reset_index(drop=True)


def _bounds_4326(src):
    from rasterio.warp import transform_bounds
    return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def _to_4326(xs, ys, crs):
    from rasterio.warp import transform as warp_transform
    lon, lat = warp_transform(crs, "EPSG:4326", list(xs), list(ys))
    return np.asarray(lon), np.asarray(lat)


def resolve_sources(root: Path):
    return {crop: [str(root / p) for p in paths if (root / p).exists()]
            for crop, paths in SOURCES.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("pilot", "extension", "all"), default="pilot")
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS_PER_TILE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cells = gpd.read_file(TILES)
    if args.batch != "all":
        cells = cells[cells.batch == args.batch]

    sources = resolve_sources(BASE)
    rng = np.random.default_rng(0)
    frames = []
    for tile_id in cells.tile_id:
        tifs = sorted((RAW / tile_id).glob("sentinel_*m_tile_*.tif"))
        if not tifs:
            print(f"{tile_id}: no imagery yet, skipped", flush=True)
            continue
        for tif in tifs:
            table = sample_tile(tif, sources, rng, args.max_per_class)
            if table.empty:
                print(f"{tile_id}: no labelled pixels", flush=True)
                continue
            frames.append(table)
            print(f"{tile_id}: {len(table):,} rows "
                  f"{table.crop.value_counts().to_dict()}", flush=True)

    if not frames:
        print("nothing sampled")
        return 1
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"\n{len(out):,} rows -> {args.out}")
    print(out.groupby("crop").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
