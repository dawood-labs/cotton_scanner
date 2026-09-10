"""Build the v2 training set from the GEE NDVI already exported for the 23 AOIs.

Where the labels come from, and why this is different from v1:

v1 labelled k-means clusters by eye. The audit found the cotton side full of curves that
peak in September, where rice peaks. Here nothing is labelled by eye. Every pixel takes
its class from a polygon somebody surveyed -- cotton from the four mill surveys, rice /
fall maize / sugarcane from the national scans, orchards from the exclusion mask.

Why the GEE rasters rather than a fresh STAC pull: they already exist, they cover both
provinces, and they cost no download time. The one thing they are not is the sensor path
inference uses, so the smoothing here is done over exactly the 35 dates cropstack
smooths over, and the comparison against v1 is run on STAC-derived validation curves
where any remaining domain gap will show.

Anything inside a validation AOI is cut out of the training geometry before a single
pixel is read. That exclusion is the only reason the later numbers mean anything.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from scipy.ndimage import binary_erosion

sys.path.insert(0, "/home/jovyan/FAO/optimized_code_testing/cropstack")
from inference_workers import get_penalty_matrix, smooth_ndvi_block  # noqa: E402

BASE = Path("/home/jovyan/FAO/cotton")
GEE = BASE / "training_v2/gee"
OUT = BASE / "training_v2/labelled_pixels.parquet"
CLUSTERS = BASE / "model_training_cluster_cotton_data/model_training_cluster_cotton_data.shp"

# The model's feature vector: 35 eight-day composites, named by the window's end date.
DATES = [str(d.date()) for d in pd.date_range("2025-04-01", "2025-12-29", freq="8D")]

CODES = {"cotton": 1, "rice": 2, "sugarcane": 3, "fall_maize": 5, "orchard": 6}
COTTON_GT = [
    "validation_data/Al-Moiz-2-Cotton-2025/Al-Moiz-2-Cotton-2025.shp",
    "validation_data/Baba-Fareed-Cotton-2025/Baba-Fareed-Cotton-2025.shp",
    "validation_data/Faran-Cotton-2025/Faran-Cotton-2025.shp",
    "validation_data/Layyah-Cotton-2025/Layyah-Cotton-2025.shp",
]
OTHER_SCANS = {
    "rice": "validation_data/Corteva_Rice_2025_2nd_Scan/Corteva_Rice_2025_2nd_Scan.shp",
    "fall_maize": "validation_data/Corteva-Fall-Maize-2025/Corteva-Fall-Maize-2025.shp",
    "sugarcane": "validation_data/Sugarcane_3m-10m_Pakistan-Scan_2025/Sugarcane_3m-10m_Pakistan-Scan_2025.shp",
    "orchard": "validation_data/orchard_exclusion_mask/orchard_exclusion_mask.gpkg",
}
VALIDATION = [
    "validation_data/cotton_val_aois/cotton_val_aois.shp",
    "validation_data/aois/layyah_orchards.gpkg",
]
AMBIGUOUS = 255


def training_geometry(cluster_id: int):
    """The AOI polygon with every validation AOI subtracted."""
    clusters = gpd.read_file(CLUSTERS)
    geom = clusters[clusters.cluster_id == cluster_id].geometry.iloc[0]
    for path in VALIDATION:
        for other in gpd.read_file(BASE / path).to_crs(4326).geometry:
            if geom.intersects(other):
                geom = geom.difference(other)
    return geom


def load_labels(geom):
    """Every surveyed polygon inside the training geometry, tagged with its class."""
    bounds = tuple(geom.bounds)
    parts = []
    for label, paths in [("cotton", COTTON_GT)] + [(k, [v]) for k, v in OTHER_SCANS.items()]:
        for path in paths:
            try:
                part = gpd.read_file(BASE / path, bbox=bounds)
            except Exception:
                continue
            if part.empty:
                continue
            part = part.to_crs(4326)
            invalid = ~part.geometry.is_valid
            if invalid.any():
                part.loc[invalid, "geometry"] = part.loc[invalid, "geometry"].make_valid()
                part = part[part.geometry.geom_type.isin(("Polygon", "MultiPolygon"))]
            part = gpd.clip(part, geom)
            part = part[~part.geometry.is_empty & part.geometry.notna()]
            if part.empty:
                continue
            keep = part[["geometry"]].copy()
            keep["crop"] = label
            parts.append(keep)
    if not parts:
        return gpd.GeoDataFrame(columns=["geometry", "crop"], crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)


def band_indices(descriptions):
    """Positions of the 35 model dates in this raster, or None if any is absent."""
    available = {}
    for i, name in enumerate(descriptions):
        if name and name.startswith("ndvi_"):
            available[name[5:].replace("_", "-")] = i
    missing = [d for d in DATES if d not in available]
    if missing:
        return None, missing
    return [available[d] for d in DATES], []


def sample_grid(raster_path: Path, labels, geom, per_class: int, rng) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        indices, missing = band_indices(src.descriptions)
        if indices is None:
            print(f"    {raster_path.name}: missing {len(missing)} dates, skipped", flush=True)
            return pd.DataFrame()

        inside = geometry_mask([geom], out_shape=(src.height, src.width),
                               transform=src.transform, invert=True)
        if not inside.any():
            return pd.DataFrame()

        burned = np.zeros((src.height, src.width), dtype="uint8")
        for label, code in CODES.items():
            part = labels[labels["crop"] == label]
            if part.empty:
                continue
            shapes = [(g, code) for g in part.geometry if g is not None and not g.is_empty]
            layer = rasterize(shapes, out_shape=burned.shape, transform=src.transform,
                              fill=0, dtype="uint8", all_touched=False)
            # A field edge is half one crop and half the next, and those mixed pixels are
            # exactly what a classifier learns the wrong boundary from. One pixel of
            # erosion costs the outer 10 m of every field and buys a clean interior.
            layer = np.where(binary_erosion(layer > 0, structure=np.ones((3, 3))), layer, 0)
            clash = (burned > 0) & (layer > 0)
            burned = np.where(layer > 0, layer, burned)
            burned[clash] = AMBIGUOUS      # two scans claim it; neither gets it
        burned[~inside] = 0

        usable = (burned > 0) & (burned != AMBIGUOUS)
        if not usable.any():
            return pd.DataFrame()

        rows, cols = np.where(usable)
        codes = burned[rows, cols]
        keep_rows, keep_cols, keep_codes = [], [], []
        for code in np.unique(codes):
            at = np.where(codes == code)[0]
            if len(at) > per_class:
                at = rng.choice(at, size=per_class, replace=False)
            keep_rows.append(rows[at]); keep_cols.append(cols[at]); keep_codes.append(codes[at])
        rows = np.concatenate(keep_rows); cols = np.concatenate(keep_cols)
        codes = np.concatenate(keep_codes)

        # Smooth over exactly the 35 dates the inference pipeline smooths over. Smoothing a
        # longer series and then slicing gives different values at the ends, and the model
        # would be trained on numbers inference never produces.
        stack = src.read([i + 1 for i in indices]).astype(np.float32)
        # The GEE export declares no nodata value and writes NaN for masked pixels, which
        # is already what the smoother treats as missing. Only convert when a raster does
        # declare one.
        if src.nodata is not None and not np.isnan(src.nodata):
            stack[stack == src.nodata] = np.nan
        penalty = get_penalty_matrix(len(DATES), 0.5, 2)
        smoothed = smooth_ndvi_block(stack, penalty, (-1.0, 1.0), missing_value=np.nan)

        curves = smoothed[:, rows, cols].T
        good = ~np.all(np.isnan(curves), axis=1)
        curves, rows, cols, codes = curves[good], rows[good], cols[good], codes[good]

        frame = pd.DataFrame(np.nan_to_num(curves, nan=0.0), columns=DATES)
        frame["label"] = codes.astype(int)
        inverse = {v: k for k, v in CODES.items()}
        frame["crop"] = [inverse[c] for c in codes]
        frame["tile"] = raster_path.parent.name + "/" + raster_path.stem
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        frame["lon"] = xs
        frame["lat"] = ys
        return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class-per-grid", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--aois", nargs="*", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    folders = sorted(GEE.glob("AOI_*_5000_cotton"))
    if args.aois:
        wanted = {f"AOI_{a}_5000_cotton" for a in args.aois}
        folders = [f for f in folders if f.name in wanted]

    frames = []
    for folder in folders:
        cluster_id = int(folder.name.split("_")[1])
        geom = training_geometry(cluster_id)
        if geom.is_empty:
            print(f"{folder.name}: entirely inside a validation AOI, skipped", flush=True)
            continue
        labels = load_labels(geom)
        if labels.empty:
            print(f"{folder.name}: no surveyed polygons", flush=True)
            continue
        acres = labels.to_crs(labels.estimate_utm_crs()).area.groupby(labels["crop"].values).sum() / 4046.86
        print(f"{folder.name}: " + ", ".join(f"{k} {v:,.0f}ac" for k, v in acres.items()), flush=True)

        for grid in sorted(folder.glob("ndvi_grid_*.tif")):
            frame = sample_grid(grid, labels, geom, args.per_class_per_grid, rng)
            if not frame.empty:
                frames.append(frame)

    if not frames:
        print("nothing sampled")
        return 1
    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out)
    print("\nrows per class:")
    print(table.groupby("crop").size().sort_values(ascending=False).to_string())
    print(f"\n{len(table):,} rows over {table['tile'].nunique()} grids -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
