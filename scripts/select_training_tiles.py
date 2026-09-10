"""Choose the 0.1 deg cells to buy imagery for, so the retrain sees confusion, not cotton.

The v1 training set was visually-labelled k-means clusters, and its failure mode is
visible in validation: rice and orchards get mapped as cotton. That is what a training
set built only from cotton polygons produces -- the model never saw the crops it has to
say no to. So a cell is only worth downloading if it holds surveyed cotton AND at least
one other crop scan; a cell of pure cotton teaches the model nothing it does not already
know.

Cost control. The other-crop scans are 320 MB - 2 GB national files, so every read here
is bbox-filtered against the shapefile's own spatial index. Acreage is then taken twice:
once cheaply, by dropping each polygon's centroid into a cell and summing its full area,
which is wrong at cell edges but right enough to rank; and then exactly, by clipping to
the cell polygon, but only for the handful of cells the cheap pass shortlists.

Validation leak. Any cell touching a validation AOI (plus a 0.02 deg margin, since a
Sentinel tile is snapped to the pixel grid and does not land exactly on the cell edge) is
dropped outright. BUG-2 in CONTEXT.md is what happens when that is not done.

Output: training_tiles.gpkg, with a `batch` column. The pilot batch is the six cells we
acquire first; the extension batch is ranked and waiting.
"""
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

BASE = Path("/home/jovyan/FAO/cotton")
OUT = BASE / "training_v2/tiles/training_tiles.gpkg"

COTTON_SOURCES = {
    "al_moiz": "validation_data/Al-Moiz-2-Cotton-2025/Al-Moiz-2-Cotton-2025.shp",
    "baba_fareed": "validation_data/Baba-Fareed-Cotton-2025/Baba-Fareed-Cotton-2025.shp",
    "faran": "validation_data/Faran-Cotton-2025/Faran-Cotton-2025.shp",
    "layyah": "validation_data/Layyah-Cotton-2025/Layyah-Cotton-2025.shp",
}
OTHER_SOURCES = {
    "rice": "validation_data/Corteva_Rice_2025_2nd_Scan/Corteva_Rice_2025_2nd_Scan.shp",
    "fall_maize": "validation_data/Corteva-Fall-Maize-2025/Corteva-Fall-Maize-2025.shp",
    "sugarcane": "validation_data/Sugarcane_3m-10m_Pakistan-Scan_2025/Sugarcane_3m-10m_Pakistan-Scan_2025.shp",
    "orchard": "validation_data/orchard_exclusion_mask/orchard_exclusion_mask.gpkg",
}
VAL_AOIS = [
    "validation_data/cotton_val_aois/cotton_val_aois.shp",
    "validation_data/aois/layyah_orchards.gpkg",
]
CLASSES = ["cotton", "rice", "sugarcane", "fall_maize", "orchard"]

CELL_DEG = 0.1
VAL_BUFFER_DEG = 0.02
SINDH_LAT = 28.5           # Sindh sows cotton ~6 weeks before Punjab; its curve is a different shape
ACRES_PER_M2 = 1 / 4046.86

# Saturation point for the score. sample_training_pixels.py caps samples per class per
# tile, so acreage past roughly this much buys no extra training rows -- which is exactly
# the property that makes a second class worth more than more of the first.
TARGET_ACRES = 400.0
MIN_COTTON_ACRES = 60.0    # below this a cell cannot supply a usable cotton sample after erosion
SHORTLIST = 60             # cells promoted from the cheap pass to the exact clip

N_SELECT = 16
MIN_SINDH = 5
# ...and a ceiling, because the Sindh cells score high as a block (cotton + sugarcane +
# orchard all overlap there) and would otherwise take most of the selection. Two sowing
# calendars means the training set needs both, not one of them thoroughly.
MAX_SINDH = 8
PILOT_SIZE = 6
PILOT_MIN_SINDH = 2
PILOT_MAX_SINDH = 3
PILOT_REQUIRED = ("cotton", "rice", "sugarcane", "orchard")


def snap_out(bounds):
    """Grow a bbox to whole 0.1 deg cell edges.

    Region bboxes overlap where two mills survey next to each other. Snapping out means
    any cell a region touches is covered by that region completely, so the same cell
    counted from two regions gets the same answer and the duplicate can just be skipped.
    """
    minx, miny, maxx, maxy = bounds
    return (np.floor(minx / CELL_DEG) * CELL_DEG, np.floor(miny / CELL_DEG) * CELL_DEG,
            np.ceil(maxx / CELL_DEG) * CELL_DEG, np.ceil(maxy / CELL_DEG) * CELL_DEG)


def cell_of(xs, ys):
    return (np.floor(np.asarray(xs) / CELL_DEG).astype(int),
            np.floor(np.asarray(ys) / CELL_DEG).astype(int))


def cell_box(i, j):
    return box(i * CELL_DEG, j * CELL_DEG, (i + 1) * CELL_DEG, (j + 1) * CELL_DEG)


def read_bbox(path, bounds, keep=None):
    """bbox read + validity repair. The national scans carry self-intersecting rings that
    make GEOS refuse any later clip or area call."""
    columns = list(keep) if keep else []
    gdf = gpd.read_file(BASE / path, bbox=tuple(bounds), columns=columns)
    if gdf.empty:
        return gdf
    gdf = gdf.to_crs(4326)
    bad = ~gdf.geometry.is_valid
    if bad.any():
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].make_valid()
    return gdf[gdf.geometry.geom_type.isin(("Polygon", "MultiPolygon"))].reset_index(drop=True)


def acres(gdf):
    if gdf.empty:
        return pd.Series(dtype="float64")
    return gdf.to_crs(gdf.estimate_utm_crs()).area * ACRES_PER_M2


def cheap_pass():
    """Per-cell acreage by centroid membership, plus the region reads cached for reuse."""
    regions, cotton_parts = {}, []
    for name, path in COTTON_SOURCES.items():
        gdf = read_bbox(path, gpd.read_file(BASE / path, columns=[]).total_bounds)
        regions[name] = snap_out(gdf.total_bounds)
        cotton_parts.append(gdf[["geometry"]])
    cotton = gpd.GeoDataFrame(pd.concat(cotton_parts, ignore_index=True), crs=4326)

    cache = {"cotton": {"all": cotton}}
    counts = {}
    districts = {}

    def add(label, gdf, done):
        if gdf.empty:
            return
        area = acres(gdf).values
        centroids = gdf.geometry.centroid
        ci, cj = cell_of(centroids.x.values, centroids.y.values)
        frame = pd.DataFrame({"i": ci, "j": cj, "acres": area})
        for (i, j), part in frame.groupby(["i", "j"]):
            if (i, j) in done:
                continue
            done.add((i, j))
            counts.setdefault((i, j), {c: 0.0 for c in CLASSES})[label] += float(part.acres.sum())

    add("cotton", cotton, set())
    for label, path in OTHER_SOURCES.items():
        cache[label] = {}
        done = set()
        keep = ["district"] if label == "orchard" else None
        for name, bounds in regions.items():
            gdf = read_bbox(path, bounds, keep=keep)
            cache[label][name] = gdf
            if label == "orchard" and not gdf.empty and "district" in gdf:
                centroids = gdf.geometry.centroid
                ci, cj = cell_of(centroids.x.values, centroids.y.values)
                for (i, j), part in pd.DataFrame({"i": ci, "j": cj, "d": gdf["district"].values}).groupby(["i", "j"]):
                    districts.setdefault((i, j), part.d.mode().iloc[0] if len(part.d.mode()) else "")
            add(label, gdf, done)
        print(f"  {label}: {sum(len(g) for g in cache[label].values()):,} features in range", flush=True)

    return counts, cache, regions, districts


def val_mask():
    parts = []
    for path in VAL_AOIS:
        gdf = gpd.read_file(BASE / path).to_crs(4326)
        parts.append(gdf[["geometry"]])
    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    return merged.geometry.buffer(VAL_BUFFER_DEG).union_all()


def score(row):
    """Concave in each class, so breadth beats depth.

    sqrt of a saturating utility: a class going 0 -> 10% of target adds 0.32, the same
    class going 90% -> 100% adds 0.03. Ten times more sugarcane in a cell that already
    has sugarcane is therefore worth far less than any rice at all. Cotton gets extra
    weight because it is the class being predicted.
    """
    utility = {c: min(row[f"{c}_acres"], TARGET_ACRES) / TARGET_ACRES for c in CLASSES}
    return 1.5 * np.sqrt(utility["cotton"]) + sum(np.sqrt(utility[c]) for c in CLASSES if c != "cotton")


def exact_pass(shortlist, cache, regions):
    rows = []
    for (i, j), cheap in shortlist:
        geom = cell_box(i, j)
        row = {"i": i, "j": j, "geometry": geom,
               "lon": (i + 0.5) * CELL_DEG, "lat": (j + 0.5) * CELL_DEG}
        for label in CLASSES:
            total = 0.0
            for name, gdf in cache[label].items():
                if gdf.empty:
                    continue
                near = gdf.iloc[list(gdf.sindex.query(geom, predicate="intersects"))]
                if near.empty:
                    continue
                clipped = gpd.clip(near, geom)
                clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
                if not clipped.empty:
                    total += float(acres(clipped).sum())
                if label == "cotton":
                    break          # cotton is held as one merged layer, not per region
            row[f"{label}_acres"] = round(total, 1)
        rows.append(row)
    return gpd.GeoDataFrame(rows, crs=4326)


def greedy(candidates, want, taken, separation, max_sindh=None):
    """Take the best-scoring cells that are at least `separation` cells from each other.

    There is no national district layer on this box, so cell separation stands in for
    "different district": two cells three steps apart on a 0.1 deg grid are ~33 km apart,
    which in the Punjab cotton belt is a different tehsil and usually a different mill's
    catchment. Adjacent cells would mostly re-sample the same fields.
    """
    chosen = list(taken)
    for _, cell in candidates.iterrows():
        if len(chosen) >= want:
            break
        if any(cell.name == c.name for c in chosen):
            continue
        if max_sindh is not None and cell.lat < SINDH_LAT:
            if sum(c.lat < SINDH_LAT for c in chosen) >= max_sindh:
                continue
        if all(max(abs(cell.i - c.i), abs(cell.j - c.j)) >= separation for c in chosen):
            chosen.append(cell)
    return chosen


def select(cells):
    cells = cells.sort_values("score", ascending=False)
    sindh = cells[cells.lat < SINDH_LAT]

    chosen = []
    for separation in (3, 2, 1):
        chosen = greedy(sindh, MIN_SINDH, chosen, separation)
        if len(chosen) >= MIN_SINDH:
            break
    for separation in (3, 2, 1):
        chosen = greedy(cells, N_SELECT, chosen, separation, max_sindh=MAX_SINDH)
        if len(chosen) >= N_SELECT:
            break
    return cells.loc[[c.name for c in chosen]]


def split_batches(selected):
    """Pilot = the smallest diverse subset. Coverage of the four classes first, then score.

    The point of the pilot is to answer "do six cells carry enough of each class", so it
    is picked to span classes rather than to be the six highest scores.
    """
    present = {idx: {c for c in CLASSES if row[f"{c}_acres"] > 0} for idx, row in selected.iterrows()}
    is_sindh = selected.lat < SINDH_LAT

    chosen, covered = [], set()
    while len(chosen) < PILOT_SIZE:
        remaining = PILOT_SIZE - len(chosen)
        need_sindh = PILOT_MIN_SINDH - sum(is_sindh[i] for i in chosen)
        pool = [i for i in selected.index if i not in chosen]
        if sum(is_sindh[i] for i in chosen) >= PILOT_MAX_SINDH:
            pool = [i for i in pool if not is_sindh[i]] or pool
        if need_sindh >= remaining:
            pool = [i for i in pool if is_sindh[i]] or pool
        missing = set(PILOT_REQUIRED) - covered
        pool.sort(key=lambda i: (-len(present[i] & missing), -selected.score[i]))
        chosen.append(pool[0])
        covered |= present[pool[0]]

    # If a required class still has no home, trade the weakest pilot cell for one that
    # carries it. Only ever a swap, so the pilot stays at PILOT_SIZE.
    for label in PILOT_REQUIRED:
        if label in covered:
            continue
        donors = [i for i in selected.index if i not in chosen and label in present[i]]
        if not donors:
            continue
        drop = min(chosen, key=lambda i: selected.score[i])
        take = max(donors, key=lambda i: selected.score[i])
        chosen[chosen.index(drop)] = take
        covered = set().union(*(present[i] for i in chosen))

    return selected.assign(batch=["pilot" if i in chosen else "extension" for i in selected.index])


def main():
    print("reading sources (bbox-filtered)", flush=True)
    counts, cache, regions, districts = cheap_pass()

    cheap = [((i, j), v) for (i, j), v in counts.items() if v["cotton"] >= MIN_COTTON_ACRES]
    print(f"cells with cotton: {len([1 for v in counts.values() if v['cotton'] > 0]):,}; "
          f"above {MIN_COTTON_ACRES:.0f} acres: {len(cheap):,}", flush=True)

    frame = pd.DataFrame([{**{f"{c}_acres": v[c] for c in CLASSES}} for _, v in cheap])
    frame["score"] = frame.apply(score, axis=1)
    order = frame.score.sort_values(ascending=False).index[:SHORTLIST]
    shortlist = [cheap[k] for k in order]
    print(f"exact clip on {len(shortlist)} shortlisted cells", flush=True)

    cells = exact_pass(shortlist, cache, regions)
    cells["district"] = [districts.get((i, j), "") for i, j in zip(cells.i, cells.j)]
    cells["tile_id"] = [f"cell_{i:04d}_{j:04d}" for i, j in zip(cells.i, cells.j)]

    keep_out = ~cells.geometry.intersects(val_mask())
    print(f"dropped {int((~keep_out).sum())} cells for touching a validation AOI "
          f"(+{VAL_BUFFER_DEG} deg)", flush=True)
    cells = cells[keep_out & (cells.cotton_acres >= MIN_COTTON_ACRES)].copy()

    cells["score"] = cells.apply(score, axis=1).round(3)
    cells["n_classes"] = (cells[[f"{c}_acres" for c in CLASSES]] > 0).sum(axis=1)
    cells["province"] = np.where(cells.lat < SINDH_LAT, "Sindh", "Punjab")

    selected = split_batches(select(cells))
    selected = selected.sort_values(["batch", "score"], ascending=[True, False])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    columns = ["tile_id", "batch", "province", "district", "lon", "lat", "score",
               "n_classes"] + [f"{c}_acres" for c in CLASSES] + ["geometry"]
    selected[columns].to_file(OUT, driver="GPKG")

    show = selected[["tile_id", "batch", "province", "district", "lon", "lat", "score"]
                    + [f"{c}_acres" for c in CLASSES]]
    for batch in ("pilot", "extension"):
        part = show[show.batch == batch]
        print(f"\n=== {batch} ({len(part)} cells) ===")
        print(part.to_string(index=False))
        totals = part[[f"{c}_acres" for c in CLASSES]].sum().round(0)
        print(f"acres: {totals.to_dict()}")
        print(f"Sindh {int((part.province == 'Sindh').sum())} / Punjab {int((part.province == 'Punjab').sum())}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
