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

Negatives are looked for somewhere else entirely. This is a per-pixel classifier, so a
rice pixel teaches the model what rice looks like regardless of whether cotton grows in
the same cell -- co-occurrence only buys shared atmosphere and shared acquisition dates,
which is second order. Cells scored for cotton therefore miss the ones that matter most:
those thick with rice and fall maize and holding no cotton at all. So the grid is
extended a second time over the rice and fall-maize scans' own extents, and ranked on
rice and maize acreage alone.

Output: training_tiles.gpkg, with a `batch` column. The pilot batch is the six cells we
acquire first; the extension batch is ranked and waiting; the negatives batch is the
rice/maize supply, independent of both.
"""
import argparse
import os
import sys
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

# Latitude alone gets the province wrong exactly where it matters. Rahim Yar Khan is
# Punjab but reaches below 28.5, so the proxy called a Punjab cell Sindh and the rice
# batch ended up entirely in Punjab. The orchard mask carries a real province attribute
# for 20 districts; where a cell touches one, that wins over the proxy.
SINDH_CERTAIN_LAT = 28.0   # unambiguous Sindh, used only to guarantee a Sindh shortlist

NEGATIVE_CLASSES = ("rice", "fall_maize")
N_NEGATIVES = 4
NEG_SHORTLIST = 40
NEG_SHORTLIST_SINDH = 20   # ...plus this many from below SINDH_CERTAIN_LAT, so the
                           # Sindh-first pick below has a pool to draw from at all
NEG_MIN_ACRES = 200.0      # a cell worth a download must carry a real block of the crop
# Wide, because these four are the whole rice/maize supply. Five cells is ~55 km, far
# enough that no two share a district or a sowing week.
NEG_SEPARATION = 5
NATIONAL_CHUNK_DEG = 2.0   # bbox-read stride over the national scans
# Walking both national scans takes ~15 minutes, and nothing downstream of it changes
# when the ranking rules do. Cached so re-runs are seconds; delete the file or pass
# --refresh after the scans themselves change.
NATIONAL_CACHE = BASE / "training_v2/tiles/national_cell_acres.parquet"


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


def national_acres(label: str):
    """Per-cell acreage for one national scan over its whole extent, read in strides.

    The scan is never opened whole -- pyogrio reports its extent from the header, and the
    file is then walked in NATIONAL_CHUNK_DEG blocks through its spatial index. A polygon
    straddling a block edge is returned by both reads, so each block keeps only the
    polygons whose centroid falls inside it; that is the same rule the cell counting uses,
    so nothing is counted twice and nothing is lost.
    """
    import pyogrio

    path = BASE / OTHER_SOURCES[label]
    minx, miny, maxx, maxy = pyogrio.read_info(path)["total_bounds"]
    counts = {}
    xs = np.arange(np.floor(minx), np.ceil(maxx), NATIONAL_CHUNK_DEG)
    ys = np.arange(np.floor(miny), np.ceil(maxy), NATIONAL_CHUNK_DEG)
    for x0 in xs:
        for y0 in ys:
            x1, y1 = x0 + NATIONAL_CHUNK_DEG, y0 + NATIONAL_CHUNK_DEG
            gdf = read_bbox(OTHER_SOURCES[label], (x0, y0, x1, y1))
            if gdf.empty:
                continue
            centroids = gdf.geometry.centroid
            cx, cy = centroids.x.values, centroids.y.values
            own = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
            if not own.any():
                continue
            gdf, cx, cy = gdf[own], cx[own], cy[own]
            area = acres(gdf).values
            ci, cj = cell_of(cx, cy)
            for (i, j), part in pd.DataFrame({"i": ci, "j": cj, "acres": area}).groupby(["i", "j"]):
                counts[(i, j)] = counts.get((i, j), 0.0) + float(part.acres.sum())
    print(f"  {label}: {len(counts):,} cells over its own extent", flush=True)
    return counts


def exact_cell(i, j):
    """Exact clipped acreage for one cell, read straight from the sources.

    Used for the negatives, which sit outside the cotton regions the cheap pass cached.
    A 0.1 deg bbox through a shapefile index is a few hundred milliseconds, so reading
    per cell is cheaper than caching a country.
    """
    geom = cell_box(i, j)
    bounds = geom.bounds
    row = {"i": i, "j": j, "geometry": geom,
           "lon": (i + 0.5) * CELL_DEG, "lat": (j + 0.5) * CELL_DEG,
           "district": "", "orchard_province": ""}
    for label in CLASSES:
        paths = list(COTTON_SOURCES.values()) if label == "cotton" else [OTHER_SOURCES[label]]
        total = 0.0
        for path in paths:
            keep = ["district", "province"] if label == "orchard" else None
            gdf = read_bbox(path, bounds, keep=keep)
            if gdf.empty:
                continue
            if label == "orchard" and "district" in gdf and not row["district"]:
                modes = gdf["district"].mode()
                row["district"] = modes.iloc[0] if len(modes) else ""
                pmodes = gdf["province"].mode() if "province" in gdf else []
                row["orchard_province"] = pmodes.iloc[0] if len(pmodes) else ""
            clipped = gpd.clip(gdf, geom)
            clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
            if not clipped.empty:
                total += float(acres(clipped).sum())
        row[f"{label}_acres"] = round(total, 1)
    return row


def resolve_province(orchard_province, lat):
    """Real province where the orchard mask knows it, latitude proxy everywhere else."""
    known = str(orchard_province or "").strip()
    return known if known in ("Punjab", "Sindh") else ("Sindh" if lat < SINDH_LAT else "Punjab")


def negative_score(row):
    """Rice and maize only. Cotton is deliberately not in this score."""
    utility = {c: min(row[f"{c}_acres"], TARGET_ACRES) / TARGET_ACRES for c in NEGATIVE_CLASSES}
    return sum(np.sqrt(u) for u in utility.values())


def national_counts(refresh: bool):
    """Per-cell rice and maize acreage nationally, cached to parquet."""
    if NATIONAL_CACHE.exists() and not refresh:
        cached = pd.read_parquet(NATIONAL_CACHE)
        print(f"  reusing {NATIONAL_CACHE.name} ({len(cached):,} cells)", flush=True)
        return {(int(r.i), int(r.j)): {c: float(getattr(r, c)) for c in CLASSES}
                for r in cached.itertuples()}

    counts = {}
    for label in NEGATIVE_CLASSES:
        for cell, value in national_acres(label).items():
            counts.setdefault(cell, {c: 0.0 for c in CLASSES})[label] = value
    NATIONAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"i": i, "j": j, **v} for (i, j), v in counts.items()]).to_parquet(NATIONAL_CACHE)
    return counts


def pick_negatives(exclude, mask, refresh=False):
    """The rice/maize cells, ranked on rice and maize alone and spread across the country."""
    counts = national_counts(refresh)

    cheap = [(cell, v) for cell, v in counts.items()
             if sum(v[c] for c in NEGATIVE_CLASSES) >= NEG_MIN_ACRES]
    frame = pd.DataFrame([{f"{c}_acres": v[c] for c in CLASSES} for _, v in cheap])
    frame["score"] = frame.apply(negative_score, axis=1)
    frame["lat"] = [(j + 0.5) * CELL_DEG for (_, j), _ in cheap]
    ranked = frame.score.sort_values(ascending=False)
    # Top cells nationally, plus the top cells that are unambiguously in Sindh. Without
    # the second half the national ranking is all Punjab and the Sindh pick below has
    # nothing to choose from.
    order = list(ranked.index[:NEG_SHORTLIST])
    sindh_rank = frame[frame.lat < SINDH_CERTAIN_LAT].score.sort_values(ascending=False)
    order += [k for k in sindh_rank.index[:NEG_SHORTLIST_SINDH] if k not in order]

    rows = []
    for k in order:
        (i, j), _ = cheap[k]
        if f"cell_{i:04d}_{j:04d}" in exclude:
            continue
        rows.append(exact_cell(i, j))
    cells = gpd.GeoDataFrame(rows, crs=4326)
    cells["tile_id"] = [f"cell_{i:04d}_{j:04d}" for i, j in zip(cells.i, cells.j)]
    cells = cells[~cells.geometry.intersects(mask)].copy()
    cells["score"] = cells.apply(negative_score, axis=1).round(3)
    cells["n_classes"] = (cells[[f"{c}_acres" for c in CLASSES]] > 0).sum(axis=1)
    cells["province"] = [resolve_province(p, lat)
                         for p, lat in zip(cells.orchard_province, cells.lat)]
    cells = cells.drop(columns=["orchard_province"]).sort_values("score", ascending=False)

    # One from each province before anything else. Punjab and Sindh transplant rice weeks
    # apart, so four cells from one province would teach the model one rice curve and
    # call it rice.
    chosen = []
    for province in ("Punjab", "Sindh"):
        chosen = greedy(cells[cells.province == province], len(chosen) + 1, chosen, NEG_SEPARATION)
    for separation in (NEG_SEPARATION, 3, 2):
        chosen = greedy(cells, N_NEGATIVES, chosen, separation)
        if len(chosen) >= N_NEGATIVES:
            break
    return cells.loc[[c.name for c in chosen]].assign(batch="negatives")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-walk the national scans instead of using the cached counts")
    args = parser.parse_args()

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

    mask = val_mask()
    keep_out = ~cells.geometry.intersects(mask)
    print(f"dropped {int((~keep_out).sum())} cells for touching a validation AOI "
          f"(+{VAL_BUFFER_DEG} deg)", flush=True)
    cells = cells[keep_out & (cells.cotton_acres >= MIN_COTTON_ACRES)].copy()

    cells["score"] = cells.apply(score, axis=1).round(3)
    cells["n_classes"] = (cells[[f"{c}_acres" for c in CLASSES]] > 0).sum(axis=1)
    cells["province"] = np.where(cells.lat < SINDH_LAT, "Sindh", "Punjab")

    selected = split_batches(select(cells))

    print("\nextending the grid over the rice and fall-maize scans", flush=True)
    negatives = pick_negatives(set(selected.tile_id), mask, refresh=args.refresh)
    selected = pd.concat([selected, negatives], ignore_index=True)
    selected = gpd.GeoDataFrame(selected, crs=4326).sort_values(
        ["batch", "score"], ascending=[True, False])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    columns = ["tile_id", "batch", "province", "district", "lon", "lat", "score",
               "n_classes"] + [f"{c}_acres" for c in CLASSES] + ["geometry"]
    selected[columns].to_file(OUT, driver="GPKG")

    show = selected[["tile_id", "batch", "province", "district", "lon", "lat", "score"]
                    + [f"{c}_acres" for c in CLASSES]]
    for batch in ("pilot", "extension", "negatives"):
        part = show[show.batch == batch]
        print(f"\n=== {batch} ({len(part)} cells) ===")
        print(part.to_string(index=False))
        totals = part[[f"{c}_acres" for c in CLASSES]].sum().round(0)
        print(f"acres: {totals.to_dict()}")
        print(f"Sindh {int((part.province == 'Sindh').sum())} / Punjab {int((part.province == 'Punjab').sum())}")
    pilot_rice = (selected[(selected.batch == "pilot") & (selected.rice_acres > 0)])
    print(f"\nnote: the pilot's rice comes from {len(pilot_rice)} cell(s) "
          f"({', '.join(pilot_rice.tile_id)}); the negatives batch is the real rice supply")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
    # GDAL/PROJ intermittently aborts in a C++ static destructor after main() has
    # returned and everything is already written ("terminate called without an active
    # exception", roughly one run in three on this box, always with no Python frame on
    # the stack). That turns a finished run into exit 134, which a caller reads as
    # failure. Flush and leave without running the C++ teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
