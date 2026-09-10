"""Self-test for sample_training_pixels on a synthetic tile, no imagery required.

The three rules that cost samples -- erosion, ambiguous-pixel dropping, and the class
codes -- are all silent when they misfire: the parquet still has rows and the model still
trains. So they get checked against a tile whose right answer can be counted by hand.

Layout, on a 20x20 pixel tile at 10 m:
  cotton    rows/cols  2..7   (6x6, so 4x4 = 16 pixels survive a 1-pixel erosion)
  rice      rows/cols 12..17  (6x6 -> 16 pixels)
  sugarcane rows/cols 12..17 but shifted 3 columns, so it overlaps rice
Every pixel where the eroded rice and eroded sugarcane masks meet must be dropped.
"""
import shutil
import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_training_pixels as S  # noqa: E402

SIZE = 20
PIXEL_DEG = 0.0001            # ~11 m at this latitude; the test never leaves 4326
ORIGIN = (71.0, 31.0)
DATES = ["2025-04-01", "2025-04-09", "2025-04-17"]


def px_box(r0, c0, r1, c1, transform):
    """Pixel index rectangle (inclusive) as a polygon, inset so rasterize is unambiguous."""
    x0, y0 = transform * (c0 + 0.1, r0 + 0.1)
    x1, y1 = transform * (c1 + 0.9, r1 + 0.9)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def build(root: Path):
    transform = from_origin(ORIGIN[0], ORIGIN[1], PIXEL_DEG, PIXEL_DEG)
    tile_dir = root / "tiles/raw/cell_test"
    tile_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    red = rng.integers(400, 600, size=(len(DATES), SIZE, SIZE)).astype("uint16")
    nir = rng.integers(2000, 4000, size=(len(DATES), SIZE, SIZE)).astype("uint16")

    names, stack = [], []
    for d, date in enumerate(DATES):
        tag = date.replace("-", "_")
        stack.append(red[d]); names.append(f"red_{tag}")
        stack.append(nir[d]); names.append(f"nir_{tag}")

    tif = tile_dir / "sentinel_10m_tile_0001.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=SIZE, width=SIZE,
                       count=len(names), dtype="uint16", crs="EPSG:4326",
                       transform=transform, tiled=True,
                       blockxsize=16, blockysize=16) as dst:
        for i, (band, name) in enumerate(zip(stack, names), start=1):
            dst.write(band, i)
            dst.set_band_description(i, name)

    shapes = {
        "cotton": px_box(2, 2, 7, 7, transform),
        "rice": px_box(12, 12, 17, 17, transform),
        "sugarcane": px_box(12, 15, 17, 19, transform),   # overlaps rice on cols 15..17
    }
    sources = {}
    for crop, geom in shapes.items():
        path = root / f"{crop}.gpkg"
        gpd.GeoDataFrame([{"crop": crop}], geometry=[geom], crs=4326).to_file(path, driver="GPKG")
        sources[crop] = [str(path)]
    return tif, sources


def eroded_pixels(r0, c0, r1, c1):
    """Which pixels a rectangle keeps after a 1-pixel erosion."""
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def main():
    root = Path(tempfile.mkdtemp(prefix="sample_pixels_test_"))
    failures = []
    try:
        tif, sources = build(root)
        table = S.sample_tile(tif, sources, max_per_class=10_000)

        expected_cols = DATES + ["label", "crop", "lon", "lat", "tile"]
        if set(table.columns) != set(expected_cols):
            failures.append(f"columns {sorted(table.columns)} != {sorted(expected_cols)}")

        cotton = eroded_pixels(2, 2, 7, 7)
        rice = eroded_pixels(12, 12, 17, 17)
        cane = eroded_pixels(12, 15, 17, 19)
        overlap = rice & cane

        counts = table.crop.value_counts().to_dict()
        for crop, expect in (("cotton", len(cotton)),
                             ("rice", len(rice - overlap)),
                             ("sugarcane", len(cane - overlap))):
            if counts.get(crop, 0) != expect:
                failures.append(f"{crop}: {counts.get(crop, 0)} rows, expected {expect}")

        if len(overlap) == 0:
            failures.append("test is not exercising ambiguity: masks do not overlap")
        if len(table) != len(cotton) + len(rice - overlap) + len(cane - overlap):
            failures.append(f"total rows {len(table)} includes something unexpected")

        # Erosion, stated as the property rather than the count: the un-eroded cotton
        # rectangle is 36 pixels, the eroded one 16. A regression that skipped erosion
        # would sail past the column checks.
        if counts.get("cotton", 0) >= 36:
            failures.append("cotton mask was not eroded")

        codes = table.groupby("crop").label.unique().to_dict()
        for crop, code in (("cotton", 1), ("rice", 2), ("sugarcane", 3)):
            if list(codes.get(crop, [])) != [code]:
                failures.append(f"{crop} code {codes.get(crop)} != [{code}]")
        if S.CLASS_CODES != {"cotton": 1, "rice": 2, "sugarcane": 3, "fall_maize": 5, "orchard": 6}:
            failures.append(f"class codes changed: {S.CLASS_CODES}")

        if table.tile.unique().tolist() != ["cell_test"]:
            failures.append(f"tile column {table.tile.unique().tolist()}")
        if not table[DATES].notna().all().all():
            failures.append("NDVI columns contain NaN")
        # NIR was drawn far above red, so every synthetic pixel must smooth to high NDVI.
        if not (table[DATES].values > 0.5).all():
            failures.append("smoothed NDVI is not where the synthetic reflectance puts it")

        lon_ok = table.lon.between(ORIGIN[0], ORIGIN[0] + SIZE * PIXEL_DEG).all()
        lat_ok = table.lat.between(ORIGIN[1] - SIZE * PIXEL_DEG, ORIGIN[1]).all()
        if not (lon_ok and lat_ok):
            failures.append("lon/lat fall outside the tile")

        # Round-trip through parquet, since that is the actual deliverable.
        out = root / "labelled_pixels.parquet"
        table.to_parquet(out)
        if not pd.read_parquet(out).equals(table):
            failures.append("parquet round-trip changed the table")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print("sample_training_pixels self-test passed")
    return 0


if __name__ == "__main__":
    code = main()
    # GDAL/PROJ intermittently aborts in a C++ static destructor after main() has
    # returned and everything is already written ("terminate called without an active
    # exception", roughly one run in three on this box, always with no Python frame on
    # the stack). That turns a finished run into exit 134, which a caller reads as
    # failure. Flush and leave without running the C++ teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
