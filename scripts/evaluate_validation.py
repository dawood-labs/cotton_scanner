"""Score the cotton map against the mill ground truth and the other-crop scans.

Two numbers, and they come from different sources because the sources say different
things:

  * recall  -- of the pixels the mill surveyed as cotton, how many did the model map as
    cotton. The mill data is positives only, so this is all it can tell us.
  * commission per crop -- of the pixels a rice / fall-maize / sugarcane / orchard scan
    claims, how many did the model map as cotton. Those pixels are cotton in nobody's
    account, so anything mapped there is a false positive with a name attached.

Both are computed on the sieved classification raster, at its own 10 m grid, so no
polygon is being compared against a differently-generalized polygon.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

BASE = Path("/home/jovyan/FAO/cotton")
RUNS = BASE / "validation_runs"
REFS = BASE / "validation_data/reference_crops"
COTTON_CLASS = 1
NODATA = 255
CLASS_ORDER = ["cotton", "rice", "sugarcane", "fall_maize", "orchard"]


def classification_raster(aoi_dir: Path) -> Path:
    """The sieved map is the pipeline's own product; fall back to the raw one."""
    ndvi_dirs = sorted(aoi_dir.glob("1_ndvi_run_*"))
    if not ndvi_dirs:
        raise FileNotFoundError(f"no NDVI run under {aoi_dir}")
    latest = ndvi_dirs[-1]
    sieved = list(latest.glob("*_sieved_*.tif"))
    return sieved[0] if sieved else next(latest.glob("*_rf_classification_map.tif"))


def score_one(slug: str) -> pd.DataFrame:
    raster_path = classification_raster(RUNS / slug)
    ref_path = REFS / f"{slug}.gpkg"
    refs = gpd.read_file(ref_path)

    with rasterio.open(raster_path) as src:
        classes = src.read(1)
        transform, shape, crs = src.transform, classes.shape, src.crs

    refs = refs.to_crs(crs)
    is_cotton = classes == COTTON_CLASS
    mapped = classes != NODATA

    rows = []
    for label in CLASS_ORDER:
        part = refs[refs["crop"] == label]
        if part.empty:
            continue
        # all_touched=False: a polygon owns the pixels whose centres it contains, which is
        # what keeps a thin field from claiming the road beside it.
        burned = rasterize(
            ((geom, 1) for geom in part.geometry if geom is not None and not geom.is_empty),
            out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=False,
        ).astype(bool)
        burned &= mapped                       # pixels outside the mapped area say nothing
        total = int(burned.sum())
        if total == 0:
            continue
        hit = int((burned & is_cotton).sum())
        rows.append({
            "aoi": slug,
            "reference": label,
            "ref_pixels": total,
            "ref_acres": round(total * 0.02471, 1),      # 10 m pixel = 0.02471 acres
            "mapped_cotton_pixels": hit,
            "pct_mapped_cotton": round(100 * hit / total, 1),
        })

    aoi_cotton_pct = round(100 * is_cotton.sum() / max(1, mapped.sum()), 1)
    for row in rows:
        row["aoi_cotton_pct"] = aoi_cotton_pct
    return pd.DataFrame(rows)


def main(slugs):
    frames = [score_one(s) for s in slugs if (RUNS / s).exists() and (REFS / f"{s}.gpkg").exists()]
    if not frames:
        print("nothing scored yet")
        return
    table = pd.concat(frames, ignore_index=True)
    out = BASE / "validation_runs/validation_scores.csv"
    table.to_csv(out, index=False)

    print(table.to_string(index=False))
    print()
    pivot = table.pivot(index="aoi", columns="reference", values="pct_mapped_cotton")
    print("percent of each reference class mapped as cotton")
    print(pivot.reindex(columns=[c for c in CLASS_ORDER if c in pivot.columns]).to_string())
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    default = ["al_moiz_2_1", "baba_fareed_1", "baba_fareed_2", "faran_1", "layyah_1"]
    main(sys.argv[1:] or default)
