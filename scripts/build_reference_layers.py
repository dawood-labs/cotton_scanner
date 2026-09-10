"""Clip the other-crop scans and the orchard mask to the cotton validation AOIs.

The mill ground truth only marks cotton, so it can measure recall and nothing else.
The rice / fall-maize / sugarcane scans and the orchard mask supply the other half:
areas that are known NOT to be cotton, which is where the false positives live.

Each source is read with a bounding-box filter so the 2 GB national scans never come
into memory whole. Output: one GeoPackage per AOI with a `crop` column.
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE = Path("/home/jovyan/FAO/cotton")
AOIS = BASE / "validation_data/cotton_val_aois/cotton_val_aois.shp"
# An extra AOI built for this work rather than supplied: the densest overlap of orchard
# blocks and surveyed cotton inside Layyah district. Orchards carry a cotton-like NDVI
# curve, and none of the five mill AOIs contains enough of them to test that.
EXTRA_AOIS = BASE / "validation_data/aois/layyah_orchards.gpkg"
OUT = BASE / "validation_data/reference_crops"

SOURCES = {
    "rice": "validation_data/Corteva_Rice_2025_2nd_Scan/Corteva_Rice_2025_2nd_Scan.shp",
    "fall_maize": "validation_data/Corteva-Fall-Maize-2025/Corteva-Fall-Maize-2025.shp",
    "sugarcane": "validation_data/Sugarcane_3m-10m_Pakistan-Scan_2025/Sugarcane_3m-10m_Pakistan-Scan_2025.shp",
    "orchard": "validation_data/orchard_exclusion_mask/orchard_exclusion_mask.gpkg",
}
COTTON_GT = {
    "Al-Moiz-2-1": "validation_data/Al-Moiz-2-Cotton-2025/Al-Moiz-2-Cotton-2025.shp",
    "Baba-Fareed-1": "validation_data/Baba-Fareed-Cotton-2025/Baba-Fareed-Cotton-2025.shp",
    "Baba-Fareed-2": "validation_data/Baba-Fareed-Cotton-2025/Baba-Fareed-Cotton-2025.shp",
    "Faran-1": "validation_data/Faran-Cotton-2025/Faran-Cotton-2025.shp",
    "Layyah-1": "validation_data/Layyah-Cotton-2025/Layyah-Cotton-2025.shp",
    "Layyah-orchards": "validation_data/Layyah-Cotton-2025/Layyah-Cotton-2025.shp",
}


def slug(mill):
    return mill.replace("-", "_").lower()


def clip_to(path, geom, bounds, label):
    """Read only what the bbox touches, then clip hard to the AOI polygon."""
    try:
        part = gpd.read_file(BASE / path, bbox=bounds)
    except Exception as exc:                       # a source that is absent is not fatal
        print(f"    {label}: unreadable ({exc})", flush=True)
        return None
    if part.empty:
        print(f"    {label}: 0", flush=True)
        return None
    part = part.to_crs(4326)
    # The national scans carry self-intersecting rings that make GEOS refuse the clip.
    # make_valid is the cheapest repair that keeps the polygon where it was.
    invalid = ~part.geometry.is_valid
    if invalid.any():
        part.loc[invalid, "geometry"] = part.loc[invalid, "geometry"].make_valid()
        part = part[part.geometry.geom_type.isin(("Polygon", "MultiPolygon"))]
    part = gpd.clip(part, geom)
    part = part[~part.geometry.is_empty & part.geometry.notna()]
    if part.empty:
        print(f"    {label}: 0", flush=True)
        return None
    keep = part[["geometry"]].copy()
    keep["crop"] = label
    utm = keep.to_crs(keep.estimate_utm_crs())
    print(f"    {label}: {len(keep)} feats, {utm.area.sum()/4046.86:,.0f} acres", flush=True)
    return keep


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    aois = gpd.read_file(AOIS)
    if EXTRA_AOIS.exists():
        aois = pd.concat([aois, gpd.read_file(EXTRA_AOIS)], ignore_index=True)
    for _, row in aois.iterrows():
        mill = row["mill"]
        if only and slug(mill) not in only:
            continue
        print(f"AOI {mill}", flush=True)
        bounds = tuple(row.geometry.bounds)
        parts = []
        cotton = clip_to(COTTON_GT[mill], row.geometry, bounds, "cotton")
        if cotton is not None:
            parts.append(cotton)
        for label, path in SOURCES.items():
            part = clip_to(path, row.geometry, bounds, label)
            if part is not None:
                parts.append(part)
        if not parts:
            print("    nothing to write", flush=True)
            continue
        merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
        merged.to_file(OUT / f"{slug(mill)}.gpkg", driver="GPKG")
        print(f"    -> {OUT / f'{slug(mill)}.gpkg'}", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
