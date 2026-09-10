"""Erase mapped cotton wherever the orchard mask says orchard.

Orchards carry an NDVI curve close enough to cotton that the classifier takes them, and
no amount of retraining on field crops fixes a perennial that simply looks like one. The
orchard blocks are surveyed independently, for 20 districts, so where the mask has an
opinion it beats the model's.

Applied to the raster rather than the polygons: a block that covers most of a mapped
field should shrink that field, not delete it or survive it whole, and only the pixel
grid can express that.

Reports acres removed and acres left, because a mask that removes almost nothing and a
mask that removes almost everything both mean something went wrong.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

PIXEL_ACRES = 0.02471          # 10 m x 10 m
DEFAULT_MASK = Path("/home/jovyan/FAO/cotton/validation_data/orchard_exclusion_mask/orchard_exclusion_mask.gpkg")


def apply_mask(raster_path: Path, mask_path: Path, out_path: Path,
               cotton_class: int, background_class: int, nodata: int,
               aoi_path: Path | None = None) -> dict:
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        classes = src.read(1)
        bounds, transform, crs = src.bounds, src.transform, src.crs

    blocks = gpd.read_file(mask_path, bbox=tuple(bounds))
    is_cotton = classes == cotton_class

    # The classification raster covers the whole 0.1 degree tile grid, which overhangs the
    # AOI -- on Faran-1 that is 11,679 raster acres against 3,278 inside the AOI. Counting
    # the overhang would make the mask look like it removed far less than it did.
    if aoi_path is not None:
        from rasterio.features import geometry_mask
        aoi = gpd.read_file(aoi_path).to_crs(crs)
        scored = geometry_mask(aoi.geometry, out_shape=classes.shape,
                               transform=transform, invert=True)
    else:
        scored = np.ones(classes.shape, dtype=bool)

    before = int((is_cotton & scored).sum())

    if blocks.empty:
        removed = 0
        print(f"orchard mask has no blocks over {raster_path.name} -- nothing to remove")
    else:
        blocks = blocks.to_crs(crs)
        # all_touched: an orchard's canopy bleeds into the pixels its outline clips, and
        # those edge pixels are exactly the ones that get mapped as cotton.
        orchard = rasterize(
            ((geom, 1) for geom in blocks.geometry if geom is not None and not geom.is_empty),
            out_shape=classes.shape, transform=transform, fill=0, dtype="uint8",
            all_touched=True,
        ).astype(bool)
        hit = is_cotton & orchard
        removed = int((hit & scored).sum())
        classes[hit] = background_class          # masked everywhere, counted inside the AOI

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(compress="lzw", tiled=True, blockxsize=256, blockysize=256, bigtiff="YES")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(classes, 1)
        dst.set_band_description(1, "RF_Classification_orchard_masked")

    stats = {
        "raster": str(raster_path),
        "cotton_acres_before": round(before * PIXEL_ACRES, 1),
        "cotton_acres_removed": round(removed * PIXEL_ACRES, 1),
        "cotton_acres_after": round((before - removed) * PIXEL_ACRES, 1),
        "removed_pct": round(100 * removed / before, 1) if before else 0.0,
        "orchard_blocks": int(len(blocks)),
        "output": str(out_path),
    }
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rasters", nargs="+", type=Path)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--cotton-class", type=int, default=1)
    parser.add_argument("--background-class", type=int, default=4)
    parser.add_argument("--nodata", type=int, default=255)
    parser.add_argument("--aoi", type=Path, default=None,
                        help="count acres inside this AOI only; the raster overhangs it")
    parser.add_argument("--suffix", default="_orchard_masked")
    args = parser.parse_args()

    for raster in args.rasters:
        out = raster.parent / f"{raster.stem}{args.suffix}{raster.suffix}"
        print(raster.name)
        apply_mask(raster, args.mask, out, args.cotton_class,
                   args.background_class, args.nodata, args.aoi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
