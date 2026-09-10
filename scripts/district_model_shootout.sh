#!/usr/bin/env bash
# Score every model on one whole district, on ground none of them was trained on.
#
# Five full inferences over 79 tiles would be ten hours on two cores, and would answer
# less than this does: the maps only say cotton or not, while sampled curves let every
# model be read at matched recall. So the district's imagery is sampled inside the
# surveyed polygons once, and each model is scored against that.
set -u
cd /home/jovyan/FAO/cotton
NAME=${1:-Layyah}
slug=$(echo "$NAME" | tr 'A-Z' 'a-z')_district

echo "=== WAIT FOR TILES $(date -Is) ==="
while pgrep -f "run.py --crop cotton --year 2025 --district $NAME" >/dev/null; do sleep 60; done
echo "=== DISTRICT RUN FINISHED $(date -Is) ==="

echo "=== REFERENCE LAYERS $(date -Is) ==="
python - "$NAME" "$slug" <<'PY'
import sys
from pathlib import Path
import geopandas as gpd, pandas as pd
sys.path.insert(0, "/home/jovyan/FAO/cotton/cotton_scanner/scripts")
import build_reference_layers as brl

name, slug = sys.argv[1], sys.argv[2]
aoi = gpd.read_file(f"/home/jovyan/FAO/cotton/timeseries_model_val_data/{name}/{name}.shp").to_crs(4326)
geom = aoi.geometry.union_all()
brl.OUT.mkdir(parents=True, exist_ok=True)
parts = []
cotton = brl.clip_to(brl.COTTON_GT[f"{name}-1"], geom, tuple(geom.bounds), "cotton") \
    if f"{name}-1" in brl.COTTON_GT else None
if cotton is not None:
    parts.append(cotton)
for label, path in brl.SOURCES.items():
    part = brl.clip_to(path, geom, tuple(geom.bounds), label)
    if part is not None:
        parts.append(part)
merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
merged.to_file(brl.OUT / f"{slug}.gpkg", driver="GPKG")
print("wrote", brl.OUT / f"{slug}.gpkg")
PY

echo "=== SAMPLE CURVES $(date -Is) ==="
mkdir -p validation_data/district_curves
ln -sfn /home/jovyan/FAO/cotton/district_runs/"$NAME" /home/jovyan/FAO/cotton/district_runs/"$slug"
python cotton_scanner/scripts/sample_reference_curves.py "$slug" \
    --runs-dir /home/jovyan/FAO/cotton/district_runs \
    --refs-dir /home/jovyan/FAO/cotton/validation_data/reference_crops \
    --out-dir /home/jovyan/FAO/cotton/validation_data/district_curves

echo "=== SHOOTOUT $(date -Is) ==="
MODELS="v1=/home/jovyan/FAO/cotton/timeseries_model/model_v1/best_rf_classifier.joblib"
for m in model_v2 model_v2_mined model_v2_no_other model_v2_clean_other model_v2_tuned; do
    f=/home/jovyan/FAO/cotton/timeseries_model/$m/cotton_rf_v2.joblib
    [ -f "$f" ] && MODELS="$MODELS ${m#model_}=$f"
done
echo "models: $MODELS"
python cotton_scanner/scripts/compare_at_matched_recall.py --models $MODELS \
    --curves /home/jovyan/FAO/cotton/validation_data/district_curves --csv validation_runs/district_shootout.csv
echo "=== SHOOTOUT DONE $(date -Is) ==="
