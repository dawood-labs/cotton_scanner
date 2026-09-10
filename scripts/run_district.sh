#!/usr/bin/env bash
# Full-district run with a timestamp on every phase.
#
# The per-AOI runs were 2 to 9 tiles each, which is small enough to hide anything that
# only shows up at scale -- a pool that deadlocks, a mosaic that will not fit in memory,
# a per-tile cost that is fine at 4 tiles and ruinous at 60. This is the run that finds
# those, so it is timed at every step rather than only end to end.
set -u
export PYTHONPATH=/home/jovyan/shared/git/standard-libraries/.worktrees/824850c677f49ef5b23af6040e9d2b165e586996
CROPSTACK=/home/jovyan/FAO/optimized_code_testing/cropstack
PY=$CROPSTACK/../cropstack_venv/bin/python
NAME=${1:-Layyah}
AOI=${2:-/home/jovyan/FAO/cotton/timeseries_model_val_data/$NAME/$NAME.shp}
OUT=/home/jovyan/FAO/cotton/district_runs/$NAME

mkdir -p "$OUT"
echo "=== START $NAME $(date -Is) ==="
df -h /home/jovyan | tail -1
cd "$CROPSTACK" || exit 1
"$PY" run.py --crop cotton --year 2025 --district "$NAME" \
    --aoi "$AOI" --out "$OUT" \
    --key /home/jovyan/FAO/cotton/scripts/gcs_data_downloader_ee_farmdar.json \
    --set run_static_model=false \
    --set ndvi_worker_count=2 \
    --set delete_raw_ndvi_tiles=false
status=$?
echo "=== END $NAME exit=$status $(date -Is) ==="
df -h /home/jovyan | tail -1
du -sh "$OUT" 2>/dev/null
