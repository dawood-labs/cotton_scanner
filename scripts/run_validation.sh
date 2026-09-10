#!/usr/bin/env bash
# Run the cotton NDVI pipeline over every validation AOI, smallest first.
# Sequential on purpose: this box has 2 cores, so two districts at once only
# makes both slower and risks the memory budget.
set -u
export PYTHONPATH=/home/jovyan/shared/git/standard-libraries/.worktrees/824850c677f49ef5b23af6040e9d2b165e586996
CROPSTACK=/home/jovyan/FAO/optimized_code_testing/cropstack
PY=/home/jovyan/FAO/optimized_code_testing/cropstack_venv/bin/python
AOIS=/home/jovyan/FAO/cotton/validation_data/aois
KEY=/home/jovyan/FAO/cotton/scripts/gcs_data_downloader_ee_farmdar.json
OUT=/home/jovyan/FAO/cotton/validation_runs

cd "$CROPSTACK" || exit 1
for name in "$@"; do
    echo "=================== $name  $(date -Is) ==================="
    "$PY" run.py --crop cotton --year 2025 --district "$name" \
        --aoi "$AOIS/$name.gpkg" --key "$KEY" --out "$OUT/$name" \
        --set run_static_model=false \
        --set ndvi_worker_count=2 \
        --set delete_raw_ndvi_tiles=false
    status=$?
    echo "--- $name exit=$status $(date -Is)"
    # Park the run in GCS as soon as it is finished. The raw Sentinel tiles are the only
    # expensive thing here; caching them means a changed window or model costs compute,
    # not another download. Local copies are kept while there is room to keep them.
    if [ $status -eq 0 ]; then
        python /home/jovyan/FAO/cotton/cotton_scanner/scripts/gcs_cache.py push "$OUT/$name" \
            || echo "--- $name: cache push failed, artifacts are still local"
    fi
done
