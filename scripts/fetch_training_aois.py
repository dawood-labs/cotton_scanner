"""Pull only the training AOIs that carry labelled ground, and only their NDVI grids.

The GEE export for all 23 cotton training AOIs is 14.7 GB. This box has about 30 GB
free and a validation cache to keep, so taking the lot is not sensible when nine of the
AOIs contribute almost nothing labelled. `aoi_label_coverage.csv` says how many acres of
each class sit inside each AOI once the validation AOIs are cut out of it; this fetches
the ones worth the disk.
"""
import sys
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

BUCKET = "farmdar_data_catalog"
PREFIX = "fao_cotton_training_data_2"
KEY = "/home/jovyan/FAO/cotton/scripts/gcs_data_downloader_ee_farmdar.json"
DEST = Path("/home/jovyan/FAO/cotton/training_v2/gee")

# Chosen from aoi_label_coverage.csv: every AOI holding surveyed cotton, plus the ones
# that carry the bulk of the rice, sugarcane, fall maize and orchard, spread across both
# provinces so neither sowing calendar is under-represented.
SELECTED = [1, 5, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23]


def main(clusters):
    cred = service_account.Credentials.from_service_account_file(KEY)
    client = storage.Client(credentials=cred, project=cred.project_id)

    total = 0
    for cluster in clusters:
        folder = f"AOI_{cluster}_5000_cotton"
        out_dir = DEST / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        got = 0
        for blob in client.list_blobs(BUCKET, prefix=f"{PREFIX}/{folder}/"):
            out = out_dir / Path(blob.name).name
            if out.exists() and out.stat().st_size == blob.size:
                continue
            blob.download_to_filename(out)
            got += 1
            total += blob.size
        print(f"{folder}: {got} new files", flush=True)
    print(f"downloaded {total/1e9:.2f} GB into {DEST}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SELECTED)
