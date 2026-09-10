"""Download the cotton validation datasets from GCS into a local folder.

Everything the validation work needs lives under one bucket prefix. This pulls
whole prefixes (a shapefile is never a single file) and skips anything already
present with a matching byte size, so it is safe to re-run.
"""
import sys
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

BUCKET = "farmdar_data_catalog"
KEY = "/home/jovyan/FAO/cotton/scripts/gcs_data_downloader_ee_farmdar.json"
DEST = Path("/home/jovyan/FAO/cotton/validation_data")

PREFIXES = [
    "fao_cotton_testing_data/cotton_val_aois/",
    "fao_cotton_testing_data/Al-Moiz-2-Cotton-2025/",
    "fao_cotton_testing_data/Baba-Fareed-Cotton-2025/",
    "fao_cotton_testing_data/Faran-Cotton-2025/",
    "fao_cotton_testing_data/Layyah-Cotton-2025/",
    "fao_cotton_testing_data/Corteva_Rice_2025_2nd_Scan/",
    "fao_cotton_testing_data/Corteva-Fall-Maize-2025/",
    "fao_cotton_testing_data/Sugarcane_3m-10m_Pakistan-Scan_2025/",
    "fao_cane_model_file/orchard_exclusion_mask/",
]


def main(prefixes):
    cred = service_account.Credentials.from_service_account_file(KEY)
    client = storage.Client(credentials=cred, project=cred.project_id)
    bucket = client.bucket(BUCKET)

    for prefix in prefixes:
        for blob in client.list_blobs(bucket, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            out = DEST / Path(blob.name).parent.name / Path(blob.name).name
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == blob.size:
                print(f"skip {out.name}", flush=True)
                continue
            print(f"get  {out.name} ({blob.size/1e6:.1f} MB)", flush=True)
            blob.download_to_filename(out)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or PREFIXES)
