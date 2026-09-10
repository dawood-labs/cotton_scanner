"""Park finished artifacts in GCS so a rerun never re-downloads the same imagery.

The expensive thing in this project is Sentinel acquisition, not compute: a 0.1 degree
tile is a few minutes of download and about 175 MB on disk. This box has 99 GB, so the
raw tiles cannot all stay local, and deleting them means paying the download again the
next time a window, a model or a label changes -- which is every time.

So: push what is done, keep a local copy while there is room, and pull it back on demand.
Sizes are compared rather than hashes; the acquisition writes a tile once and never
edits it, so a matching size means a matching file.
"""
import argparse
import sys
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

BUCKET = "farmdar_data_catalog"
PREFIX = "fao_cotton_scanner_cache"
KEY = "/home/jovyan/FAO/cotton/scripts/gcs_data_downloader_ee_farmdar.json"
LOCAL_ROOT = Path("/home/jovyan/FAO/cotton")
# Below this much free disk, pushing also deletes the local copy.
FREE_GIB_FLOOR = 12.0


def client():
    cred = service_account.Credentials.from_service_account_file(KEY)
    return storage.Client(credentials=cred, project=cred.project_id)


def free_gib() -> float:
    import shutil
    return shutil.disk_usage(LOCAL_ROOT).free / 2**30


def remote_name(path: Path) -> str:
    return f"{PREFIX}/{path.relative_to(LOCAL_ROOT)}"


def push(paths, evict: bool | None):
    """Upload every file under `paths`. `evict=None` means decide from free disk."""
    bucket = client().bucket(BUCKET)
    if evict is None:
        evict = free_gib() < FREE_GIB_FLOOR
        if evict:
            print(f"only {free_gib():.1f} GiB free, local copies will be removed after upload")

    sent = skipped = 0
    for root in paths:
        root = Path(root).resolve()
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            blob = bucket.blob(remote_name(path))
            if blob.exists() and blob.size == path.stat().st_size:
                skipped += 1
            else:
                blob.upload_from_filename(path)
                sent += 1
                print(f"push {remote_name(path)} ({path.stat().st_size/1e6:.0f} MB)", flush=True)
            if evict:
                path.unlink()
    print(f"pushed {sent}, already cached {skipped}, free now {free_gib():.1f} GiB")


def pull(paths):
    bucket = client().bucket(BUCKET)
    got = 0
    for root in paths:
        root = Path(root).resolve()
        for blob in client().list_blobs(BUCKET, prefix=remote_name(root)):
            local = LOCAL_ROOT / Path(blob.name).relative_to(PREFIX)
            if local.exists() and local.stat().st_size == blob.size:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local)
            got += 1
            print(f"pull {local} ({blob.size/1e6:.0f} MB)", flush=True)
    print(f"pulled {got}, free now {free_gib():.1f} GiB")


def listing(paths):
    total = 0
    for root in paths or [LOCAL_ROOT]:
        prefix = remote_name(Path(root).resolve()) if Path(root).resolve() != LOCAL_ROOT else PREFIX
        for blob in client().list_blobs(BUCKET, prefix=prefix):
            total += blob.size
            print(f"{blob.size/1e6:9.1f} MB  {blob.name}")
    print(f"cached total {total/1e9:.2f} GB, free locally {free_gib():.1f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["push", "pull", "list", "free"])
    parser.add_argument("paths", nargs="*", help="local paths under /home/jovyan/FAO/cotton")
    parser.add_argument("--evict", dest="evict", action="store_true",
                        help="delete the local copy after a successful upload")
    parser.add_argument("--keep", dest="evict", action="store_false",
                        help="never delete the local copy")
    parser.set_defaults(evict=None)
    args = parser.parse_args()

    if args.action == "free":
        print(f"{free_gib():.1f} GiB free under {LOCAL_ROOT}")
    elif args.action == "push":
        push(args.paths or [LOCAL_ROOT / "validation_runs"], args.evict)
    elif args.action == "pull":
        pull(args.paths or [LOCAL_ROOT / "validation_runs"])
    else:
        listing(args.paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
