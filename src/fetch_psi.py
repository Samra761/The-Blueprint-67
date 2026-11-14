"""Phase -1: download NASA PSI telemetry CSVs from S3 into data/raw/.

Owner: Wania (Data Engineer)

Run from the project root:
    cp .env.example .env        # then fill in bucket / prefix
    python src/fetch_psi.py                 # download everything under the prefix
    python src/fetch_psi.py --material Kapton --dry-run

Layout produced (what src/load.py expects):
    data/raw/<material>/HKPData_gmt<day>_<n>.csv

TEMPLATE NOTE: bucket and prefix come from .env. Keys under the prefix are
expected to look like  <prefix><material>/<file>.csv ; adjust KEY_TO_MATERIAL
if your mirror is organised differently.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MATERIALS = ["Cotton Lamp wick", "Kapton", "Pyrell", "Silicone", "Teflon"]


def s3_client():
    anonymous = os.getenv("PSI_S3_ANONYMOUS", "1") == "1"
    cfg = Config(signature_version=UNSIGNED) if anonymous else None
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"), config=cfg)


def KEY_TO_MATERIAL(key: str, prefix: str) -> str | None:
    """'SAME/Kapton/HKPData_gmt214_1.csv' -> 'Kapton' (None if not a known material)."""
    parts = key[len(prefix):].split("/")
    return parts[0] if len(parts) >= 2 and parts[0] in MATERIALS else None


def list_csv_keys(s3, bucket: str, prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".csv"):
                yield obj["Key"], obj["Size"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--material", choices=MATERIALS, help="only this material")
    ap.add_argument("--dry-run", action="store_true", help="list, don't download")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    bucket = os.environ.get("PSI_S3_BUCKET")
    prefix = os.getenv("PSI_S3_PREFIX", "")
    if not bucket or bucket == "your-psi-bucket-name":
        raise SystemExit("Set PSI_S3_BUCKET in .env (see .env.example).")

    s3 = s3_client()
    n, total = 0, 0
    for key, size in list_csv_keys(s3, bucket, prefix):
        material = KEY_TO_MATERIAL(key, prefix)
        if material is None or (args.material and material != args.material):
            continue
        dest = RAW / material / Path(key).name
        if dest.exists() and not args.force:
            continue
        print(f"{'[dry] ' if args.dry_run else ''}{key}  ->  {dest.relative_to(ROOT)}  ({size/1e6:.1f} MB)")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
        n, total = n + 1, total + size
    print(f"\n{n} files, {total/1e6:.1f} MB {'listed' if args.dry_run else 'downloaded'}.")


if __name__ == "__main__":
    main()
