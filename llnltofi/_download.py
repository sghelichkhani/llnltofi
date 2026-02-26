"""Download and cache remote data files."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_MANIFEST = Path(__file__).resolve().parent / "datasets.json"


def _load_manifest() -> dict:
    with open(_MANIFEST) as f:
        return json.load(f)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_via_boto3(s3_config: dict, filename: str, dest: Path) -> None:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=s3_config["endpoint_url"],
        config=Config(signature_version=UNSIGNED),
    )
    s3_key = s3_config["prefix"] + filename
    try:
        from tqdm import tqdm

        head = client.head_object(Bucket=s3_config["bucket"], Key=s3_key)
        total = head["ContentLength"]
        with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
            client.download_file(
                s3_config["bucket"],
                s3_key,
                str(dest),
                Callback=lambda n: pbar.update(n),
            )
    except ImportError:
        client.download_file(s3_config["bucket"], s3_key, str(dest))


def _download_via_https(cdn_url: str, filename: str, dest: Path) -> None:
    url = cdn_url + filename
    try:
        from tqdm import tqdm

        resp = urllib.request.urlopen(url)
        total = int(resp.headers.get("Content-Length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar,
        ):
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    except ImportError:
        urllib.request.urlretrieve(url, dest)


def ensure_data(filename: str = "grid_data.npz") -> Path:
    """Return the local path to *filename*, downloading if necessary.

    Downloads from S3 (via boto3) or the CDN (via urllib) and verifies the
    SHA-256 hash against the manifest.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DATA_DIR / filename
    manifest = _load_manifest()
    expected_hash = manifest["files"][filename]["sha256"]

    if dest.exists() and _file_sha256(dest) == expected_hash:
        return dest

    try:
        _download_via_boto3(manifest["s3"], filename, dest)
    except ImportError:
        cdn_url = manifest.get("cdn_url", "")
        if not cdn_url:
            raise RuntimeError(
                "boto3 is not installed and no CDN URL is configured. "
                "Install boto3 or set cdn_url in datasets.json."
            )
        _download_via_https(cdn_url, filename, dest)

    actual = _file_sha256(dest)
    if actual != expected_hash:
        dest.unlink()
        raise RuntimeError(
            f"Hash mismatch for {filename}: expected {expected_hash}, got {actual}"
        )

    return dest
