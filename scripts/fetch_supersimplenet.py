from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
from pathlib import Path

import requests

COMMIT = "98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d"
SHA256 = "83c5a246c62950ebaacfa4990a505c2d4bf7601b3c7a67dc0609372858f356cf"
URL = f"https://api.github.com/repos/blaz-r/SuperSimpleNet/tarball/{COMMIT}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / f"SuperSimpleNet-{COMMIT}.tar.gz"
    verify = os.environ.get("SSL_CERT_FILE", True)
    with requests.get(URL, stream=True, timeout=120, verify=verify) as response:
        response.raise_for_status()
        with archive.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"Archive SHA-256 mismatch: {digest}")
    extract_to = args.destination / f"SuperSimpleNet-{COMMIT}"
    extract_to.mkdir(exist_ok=True)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            relative = Path(*Path(member.name).parts[1:])
            if not relative.parts:
                continue
            target = (extract_to / relative).resolve()
            if not target.is_relative_to(extract_to.resolve()):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            member.name = str(relative)
            handle.extract(member, extract_to)
    print(extract_to)


if __name__ == "__main__":
    main()
