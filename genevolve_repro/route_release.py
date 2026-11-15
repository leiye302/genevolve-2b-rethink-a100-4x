"""Fetch the public Route-SFT pack without credentials or local project files."""

from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from .common import ROOT, read_json
from .sft_export import file_sha256


NAMES = {"manifest.json", "train.jsonl", "batches.jsonl"}
RELEASE_PAGE = "https://github.com/leiye302/genevolve-2b-rethink-a100-4x/releases"


def release_lock():
    spec = read_json(ROOT / "configs" / "route_release.json")

    def valid_file(entry):
        return (type(entry.get("bytes")) is int and 0 < entry["bytes"] < 2**30
                and re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", "")))

    if (not re.fullmatch(r"[A-Za-z0-9_-]+", spec["tag"])
            or not re.fullmatch(r"[A-Za-z0-9_-]+\.zip", spec["name"])
            or not valid_file(spec) or set(spec["files"]) != NAMES
            or not all(valid_file(entry) for entry in spec["files"].values())):
        raise ValueError("Invalid Route-SFT release lock.")
    return spec


def matches(path, spec):
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == spec["bytes"] and file_sha256(path) == spec["sha256"])


def copy_bounded(src, dst, size):
    remaining = size
    with dst.open("wb") as file:
        while chunk := src.read(min(1024**2, remaining + 1)):
            remaining -= len(chunk)
            if remaining < 0:
                raise ValueError("Route-SFT artifact exceeds its locked size.")
            file.write(chunk)
    if remaining:
        raise ValueError("Incomplete Route-SFT artifact.")


def verify_files(folder, spec):
    for name, lock in spec["files"].items():
        if not matches(folder / name, lock):
            raise ValueError(f"Route-SFT SHA-256 or size mismatch: {folder / name}. "
                             "Use a fresh WORK directory or supply a verified pack via --data.")


def download_archive(archive, spec):
    url = f"{RELEASE_PAGE}/download/{spec['tag']}/{spec['name']}"
    request = Request(url, headers={"User-Agent": "GenEvolve-Route-SFT-Downloader/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                copy_bounded(response, archive, spec["bytes"])
            break
        except HTTPError as error:
            error.close()
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise ValueError(f"Could not download Route-SFT from GitHub (HTTP {error.code}). "
                                 f"Download {spec['name']} from {RELEASE_PAGE}/tag/{spec['tag']} "
                                 "and pass its extracted folder via --data.") from None
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise ValueError("Could not reach GitHub Releases. Download the pinned Route-SFT "
                                 "asset on a connected machine and pass its extracted folder via --data.") from None
        sleep(attempt + 1)
    if not matches(archive, spec):
        raise ValueError("Downloaded Route-SFT archive SHA-256 mismatch.")


def unpack_archive(archive, folder, spec):
    folder.mkdir()
    try:
        with ZipFile(archive) as zip_file:
            names = zip_file.namelist()
            if len(names) != len(NAMES) or set(names) != NAMES:
                raise ValueError("Unexpected Route-SFT archive members.")
            for name in sorted(NAMES):
                size = spec["files"][name]["bytes"]
                with zip_file.open(name) as stream:
                    copy_bounded(stream, folder / name, size)
    except BadZipFile as error:
        raise ValueError("Invalid Route-SFT zip archive.") from error
    verify_files(folder, spec)


def prepare_route(work):
    """Cache the fully validated release atomically; never overwrite existing data."""
    from filelock import FileLock

    spec = release_lock()
    folder = Path(work) / "route_data" / spec["tag"]
    folder.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(folder.parent / ".download.lock"):
        if folder.is_symlink():
            raise ValueError("Expected a local Route-SFT directory, not a symlink.")
        if not folder.exists():
            print(f"Downloading {spec['name']} ({spec['bytes']:,} bytes) from GitHub Releases.", flush=True)
            with TemporaryDirectory(prefix=".route-download-", dir=folder.parent) as scratch:
                scratch = Path(scratch)
                archive, staging = scratch / spec["name"], scratch / "pack"
                download_archive(archive, spec)
                unpack_archive(archive, staging, spec)
                staging.rename(folder)
        else:
            verify_files(folder, spec)
    return folder
