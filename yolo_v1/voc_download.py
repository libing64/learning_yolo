"""Download PASCAL VOC 2007 + 2012 into ~/dataset/VOC (or --root)."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

VOC_ARCHIVES = {
    "VOCtrainval_06-Nov-2007.tar": {
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "urls": [
            "https://data.pjreddie.com/files/VOCtrainval_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        ],
    },
    "VOCtest_06-Nov-2007.tar": {
        "md5": "b6e924de25625d8de591ea690078ad9f",
        "urls": [
            "https://data.pjreddie.com/files/VOCtest_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        ],
    },
    "VOCtrainval_11-May-2012.tar": {
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "urls": [
            "https://data.pjreddie.com/files/VOCtrainval_11-May-2012.tar",
            "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        ],
    },
}


def md5sum(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _progress(prefix: str):
    last = {"p": -1}

    def hook(count, block_size, total):
        if total <= 0:
            return
        pct = int(count * block_size * 100 / total)
        if pct != last["p"] and pct % 2 == 0:
            last["p"] = pct
            print(f"\r{prefix}: {pct:3d}%", end="", flush=True)

    return hook


def download_file(urls: list[str], dest: Path, expected_md5: str) -> None:
    if dest.exists() and md5sum(dest) == expected_md5:
        print(f"[skip] {dest.name} already present and verified")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"Downloading {dest.name} from {url}")
            urlretrieve(url, dest, reporthook=_progress(dest.name))
            print()
            got = md5sum(dest)
            if got != expected_md5:
                raise RuntimeError(f"md5 mismatch for {dest.name}: {got} != {expected_md5}")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"\nFailed {url}: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"Could not download {dest.name}") from last_err


def extract_tar(archive: Path, root: Path) -> None:
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive, "r") as tar:
        tar.extractall(root)


def voc_ready(root: Path) -> bool:
    voc07 = root / "VOCdevkit" / "VOC2007"
    voc12 = root / "VOCdevkit" / "VOC2012"
    needed = [
        voc07 / "ImageSets" / "Main" / "trainval.txt",
        voc07 / "ImageSets" / "Main" / "test.txt",
        voc12 / "ImageSets" / "Main" / "trainval.txt",
    ]
    return all(p.exists() for p in needed)


def download_voc(root: str | Path, keep_tars: bool = True) -> Path:
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if voc_ready(root):
        print(f"VOC already extracted at {root}")
        return root
    for name, meta in VOC_ARCHIVES.items():
        archive = root / name
        download_file(meta["urls"], archive, meta["md5"])
        extract_tar(archive, root)
        if not keep_tars:
            archive.unlink()
    if not voc_ready(root):
        raise RuntimeError(f"Download finished but VOC files are incomplete under {root}")
    print(f"VOC ready at {root / 'VOCdevkit'}")
    return root


def main():
    parser = argparse.ArgumentParser(description="Download PASCAL VOC 2007 + 2012")
    parser.add_argument("--root", default=str(Path.home() / "dataset" / "VOC"))
    parser.add_argument("--delete-tars", action="store_true")
    args = parser.parse_args()
    download_voc(args.root, keep_tars=not args.delete_tars)


if __name__ == "__main__":
    main()
