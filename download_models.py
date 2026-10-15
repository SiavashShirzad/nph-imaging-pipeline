#!/usr/bin/env python3
"""Download the trained models from the GitHub release and unpack them into model/."""
import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE_URL = ('https://github.com/SiavashShirzad/nph-imaging-pipeline/releases/download/'
               'v1.0/nph-models-v1.0.zip')
SHA256 = '6560b198313d82774a9d14929a41ad8cf8c2916cd2cfcc2fa4fd295f8aa90639'
REQUIRED = (
    'model/acpc/AC_PC_IVF_BestModel.pth',
    'model/icv/fold_all/checkpoint_final.pth',
    'model/ventricles/fold_0/checkpoint_best.pth',
    'model/ca/best_pretrained_segmentation_model_2.pth',
    'model/desi/best_model.pth',
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target):
    with urllib.request.urlopen(url) as response, open(target, 'wb') as out:
        total = int(response.headers.get('Content-Length', 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', help="install from an already downloaded nph-models-v1.0.zip")
    parser.add_argument('--force', action='store_true', help="reinstall even if the models are present")
    args = parser.parse_args()

    if all((ROOT / f).is_file() for f in REQUIRED) and not args.force:
        print("models already installed in", ROOT / 'model')
        return

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(args.zip) if args.zip else Path(tmp) / 'nph-models.zip'
        if not args.zip:
            print("downloading", RELEASE_URL)
            download(RELEASE_URL, archive)
        print("verifying checksum")
        if sha256(archive) != SHA256:
            sys.exit("checksum mismatch: the download is incomplete or not the v1.0 model bundle")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if not member.startswith('model/') or '..' in Path(member).parts:
                    sys.exit(f"unexpected file in archive: {member}")
            for member in bundle.infolist():
                target = ROOT / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, open(target, 'wb') as out:
                    shutil.copyfileobj(source, out)

    missing = [f for f in REQUIRED if not (ROOT / f).is_file()]
    if missing:
        sys.exit(f"installation incomplete, missing: {', '.join(missing)}")
    print("models installed in", ROOT / 'model')


if __name__ == '__main__':
    main()
