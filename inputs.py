import os
from pathlib import Path

import nibabel as nib

NIFTI_SUFFIXES = ('.nii', '.nii.gz')


def is_nifti(path):
    return str(path).lower().endswith(NIFTI_SUFFIXES)


def is_dicom_dir(path, probe=25):
    import pydicom

    files = [f for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    for name in files[:probe]:
        try:
            pydicom.dcmread(os.path.join(path, name), stop_before_pixels=True)
            return True
        except Exception:
            continue
    return False


def discover(path):
    path = Path(path)
    if path.is_file():
        return [path] if is_nifti(path) else []
    if not path.is_dir():
        return []
    if is_dicom_dir(path):
        return [path]

    found = []
    for dirpath, dirnames, filenames in os.walk(path):
        current = Path(dirpath)
        if current != path and is_dicom_dir(current):
            found.append(current)
            dirnames[:] = []
            continue
        found.extend(current / f for f in sorted(filenames) if is_nifti(f))
        dirnames.sort()
    return sorted(found)


def load_scan(path):
    path = Path(path)
    if path.is_dir():
        import dicom2nifti
        img = dicom2nifti.dicom_series_to_nifti(str(path), output_file=None, reorient_nifti=False)['NII']
    else:
        img = nib.load(str(path))
    return nib.as_closest_canonical(img)


def case_name(path, root=None):
    path = Path(path)
    stem = path.name
    for suffix in NIFTI_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    if root is not None:
        try:
            parents = Path(path).relative_to(root).parent.parts
        except ValueError:
            parents = ()
        if parents:
            stem = '_'.join(parents + (stem,))
    return stem.replace(os.sep, '_').replace(' ', '_')
