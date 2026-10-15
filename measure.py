import numpy as np


def _frontal_horn_mask(vent_mask, ac_mm, pc_mm, spacing, frac, ivf_mm=None):
    """Ventricle voxels anterior to a coronal cut and superior to the AC-PC line.

    The cut is perpendicular to the AC-PC line and passes through the interventricular
    foramen when it is given, otherwise through the point `frac` of the way from AC to PC.
    """
    axis = pc_mm - ac_mm
    u = axis / np.linalg.norm(axis)
    sup = np.array([0.0, 0.0, 1.0])
    v = sup - np.dot(sup, u) * u
    v /= np.linalg.norm(v)
    if ivf_mm is not None:
        cut_mm = ac_mm + np.dot(ivf_mm - ac_mm, u) * u
    else:
        cut_mm = ac_mm + frac * axis

    vox = np.argwhere(vent_mask)
    mm = vox * spacing
    anterior = (mm - cut_mm).dot(u) <= 0
    superior = (mm - cut_mm).dot(v) >= 0
    selected = vox[anterior & superior]

    horns = np.zeros_like(vent_mask, dtype=bool)
    horns[selected[:, 0], selected[:, 1], selected[:, 2]] = True

    filled = horns.copy()
    for y in range(vent_mask.shape[1]):
        for z in range(vent_mask.shape[2]):
            xs = np.where(horns[:, y, z])[0]
            if xs.size:
                filled[xs.min():xs.max() + 1, y, z] = True
    return filled, cut_mm


def evans_index(aligned_seg, aligned_icv, aligned_ac_vox, aligned_pc_vox, spacing, frac=0.2,
                aligned_ivf_vox=None):
    vent_mask = aligned_seg == 1
    if not vent_mask.any():
        raise ValueError("empty ventricle segmentation")

    filled, cut_mm = _frontal_horn_mask(vent_mask,
                                np.asarray(aligned_ac_vox, float) * spacing,
                                np.asarray(aligned_pc_vox, float) * spacing,
                                spacing, frac,
                                None if aligned_ivf_vox is None else np.asarray(aligned_ivf_vox, float) * spacing)

    widths = filled.sum(axis=0)
    vent_y, axial_z = np.unravel_index(np.argmax(widths), widths.shape)

    horn_line = np.zeros_like(filled)
    horn_line[:, vent_y, axial_z] = filled[:, vent_y, axial_z]

    icv = aligned_icv > 0
    brain_y = int(icv.sum(axis=0)[:, axial_z].argmax())
    brain_line = np.zeros_like(icv)
    brain_line[:, brain_y, axial_z] = icv[:, brain_y, axial_z]

    horn_x = np.where(horn_line[:, vent_y, axial_z])[0]
    brain_x = np.where(brain_line[:, brain_y, axial_z])[0]
    if brain_x.size == 0:
        raise ValueError("no intracranial voxels on the measurement slice")

    return {
        'evans_index': float(horn_x.size / brain_x.size),
        'cut_vox': [round(float(c), 3) for c in cut_mm / spacing],
        'horn_cut': 'IVF' if aligned_ivf_vox is not None else f'{frac:g} AC-PC',
        'axial_z': int(axial_z),
        'vent_y': int(vent_y),
        'brain_y': brain_y,
        'horn_x': (int(horn_x.min()), int(horn_x.max())),
        'brain_x': (int(brain_x.min()), int(brain_x.max())),
        'horn_width_mm': float((horn_x.max() - horn_x.min()) * spacing[0]),
        'brain_width_mm': float((brain_x.max() - brain_x.min()) * spacing[0]),
    }
