import numpy as np
from scipy.ndimage import affine_transform


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def acpc_rotation(v, pitch_only=False):
    """Rotation that brings the PC->AC vector `v` (RAS, mm) onto +y without changing roll.

    Pitch (about the left-right axis) removes the superior component, then yaw (about the
    superior axis) removes the left-right component. Applied in that order, the left-right axis
    of the scan stays in the axial plane, so roll is untouched. With `pitch_only` the yaw step is
    skipped, for volumes whose yaw and roll are already set by registration.
    Returns the rotation and the applied (pitch, yaw) in degrees.
    """
    pitch = -np.arctan2(v[2], v[1])
    R_pitch = _rotation_x(pitch)
    w = R_pitch @ v
    yaw = 0.0 if pitch_only else np.arctan2(w[0], w[1])
    return _rotation_z(yaw) @ R_pitch, float(np.degrees(pitch)), float(np.degrees(yaw))


def acpc_transform(pc_vox, ac_vox, vol_shape, spacing_mm, pitch_only=False):
    pc_mm = pc_vox * spacing_mm
    ac_mm = ac_vox * spacing_mm
    v = ac_mm - pc_mm
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("AC and PC landmarks coincide")
    R, pitch_deg, yaw_deg = acpc_rotation(v, pitch_only)

    corners = np.array([[i, j, k]
                        for i in [0, vol_shape[0] - 1]
                        for j in [0, vol_shape[1] - 1]
                        for k in [0, vol_shape[2] - 1]])
    moved = (R @ (corners * spacing_mm - pc_mm).T).T
    min_mm = moved.min(axis=0)
    new_shape = np.ceil((moved.max(axis=0) - min_mm) / spacing_mm).astype(int) + 1

    S_inv = np.diag(1 / spacing_mm)
    matrix_vox = S_inv @ R.T @ np.diag(spacing_mm)
    offset_vox = S_inv @ (R.T @ min_mm + pc_mm)

    aligned_pc_vox = -min_mm / spacing_mm
    aligned_ac_vox = (R @ (ac_mm - pc_mm) - min_mm) / spacing_mm

    return {
        'pc_mm': pc_mm,
        'min_mm': min_mm,
        'spacing_mm': np.asarray(spacing_mm, float),
        'matrix': matrix_vox,
        'offset': offset_vox,
        'shape': tuple(int(s) for s in new_shape),
        'rotation': R,
        'pitch_deg': pitch_deg,
        'yaw_deg': yaw_deg,
        'pitch_only': bool(pitch_only),
        'aligned_pc_vox': aligned_pc_vox,
        'aligned_ac_vox': aligned_ac_vox,
    }


def apply_transform(volume, transform, order=1):
    return affine_transform(volume,
                            matrix=transform['matrix'],
                            offset=transform['offset'],
                            output_shape=transform['shape'],
                            order=order)


def transform_point(point_vox, transform):
    """Map a voxel coordinate of the input volume into the aligned volume."""
    spacing = transform['spacing_mm']
    moved = transform['rotation'] @ (np.asarray(point_vox, float) * spacing - transform['pc_mm'])
    return (moved - transform['min_mm']) / spacing
