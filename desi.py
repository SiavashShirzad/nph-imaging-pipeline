import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import (DESI_HC_ANGLE_DEG, DESI_NUM_CLASSES, DESI_SLICE_PAD_FRACTION,
                     IMAGENET_MEAN, IMAGENET_STD, SEGMENTATION_INPUT_SIZE)
from segmentation import largest_component

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(SEGMENTATION_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def volume_ml(mask, spacing):
    return float(np.count_nonzero(mask) * np.prod(spacing) / 1000.0)


def load_desi_model(weights, device, num_classes=DESI_NUM_CLASSES):
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None,
                     in_channels=3, classes=num_classes, activation=None)
    try:
        state = torch.load(str(weights), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(weights), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    return model.to(device).eval()


def _slice_range(aligned_t1, ac_vox, pc_vox):
    y_ac, y_pc = int(ac_vox[1]), int(pc_vox[1])
    start, end = min(y_ac, y_pc), max(y_ac, y_pc)
    pad = max(1, int(DESI_SLICE_PAD_FRACTION * aligned_t1.shape[1]))
    start = max(0, start - pad)
    end = min(aligned_t1.shape[1] - 1, end + pad)
    return list(range(start, end + 1))


def _segment_slices(aligned_t1, slices, model, device, batch_size=16):
    batch_input = []
    for y in slices:
        coronal = np.flipud(aligned_t1[:, y, :].T)
        low, high = coronal.min(), coronal.max()
        normalized = (coronal - low) / (high - low) * 255.0 if high > low else coronal
        batch_input.append(INFERENCE_TRANSFORM(np.stack([normalized.astype(np.uint8)] * 3, axis=-1)))

    predictions = np.zeros(aligned_t1.shape, dtype=np.uint8)
    target = (aligned_t1.shape[2], aligned_t1.shape[0])

    with torch.inference_mode():
        for start in range(0, len(batch_input), batch_size):
            batch = torch.stack(batch_input[start:start + batch_size]).to(device)
            predicted = model(batch).argmax(dim=1)
            resized = F.interpolate(predicted.unsqueeze(1).float(), size=target, mode="nearest").squeeze(1)
            for offset, slice_prediction in enumerate(resized):
                index = start + offset
                if index < len(slices):
                    predictions[:, slices[index], :] = np.flipud(
                        slice_prediction.cpu().numpy()).T.astype(np.uint8)
    return predictions


def _filter_high_convexity(candidate, ac_vox, pc_vox, zooms, angle_deg=DESI_HC_ANGLE_DEG):
    """Keep candidate voxels within a cone above the AC-PC line."""
    indices = np.argwhere(candidate)
    if indices.size == 0:
        return np.zeros_like(candidate)

    ac_mm, pc_mm = ac_vox * zooms, pc_vox * zooms
    base_points = np.array([ac_mm + t * (pc_mm - ac_mm) for t in np.linspace(0, 1, 15)])

    points_mm = indices * zooms
    distances = np.linalg.norm(points_mm[:, None, :] - base_points[None, :, :], axis=2)
    nearest = base_points[distances.argmin(axis=1)]

    offsets = points_mm - nearest
    norms = np.linalg.norm(offsets, axis=1)
    valid = norms > 0
    cosine = np.zeros(len(indices))
    cosine[valid] = offsets[valid, 2] / norms[valid]

    keep = valid & (offsets[:, 2] > 0) & (cosine >= np.cos(np.radians(angle_deg)))
    mask = np.zeros_like(candidate)
    selected = indices[keep]
    if selected.size:
        mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return mask


def compute_desi(aligned_t1, ac_vox, pc_vox, zooms, model, device):
    """Ratio of Sylvian fissure volume to high-convexity sulcal volume."""
    slices = _slice_range(aligned_t1, ac_vox, pc_vox)
    predictions = _segment_slices(aligned_t1, slices, model, device)

    right_sylvian = largest_component(predictions == 1) > 0
    left_sylvian = largest_component(predictions == 2) > 0
    high_convexity = _filter_high_convexity(predictions == 3, ac_vox, pc_vox, zooms)

    sylvian_volume = volume_ml(right_sylvian | left_sylvian, zooms)
    convexity_volume = volume_ml(high_convexity, zooms)
    ratio = sylvian_volume / convexity_volume if convexity_volume > 0 else None

    return {
        "desi_ratio": None if ratio is None else float(ratio),
        "sylvian_volume_ml": sylvian_volume,
        "high_convexity_volume_ml": convexity_volume,
        "n_slices": len(slices),
        "sylvian_mask": right_sylvian | left_sylvian,
        "high_convexity_mask": high_convexity,
    }
