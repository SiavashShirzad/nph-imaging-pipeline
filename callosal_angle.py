import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import measure
from sklearn.svm import SVC

from config import SEGMENTATION_INPUT_SIZE


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, out_channels=3, encoder_name="efficientnet_b0", pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained,
                                         features_only=True, out_indices=(0, 1, 2, 3, 4))
        channels = self.encoder.feature_info.channels()
        self.up_conv4 = nn.ConvTranspose2d(channels[-1], channels[-2], 2, stride=2)
        self.conv4 = DoubleConv(channels[-2] * 2, channels[-2])
        self.up_conv3 = nn.ConvTranspose2d(channels[-2], channels[-3], 2, stride=2)
        self.conv3 = DoubleConv(channels[-3] * 2, channels[-3])
        self.up_conv2 = nn.ConvTranspose2d(channels[-3], channels[-4], 2, stride=2)
        self.conv2 = DoubleConv(channels[-4] * 2, channels[-4])
        self.up_conv1 = nn.ConvTranspose2d(channels[-4], channels[-5], 2, stride=2)
        self.conv1 = DoubleConv(channels[-5] * 2, channels[-5])
        self.final_conv = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x):
        input_shape = x.shape[2:]
        f0, f1, f2, f3, f4 = self.encoder(x)

        x = self.up_conv4(f4)
        if x.shape[2:] != f3.shape[2:]:
            x = F.interpolate(x, size=f3.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv4(torch.cat([x, f3], dim=1))

        x = self.up_conv3(x)
        if x.shape[2:] != f2.shape[2:]:
            x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv3(torch.cat([x, f2], dim=1))

        x = self.up_conv2(x)
        if x.shape[2:] != f1.shape[2:]:
            x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv2(torch.cat([x, f1], dim=1))

        x = self.up_conv1(x)
        if x.shape[2:] != f0.shape[2:]:
            x = F.interpolate(x, size=f0.shape[2:], mode="bilinear", align_corners=False)
        x = self.conv1(torch.cat([x, f0], dim=1))

        x = self.final_conv(x)
        return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)


def load_ca_model(weights, device):
    model = EfficientUNet(out_channels=3, pretrained=False)
    try:
        state = torch.load(str(weights), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(weights), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()}, strict=False)
    return model.to(device).eval()


def resize_with_padding(image, target_size=SEGMENTATION_INPUT_SIZE):
    height, width = image.shape[:2]
    scale = min(target_size[0] / height, target_size[1] / width)
    new_height, new_width = int(height * scale), int(width * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    shape = (target_size[0], target_size[1], image.shape[2]) if image.ndim == 3 else target_size
    canvas = np.zeros(shape, dtype=np.float32)
    top, left = (target_size[0] - new_height) // 2, (target_size[1] - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def cast_ray(origin, direction, mask, step=0.5, max_distance=224):
    height, width = mask.shape
    distance = 0.0
    while distance < max_distance:
        point = origin + distance * direction
        column, row = int(round(point[0])), int(round(point[1]))
        if not (0 <= column < width and 0 <= row < height):
            return None
        if mask[row, column]:
            return point
        distance += step
    return None


def best_line_from_base(base, edge_points, tolerance=1.0):
    """Direction from the base point that most edge points lie along."""
    if base is None or edge_points is None or len(edge_points) == 0:
        return None, None, None

    offsets = edge_points - base
    norms = np.linalg.norm(offsets, axis=1)
    valid = norms > 1e-6
    if not valid.any():
        return None, None, None
    directions = offsets[valid] / norms[valid, None]

    projections = offsets @ directions.T
    residuals = np.linalg.norm(offsets[:, None, :] - projections[:, :, None] * directions[None, :, :], axis=2)
    inlier_counts = (residuals < tolerance).sum(axis=0)

    best = int(inlier_counts.argmax())
    direction = directions[best]
    inliers = offsets[residuals[:, best] < tolerance]
    endpoint = base + direction * (inliers @ direction).max()
    return direction, base, endpoint


def _midline_from_svm(mask1, mask2, contours1, contours2):
    points, labels = [], []
    for contour in contours1:
        points.append(np.fliplr(contour))
        labels.extend([0] * contour.shape[0])
    for contour in contours2:
        points.append(np.fliplr(contour))
        labels.extend([1] * contour.shape[0])
    if not points:
        raise ValueError("No ventricle contours were detected on the PC slice.")

    classifier = SVC(kernel="linear", C=1e5)
    classifier.fit(np.concatenate(points, axis=0), np.array(labels))
    weights, bias = classifier.coef_[0], classifier.intercept_[0]

    height, width = mask1.shape
    intersections = []
    inside = lambda p: 0 <= p[0] < width and 0 <= p[1] < height
    if abs(weights[1]) > 1e-6:
        for point in ((0, -bias / weights[1]), (width - 1, -(weights[0] * (width - 1) + bias) / weights[1])):
            if inside(point):
                intersections.append(point)
    if abs(weights[0]) > 1e-6:
        for point in ((-bias / weights[0], 0), (-(weights[1] * (height - 1) + bias) / weights[0], height - 1)):
            if inside(point):
                intersections.append(point)
    segment = intersections[:2] if len(intersections) >= 2 else [(0, 0), (width - 1, height - 1)]
    return weights, bias, segment


def _base_points(contours1, contours2, mask1, mask2, max_iterations=20):
    height, width = mask1.shape
    base1 = base2 = None
    threshold = 3.0

    for _ in range(max_iterations):
        distance_to_2 = distance_transform_edt(~mask2)
        distance_to_1 = distance_transform_edt(~mask1)

        candidates = []
        for contours, distance_map in ((contours1, distance_to_2), (contours2, distance_to_1)):
            found = []
            for contour in contours:
                points = np.fliplr(contour)
                distances = np.array([
                    distance_map[int(round(y)), int(round(x))] if 0 <= x < width and 0 <= y < height else 999
                    for x, y in points
                ])
                near = np.nonzero(distances < threshold)[0]
                if near.size:
                    found.append(points[near][np.argmin(points[near][:, 1])])
            candidates.append(min(found, key=lambda p: p[1]) if found else None)

        base1, base2 = candidates
        if base1 is not None and base2 is not None:
            break
        threshold += 1.0
    return base1, base2


def compute_callosal_angle(aligned_t1, pc_vox, model, device):
    """Segment the lateral ventricles on the PC coronal slice and measure the callosal angle."""
    slice_index = int(round(pc_vox[1]))
    slice_index = int(np.clip(slice_index, 0, aligned_t1.shape[1] - 1))
    coronal = np.flipud(aligned_t1[:, slice_index, :].T)

    low, high = coronal.min(), coronal.max()
    normalized = (coronal - low) / (high - low) if high > low else coronal
    padded = resize_with_padding(np.stack([normalized] * 3, axis=-1))

    with torch.inference_mode():
        logits = model(torch.from_numpy(padded.transpose(2, 0, 1))[None].to(device))
        prediction = logits.argmax(dim=1).cpu().numpy().squeeze()

    mask1, mask2 = prediction == 1, prediction == 2
    contours1 = measure.find_contours(mask1.astype(float), 0.5)
    contours2 = measure.find_contours(mask2.astype(float), 0.5)

    weights, bias, segment = _midline_from_svm(mask1, mask2, contours1, contours2)

    p0, p1 = np.array(segment[0], dtype=float), np.array(segment[1], dtype=float)
    direction = p1 - p0
    direction /= np.linalg.norm(direction) + 1e-6
    normal = np.array([-direction[1], direction[0]])
    midpoint = (p0 + p1) / 2
    if np.dot(weights, midpoint + 0.5 * normal) + bias > 0:
        ray1, ray2, array1, array2 = -normal, normal, mask1, mask2
    else:
        ray1, ray2, array1, array2 = normal, -normal, mask1, mask2

    edges1, edges2 = [], []
    for point in np.linspace(p0, p1, 100):
        hit1 = cast_ray(point, ray1, array1)
        hit2 = cast_ray(point, ray2, array2)
        if hit1 is not None:
            edges1.append(hit1)
        if hit2 is not None:
            edges2.append(hit2)
    edges1, edges2 = np.array(edges1), np.array(edges2)

    base1, base2 = _base_points(contours1, contours2, mask1, mask2)

    superior = lambda edges, base: (np.array([]) if base is None or edges.size == 0
                                    else edges[edges[:, 1] < base[1]])
    direction1, start1, end1 = best_line_from_base(base1, superior(edges1, base1))
    direction2, start2, end2 = best_line_from_base(base2, superior(edges2, base2))

    angle = None
    if direction1 is not None and direction2 is not None:
        angle = float(np.degrees(np.arccos(np.clip(np.dot(direction1, direction2), -1.0, 1.0))))

    return {
        "callosal_angle_deg": angle,
        "slice_index": slice_index,
        "image": padded[:, :, 0],
        "contours1": contours1,
        "contours2": contours2,
        "midline": segment,
        "base1": base1,
        "base2": base2,
        "line1": (start1, end1),
        "line2": (start2, end2),
    }
