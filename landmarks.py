import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18


def soft_argmax_3d(heatmaps):
    N, K, D, H, W = heatmaps.shape
    softmax = F.softmax(heatmaps.view(N, K, -1), dim=2)
    device = heatmaps.device
    zz, yy, xx = torch.meshgrid(
        torch.linspace(0, D - 1, D, device=device),
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing='ij'
    )
    exp_z = torch.sum(softmax * zz.reshape(-1), dim=2)
    exp_y = torch.sum(softmax * yy.reshape(-1), dim=2)
    exp_x = torch.sum(softmax * xx.reshape(-1), dim=2)
    return torch.stack([exp_z, exp_y, exp_x], dim=2).view(N, -1)


class Keypoint3DResNetUNet(nn.Module):
    def __init__(self, num_keypoints=3):
        super().__init__()
        self.num_keypoints = num_keypoints

        backbone = r3d_18(weights=None)
        backbone.fc = nn.Identity()
        old_conv = backbone.stem[0]
        backbone.stem[0] = nn.Conv3d(old_conv.in_channels, old_conv.out_channels,
                                     kernel_size=old_conv.kernel_size, stride=(2, 2, 2),
                                     padding=old_conv.padding, bias=False)
        self.backbone = backbone

        self.skip_conv0 = nn.Conv3d(64, self.num_keypoints, kernel_size=1)
        self.skip_conv2 = nn.Conv3d(256, 128, kernel_size=1)
        self.skip_conv3 = nn.Conv3d(128, 64, kernel_size=1)
        self.skip_conv4 = nn.Conv3d(64, 32, kernel_size=1)

        self.attn_skip_conv1 = nn.Conv3d(256, 128, kernel_size=1)
        self.attn_skip_conv2 = nn.Conv3d(128, 64, kernel_size=1)
        self.attn_skip_conv3 = nn.Conv3d(64, 32, kernel_size=1)
        self.attn_skip_conv4 = nn.Conv3d(64, self.num_keypoints, kernel_size=1)

        self.up1 = nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2)
        self.conv1 = self._block(128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self._block(64)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv3 = self._block(32)
        self.up4 = nn.ConvTranspose3d(32, self.num_keypoints, kernel_size=2, stride=2)
        self.conv4 = self._block(self.num_keypoints)
        self.multi_heatmap_head = nn.Conv3d(self.num_keypoints, num_keypoints, kernel_size=1)

        self.attn_up1 = nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2)
        self.attn_conv1 = self._block(128)
        self.attn_up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.attn_conv2 = self._block(64)
        self.attn_up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.attn_conv3 = self._block(32)
        self.attn_up4 = nn.ConvTranspose3d(32, self.num_keypoints, kernel_size=2, stride=2)
        self.attn_conv4 = self._block(self.num_keypoints)
        self.attn_heatmap_head = nn.Conv3d(self.num_keypoints, 1, kernel_size=1)

    @staticmethod
    def _block(channels):
        return nn.Sequential(
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        stem_out = self.backbone.stem(x)
        out1 = self.backbone.layer1(stem_out)
        out2 = self.backbone.layer2(out1)
        out3 = self.backbone.layer3(out2)
        out4 = self.backbone.layer4(out3)

        attn1 = self.attn_conv1(self.attn_up1(out4)) + self.attn_skip_conv1(out3)
        attn2 = self.attn_conv2(self.attn_up2(attn1)) + self.attn_skip_conv2(out2)
        attn3 = self.attn_conv3(self.attn_up3(attn2)) + self.attn_skip_conv3(out1)
        attn4 = self.attn_conv4(self.attn_up4(attn3))
        attn4 = attn4 + F.interpolate(self.attn_skip_conv4(stem_out),
                                      size=attn4.shape[2:], mode='trilinear', align_corners=False)
        attn_heatmap = self.attn_heatmap_head(attn4)

        up1 = self.conv1(self.up1(out4)) + self.skip_conv2(out3) + attn1
        up2 = self.conv2(self.up2(up1)) + self.skip_conv3(out2) + attn2
        up3 = self.conv3(self.up3(up2)) + self.skip_conv4(out1) + attn3
        up4 = self.conv4(self.up4(up3))
        up4 = up4 + F.interpolate(self.skip_conv0(stem_out),
                                  size=up4.shape[2:], mode='trilinear', align_corners=False) + attn4

        multi_heatmap = self.multi_heatmap_head(up4)
        coords = soft_argmax_3d(multi_heatmap)
        return attn_heatmap, multi_heatmap, coords


DEFAULT_ORDERS = {2: ('PC', 'AC'), 3: ('PC', 'AC', 'IVF')}


def load_landmark_model(weights_path, device, bn_train=False):
    """Returns the network and its landmark order.

    The order comes from the `<weights>.json` written at training time, or from the number of
    output channels for checkpoints without one. Batch-norm runs on its running statistics
    unless `bn_train` is set, which normalises each scan with its own statistics instead
    (the behaviour of the original 2-landmark release).
    """
    weights_path = Path(weights_path)
    state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    num_keypoints = state['multi_heatmap_head.weight'].shape[0]

    metadata_path = weights_path.with_suffix('.json')
    if metadata_path.is_file():
        order = tuple(json.loads(metadata_path.read_text())['landmark_order'])
    else:
        order = DEFAULT_ORDERS[num_keypoints]
    if len(order) != num_keypoints or not {'AC', 'PC'} <= set(order):
        raise ValueError(f"landmark order {order} does not match a {num_keypoints}-keypoint model")

    model = Keypoint3DResNetUNet(num_keypoints)
    model.load_state_dict(state)
    model = model.to(device).eval()
    if bn_train:
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.momentum = 0.0
    return model, order


def prepare_volume(img, volume_size=128):
    """Isotropic resample so the largest physical side spans `volume_size`, centre pad, z-score.

    `img` must already be in closest-canonical orientation. Returns the network input and the
    geometry needed to map grid coordinates back to `img` voxels.
    """
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    shape = np.array(data.shape[:3])
    physical = shape * np.array(img.header.get_zooms()[:3], dtype=np.float64)
    new_shape = np.minimum(np.round(physical / (physical.max() / volume_size)).astype(int), volume_size)

    tensor = torch.from_numpy(np.ascontiguousarray(data))[None, None]
    resized = F.interpolate(tensor, size=tuple(int(s) for s in new_shape), mode='trilinear', align_corners=False)
    before = (volume_size - new_shape) // 2
    after = volume_size - new_shape - before
    padded = F.pad(resized, (before[2], after[2], before[1], after[1], before[0], after[0]))
    padded = (padded - padded.mean()) / padded.std().clamp_min(1e-6)
    return padded, {'shape': shape, 'new_shape': new_shape, 'pad_before': before}


def grid_to_voxel(grid, geometry):
    """Exact inverse of `prepare_volume` (trilinear resize with align_corners=False, then pad)."""
    return ((grid - geometry['pad_before'] + 0.5) * geometry['shape'] / geometry['new_shape']) - 0.5


def predict_landmarks(model, order, img, device):
    """Landmarks as {name: voxel coordinate in `img`}."""
    volume, geometry = prepare_volume(img)
    with torch.no_grad():
        grid = model(volume.to(device))[2].float().view(-1, 3).cpu().numpy()
    voxels = grid_to_voxel(grid, geometry)
    return {name: voxels[k] for k, name in enumerate(order)}
