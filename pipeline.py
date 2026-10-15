import csv
import json
from pathlib import Path

import numpy as np
import torch

import alignment
import inputs
import landmarks
import measure
import plotting
import preprocess
import segmentation
from callosal_angle import compute_callosal_angle, load_ca_model
from config import (ACPC_RANGE_MM, CALLOSAL_ANGLE_THRESHOLD, CA_WEIGHTS, CONDITIONAL_STAGES,
                    DESI_WEIGHTS, EVANS_THRESHOLD, ICV_DIR, IVF_AC_RANGE_MM, LANDMARK_WEIGHTS,
                    STAGES, VENT_DIR)
from desi import compute_desi, load_desi_model


def resolve_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


class NPHPipeline:
    """AC-PC alignment, Evans Index, callosal angle and DESH ratio from a raw T1."""

    def __init__(self, models_dir, device='auto',
                 icv_folds=('all',), icv_checkpoint='checkpoint_final.pth',
                 vent_folds=(0,), vent_checkpoint='checkpoint_best.pth',
                 stages=STAGES, evans_threshold=EVANS_THRESHOLD,
                 angle_threshold=CALLOSAL_ANGLE_THRESHOLD, gate_on_evans=True, dpi=150,
                 n4=False, register_mni=False, template=None, landmark_weights=None, use_ivf=True):
        self.device = resolve_device(device)
        self.models_dir = Path(models_dir)
        self.stages = tuple(stages)
        self.evans_threshold = evans_threshold
        self.angle_threshold = angle_threshold
        self.gate_on_evans = gate_on_evans
        self.dpi = dpi
        self.n4 = n4
        self.register_mni = register_mni
        self.template_path = template
        self.template = preprocess.load_template(template) if register_mni else None
        self.template_img = preprocess.sitk_to_nib(self.template) if register_mni else None

        landmark_weights = Path(landmark_weights) if landmark_weights else self.models_dir / LANDMARK_WEIGHTS
        icv_dir = self.models_dir / ICV_DIR
        vent_dir = self.models_dir / VENT_DIR
        for path in (landmark_weights, icv_dir, vent_dir):
            if not path.exists():
                raise FileNotFoundError(f"missing model: {path}")

        self.landmark_model, self.landmark_order = landmarks.load_landmark_model(landmark_weights, self.device)
        self.use_ivf = use_ivf and 'IVF' in self.landmark_order
        self.predictor_vent = segmentation.build_predictor(vent_dir, vent_folds, vent_checkpoint, self.device)
        self.predictor_icv = segmentation.build_predictor(icv_dir, icv_folds, icv_checkpoint, self.device)

        self.ca_model = None
        self.desi_model = None
        if 'ca' in self.stages:
            self.ca_model = load_ca_model(self._require(CA_WEIGHTS, 'callosal angle model'), self.device)
        if 'desi' in self.stages:
            self.desi_model = load_desi_model(self._require(DESI_WEIGHTS, 'DESI model'), self.device)

    def _require(self, relative, description):
        path = self.models_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"missing model ({description}): {path}")
        return path

    def run(self, scan_path, out_dir, case_name=None, frac=0.2):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        case_name = case_name or inputs.case_name(scan_path)

        img = inputs.load_scan(scan_path)
        preprocessing = {'n4': False, 'mni_registration': False}
        if self.n4 or self.register_mni:
            img, preprocessing = self._preprocess(img, out_dir, case_name)
        data = img.get_fdata()
        spacing = np.array(img.header.get_zooms()[:3], dtype=float)

        points = landmarks.predict_landmarks(self.landmark_model, self.landmark_order, img, self.device)
        pc_vox, ac_vox = points['PC'], points['AC']
        ivf_vox = points['IVF'] if self.use_ivf else None
        # After MNI registration yaw and roll are already fixed, so only pitch is corrected.
        transform = alignment.acpc_transform(pc_vox, ac_vox, data.shape, spacing,
                                             pitch_only=preprocessing['mni_registration'])

        seg = segmentation.predict(self.predictor_vent, img)
        icv = segmentation.predict(self.predictor_icv, img)

        aligned = alignment.apply_transform(data, transform, order=1)
        aligned_seg = alignment.apply_transform(seg, transform, order=0)
        aligned_icv = segmentation.largest_component(alignment.apply_transform(icv, transform, order=0))
        aligned_seg = aligned_seg * (aligned_icv > 0)

        aligned_ac = transform['aligned_ac_vox']
        aligned_pc = transform['aligned_pc_vox']
        aligned_ivf = None if ivf_vox is None else alignment.transform_point(ivf_vox, transform)
        acpc_mm = float(np.linalg.norm((ac_vox - pc_vox) * spacing))
        ivf_check = None if ivf_vox is None else ivf_geometry(aligned_ac, aligned_pc, aligned_ivf, spacing)

        result = {
            'case': case_name,
            'scan': str(scan_path),
            'acpc': {
                'ac_voxel_input': [round(float(v), 3) for v in ac_vox],
                'pc_voxel_input': [round(float(v), 3) for v in pc_vox],
                'ac_voxel_aligned': [round(float(v), 3) for v in aligned_ac],
                'pc_voxel_aligned': [round(float(v), 3) for v in aligned_pc],
                'acpc_distance_mm': round(acpc_mm, 3),
                'acpc_distance_plausible': bool(ACPC_RANGE_MM[0] <= acpc_mm <= ACPC_RANGE_MM[1]),
                **({} if ivf_vox is None else {
                    'ivf_voxel_input': [round(float(v), 3) for v in ivf_vox],
                    'ivf_voxel_aligned': [round(float(v), 3) for v in aligned_ivf],
                    **ivf_check,
                }),
                'alignment': 'pitch' if transform['pitch_only'] else 'yaw+pitch',
                'pitch_deg': round(transform['pitch_deg'], 3),
                'yaw_deg': round(transform['yaw_deg'], 3),
                'aligned_shape': [int(s) for s in transform['shape']],
                'spacing_mm': [round(float(s), 4) for s in spacing],
            },
            'segmentation': {
                'ventricle_volume_ml': round(float(np.count_nonzero(aligned_seg == 1)
                                                   * np.prod(spacing) / 1000.0), 2),
                'icv_volume_ml': round(float(np.count_nonzero(aligned_icv)
                                             * np.prod(spacing) / 1000.0), 2),
            },
            'stages_requested': list(self.stages),
            'preprocessing': preprocessing,
        }
        panels = []

        if 'align' in self.stages:
            figure = plotting.alignment_figure(
                {'t1': data, 'ventricles': (seg == 1).astype(np.uint8),
                 'icv': (icv > 0).astype(np.uint8), 'ac': ac_vox, 'pc': pc_vox, 'ivf': ivf_vox},
                {'t1': aligned, 'ventricles': (aligned_seg == 1).astype(np.uint8),
                 'icv': (aligned_icv > 0).astype(np.uint8), 'ac': aligned_ac, 'pc': aligned_pc,
                 'ivf': aligned_ivf},
                spacing)
            panels.append(('Alignment', plotting.save(figure, out_dir / f"{case_name}_alignment.png", self.dpi)))

        evans = None
        if 'evans' in self.stages:
            try:
                evans = measure.evans_index(aligned_seg, aligned_icv, aligned_ac, aligned_pc,
                                            spacing, frac=frac,
                                            aligned_ivf_vox=aligned_ivf if ivf_check and ivf_check['ivf_plausible']
                                            else None)
                result['evans'] = dict(evans)
                result['evans']['above_threshold'] = bool(evans['evans_index'] >= self.evans_threshold)
                path = out_dir / f"{case_name}_evans.png"
                plotting.save_report(path, case_name, aligned, aligned_icv, aligned_seg,
                                     spacing, aligned_ac, aligned_pc, evans, aligned_ivf)
                panels.append(('Evans Index', path))
            except Exception as error:
                result['evans'] = {'error': str(error)}

        for stage in self._conditional(evans):
            if stage == 'ca':
                try:
                    angle_result = compute_callosal_angle(aligned, aligned_pc, self.ca_model, self.device)
                    angle = angle_result['callosal_angle_deg']
                    result['callosal_angle'] = {
                        'callosal_angle_deg': None if angle is None else round(angle, 2),
                        'slice_index': angle_result['slice_index'],
                        'below_threshold': None if angle is None else bool(angle < self.angle_threshold),
                    }
                    figure = plotting.callosal_angle_figure(angle_result)
                    panels.append(('Callosal angle',
                                   plotting.save(figure, out_dir / f"{case_name}_callosal_angle.png", self.dpi)))
                except Exception as error:
                    result['callosal_angle'] = {'error': str(error)}
            else:
                try:
                    desi_result = compute_desi(aligned, aligned_ac, aligned_pc, spacing,
                                               self.desi_model, self.device)
                    result['desi'] = {k: v for k, v in desi_result.items()
                                      if not isinstance(v, np.ndarray)}
                    figure = plotting.desi_figure(aligned, aligned_pc, desi_result, spacing)
                    panels.append(('DESH ratio',
                                   plotting.save(figure, out_dir / f"{case_name}_desi.png", self.dpi)))
                except Exception as error:
                    result['desi'] = {'error': str(error)}

        summary = plotting.summary_report(case_name, result, panels,
                                          self.evans_threshold, self.angle_threshold)
        summary_path = plotting.save(summary, out_dir / f"{case_name}_report.png", self.dpi)

        json_path = out_dir / f"{case_name}_results.json"
        json_path.write_text(json.dumps(result, indent=2))

        result['image'] = str(summary_path)
        result['json'] = str(json_path)
        return result

    def _preprocess(self, raw, out_dir, case_name):
        import nibabel as nib
        import SimpleITK as sitk

        img, transform, info = preprocess.preprocess(raw, n4=self.n4, register=self.register_mni,
                                                     template=self.template)
        if self.register_mni:
            info['template'] = str(self.template_path) if self.template_path else 'MNI152 ICBM 2009c (nilearn)'

        image_path = out_dir / f"{case_name}_preprocessed.nii.gz"
        nib.save(nib.Nifti1Image(img.get_fdata().astype(np.float32), img.affine), str(image_path))
        info['image'] = str(image_path)
        if transform is not None:
            transform_path = out_dir / f"{case_name}_to_mni.tfm"
            sitk.WriteTransform(transform, str(transform_path))
            info['transform'] = str(transform_path)

        figure = plotting.preprocessing_figure(raw, img, self.template_img, info)
        info['figure'] = str(plotting.save(figure, out_dir / f"{case_name}_preprocessing.png", self.dpi))
        return img, info

    def _conditional(self, evans):
        requested = [s for s in CONDITIONAL_STAGES if s in self.stages]
        if not requested or not self.gate_on_evans or evans is None:
            return requested
        if evans['evans_index'] >= self.evans_threshold:
            return requested
        return []


def ivf_geometry(ac_vox, pc_vox, ivf_vox, spacing):
    """Position of the IVF relative to AC and PC, with the anatomical checks used in training QC."""
    ac, pc, ivf = (np.asarray(p, float) * spacing for p in (ac_vox, pc_vox, ivf_vox))
    axis = (ac - pc) / np.linalg.norm(ac - pc)
    up = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
    up /= np.linalg.norm(up)
    distance = float(np.linalg.norm(ivf - ac))
    height = float((ivf - pc) @ up)
    plausible = (IVF_AC_RANGE_MM[0] <= distance <= IVF_AC_RANGE_MM[1] and height > 0
                 and float((ivf - pc) @ axis) > 0)
    return {'ivf_ac_distance_mm': round(distance, 3), 'ivf_height_mm': round(height, 3),
            'ivf_plausible': bool(plausible)}


def summary_row(result):
    evans = result.get('evans') or {}
    angle = result.get('callosal_angle') or {}
    desi = result.get('desi') or {}
    pre = result.get('preprocessing') or {}
    return {
        'case': result['case'],
        'scan': result['scan'],
        'preprocessing': '+'.join(k for k in ('n4', 'mni') if pre.get('mni_registration' if k == 'mni' else k))
                         or 'none',
        'evans_index': None if 'evans_index' not in evans else round(evans['evans_index'], 4),
        'frontal_horn_mm': None if 'horn_width_mm' not in evans else round(evans['horn_width_mm'], 2),
        'inner_skull_mm': None if 'brain_width_mm' not in evans else round(evans['brain_width_mm'], 2),
        'axial_slice': evans.get('axial_z'),
        'callosal_angle_deg': angle.get('callosal_angle_deg'),
        'desi_ratio': desi.get('desi_ratio'),
        'sylvian_volume_ml': desi.get('sylvian_volume_ml'),
        'high_convexity_volume_ml': desi.get('high_convexity_volume_ml'),
        'ventricle_volume_ml': result['segmentation']['ventricle_volume_ml'],
        'icv_volume_ml': result['segmentation']['icv_volume_ml'],
        'acpc_distance_mm': result['acpc']['acpc_distance_mm'],
        'acpc_distance_plausible': result['acpc']['acpc_distance_plausible'],
        'ivf_ac_distance_mm': result['acpc'].get('ivf_ac_distance_mm'),
        'ivf_plausible': result['acpc'].get('ivf_plausible'),
        'alignment': result['acpc']['alignment'],
        'horn_cut': evans.get('horn_cut'),
        'errors': '; '.join(f"{k}: {v['error']}" for k, v in
                            (('evans', evans), ('ca', angle), ('desi', desi)) if 'error' in v),
        'image': result.get('image'),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
