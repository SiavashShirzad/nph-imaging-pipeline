# NPH Imaging Pipeline

Developed by the **[Brain Signatures Lab](https://www.brainsigns.net/)**.

Automated imaging markers of normal pressure hydrocephalus from a raw 3D T1-weighted MRI. The
pipeline detects the anterior and posterior commissures and the interventricular foramen (IVF) with a
3D keypoint network, rigidly aligns
the scan to the AC-PC plane, segments the lateral ventricles and the intracranial volume with
nnU-Net, and measures the Evans Index, the callosal angle and the DESH ratio.

Nothing has to be prepared by hand: ventricular and ICV masks are produced by the pipeline, not
supplied. Suggested input is an MNI-registered, N4 bias-corrected scan; both steps can be run by
the pipeline itself with `--preprocess`.

## Pipeline

0. **Preprocess** (optional) — N4 bias field correction in native space, then rigid (6 DOF)
   Mattes mutual-information registration to the MNI152 template and a single resampling onto the
   template grid. Enabled with `--preprocess`, or one step at a time with `--n4` / `--mni`.
1. **Align** — the BrainSignsNET AC-PC-IVF keypoint network detects AC, PC and the IVF, and the
   volume and its masks are rigidly resampled so the AC-PC line points anteriorly. Without MNI
   registration both yaw and pitch are corrected; after MNI registration, which already sets yaw
   and roll, only pitch is corrected. Roll is never changed by this step.
2. **Evans** — the lateral ventricles and the ICV are segmented with nnU-Net, and the maximal
   frontal horn width and maximal inner skull diameter are measured on the aligned axial slice.
   The frontal horns are the ventricle anterior to a coronal plane through the detected IVF; if the
   IVF fails its anatomical check, the plane falls back to `--frac` of the way from AC to PC.
3. **Callosal angle** — the lateral ventricles are segmented on the coronal slice through the PC
   and the angle between the superior ventricular walls is measured.
4. **DESH ratio** — Sylvian fissures and high-convexity sulci are segmented across the coronal
   slices spanning AC to PC, and the ratio of their volumes is reported.

Stages 3 and 4 run only when the Evans Index reaches the threshold (0.30). The threshold is
configurable, the gate can be switched off, and any stage can be run on its own.

## Install

```bash
git clone https://github.com/SiavashShirzad/nph-imaging-pipeline.git
cd nph-imaging-pipeline
pip install -r requirements.txt
python download_models.py
```

`download_models.py` fetches the trained models (about 450 MB) from the
[v1.0 release](https://github.com/SiavashShirzad/nph-imaging-pipeline/releases/tag/v1.0), verifies
the checksum and unpacks them into `model/`; see [model/README.md](model/README.md) for the layout
and for installing from a manually downloaded zip (`python download_models.py --zip <file>`).

Python 3.10 or newer is required. A CUDA GPU is strongly recommended; CPU inference works but is
slow.

That is the whole setup: install the requirements, download the models, and run `run_pipeline.py`.
There is nothing to install or import — the scripts sit next to each other in one folder:

```
nph-imaging-pipeline/
    run_pipeline.py     run this
    config.py           thresholds, model paths and stage names
    inputs.py           scan discovery, NIfTI and DICOM loading
    landmarks.py        AC / PC / IVF detection model
    alignment.py        rigid AC-PC alignment
    segmentation.py     nnU-Net ventricle and ICV segmentation
    measure.py          Evans Index
    callosal_angle.py   callosal angle
    desi.py             DESH ratio
    plotting.py         figures and reports
    preprocess.py       optional N4 bias correction and rigid MNI152 registration
    pipeline.py         stage orchestration
    banner.py           Brain Signatures Lab banner
    download_models.py  fetches the trained models into model/
    model/              the trained models
    requirements.txt
```

`run_pipeline.py` works from any working directory, either from inside the folder or by absolute
path; the default model folder is resolved relative to the script, not to the current directory.

## Usage

```bash
# single NIfTI
python run_pipeline.py /data/sub-01_T1w.nii.gz -o results

# single DICOM series folder
python run_pipeline.py /data/dicom/sub-01/MPRAGE -o results

# a folder of scans (NIfTI files and/or DICOM series subfolders, searched recursively)
python run_pipeline.py /data/cohort -o results --csv results/nph.csv
```

Options:

| flag | default | meaning |
| --- | --- | --- |
| `-o, --out` | `output` | folder where per-case results are written |
| `-m, --models` | `model` | folder holding `acpc/`, `icv/`, `ventricles/`, `ca/`, `desi/` |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--csv` | `<out>/results.csv` | summary table of all processed scans |
| `--stages` | all | any of `align evans ca desi` |
| `--evans-threshold` | `0.30` | Evans Index gating the callosal angle and DESH ratio |
| `--no-threshold` | off | run every requested stage regardless of the Evans Index |
| `--angle-threshold` | `90` | callosal angle flagged below this value |
| `--frac` | `0.2` | AC-PC fraction for the frontal horn cut, used only without a plausible IVF |
| `--landmark-weights` | `<models>/acpc/AC_PC_IVF_BestModel.pth` | AC/PC/IVF landmark checkpoint |
| `--no-ivf` | off | ignore the detected IVF; the frontal horn cut uses `--frac` |
| `--preprocess` | off | N4 bias correction and rigid MNI152 registration (`--n4 --mni`) |
| `--n4` | off | N4 bias field correction only |
| `--mni` | off | rigid registration and resampling to MNI152 only |
| `--template` | nilearn MNI152 | T1 template for `--mni`; the default is the brain-only ICBM 2009c 1 mm |
| `--dpi` | `150` | figure resolution |
| `--overwrite` | off | recompute cases whose report already exists |

Naming stages explicitly disables the Evans gate, so a stage always runs when it is asked for:

```bash
python run_pipeline.py /data/sub-01_T1w.nii.gz --stages evans
python run_pipeline.py /data/sub-01_T1w.nii.gz --stages ca desi
python run_pipeline.py /data/cohort --evans-threshold 0.30
python run_pipeline.py /data/cohort --no-threshold
python run_pipeline.py /data/cohort --preprocess
python run_pipeline.py /data/cohort --preprocess --template /data/MNI152_T1_1mm.nii.gz
```

Call it from Python instead, with the repository folder as the working directory:

```python
from pipeline import NPHPipeline

pipeline = NPHPipeline("model", device="cuda")
result = pipeline.run("/data/sub-01_T1w.nii.gz", "results/sub-01", case_name="sub-01")
print(result["evans"]["evans_index"], result["callosal_angle"]["callosal_angle_deg"])
```

## Input requirements

3D T1-weighted MPRAGE/SPGR scans, either as NIfTI or as a DICOM series folder. Volumes are
reoriented to the closest canonical (RAS) orientation before processing, so arbitrary acquisition
orientations are handled.

## Output

```
output/
    results.csv                     one row per case
    <case>/
        <case>_results.json         all measurements and geometry
        <case>_preprocessed.nii.gz  N4-corrected / MNI-registered T1 (with preprocessing)
        <case>_to_mni.tfm           native-to-template rigid transform (SimpleITK, with --mni)
        <case>_preprocessing.png    input vs preprocessed, template outline overlaid
        <case>_report.png           single-page summary
        <case>_alignment.png        original vs aligned, three planes, AC and PC marked
        <case>_evans.png            aligned views, sagittal midline, annotated measurement slice
        <case>_callosal_angle.png   ventricle contours, midline and fitted walls
        <case>_desi.png             Sylvian and high-convexity projections
```

The summary page carries a metric strip — Evans Index, callosal angle, DESH ratio, frontal horn
width — colour-coded against the configured cut-offs, above the stage figures.

Measurements in the JSON:

| Field | Meaning |
| --- | --- |
| `preprocessing` | steps applied, registration rotation/translation, metric and output paths |
| `acpc.ac_voxel_input` / `pc_voxel_input` | AC and PC in the input volume's voxel grid (the preprocessed volume when preprocessing is on) |
| `acpc.ac_voxel_aligned` / `pc_voxel_aligned` | The same points after AC-PC alignment |
| `acpc.acpc_distance_mm` | AC-PC separation; flagged when outside 15-40 mm |
| `acpc.ivf_voxel_input` / `ivf_voxel_aligned` | IVF in the input and aligned grids |
| `acpc.ivf_ac_distance_mm` / `ivf_height_mm` / `ivf_plausible` | IVF distance from AC, height above the AC-PC line, and whether it passes the training QC (2-15 mm from AC, above the line, anterior to PC) |
| `acpc.alignment` / `pitch_deg` / `yaw_deg` | `pitch` after MNI registration, otherwise `yaw+pitch`, and the angles applied |
| `evans.horn_cut` / `cut_vox` | `IVF`, or the `--frac` fallback, and the cut point in the aligned grid |
| `evans.evans_index` | Frontal horn width over inner skull width |
| `evans.horn_width_mm` / `brain_width_mm` | The two measured widths |
| `evans.axial_z` / `vent_y` / `brain_y` | Slice and rows the measurement was taken on |
| `callosal_angle.callosal_angle_deg` | Angle between the superior ventricular walls |
| `desi.desi_ratio` | Sylvian volume over high-convexity volume |
| `desi.sylvian_volume_ml` / `high_convexity_volume_ml` | Component volumes |

## Citation

This pipeline is developed by the **[Brain Signatures Lab](https://www.brainsigns.net/)** and
implements four published methods. If you use it, please cite the papers behind the measurements
you report.

| Pipeline step | Paper |
| --- | --- |
| AC, PC and IVF detection, AC-PC alignment | BrainSignsNET (*medRxiv* 2025) |
| Evans Index | *Neuroscience Informatics* 2026 |
| Callosal angle | *Fluids and Barriers of the CNS* 2026 |
| DESI (DESH ratio) | *Brain Research Bulletin* 2026 |

1. **BrainSignsNET: AC-PC (and IVF) landmark detection.** Shirzadeh Barough S, Ventura C, Bilgel
   M, Albert MS, Miller MI, Moghekar A. BrainSignsNET: deep learning-based 3D anatomical landmark
   detection in human brain imaging. *medRxiv*. 2025. doi:10.1101/2025.07.31.25332457.
   [medRxiv](https://www.medrxiv.org/content/10.1101/2025.07.31.25332457v1)
2. **Evans Index.** Shirzadeh Barough S, Bilgel M, Moghekar A, Ventura C, Albert MS, Luciano MG,
   Moghekar A. Fully automated deep learning-based pipeline for Evans Index measurement from raw 3D
   MRI. *Neuroscience Informatics*. 2026;6(2):100275. doi:10.1016/j.neuri.2026.100275.
   [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772528626000191)
3. **Callosal angle.** Shirzadeh Barough S, Bilgel M, Ventura C, An L, Moghekar A, Albert MS,
   Miller MI, Luciano MG, Moghekar A. Automated deep learning pipeline for callosal angle
   quantification. *Fluids and Barriers of the CNS*. 2026;23(1):17. doi:10.1186/s12987-025-00750-w.
   [Springer](https://link.springer.com/article/10.1186/s12987-025-00750-w)
4. **DESI.** Shirzadeh Barough S, Ohno S, Bilgel M, Moghekar A, Sair HI, Luciano MG, Moghekar A.
   Disproportionately elevated sulcal index (DESI): an automatically driven index representing
   disproportionate subarachnoid space enlargement in brain MRI scans. *Brain Research Bulletin*.
   2026;241:111928. doi:10.1016/j.brainresbull.2026.111928.
   [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0361923026002145)

BibTeX:

```bibtex
@article{ShirzadehBarough2026EvansIndex,
  title   = {Fully automated deep learning-based pipeline for Evans Index measurement from raw 3D MRI},
  author  = {Shirzadeh Barough, Siavash and Bilgel, Murat and Moghekar, Ameya and Ventura, Catalina and Albert, Marilyn S. and Luciano, Mark G. and Moghekar, Abhay},
  journal = {Neuroscience Informatics},
  volume  = {6},
  number  = {2},
  pages   = {100275},
  year    = {2026},
  issn    = {2772-5286},
  doi     = {10.1016/j.neuri.2026.100275},
  url     = {https://www.sciencedirect.com/science/article/pii/S2772528626000191}
}

@article{ShirzadehBarough2026CallosalAngle,
  title   = {Automated deep learning pipeline for callosal angle quantification},
  author  = {Shirzadeh Barough, Siavash and Bilgel, Murat and Ventura, Catalina and An, Lucas and Moghekar, Ameya and Albert, Marilyn S. and Miller, Michael I. and Luciano, Mark G. and Moghekar, Abhay},
  journal = {Fluids and Barriers of the CNS},
  volume  = {23},
  number  = {1},
  pages   = {17},
  year    = {2026},
  issn    = {2045-8118},
  doi     = {10.1186/s12987-025-00750-w},
  url     = {https://link.springer.com/article/10.1186/s12987-025-00750-w}
}

@article{ShirzadehBarough2026DESI,
  title   = {Disproportionately elevated sulcal index (DESI): An automatically driven index representing disproportionate subarachnoid space enlargement in brain MRI scans},
  author  = {Shirzadeh Barough, Siavash and Ohno, Shin and Bilgel, Murat and Moghekar, Anaya and Sair, Haris I. and Luciano, Mark G. and Moghekar, Abhay},
  journal = {Brain Research Bulletin},
  volume  = {241},
  pages   = {111928},
  year    = {2026},
  issn    = {0361-9230},
  doi     = {10.1016/j.brainresbull.2026.111928},
  url     = {https://www.sciencedirect.com/science/article/pii/S0361923026002145}
}

@article{ShirzadehBarough2025BrainSignsNET,
  title   = {BrainSignsNET: Deep Learning-Based 3D Anatomical Landmark Detection in Human Brain Imaging},
  author  = {Shirzadeh Barough, Siavash and Ventura, Catalina and Bilgel, Murat and Albert, Marilyn S. and Miller, Michael I. and Moghekar, Abhay},
  journal = {medRxiv},
  pages   = {2025.07.31.25332457},
  year    = {2025},
  doi     = {10.1101/2025.07.31.25332457},
  url     = {https://www.medrxiv.org/content/10.1101/2025.07.31.25332457v1}
}
```

## License

Free for non-commercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE); this
covers the code and the trained models. For commercial use, contact the
[Brain Signatures Lab](https://www.brainsigns.net/).

## Disclaimer

Research software. Not a medical device and not intended for clinical decision making. Thresholds
are reporting aids, not diagnostic criteria.
