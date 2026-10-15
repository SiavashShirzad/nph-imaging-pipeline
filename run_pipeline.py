#!/usr/bin/env python3
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inputs
from config import CALLOSAL_ANGLE_THRESHOLD, EVANS_THRESHOLD, MODEL_DIR, STAGES


def parse_args():
    p = argparse.ArgumentParser(
        description="NPH imaging pipeline: AC-PC detection and alignment, automated ventricle and "
                    "ICV segmentation, Evans Index, callosal angle and DESH ratio.")
    p.add_argument('input', nargs='+',
                   help="NIfTI file, DICOM series folder, or a folder containing either")
    p.add_argument('-o', '--out', default='output', help="output folder")
    p.add_argument('-m', '--models', default=str(MODEL_DIR),
                   help="folder holding acpc/, icv/, ventricles/, ca/ and desi/")
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--csv', default=None, help="write a summary table to this path")
    p.add_argument('--frac', type=float, default=0.2,
                   help="AC-PC fraction defining the frontal horn cut plane when no plausible IVF "
                        "is available (default: 0.2)")

    p.add_argument('--stages', nargs='+', choices=STAGES, default=None,
                   help=f"stages to run (default: {' '.join(STAGES)})")
    p.add_argument('--evans-threshold', type=float, default=EVANS_THRESHOLD,
                   help=f"Evans Index above which the callosal angle and DESH ratio are computed "
                        f"(default: {EVANS_THRESHOLD})")
    p.add_argument('--no-threshold', action='store_true',
                   help="run the callosal angle and DESH ratio regardless of the Evans Index")
    p.add_argument('--angle-threshold', type=float, default=CALLOSAL_ANGLE_THRESHOLD,
                   help=f"callosal angle flagged below this value (default: {CALLOSAL_ANGLE_THRESHOLD})")

    pre = p.add_argument_group('preprocessing (off by default)')
    pre.add_argument('--preprocess', action='store_true',
                     help="N4 bias correction followed by rigid MNI152 registration "
                          "(same as --n4 --mni)")
    pre.add_argument('--n4', action='store_true', help="N4 bias field correction")
    pre.add_argument('--mni', action='store_true',
                     help="rigid (6 DOF) registration and resampling to the MNI152 template")
    pre.add_argument('--template', default=None,
                     help="T1 template for --mni (default: MNI152 ICBM 2009c 1 mm bundled with nilearn)")

    p.add_argument('--landmark-weights', default=None,
                   help="AC/PC/IVF landmark checkpoint (default: <models>/acpc/AC_PC_IVF_BestModel.pth)")
    p.add_argument('--no-ivf', action='store_true',
                   help="ignore the detected IVF; the frontal horn cut uses --frac")
    p.add_argument('--icv-folds', default='all')
    p.add_argument('--icv-checkpoint', default='checkpoint_final.pth')
    p.add_argument('--vent-folds', default='0')
    p.add_argument('--vent-checkpoint', default='checkpoint_best.pth')
    p.add_argument('--dpi', type=int, default=150, help="figure resolution (default: 150)")
    p.add_argument('--overwrite', action='store_true',
                   help="recompute cases whose report already exists")
    return p.parse_args()


def parse_folds(text):
    return tuple(f if f == 'all' else int(f) for f in text.split(','))


def collect(paths):
    cases = []
    for raw in paths:
        root = Path(raw)
        found = inputs.discover(root)
        if not found:
            print(f"[warn] nothing usable found in {raw}", file=sys.stderr)
        base = root if root.is_dir() else root.parent
        for scan in found:
            cases.append((scan, inputs.case_name(scan, base)))
    return cases


def main():
    args = parse_args()
    from banner import print_banner
    print_banner()
    cases = collect(args.input)
    if not cases:
        sys.exit("no scans found")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.template and not (args.mni or args.preprocess):
        sys.exit("--template requires --mni or --preprocess")

    from pipeline import NPHPipeline, summary_row, write_csv

    stages = tuple(args.stages) if args.stages else STAGES
    gate = not (args.no_threshold or args.stages is not None)

    pipeline = NPHPipeline(
        args.models,
        device=args.device,
        landmark_weights=args.landmark_weights,
        use_ivf=not args.no_ivf,
        icv_folds=parse_folds(args.icv_folds),
        icv_checkpoint=args.icv_checkpoint,
        vent_folds=parse_folds(args.vent_folds),
        vent_checkpoint=args.vent_checkpoint,
        stages=stages,
        evans_threshold=args.evans_threshold,
        angle_threshold=args.angle_threshold,
        gate_on_evans=gate,
        dpi=args.dpi,
        n4=args.n4 or args.preprocess,
        register_mni=args.mni or args.preprocess,
        template=args.template,
    )

    rows = []
    for i, (scan, name) in enumerate(cases, 1):
        case_dir = out_dir / name
        if (case_dir / f"{name}_report.png").exists() and not args.overwrite:
            print(f"[{i}/{len(cases)}] skip {name} (report exists)")
            continue
        print(f"[{i}/{len(cases)}] {name}")
        try:
            result = pipeline.run(scan, case_dir, case_name=name, frac=args.frac)
        except Exception:
            traceback.print_exc()
            print(f"[fail] {scan}", file=sys.stderr)
            continue

        row = summary_row(result)
        if row['evans_index'] is not None:
            print(f"        Evans Index = {row['evans_index']:.3f}", end='')
            if row['callosal_angle_deg'] is not None:
                print(f"   Callosal angle = {row['callosal_angle_deg']:.1f} deg", end='')
            if row['desi_ratio'] is not None:
                print(f"   DESH ratio = {row['desi_ratio']:.2f}", end='')
            print()
        if row['ivf_plausible'] is False:
            print(f"        [warn] IVF position implausible ({row['ivf_ac_distance_mm']:.1f} mm from AC); "
                  f"frontal horn cut fell back to --frac", file=sys.stderr)
        if not row['acpc_distance_plausible']:
            print(f"        [warn] AC-PC distance {row['acpc_distance_mm']:.1f} mm is outside "
                  f"15-40 mm; check the alignment figure", file=sys.stderr)
        print(f"        -> {result['image']}")
        rows.append(row)

    if rows:
        write_csv(args.csv or (out_dir / 'results.csv'), rows)
        print(f"summary written to {args.csv or (out_dir / 'results.csv')}")


if __name__ == '__main__':
    main()
