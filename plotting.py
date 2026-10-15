import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from config import CALLOSAL_ANGLE_THRESHOLD, EVANS_THRESHOLD

FLAG = '#c0392b'
OK = '#1e8449'
MUTED = '#5d6d7e'
PANEL = '#f4f6f8'
IVF_COLOR = 'gold'


def _window(volume, low=1, high=99):
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return None, None
    return np.percentile(finite, low), np.percentile(finite, high)


def _bounds(mask, margin=8):
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return [(0, s) for s in mask.shape]
    return [(max(int(i.min()) - margin, 0), min(int(i.max()) + margin + 1, s))
            for i, s in zip(idx, mask.shape)]


def _panel(ax, image, extent, aspect, title, vmin, vmax):
    ax.imshow(image, cmap='gray', origin='lower', aspect=aspect,
              extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def _extent(a, b):
    return (a[0] - 0.5, a[1] - 0.5, b[0] - 0.5, b[1] - 0.5)


def save_report(out_path, case_name, aligned, aligned_icv, aligned_seg,
                spacing, ac_vox, pc_vox, measurement, ivf_vox=None):
    vmin, vmax = _window(aligned)
    aspect_axial = spacing[1] / spacing[0]
    aspect_sagittal = spacing[2] / spacing[1]
    aspect_coronal = spacing[2] / spacing[0]

    bx, by, bz = _bounds(aligned_icv > 0)
    x_idx = int(round(ac_vox[0]))
    y_idx = int(round(ac_vox[1]))
    z_idx = int(round(ac_vox[2]))
    axial_z = measurement['axial_z']

    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.35], hspace=0.12, wspace=0.3)

    ax = fig.add_subplot(grid[0, 0:2])
    _panel(ax, aligned[bx[0]:bx[1], by[0]:by[1], z_idx].T, _extent(bx, by),
           aspect_axial, f"Aligned axial (z={z_idx})", vmin, vmax)
    ax.scatter(ac_vox[0], ac_vox[1], marker='x', s=60, color='deepskyblue', label='AC')
    ax.scatter(pc_vox[0], pc_vox[1], marker='o', s=40, color='red', label='PC')
    if ivf_vox is not None:
        ax.scatter(ivf_vox[0], ivf_vox[1], marker='^', s=45, color=IVF_COLOR, label='IVF')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.35)

    ax = fig.add_subplot(grid[0, 2:4])
    _panel(ax, aligned[x_idx, by[0]:by[1], bz[0]:bz[1]].T, _extent(by, bz),
           aspect_sagittal, f"Aligned sagittal (x={x_idx})", vmin, vmax)
    ax.scatter(ac_vox[1], ac_vox[2], marker='x', s=60, color='deepskyblue')
    ax.scatter(pc_vox[1], pc_vox[2], marker='o', s=40, color='red')
    if ivf_vox is not None:
        ax.scatter(ivf_vox[1], ivf_vox[2], marker='^', s=45, color=IVF_COLOR)

    ax = fig.add_subplot(grid[0, 4:6])
    _panel(ax, aligned[bx[0]:bx[1], y_idx, bz[0]:bz[1]].T, _extent(bx, bz),
           aspect_coronal, f"Aligned coronal (y={y_idx})", vmin, vmax)
    ax.scatter(ac_vox[0], ac_vox[2], marker='x', s=60, color='deepskyblue')
    ax.scatter(pc_vox[0], pc_vox[2], marker='o', s=40, color='red')
    if ivf_vox is not None:
        ax.scatter(ivf_vox[0], ivf_vox[2], marker='^', s=45, color=IVF_COLOR)

    ax = fig.add_subplot(grid[1, 0:3])
    _panel(ax, aligned[x_idx, by[0]:by[1], bz[0]:bz[1]].T, _extent(by, bz),
           aspect_sagittal, "Sagittal midline (AC-PC plane)", vmin, vmax)
    ax.plot([ac_vox[1], pc_vox[1]], [ac_vox[2], pc_vox[2]],
            color='yellow', linestyle='--', alpha=0.8, linewidth=1.2)
    ax.scatter(ac_vox[1], ac_vox[2], marker='x', s=70, color='deepskyblue', label='AC', zorder=3)
    ax.scatter(pc_vox[1], pc_vox[2], marker='o', s=45, color='red', label='PC', zorder=3)
    if ivf_vox is not None:
        ax.scatter(ivf_vox[1], ivf_vox[2], marker='^', s=55, color=IVF_COLOR, label='IVF', zorder=3)
    if 'cut_vox' in measurement:
        ax.axvline(measurement['cut_vox'][1], color='magenta', linewidth=1.0, alpha=0.7, linestyle=':',
                   label=f"Frontal horn cut ({measurement.get('horn_cut', '')})")
    ax.axhline(axial_z, color='orange', linewidth=1.0, alpha=0.6)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.35)

    ax = fig.add_subplot(grid[1, 3:6])
    _panel(ax, aligned[bx[0]:bx[1], by[0]:by[1], axial_z].T, _extent(bx, by),
           aspect_axial, f"Measurement slice (z={axial_z})", vmin, vmax)
    xs = np.arange(bx[0], bx[1])
    ys = np.arange(by[0], by[1])
    ax.contour(xs, ys, (aligned_icv[bx[0]:bx[1], by[0]:by[1], axial_z] > 0).T.astype(float),
               levels=[0.5], colors='dodgerblue', linewidths=1.2, alpha=0.7)
    ax.contour(xs, ys, (aligned_seg[bx[0]:bx[1], by[0]:by[1], axial_z] == 1).T.astype(float),
               levels=[0.5], colors='cyan', linewidths=1.2, alpha=0.7)

    vx0, vx1 = measurement['horn_x']
    vy = measurement['vent_y']
    ax.plot([vx0, vx1], [vy, vy], color='magenta', linewidth=2.5, label='Frontal horn width')
    ax.text((vx0 + vx1) / 2, vy + 3, f"{measurement['horn_width_mm']:.1f} mm",
            color='magenta', ha='center', fontweight='bold', fontsize=9)

    bx0, bx1 = measurement['brain_x']
    brain_y = measurement['brain_y']
    ax.plot([bx0, bx1], [brain_y, brain_y], color='red', linewidth=2.5, label='Inner skull diameter')
    ax.text((bx0 + bx1) / 2, brain_y + 3, f"{measurement['brain_width_mm']:.1f} mm",
            color='red', ha='center', fontweight='bold', fontsize=9)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.35)

    fig.suptitle(f"{case_name}   |   Evans Index = {measurement['evans_index']:.3f}", fontsize=15)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _overlay(color):
    return ListedColormap([(0, 0, 0, 0), color])


def save(fig, path, dpi=150):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def alignment_figure(original, aligned, spacing):
    """Original versus AC-PC aligned volume in all three planes, with AC and PC marked."""
    aspects = {'axial': spacing[1] / spacing[0], 'sagittal': spacing[2] / spacing[1],
               'coronal': spacing[2] / spacing[0]}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

    for row, (label, state) in enumerate((('Original', original), ('AC-PC aligned', aligned))):
        volume, vent, icv, ac, pc = (state['t1'], state['ventricles'], state['icv'],
                                     state['ac'], state['pc'])
        ivf = state.get('ivf')
        axes_of = {'sagittal': (1, 2), 'coronal': (0, 2), 'axial': (0, 1)}
        vmin, vmax = _window(volume)
        for column, view in enumerate(('sagittal', 'coronal', 'axial')):
            ax = axes[row, column]
            if view == 'sagittal':
                index = int(round(np.clip(ac[0], 0, volume.shape[0] - 1)))
                image, v, b = volume[index].T, vent[index].T, icv[index].T
                ac_xy, pc_xy = (ac[1], ac[2]), (pc[1], pc[2])
            elif view == 'coronal':
                index = int(round(np.clip(pc[1], 0, volume.shape[1] - 1)))
                image, v, b = volume[:, index].T, vent[:, index].T, icv[:, index].T
                ac_xy, pc_xy = (ac[0], ac[2]), (pc[0], pc[2])
            else:
                index = int(round(np.clip(ac[2], 0, volume.shape[2] - 1)))
                image, v, b = volume[:, :, index].T, vent[:, :, index].T, icv[:, :, index].T
                ac_xy, pc_xy = (ac[0], ac[1]), (pc[0], pc[1])

            aspect = aspects[view]
            ax.imshow(image, cmap='gray', origin='lower', aspect=aspect, vmin=vmin, vmax=vmax)
            if np.any(b):
                ax.contour(np.asarray(b, float), levels=[0.5], colors=['dodgerblue'],
                           linewidths=1.0, origin='lower', alpha=0.8)
            ax.imshow(np.ma.masked_where(v == 0, v), cmap=_overlay('cyan'), alpha=0.5,
                      origin='lower', vmin=0, vmax=1, aspect=aspect)
            if view == 'sagittal':
                ax.plot([pc_xy[0], ac_xy[0]], [pc_xy[1], ac_xy[1]], color='yellow',
                        ls='--', lw=1.2, alpha=0.85, zorder=3)
            ax.scatter(*ac_xy, marker='x', s=65, color='deepskyblue', zorder=4)
            ax.scatter(*pc_xy, marker='o', s=42, color='red', zorder=4)
            if ivf is not None:
                ax.scatter(*(ivf[i] for i in axes_of[view]), marker='^', s=45, color=IVF_COLOR, zorder=4)
            ax.set_title(f"{label} — {view}", fontsize=10)
            ax.axis('off')

    handles = [Line2D([], [], marker='x', color='deepskyblue', lw=0, markersize=8, label='AC'),
               Line2D([], [], marker='o', color='red', lw=0, markersize=8, label='PC'),
               *([Line2D([], [], marker='^', color=IVF_COLOR, lw=0, markersize=8, label='IVF')]
                 if aligned.get('ivf') is not None else []),
               Line2D([], [], color='cyan', lw=6, alpha=0.6, label='Ventricles'),
               Line2D([], [], color='dodgerblue', lw=1.6, label='ICV boundary')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('AC-PC alignment and segmentation', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def _three_planes(volume, spacing):
    centre = [s // 2 for s in volume.shape]
    return (('sagittal', volume[centre[0]].T, spacing[2] / spacing[1]),
            ('coronal', volume[:, centre[1]].T, spacing[2] / spacing[0]),
            ('axial', volume[:, :, centre[2]].T, spacing[1] / spacing[0]))


def preprocessing_figure(raw, processed, template, info):
    """Input versus preprocessed volume at the volume centre; template outline when registered."""
    steps = [name for name, done in (('N4', info.get('n4')), ('MNI152 rigid', info.get('mni_registration')))
             if done]
    rows = (('Input', raw), (' + '.join(steps), processed))
    outline, level = None, None
    if template is not None:
        outline = template.get_fdata()
        level = 0.3 * np.percentile(outline[outline > 0], 99)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for row, (label, img) in enumerate(rows):
        volume = img.get_fdata()
        vmin, vmax = _window(volume)
        spacing = img.header.get_zooms()[:3]
        planes = _three_planes(volume, spacing)
        edges = _three_planes(outline, spacing) if outline is not None and row == 1 else None
        for column, (view, image, aspect) in enumerate(planes):
            ax = axes[row, column]
            ax.imshow(image, cmap='gray', origin='lower', aspect=aspect, vmin=vmin, vmax=vmax)
            if edges is not None:
                ax.contour(edges[column][1], levels=[level], colors=['orange'],
                           linewidths=0.9, origin='lower', alpha=0.85)
            ax.set_title(f"{label} — {view}", fontsize=10)
            ax.axis('off')

    if outline is not None:
        fig.legend(handles=[Line2D([], [], color='orange', lw=1.6, label='MNI152 template outline')],
                   loc='lower center', frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
        detail = (f"rotation {', '.join(f'{a:.1f}' for a in info['rotation_deg'])}\u00b0   "
                  f"translation {', '.join(f'{t:.1f}' for t in info['translation_mm'])} mm")
    else:
        detail = ''
    fig.suptitle(f"Preprocessing\n{detail}".strip(), fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def callosal_angle_figure(result):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(result['image'], cmap='gray', extent=[0, 224, 224, 0])

    for contour in result['contours1']:
        ax.plot(contour[:, 1], contour[:, 0], color='#e74c3c', lw=1.4, alpha=0.85)
    for contour in result['contours2']:
        ax.plot(contour[:, 1], contour[:, 0], color='#2980b9', lw=1.4, alpha=0.85)

    (x0, y0), (x1, y1) = result['midline']
    ax.plot([x0, x1], [y0, y1], color='green', ls='--', lw=1.4, alpha=0.7)

    for base, colour in ((result['base1'], '#e74c3c'), (result['base2'], '#2980b9')):
        if base is not None:
            ax.plot(base[0], base[1], 'o', color=colour, markersize=7)
    for start, end in (result['line1'], result['line2']):
        if start is not None and end is not None:
            ax.plot([start[0], end[0]], [start[1], end[1]], color='#f1c40f', lw=3)

    angle = result['callosal_angle_deg']
    if angle is None:
        ax.set_title('Callosal angle unavailable', fontsize=15, color=MUTED)
    else:
        colour = FLAG if angle < CALLOSAL_ANGLE_THRESHOLD else OK
        ax.set_title(f"Callosal angle = {angle:.1f}\u00b0", fontsize=15, fontweight='bold', color=colour)
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.axis('off')
    fig.tight_layout()
    return fig


def desi_figure(aligned, pc_vox, result, spacing=None):
    """Depth heatmaps of the Sylvian and high-convexity compartments.

    Each mask is summed along the anterior-posterior axis, so the value at every
    pixel is the thickness of that structure through the column rather than a
    binary hit, and the third dimension survives the projection.
    """
    fig, ax = plt.subplots(figsize=(9, 8.6))
    index = int(np.clip(round(pc_vox[1]), 0, aligned.shape[1] - 1))
    vmin, vmax = _window(aligned)
    ax.imshow(aligned[:, index].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

    step = 1.0 if spacing is None else float(spacing[1])
    sylvian = result['sylvian_mask'].sum(axis=1).T * step
    convexity = result['high_convexity_mask'].sum(axis=1).T * step

    images = []
    for depth, cmap, label, tint in ((sylvian, 'Reds', 'Sylvian depth (mm)', '#a5281c'),
                                     (convexity, 'Purples', 'High-convexity depth (mm)', '#5b3a9e')):
        if depth.max() <= 0:
            continue
        image = ax.imshow(np.ma.masked_where(depth == 0, depth), cmap=cmap, alpha=0.6,
                          origin='lower', vmin=0, vmax=depth.max())
        images.append((image, label, tint))

    for image, label, tint in images:
        bar = fig.colorbar(image, ax=ax, orientation='horizontal', fraction=0.045,
                           pad=0.055, shrink=0.85)
        bar.set_label(label, fontsize=9, color=tint, fontweight='bold')
        bar.ax.tick_params(labelsize=8, colors=tint)

    ratio = result['desi_ratio']
    heading = 'DESH ratio unavailable' if ratio is None else f"DESH ratio = {ratio:.2f}"
    ax.set_title(f"{heading}\nSylvian {result['sylvian_volume_ml']:.1f} ml    "
                 f"High convexity {result['high_convexity_volume_ml']:.1f} ml",
                 fontsize=15, fontweight='bold', color=MUTED if ratio is None else 'black')
    ax.title.set_linespacing(1.6)
    ax.axis('off')
    fig.tight_layout()
    return fig


def _metric_card(ax, label, value, unit, flag, detail):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color('#d5dbe1')
    ax.set_xticks([])
    ax.set_yticks([])
    colour = MUTED if flag is None else (FLAG if flag else OK)
    ax.text(0.5, 0.78, label, ha='center', va='center', color=MUTED, fontsize=10,
            transform=ax.transAxes)
    ax.text(0.5, 0.44, value, ha='center', va='center', color=colour, fontsize=26,
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.17, f"{unit}   {detail}".strip(), ha='center', va='center', color=MUTED,
            fontsize=9, transform=ax.transAxes)


def summary_report(case_name, results, panels, evans_threshold=EVANS_THRESHOLD,
                   angle_threshold=CALLOSAL_ANGLE_THRESHOLD):
    """Single page: metric cards over the stage figures."""
    from pathlib import Path

    evans = results.get('evans') or {}
    angle = results.get('callosal_angle') or {}
    desi = results.get('desi') or {}

    cards = []
    if 'evans_index' in evans:
        cards.append(('Evans Index', f"{evans['evans_index']:.3f}", '',
                      evans['evans_index'] >= evans_threshold, f"cut-off {evans_threshold:.2f}"))
    elif evans:
        cards.append(('Evans Index', 'n/a', '', None, 'not available'))

    if 'callosal_angle_deg' in angle:
        value = angle['callosal_angle_deg']
        cards.append(('Callosal angle', 'n/a' if value is None else f"{value:.1f}\u00b0", '',
                      None if value is None else value < angle_threshold,
                      f"cut-off {angle_threshold:.0f}\u00b0"))
    elif angle:
        cards.append(('Callosal angle', 'n/a', '', None, 'not available'))

    if 'desi_ratio' in desi:
        ratio = desi['desi_ratio']
        cards.append(('DESH ratio', 'n/a' if ratio is None else f"{ratio:.2f}",
                      'Sylvian / high convexity', None,
                      f"{desi['sylvian_volume_ml']:.0f} / {desi['high_convexity_volume_ml']:.0f} ml"))
    elif desi:
        cards.append(('DESH ratio', 'n/a', '', None, 'not available'))

    if 'horn_width_mm' in evans:
        cards.append(('Frontal horn', f"{evans['horn_width_mm']:.1f}", 'mm', None,
                      f"inner skull {evans['brain_width_mm']:.1f} mm"))

    wide = [(t, p) for t, p in panels if Path(p).is_file() and t in ('Alignment', 'Evans Index')]
    narrow = [(t, p) for t, p in panels if Path(p).is_file() and t not in ('Alignment', 'Evans Index')]

    heights = [1.0] + [3.2] * len(wide) + ([3.0] if narrow else [])
    fig = plt.figure(figsize=(15, 2.2 + sum(heights[1:]) * 2.0))
    grid = fig.add_gridspec(len(heights), 12, height_ratios=heights, hspace=0.14, wspace=0.08)

    for index, card in enumerate(cards[:4]):
        _metric_card(fig.add_subplot(grid[0, index * 3:(index + 1) * 3]), *card)

    row = 1
    for _, path in wide:
        ax = fig.add_subplot(grid[row, :])
        ax.imshow(plt.imread(str(path)))
        ax.axis('off')
        row += 1

    if narrow:
        width = 6 if len(narrow) == 1 else 12 // len(narrow)
        start = 3 if len(narrow) == 1 else 0
        for position, (_, path) in enumerate(narrow):
            left = start + position * width
            ax = fig.add_subplot(grid[row, left:left + width])
            ax.imshow(plt.imread(str(path)))
            ax.axis('off')

    fig.suptitle(f"NPH imaging report — {case_name}", fontsize=17, fontweight='bold')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    return fig
