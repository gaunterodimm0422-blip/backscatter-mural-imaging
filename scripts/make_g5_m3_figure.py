"""G5: plot archived M3 histograms and fits; never refit or alter selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.special import ndtr


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(d):
    """Recompute saved likelihoods from counts, not from reported T alone."""
    x = np.array(d['x_centres_mm'])
    masks = np.array(d['band_mask'], dtype=bool)
    audit = {}
    for key, num in [('signal', 'halfvoid'), ('null', 'ref_null')]:
        f = d['fits'][key]
        nv = np.array(d['hist'][num], dtype=float)
        nr = np.array(d['hist']['ref_main'], dtype=float)
        assert masks.shape == nv.shape == nr.shape
        assert np.all(nv[~masks] == 0) and np.all(nr[~masks] == 0)
        s = d['n_primary'][num] / d['n_primary']['ref_main']
        assert s == f['s'] and nv.sum() == f['n_void'] and nr.sum() == f['n_ref']
        rho = s * np.array(f['A'])[:, None] * (1 - np.array(f['c'])[:, None]
                   * ndtr((x[None, :] - f['xe']) / f['w']))
        def nll(r):
            return float(-2 * np.sum(nv * np.log(r) - (nv + nr) * np.log1p(r)))
        n1 = nll(rho)
        n0 = nll((nv.sum(axis=1) / nr.sum(axis=1))[:, None])
        assert abs(n1 - f['nll']) < 1e-6
        assert abs(n0 - n1 - f['T']) < 1e-6
        audit[key] = dict(T_saved=f['T'], T_recomputed=n0-n1,
                          nll_recomputed=n1, used_bins=masks.sum(axis=1).tolist())
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--null-result', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    d, dn = load(a.result), load(a.null_result)
    assert 'm3_halfvoid_t20_L-15_4m_seed24091301' in d['inputs']['halfvoid']
    assert d['hist'] == dn['hist'] and d['fits']['signal'] == dn['fits']['signal']
    audit = verify(d)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((f for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'] if f in fonts), None)
    if cjk is None:
        raise RuntimeError('Chinese font unavailable; stop rather than emit missing glyphs')
    plt.rcParams.update({'font.family': [cjk, 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 10})
    colors = ['#85a9c3', '#367da6', '#245189', '#172b50']
    markers = ['o', 's', '^', 'D']
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.6), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.29, top=.72, wspace=.12)
    head = d['head_x_mm']
    xh = np.array(d['x_centres_mm'])
    x = xh + head
    mask = np.array(d['band_mask'], dtype=bool)
    rows = []
    for j, (key, num) in enumerate([('signal', 'halfvoid'), ('null', 'ref_null')]):
        ax, f = axes[j], d['fits'][key]
        s = d['n_primary'][num]/d['n_primary']['ref_main']
        ax.axhline(1, color='#999999', lw=.9, ls='--')
        for b, (lo, hi) in enumerate(d['bands_mm']):
            nv, nr = np.array(d['hist'][num][b], float), np.array(d['hist']['ref_main'][b], float)
            use = mask[b] & (nr > 0)
            # Display-only first-order propagation for independent Poisson counts.
            r = nv[use] / (s * nr[use])
            err = np.sqrt(nv[use]/(s*nr[use])**2 + nv[use]**2/(s*s*nr[use]**3))
            ax.errorbar(x[use], r, yerr=err, fmt=markers[b], color=colors[b],
                        ms=3.6, elinewidth=.75, capsize=1.8, label=f'{lo}–{hi} mm')
            xx = np.linspace(xh[use].min(), xh[use].max(), 400)
            yy = f['A'][b]*(1-f['c'][b]*ndtr((xx-f['xe'])/f['w']))
            ax.plot(xx+head, yy, color=colors[b], lw=1.45, ls='-' if j == 0 else '--')
            for idx, rr, ee in zip(np.flatnonzero(use), r, err):
                rows.append([key,lo,hi,xh[idx],x[idx],int(nv[idx]),int(nr[idx]),s,rr,ee])
        ax.set_xlim(x.min()-2, x.max()+2)
        ax.set_xlabel('横向位置 x（世界坐标，mm）', labelpad=9)
        ax.grid(axis='y', color='#eeeeee', lw=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('按初级事例数归一的计数比值')
    # Use all uncertainty bounds; do not hide low-statistics edge bins.
    axes[0].set_ylim(min(.55, min(r[-2]-r[-1] for r in rows)-.04),
                     max(1.3, max(r[-2]+r[-1] for r in rows)+.04))
    f = d['fits']['signal']
    xe = head+f['xe']
    axes[0].axvspan(head+f['xe_lo'], head+f['xe_hi'], color='#bc6542', alpha=.13)
    axes[0].axvline(d['truth_edge_world_mm'], color='#666666', ls=':', lw=1.1)
    axes[0].axvline(xe, color='#bc6542', lw=1.2)
    axes[0].set_title('a  空鼓场 / 完好参照场', loc='left', fontsize=12, pad=10)
    axes[1].set_title('b  独立完好场 / 完好参照场（负对照）', loc='left', fontsize=12, pad=10)
    fig.suptitle('M3 构型：空鼓横向边界定位与负对照', x=.075, ha='left', y=.97, fontsize=17)
    fig.text(.075,.895, f'拟合边界 {xe:.2f} mm；真值 {d["truth_edge_world_mm"]:.2f} mm\n'
             f'68% 区间 [{head+f["xe_lo"]:.2f}, {head+f["xe_hi"]:.2f}] mm；T = {f["T"]:.2f} ≥ 30',
             va='top', fontsize=10)
    fn = d['fits']['null']
    fig.text(.555,.895, f'T = {fn["T"]:.2f} < 30；本次负对照未检出边界\n'
             f'距阈值仅 {30-fn["T"]:.2f}；不能据此宣称误报率为零', va='top', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.135),
               ncol=4, frameon=False, title='重建深度带（不是真实深度分层）', fontsize=9)
    fig.text(.075,.075,'点：原始分箱计数比；误差棒：独立泊松计数的一阶传播 ±1σ。线：存档拟合，不重新拟合、不平滑数据。', fontsize=9)
    fig.text(.075,.043,'左图棕线/色带：拟合边界/68%区间；灰虚线：真值，仅供评价。右图虚曲线：未达检出阈值的台阶模型拟合。', fontsize=9)
    fig.text(.075,.012,'每档 400 万初级事例；空鼓厚 20 mm、深 20–40 mm。边界估计偏差不等于成像 FWHM；本图不证明实物性能。', fontsize=9)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(a.out/f'G5_M3_edge_ratio.{ext}', dpi=240, facecolor='white')
    plt.close(fig)
    with (a.out/'G5_M3_source_data.csv').open('w',encoding='utf-8-sig',newline='') as fh:
        writer=csv.writer(fh)
        writer.writerow(['panel','depth_lo_mm','depth_hi_mm','x_head_mm','x_world_mm','n_num','n_ref','exposure_ratio','ratio','ratio_sigma_poisson'])
        writer.writerows(rows)
    audit.update(source=str(a.result.resolve()),source_sha256=hashlib.sha256(a.result.read_bytes()).hexdigest(),
                 null_source=str(a.null_result.resolve()),null_sha256=hashlib.sha256(a.null_result.read_bytes()).hexdigest(),
                 plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 inputs=d['inputs'],primary_events=d['n_primary'],
                 null_trials=dn['null_calibration']['trials'],null_max=dn['null_calibration']['max'],
                 world_edge_mm=xe,world_edge_ci68_mm=[head+f['xe_lo'],head+f['xe_hi']],
                 contrast=f['c'],refit=False,truth_use='Reference line and error evaluation only; no truth-based event filtering in this plotting script')
    (a.out/'G5_M3_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
