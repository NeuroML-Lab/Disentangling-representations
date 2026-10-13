"""
Mante modularity figure — 3 panels (no schematic).

a) Double dissociation: accuracy after ablating motion/color subspace (3 levels)
b) DDI vs dimension
c) Accuracy after ablation vs dimension — shows dissociation emerging
"""
import json, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 5.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5,
})

CM, CC = '#2166AC', '#B2182B'  # motion blue, color red
CU, CO, CL = '#999999', '#4477AA', '#CC3311'


def load(dd, ns):
    R = []
    for s in range(ns):
        p = os.path.join(dd, f"seed{s}", "modularity_results.json")
        if os.path.exists(p):
            with open(p) as f:
                R.append(json.load(f))
    return R


def get_causal(R, dim, level):
    dk = f"dim_{dim}"
    o = {k: [] for k in ["bl_m", "bl_c", "aA_m", "aA_c", "aB_m", "aB_c"]}
    for r in R:
        if dk not in r or level not in r[dk]:
            continue
        c = r[dk][level]["causal"]
        o["bl_m"].append(c["none"]["acc_motion_nonzero"])
        o["bl_c"].append(c["none"]["acc_color_nonzero"])
        o["aA_m"].append(c["ablate_A"]["acc_motion_nonzero"])
        o["aA_c"].append(c["ablate_A"]["acc_color_nonzero"])
        o["aB_m"].append(c["ablate_B"]["acc_motion_nonzero"])
        o["aB_c"].append(c["ablate_B"]["acc_color_nonzero"])
    return {k: np.array(v) for k, v in o.items()}


def assign(d):
    """Assign which ablation (A or B) corresponds to motion vs color subspace."""
    if np.mean(d["bl_m"] - d["aB_m"]) > np.mean(d["bl_m"] - d["aA_m"]):
        return "aB", "aA"  # ablB hurts motion more → B is motion sub
    return "aA", "aB"


def get_ddi(R, dim, level):
    dk = f"dim_{dim}"
    out = []
    for r in R:
        if dk not in r or level not in r[dk]:
            continue
        c = r[dk][level]["causal"]
        dMA = c["none"]["acc_motion_nonzero"] - c["ablate_A"]["acc_motion_nonzero"]
        dCA = c["none"]["acc_color_nonzero"] - c["ablate_A"]["acc_color_nonzero"]
        dMB = c["none"]["acc_motion_nonzero"] - c["ablate_B"]["acc_motion_nonzero"]
        dCB = c["none"]["acc_color_nonzero"] - c["ablate_B"]["acc_color_nonzero"]
        if dMB > dMA:
            sm, sc, cm, cc = dMB, dCA, dMA, dCB
        else:
            sm, sc, cm, cc = dMA, dCB, dMB, dCA
        ddi = min(max(sm, 0), max(sc, 0)) - max(max(cm, 0), max(cc, 0))
        out.append(max(ddi, 0))
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./mante_paper")
    p.add_argument("--n_seeds", type=int, default=10)
    args = p.parse_args()
    R = load(args.data_dir, args.n_seeds)
    print(f"Loaded {len(R)} seeds")
    if not R:
        return
    n = len(R)

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35,
                  width_ratios=[1.0, 1.0])

    # ==================================================================
    # Panel A: Double dissociation at dim=50
    # Two sub-panels: ablate motion sub, ablate color sub
    # ==================================================================
    dim_show = 50
    levels = ["unit", "orthogonal", "linear_k2.0"]
    llabels = ["Unit", "Orthogonal", "Linear"]
    bw = 0.35

    for panel_idx, (abl_label, task_that_drops) in enumerate([
            ("Ablate motion sub", "motion"),
            ("Ablate color sub", "color")]):

        ax_b = fig.add_subplot(gs[0, panel_idx])
        x_pos = np.arange(len(levels))

        for li, level in enumerate(levels):
            d = get_causal(R, dim_show, level)
            if len(d["bl_m"]) == 0:
                continue
            msub, csub = assign(d)
            se = lambda v: np.std(v) / np.sqrt(n)

            if panel_idx == 0:
                abl_key = msub
            else:
                abl_key = csub

            mot_acc = np.mean(d[f"{abl_key}_m"])
            col_acc = np.mean(d[f"{abl_key}_c"])
            mot_se = se(d[f"{abl_key}_m"])
            col_se = se(d[f"{abl_key}_c"])

            ax_b.bar(x_pos[li] - bw/2 - 0.02, mot_acc, bw, yerr=mot_se,
                     color=CM, hatch='////', edgecolor='#555', linewidth=0.3,
                     capsize=2, error_kw={'linewidth': 0.5})
            ax_b.bar(x_pos[li] + bw/2 + 0.02, col_acc, bw, yerr=col_se,
                     color=CC, hatch='////', edgecolor='#555', linewidth=0.3,
                     capsize=2, error_kw={'linewidth': 0.5})

        # Baselines
        bl = get_causal(R, dim_show, "orthogonal")
        if len(bl["bl_m"]) > 0:
            ax_b.axhline(np.mean(bl["bl_m"]), color=CM, ls='-', lw=1.0, alpha=0.4,
                         label='Motion baseline')
            ax_b.axhline(np.mean(bl["bl_c"]), color=CC, ls='-', lw=1.0, alpha=0.4,
                         label='Color baseline')

        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels(llabels, fontsize=6)
        ax_b.set_ylim(0, 1.08)
        ax_b.set_title(abl_label, fontsize=9)
        if panel_idx == 0:
            ax_b.set_ylabel('Accuracy after ablation')

        if panel_idx == 1:
            leg = [
                Patch(fc=CM, hatch='////', ec='#555', lw=0.3, label='Motion task'),
                Patch(fc=CC, hatch='////', ec='#555', lw=0.3, label='Color task'),
                mlines.Line2D([], [], color=CM, ls='-', lw=1.0, alpha=0.4, label='Mot. baseline'),
                mlines.Line2D([], [], color=CC, ls='-', lw=1.0, alpha=0.4, label='Col. baseline'),
            ]
            ax_b.legend(handles=leg, fontsize=5, loc='lower right',
                        frameon=True, fancybox=False, edgecolor='#CCC')

        label = chr(ord('a') + panel_idx)
        ax_b.text(-0.15, 1.10, label, transform=ax_b.transAxes, fontsize=12,
                  fontweight='bold', va='top')

    # ==================================================================
    # Panel C: DDI vs dimension
    # ==================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    dims = [2, 5, 10, 15, 20, 50]

    for level, color, label in [("unit", CU, "Unit"),
                                ("orthogonal", CO, "Orthogonal"),
                                ("linear_k2.0", CL, "Linear")]:
        ms, ses = [], []
        for dim in dims:
            dd = get_ddi(R, dim, level)
            ms.append(np.mean(dd) if len(dd) > 0 else np.nan)
            ses.append(np.std(dd) / np.sqrt(len(dd)) if len(dd) > 0 else 0)
        ms, ses = np.array(ms), np.array(ses)
        ax_c.plot(dims, ms, 'o-', color=color, label=label, markersize=3.5, linewidth=1.3)
        ax_c.fill_between(dims, ms - ses, ms + ses, color=color, alpha=0.12)

    ax_c.set_xlabel('Subspace dimension ($d_A = d_B$)')
    ax_c.set_ylabel('DDI')
    ax_c.set_xscale('log')
    ax_c.set_xticks(dims)
    ax_c.set_xticklabels([str(d) for d in dims])
    ax_c.set_ylim(-0.02, 0.35)
    ax_c.set_title('DDI vs subspace dimension', fontsize=9)
    ax_c.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#CCC')
    ax_c.text(-0.15, 1.15, 'c', transform=ax_c.transAxes, fontsize=12,
              fontweight='bold', va='top')

    # ==================================================================
    # Panel D: Accuracy after ablation vs dim (linear k=2)
    # ==================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    dims_d = [2, 5, 10, 15, 20, 50]
    level = "linear_k2.0"

    m_after_ablM, c_after_ablM = [], []
    m_after_ablC, c_after_ablC = [], []
    m_after_ablM_se, c_after_ablM_se = [], []
    m_after_ablC_se, c_after_ablC_se = [], []

    for dim in dims_d:
        d = get_causal(R, dim, level)
        if len(d["bl_m"]) == 0:
            for lst in [m_after_ablM, c_after_ablM, m_after_ablC, c_after_ablC,
                        m_after_ablM_se, c_after_ablM_se, m_after_ablC_se, c_after_ablC_se]:
                lst.append(np.nan)
            continue
        msub, csub = assign(d)
        m_after_ablM.append(np.mean(d[f"{msub}_m"]))
        c_after_ablM.append(np.mean(d[f"{msub}_c"]))
        m_after_ablC.append(np.mean(d[f"{csub}_m"]))
        c_after_ablC.append(np.mean(d[f"{csub}_c"]))
        m_after_ablM_se.append(np.std(d[f"{msub}_m"]) / np.sqrt(n))
        c_after_ablM_se.append(np.std(d[f"{msub}_c"]) / np.sqrt(n))
        m_after_ablC_se.append(np.std(d[f"{csub}_m"]) / np.sqrt(n))
        c_after_ablC_se.append(np.std(d[f"{csub}_c"]) / np.sqrt(n))

    # Ablate motion sub: motion drops, color stays
    ax_d.errorbar(dims_d, m_after_ablM, yerr=m_after_ablM_se,
                  fmt='o-', color=CM, markersize=3.5, linewidth=1.3,
                  label='Motion (abl mot sub)', capsize=2)
    ax_d.errorbar(dims_d, c_after_ablM, yerr=c_after_ablM_se,
                  fmt='s--', color=CC, markersize=3, linewidth=0.9, alpha=0.5,
                  label='Color (abl mot sub)', capsize=2)

    # Ablate color sub: color drops, motion stays
    ax_d.errorbar(dims_d, c_after_ablC, yerr=c_after_ablC_se,
                  fmt='o-', color=CC, markersize=3.5, linewidth=1.3,
                  label='Color (abl col sub)', capsize=2)
    ax_d.errorbar(dims_d, m_after_ablC, yerr=m_after_ablC_se,
                  fmt='s--', color=CM, markersize=3, linewidth=0.9, alpha=0.5,
                  label='Motion (abl col sub)', capsize=2)

    # Baselines
    bl = get_causal(R, dims_d[0], level)
    if len(bl["bl_m"]) > 0:
        ax_d.axhline(np.mean(bl["bl_m"]), color=CM, ls=':', lw=0.6, alpha=0.4)
        ax_d.axhline(np.mean(bl["bl_c"]), color=CC, ls=':', lw=0.6, alpha=0.4)

    ax_d.set_xlabel('Subspace dimension ($d_A = d_B$)')
    ax_d.set_ylabel('Accuracy after ablation')
    ax_d.set_xscale('log')
    ax_d.set_xticks(dims_d)
    ax_d.set_xticklabels([str(d) for d in dims_d])
    ax_d.set_ylim(0.3, 1.05)
    ax_d.set_title('Dissociation emerges (Linear)', fontsize=9)
    ax_d.legend(fontsize=4.8, loc='lower left', frameon=True, fancybox=False,
                edgecolor='#CCC', ncol=1)

    # Annotations
    if not np.isnan(m_after_ablM[-1]):
        ax_d.annotate('targeted task\ndrops', xy=(50, m_after_ablM[-1]),
                      xytext=(30, 0.42), fontsize=5, color='#444', ha='center',
                      arrowprops=dict(arrowstyle='->', color='#888', lw=0.5))
    if not np.isnan(c_after_ablM[-1]):
        ax_d.annotate('other task\npreserved', xy=(50, c_after_ablM[-1]),
                      xytext=(30, 0.92), fontsize=5, color='#444', ha='center',
                      arrowprops=dict(arrowstyle='->', color='#888', lw=0.5))

    ax_d.text(-0.15, 1.15, 'd', transform=ax_d.transAxes, fontsize=12,
              fontweight='bold', va='top')

    out = os.path.join(args.data_dir, "fig_mante_main.pdf")
    plt.savefig(out, bbox_inches='tight', pad_inches=0.08)
    plt.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.08)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()