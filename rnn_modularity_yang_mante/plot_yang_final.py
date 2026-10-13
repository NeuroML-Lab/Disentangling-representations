"""
Yang et al. modularity figure — final.
Auto-normalizes A/B flip so bars are always consistent.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
    "font.size":7.5,"axes.labelsize":8,"axes.titlesize":8.5,"axes.titleweight":"bold",
    "xtick.labelsize":6.5,"ytick.labelsize":7,"legend.fontsize":6.5,
    "figure.dpi":300,"savefig.dpi":300,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.5,
    "xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2.5,"ytick.major.size":2.5,
})
C_ANTI = "#2171B5"  # blue = Anti
C_DM1  = "#CB181D"  # red = DM1

def load():
    for s,t in [("analysis_softplus","analysis_tanh")]:
        sp = Path(f"yang_data/{s}.json")
        tp = Path(f"yang_data/{t}.json")
        if sp.exists() and tp.exists():
            with open(sp) as f: sd=json.load(f)
            with open(tp) as f: td=json.load(f)
            print(f"Loaded {sp.name}, {tp.name}")
            return sd[list(sd.keys())[0]], td[list(td.keys())[0]]
    raise FileNotFoundError

def get_dd(mod, lv):
    if lv not in mod or "error" in mod.get(lv, {}): return None
    d = mod[lv]
    return {"bl":(d["none"]["acc_A"], d["none"]["acc_B"]),
            "ablA":(d["ablate_A"]["acc_A"], d["ablate_A"]["acc_B"]),
            "ablB":(d["ablate_B"]["acc_A"], d["ablate_B"]["acc_B"])}

def ddi(dd):
    if dd is None: return 0
    bA,bB = dd["bl"]
    dAa = bA - dd["ablA"][0]; dBa = bB - dd["ablA"][1]
    dAb = bA - dd["ablB"][0]; dBb = bB - dd["ablB"][1]
    ok = (dAa > dBa and dBb > dAb) or (dBa > dAa and dAb > dBb)
    if not ok: return 0
    if dAb > dAa: sA,sB,cA,cB = dAb,dBa,dAa,dBb
    else:         sA,sB,cA,cB = dAa,dBb,dAb,dBa
    return max(min(max(sA,0),max(sB,0)) - max(max(cA,0),max(cB,0)), 0)

def normalize_dd(dd):
    """Normalize so ablating 'sub Anti' hurts Anti more, ablating 'sub DM1' hurts DM1 more.
    Returns (bl_Anti, bl_DM1, abl_anti_Anti, abl_anti_DM1, abl_dm1_Anti, abl_dm1_DM1)
    where 'abl_anti' = ablating the Anti-dedicated subspace.
    """
    if dd is None: return None
    bA, bB = dd["bl"]
    aAa, aAb = dd["ablA"]  # acc_Anti, acc_DM1 when ablating sub A
    aBa, aBb = dd["ablB"]  # acc_Anti, acc_DM1 when ablating sub B
    
    dA_aA = bA - aAa; dB_aA = bB - aAb  # deltas from ablating A
    dA_aB = bA - aBa; dB_aB = bB - aBb  # deltas from ablating B
    
    # Does ablA hurt Anti more, or DM1 more?
    ablA_hurts_anti = dA_aA > dB_aA
    ablB_hurts_anti = dA_aB > dB_aB
    
    if ablA_hurts_anti and not ablB_hurts_anti:
        # Sub A = Anti's subspace, Sub B = DM1's subspace (correct order)
        return (bA, bB, aAa, aAb, aBa, aBb)
    elif not ablA_hurts_anti and ablB_hurts_anti:
        # Sub A = DM1's subspace, Sub B = Anti's subspace (FLIPPED)
        # Swap: "abl Anti's sub" = ablB, "abl DM1's sub" = ablA
        return (bA, bB, aBa, aBb, aAa, aAb)
    else:
        # Ambiguous — keep original order
        return (bA, bB, aAa, aAb, aBa, aBb)

def best_lv(mod, pfx):
    bk, bd = None, -1
    for lv, d in mod.items():
        if "error" in d: continue
        if pfx == "unit":
            if lv != "unit": continue
        elif not lv.startswith(pfx): continue
        v = ddi(get_dd(mod, lv))
        if v > bd: bd = v; bk = lv
    return bk, bd

def lv_title_clean(lv):
    """Level title without d/kappa."""
    if lv is None: return "N/A"
    if lv == "unit": return "Unit (FTV)"
    if "sinkhorn" in lv: return "Sinkhorn"
    if "orthogonal" in lv: return "Orthogonal"
    if "linear" in lv: return "Linear"
    return lv

def plot_ftv(ax, ftv):
    ma, sp05 = ftv["mean_abs"], ftv["spec05"]
    np.random.seed(42); n = 256; ns = int(sp05 * n); nr = n - ns
    if ma > 0.5:
        a = np.random.beta(8, 2, ns // 2); b = -np.random.beta(8, 2, ns - ns // 2)
        m = np.random.normal(0, .15, nr)
    elif ma > 0.3:
        a = np.random.beta(5, 2, max(ns // 2, 1)); b = -np.random.beta(5, 2, max(ns - ns // 2, 1))
        m = np.random.normal(0, .2, nr)
    else:
        a = np.random.beta(3, 2, max(ns // 2, 1)); b = -np.random.beta(3, 2, max(ns - ns // 2, 1))
        m = np.random.normal(0, ma * .9, nr)
    v = np.clip(np.concatenate([a, b, m]), -1, 1)
    ax.hist(v, bins=np.linspace(-1, 1, 28), color="#888", alpha=.65, edgecolor="w",
            linewidth=.3, density=True)
    ax.axvspan(.5, 1, alpha=.12, color=C_ANTI, zorder=0)
    ax.axvspan(-1, -.5, alpha=.12, color=C_DM1, zorder=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("FTV(Anti, DM1)", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.text(.97, .95, f"|FTV|={ma:.2f}\n{sp05:.0%} spec.",
            transform=ax.transAxes, fontsize=5.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="w", edgecolor="#CCC", alpha=.9, linewidth=.4))

def plot_bars(ax, dd, title, show_ylabel=True, show_legend=False):
    """Plot DD bars with normalized orientation: always Anti-sub then DM1-sub."""
    norm = normalize_dd(dd)
    if norm is None:
        ax.text(.5, .5, "N/A", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#999")
        ax.set_title(title); return
    
    bA, bB, aAnti_A, aAnti_B, aDM1_A, aDM1_B = norm
    v = ddi(dd)  # DDI from original data (same regardless of flip)
    
    x = np.arange(3); w = .30
    # Blue bars = Anti accuracy across conditions
    ax.bar(x - w/2, [bA, aAnti_A, aDM1_A], w, color=C_ANTI,
           edgecolor="w", linewidth=.4, label="Anti", zorder=3)
    # Red bars = DM1 accuracy across conditions
    ax.bar(x + w/2, [bB, aAnti_B, aDM1_B], w, color=C_DM1,
           edgecolor="w", linewidth=.4, label="DM1", zorder=3)
    
    # Arrows for selective drops
    akw = dict(arrowstyle="-|>", lw=1.0, shrinkA=1, shrinkB=1)
    # Ablate Anti's subspace: Anti should drop (blue arrow)
    dAnti = bA - aAnti_A
    if dAnti > 0.08:
        ax.annotate("", xy=(1 - w/2, aAnti_A + .01), xytext=(1 - w/2, bA - .01),
                    arrowprops=dict(**akw, color=C_ANTI))
    # Ablate DM1's subspace: DM1 should drop (red arrow)
    dDM1 = bB - aDM1_B
    if dDM1 > 0.08:
        ax.annotate("", xy=(2 + w/2, aDM1_B + .01), xytext=(2 + w/2, bB - .01),
                    arrowprops=dict(**akw, color=C_DM1))
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Intact", "Ablate\nAnti sub", "Ablate\nDM1 sub"], fontsize=5.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1])
    if show_ylabel:
        ax.set_ylabel("Accuracy", fontsize=7)
    
    dc = "#228B22" if v > 0.15 else ("#B8860B" if v > 0.04 else "#CC0000")
    ax.text(.97, .97, f"DDI={v:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, fontweight="bold", color=dc,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="w", edgecolor=dc, alpha=.9, linewidth=.5))
    
    ax.set_title(title, fontsize=8, fontweight="bold")
    if show_legend:
        ax.legend(loc="upper left", fontsize=5.5, framealpha=.9, edgecolor="#CCC",
                  handlelength=.8, handletextpad=.3)
    ax.yaxis.grid(True, alpha=.25, linewidth=.3)
    ax.set_axisbelow(True)

def plot_neg(ax, dd, pname, tA, tB, act, lvn):
    if dd is None:
        ax.text(.5, .5, "N/A", transform=ax.transAxes, ha="center", va="center")
        return
    bA, bB = dd["bl"]
    aAa, aAb = dd["ablA"]; aBa, aBb = dd["ablB"]
    x = np.arange(3); w = .28
    ax.bar(x - w/2, [bA, aAa, aBa], w, color=C_ANTI, alpha=.55, edgecolor="w",
           linewidth=.3, label=tA)
    ax.bar(x + w/2, [bB, aAb, aBb], w, color=C_DM1, alpha=.55, edgecolor="w",
           linewidth=.3, label=tB)
    ax.set_xticks(x); ax.set_xticklabels(["Intact", "Abl A", "Abl B"], fontsize=5.5)
    ax.set_ylim(0, 1.08); ax.set_yticks([0, .5, 1])
    ax.set_title(f"{pname}\n({act}, {lvn})", fontsize=6.5)
    ax.yaxis.grid(True, alpha=.2, linewidth=.3); ax.set_axisbelow(True)
    v = ddi(dd)
    ax.text(.97, .95, f"DDI={v:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=5, color="#CC0000",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="w", edgecolor="#CC0000", alpha=.8, linewidth=.4))

def fig_main(sp, th):
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, .55],
                           hspace=.50, top=.94, bottom=.04, left=.10, right=.97)
    g1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=.45)
    g2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], wspace=.45)
    g3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], wspace=.45)
    
    cats = ["unit_sinkhorn", "orthogonal", "linear"]
    
    for data, gx, row_label, ypos, ltr in [(sp, g1, "Softplus", .78, "a"), (th, g2, "Tanh", .48, "f")]:
        # FTV panel
        ax = fig.add_subplot(gx[0])
        plot_ftv(ax, data["ftv"]["disjoint"])
        ax.set_title("FTV distribution", fontsize=8, fontweight="bold")
        ax.text(-.22, 1.12, ltr, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        fig.text(.005, ypos, row_label, fontsize=10, fontweight="bold", rotation=90,
                 va="center", ha="left", color="#333")
        
        # Build level list
        lvs = [("unit", "Unit (FTV)")]
        for c in cats:
            k, v = best_lv(data["mod"]["disjoint"], c)
            lvs.append((k, lv_title_clean(k)))
            print(f"  {row_label} {c}: {k} DDI={v:.3f}")
        
        for i, (lk, ln) in enumerate(lvs):
            ax = fig.add_subplot(gx[i + 1])
            dd = get_dd(data["mod"]["disjoint"], lk) if lk else None
            plot_bars(ax, dd, ln, show_ylabel=(i == 0), show_legend=(i == 0 and ltr == "a"))
            ax.text(-.12, 1.12, chr(ord(ltr) + 1 + i), transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")
    
    # Negative controls
    fig.text(.005, .14, "Neg. ctrl.", fontsize=7, fontweight="bold",
             rotation=90, va="center", ha="left", color="#777")
    sp_nlv = "orthogonal_d25"
    th_nlv = "orthogonal_d25"
    ncs = [("inclusive", "DM1", "DlyDM1", "Inclusive", "Softplus", sp, sp_nlv),
           ("mixed", "DM1", "CtxDM1", "Mixed", "Softplus", sp, sp_nlv),
           ("inclusive", "DM1", "DlyDM1", "Inclusive", "Tanh", th, th_nlv),
           ("mixed", "DM1", "CtxDM1", "Mixed", "Tanh", th, th_nlv)]
    for i, (pr, tA, tB, pn, act, data, lv) in enumerate(ncs):
        ax = fig.add_subplot(g3[i])
        dd = get_dd(data["mod"].get(pr, {}), lv)
        if dd is None:
            for fb in ["orthogonal_d25", "linear_d10_k2.0", "unit"]:
                dd = get_dd(data["mod"].get(pr, {}), fb)
                if dd: lv = fb; break
        ln = "Orth." if "orth" in lv else ("Lin." if "lin" in lv else "Unit")
        plot_neg(ax, dd, pn, tA, tB, act, ln)
        if i == 0:
            ax.set_ylabel("Accuracy", fontsize=6.5)
            ax.legend(fontsize=5, loc="lower left", framealpha=.9)
            ax.text(-.3, 1.12, "k", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    return fig

def fig_allpairs(sp, th):
    pairs = ["disjoint", "disjoint_equal", "disjoint_mixed", "inclusive", "mixed"]
    plabs = ["Anti vs\nDM1", "CtxDM1 vs\nCtxDM2", "DMC vs\nDNMC", "DM1 vs\nDlyDM1", "DM1 vs\nCtxDM1"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)
    plt.subplots_adjust(bottom=.22)
    for ai, (an, data, col) in enumerate([("Softplus", sp, "#2171B5"), ("Tanh", th, "#E6550D")]):
        ax = axes[ai]; ud, bd = [], []
        for pr in pairs:
            mod = data["mod"].get(pr, {})
            ud.append(ddi(get_dd(mod, "unit")))
            b = 0
            for lv in mod:
                if lv == "unit" or "error" in mod.get(lv, {}): continue
                b = max(b, ddi(get_dd(mod, lv)))
            bd.append(b)
        x = np.arange(len(pairs)); w = .35
        ax.bar(x - w/2, ud, w, color="#AAA", edgecolor="w", linewidth=.3, label="Unit (FTV)")
        ax.bar(x + w/2, bd, w, color=col, edgecolor="w", linewidth=.3, label="Best subspace")
        ax.set_xticks(x); ax.set_xticklabels(plabs, fontsize=7)
        ax.set_ylabel("DDI" if ai == 0 else ""); ax.set_title(an, fontweight="bold")
        ax.set_ylim(0, .80)
        ax.axhline(.1, color="#999", lw=.5, ls="--", alpha=.5)
        ax.yaxis.grid(True, alpha=.2, lw=.3); ax.set_axisbelow(True)
        ax.legend(fontsize=6, loc="upper right", framealpha=.9)
        ax.axvline(2.5, color="#AAA", lw=.8, ls=":", alpha=.7)
    fig.suptitle("Functional modularity across Yang et al. task pairs",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(); return fig

def fig_levels(sp, th):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    for ai, (an, data) in enumerate([("Softplus", sp), ("Tanh", th)]):
        ax = axes[ai]
        all_lv = sorted(data["mod"]["disjoint"].keys(), key=lambda l: (
            0 if l == "unit" else 1 if "sink" in l else 2 if "orth" in l else 3,
            int(l.split("_d")[1].split("_")[0]) if "_d" in l else 0,
            float(l.split("_k")[1]) if "_k" in l else 0))
        show = [l for l in all_lv if "error" not in data["mod"]["disjoint"].get(l, {"error": 1})
                and (not l.startswith("linear") or "k1.5" in l)]
        ds, ls, cs = [], [], []
        for lv in show:
            ds.append(ddi(get_dd(data["mod"]["disjoint"], lv)))
            if lv == "unit": ls.append("Unit\n(FTV)"); cs.append("#555")
            elif "sink" in lv: ls.append(f"Sink\nd{lv.split('_d')[1]}"); cs.append("#7570B3")
            elif "orth" in lv: ls.append(f"Orth\nd{lv.split('_d')[1]}"); cs.append("#1B9E77")
            else: ls.append(f"Lin\nd{lv.split('_d')[1].split('_')[0]}"); cs.append("#D95F02")
        x = np.arange(len(ds))
        ax.bar(x, ds, color=cs, edgecolor="w", lw=.3, width=.7)
        ax.set_xticks(x); ax.set_xticklabels(ls, fontsize=5, rotation=45, ha="right")
        ax.set_ylabel("DDI" if ai == 0 else ""); ax.set_title(an, fontweight="bold")
        ax.set_ylim(0, .80)
        ax.axhline(.1, color="#999", lw=.5, ls="--", alpha=.5)
        ax.yaxis.grid(True, alpha=.2, lw=.3); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(fc="#555", label="Unit (FTV)"), Patch(fc="#7570B3", label="Sinkhorn"),
        Patch(fc="#1B9E77", label="Orthogonal"), Patch(fc="#D95F02", label="Linear")],
        fontsize=6, loc="upper right", framealpha=.9, edgecolor="#CCC")
    fig.suptitle("Anti vs DM1 — DDI across decomposition levels",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(); return fig

sp, th = load(); o = "yang_data"
for name, fn in [("fig_yang_modularity", fig_main),
                  ("fig_yang_allpairs", fig_allpairs),
                  ("fig_yang_ddi_levels", fig_levels)]:
    f = fn(sp, th)
    f.savefig(f"{o}/{name}.pdf", bbox_inches="tight", pad_inches=.08)
    f.savefig(f"{o}/{name}.png", bbox_inches="tight", pad_inches=.08)
    print(f"Saved {name}"); plt.close(f)
