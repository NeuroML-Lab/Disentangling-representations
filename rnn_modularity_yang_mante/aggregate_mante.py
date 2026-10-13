"""
Aggregate Mante modularity results across seeds.

Usage:
    python aggregate_mante.py --data_dir ./mante_paper --n_seeds 10
"""
import argparse
import json
import os
import numpy as np


def load_all_seeds(data_dir, n_seeds):
    results = []
    for seed in range(n_seeds):
        path = os.path.join(data_dir, f"seed{seed}", "modularity_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
        else:
            print(f"  Warning: seed {seed} not found")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./mante_paper")
    p.add_argument("--n_seeds", type=int, default=10)
    args = p.parse_args()

    results = load_all_seeds(args.data_dir, args.n_seeds)
    n_found = len(results)
    print(f"Found {n_found}/{args.n_seeds} seeds\n")

    if n_found == 0:
        return

    # ============================================================
    # Unit-level selectivity
    # ============================================================
    print("=" * 70)
    print("  UNIT-LEVEL SELECTIVITY (direct, no optimization)")
    print("=" * 70)

    si_vals = [r["unit_direct"]["mean_selectivity_index"] for r in results]
    frac_03 = [r["unit_direct"]["frac_selective_0.3"] for r in results]
    frac_05 = [r["unit_direct"]["frac_selective_0.5"] for r in results]

    print(f"  Mean selectivity index: {np.mean(si_vals):.3f} ± {np.std(si_vals):.3f}")
    print(f"  Frac selective >0.3:    {np.mean(frac_03):.3f} ± {np.std(frac_03):.3f}")
    print(f"  Frac selective >0.5:    {np.mean(frac_05):.3f} ± {np.std(frac_05):.3f}")

    # ============================================================
    # Causal selectivity by dim and level
    # ============================================================
    print(f"\n{'=' * 70}")
    print("  CAUSAL DOUBLE DISSOCIATION")
    print("=" * 70)

    dim_keys = sorted([k for k in results[0].keys() if k.startswith("dim_")],
                       key=lambda x: int(x.split("_")[1]))

    # For each dim and level, collect selectivity across seeds
    for dim_key in dim_keys:
        dim = int(dim_key.split("_")[1])
        print(f"\n  --- dim_A = dim_B = {dim} ({dim}/{256*100:.0f}% of network per subspace) ---")

        levels = sorted(set(
            k for r in results if dim_key in r
            for k in r[dim_key].keys()))

        print(f"  {'Level':>16} | {'Sel(ablA)':>10} | {'Sel(ablB)':>10} | "
              f"{'ΔM(ablA)':>10} | {'ΔC(ablA)':>10} | {'ΔM(ablB)':>10} | {'ΔC(ablB)':>10}")
        print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for level in levels:
            sel_A_list = []
            sel_B_list = []
            dM_ablA_list = []
            dC_ablA_list = []
            dM_ablB_list = []
            dC_ablB_list = []

            for r in results:
                if dim_key not in r or level not in r[dim_key]:
                    continue
                causal = r[dim_key][level].get("causal", {})
                if not causal:
                    continue

                aA = causal.get("ablate_A", {})
                aB = causal.get("ablate_B", {})

                dM_A = aA.get("delta_motion", 0)
                dC_A = aA.get("delta_color", 0)
                dM_B = aB.get("delta_motion", 0)
                dC_B = aB.get("delta_color", 0)

                # Corrected selectivity: which task is more affected?
                # For ablate_A: selectivity = max(|dM|, |dC|) / (|dM| + |dC|)
                total_A = abs(dM_A) + abs(dC_A)
                total_B = abs(dM_B) + abs(dC_B)
                sel_A = max(abs(dM_A), abs(dC_A)) / total_A if total_A > 0.01 else 0.5
                sel_B = max(abs(dM_B), abs(dC_B)) / total_B if total_B > 0.01 else 0.5

                sel_A_list.append(sel_A)
                sel_B_list.append(sel_B)
                dM_ablA_list.append(dM_A)
                dC_ablA_list.append(dC_A)
                dM_ablB_list.append(dM_B)
                dC_ablB_list.append(dC_B)

            if not sel_A_list:
                continue

            def fmt(vals):
                return f"{np.mean(vals):+.3f}±{np.std(vals):.3f}"

            print(f"  {level:>16} | "
                  f"{np.mean(sel_A_list):.3f}±{np.std(sel_A_list):.3f} | "
                  f"{np.mean(sel_B_list):.3f}±{np.std(sel_B_list):.3f} | "
                  f"{fmt(dM_ablA_list):>10} | {fmt(dC_ablA_list):>10} | "
                  f"{fmt(dM_ablB_list):>10} | {fmt(dC_ablB_list):>10}")

    # ============================================================
    # Summary table for paper
    # ============================================================
    print(f"\n\n{'=' * 70}")
    print("  PAPER TABLE: Best selectivity per level and dim")
    print("=" * 70)
    print(f"  {'dim':>4} | {'Orthogonal':>12} | {'Linear κ=1.5':>14} | {'Linear κ=2.0':>14}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*14}-+-{'-'*14}")

    for dim_key in dim_keys:
        dim = int(dim_key.split("_")[1])

        row = {"orthogonal": [], "linear_k1.5": [], "linear_k2.0": []}

        for r in results:
            if dim_key not in r:
                continue
            for level in row.keys():
                if level not in r[dim_key]:
                    continue
                causal = r[dim_key][level].get("causal", {})
                if not causal:
                    continue

                # Average selectivity across both ablations
                sels = []
                for abl in ["ablate_A", "ablate_B"]:
                    a = causal.get(abl, {})
                    dM = abs(a.get("delta_motion", 0))
                    dC = abs(a.get("delta_color", 0))
                    total = dM + dC
                    if total > 0.01:
                        sels.append(max(dM, dC) / total)
                if sels:
                    row[level].append(np.mean(sels))

        def fmt_sel(vals):
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

        print(f"  {dim:>4} | {fmt_sel(row['orthogonal']):>12} | "
              f"{fmt_sel(row['linear_k1.5']):>14} | {fmt_sel(row['linear_k2.0']):>14}")

    # ============================================================
    # Training accuracy summary
    # ============================================================
    print(f"\n\n{'=' * 70}")
    print("  RNN TRAINING ACCURACY")
    print("=" * 70)

    for seed in range(args.n_seeds):
        hist_path = os.path.join(args.data_dir, f"seed{seed}", "train_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)
            last = hist[-1]
            print(f"  Seed {seed}: acc={last['acc']:.3f}  "
                  f"motion={last['acc_motion']:.3f}  "
                  f"color={last['acc_color']:.3f}  "
                  f"nonzero={last['acc_nonzero']:.3f}")


if __name__ == "__main__":
    main()
