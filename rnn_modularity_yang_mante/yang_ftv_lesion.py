"""
FTV-based lesioning — replicating Yang et al. (2019) causal analysis.

For each task pair:
  1. Compute FTV for all units
  2. Assign units to clusters based on FTV threshold
  3. Lesion each cluster (zero out those units' outgoing weights)
  4. Measure task performance

This directly replicates Yang et al. Fig 2e and Fig 5b.

Usage:
    python yang_ftv_lesion.py --model_path ./yang_data/softplus_seed0_model.pt --activation softplus
    python yang_ftv_lesion.py --model_path ./yang_data/tanh_seed0_model.pt --activation tanh
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

# Import from yang_modularity
from yang_modularity import (
    MultiTaskRNN, generate_batch, eval_acc, compute_tv, compute_ftv,
    TASK_LIST, N_TASKS, N_INPUT, N_OUTPUT, N_RING, ALPHA, DT, TAU,
    _dec,
)

KEY_PAIRS = {
    "disjoint": ("Anti", "DM1"),
    "inclusive": ("DM1", "Dly_DM1"),
    "mixed": ("DM1", "Ctx_DM1"),
    "disjoint_equal": ("Ctx_DM1", "Ctx_DM2"),
    "disjoint_mixed": ("DMC", "DNMC"),
}


def ftv_lesion(model, TV, tA, tB, n_trials=500, device="cuda",
               thresholds=[0.3, 0.5, 0.7]):
    """Lesion units based on FTV at multiple thresholds.

    Groups:
      A-units: FTV > threshold (specialized for task A)
      B-units: FTV < -threshold (specialized for task B)
      Shared: |FTV| < threshold

    Lesion = zero out unit's activation at every timestep.
    """
    iA, iB = TASK_LIST.index(tA), TASK_LIST.index(tB)
    ftv, active = compute_ftv(TV, iA, iB)

    results = {}

    for thresh in thresholds:
        units_A = (ftv > thresh)
        units_B = (ftv < -thresh)
        n_A = units_A.sum().item()
        n_B = units_B.sum().item()
        n_shared = model.n_units - n_A - n_B

        mask_A = (1 - units_A.float()).to(device)  # 0 for A-units, 1 elsewhere
        mask_B = (1 - units_B.float()).to(device)
        mask_shared = torch.ones(model.n_units, device=device)
        mask_shared[units_A | units_B] = 1.0  # keep specialized
        mask_shared[~(units_A | units_B)] = 0.0  # zero shared
        mask_shared = 1 - mask_shared  # invert: 0 for shared, 1 for specialized
        # Actually: shared mask should zero shared units
        shared_units = ~(units_A | units_B) & active
        mask_shared_correct = (1 - shared_units.float()).to(device)

        thresh_results = {
            "n_A": n_A, "n_B": n_B, "n_shared": n_shared,
            "threshold": thresh,
        }

        # Test each lesion condition
        for lesion_name, mask in [
                ("none", None),
                ("lesion_A_units", mask_A),
                ("lesion_B_units", mask_B),
                ("lesion_shared", mask_shared_correct)]:

            accs = {}
            for tn in [tA, tB]:
                inp, tgt, _ = generate_batch(tn, n_trials)
                inp, tgt = inp.to(device), tgt.to(device)
                B, T, _ = inp.shape
                h = torch.zeros(B, model.n_units, device=device)
                ao = []

                model.eval()
                with torch.no_grad():
                    for t in range(T):
                        h = (1-ALPHA)*h + ALPHA*model.activation(
                            model.W_rec(h) + model.W_in(inp[:, t]))
                        if mask is not None:
                            h = h * mask  # zero out lesioned units
                        ao.append(torch.sigmoid(model.W_out(h)))

                out = torch.stack(ao, 1)
                c = tot = 0
                for b in range(n_trials):
                    tr = tgt[b, -1, 1:].cpu().numpy()
                    oR = out[b, -1, 1:].cpu().numpy()
                    if tr.max() > 0.3:
                        diff = abs(_dec(tr) - _dec(oR))
                        diff = min(diff, 2*np.pi - diff)
                        c += int(diff < np.pi/5); tot += 1
                    else:
                        c += int(out[b, -1, 0].item() > 0.5); tot += 1
                accs[tn] = c / max(tot, 1)

            thresh_results[lesion_name] = {
                "acc_A": accs[tA], "acc_B": accs[tB]
            }

        # Compute selectivity and deltas
        bl = thresh_results["none"]
        for ln in ["lesion_A_units", "lesion_B_units", "lesion_shared"]:
            r = thresh_results[ln]
            r["delta_A"] = bl["acc_A"] - r["acc_A"]
            r["delta_B"] = bl["acc_B"] - r["acc_B"]

        results[f"thresh_{thresh}"] = thresh_results

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--activation", default="softplus")
    p.add_argument("--n_units", type=int, default=256)
    p.add_argument("--n_trials", type=int, default=500)
    p.add_argument("--out_dir", default="./yang_data")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MultiTaskRNN(args.n_units, args.activation).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Check accuracy
    print("Task accuracies:")
    accs = eval_acc(model, device)
    for tn, a in sorted(accs.items(), key=lambda x: x[1]):
        print(f"  {tn}: {a:.3f}")
    print(f"  Mean: {np.mean(list(accs.values())):.3f}")

    # Compute task variance
    print("\nComputing task variance...")
    TV = compute_tv(model, device, nc=200)

    # FTV lesioning for each pair
    all_results = {}
    for pair_name, (tA, tB) in KEY_PAIRS.items():
        print(f"\n{'='*60}")
        print(f"  {pair_name}: {tA} vs {tB}")
        print(f"{'='*60}")

        # Show FTV distribution
        iA, iB = TASK_LIST.index(tA), TASK_LIST.index(tB)
        ftv, active = compute_ftv(TV, iA, iB)
        ftv_active = ftv[active].numpy()
        print(f"  FTV: mean|FTV|={np.abs(ftv_active).mean():.3f}  "
              f"spec>0.5={float((np.abs(ftv_active) > 0.5).mean()):.3f}")

        results = ftv_lesion(model, TV, tA, tB, args.n_trials, device)

        for tk, tv in results.items():
            print(f"\n  {tk}: {tv['n_A']} A-units, {tv['n_B']} B-units, "
                  f"{tv['n_shared']} shared")
            bl = tv["none"]
            print(f"    Baseline: A={bl['acc_A']:.3f}  B={bl['acc_B']:.3f}")

            for ln in ["lesion_A_units", "lesion_B_units", "lesion_shared"]:
                r = tv[ln]
                # Check: does lesioning A-units selectively hurt A?
                print(f"    {ln}: A={r['acc_A']:.3f} B={r['acc_B']:.3f}  "
                      f"(ΔA={r['delta_A']:+.3f} ΔB={r['delta_B']:+.3f})")

            # Double dissociation check
            la = tv["lesion_A_units"]
            lb = tv["lesion_B_units"]
            # Lesion A should hurt A more; lesion B should hurt B more
            dd = (la["delta_A"] > la["delta_B"]) and (lb["delta_B"] > lb["delta_A"])
            print(f"    Double dissociation: {'✓' if dd else '✗'}")

        all_results[pair_name] = results

    # Save
    out_path = os.path.join(args.out_dir,
                             f"ftv_lesion_{args.activation}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
