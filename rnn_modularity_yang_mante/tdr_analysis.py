"""
Targeted Dimensionality Reduction (TDR) on trained Mante RNNs.

Replicates Mante et al. (2013) analysis:
  1. Find regression axes for motion coherence, color coherence, context, choice
  2. Measure orthogonality between axes
  3. Ablate along TDR axes and measure causal effects
  4. Compare TDR ablation vs learned decomposition ablation

Usage:
    python tdr_analysis.py --data_dir ./mante_paper --n_seeds 10
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Import from mante_modularity
# =====================================================================
from mante_modularity import (
    ContextRNN, TrialConfig, generate_trials,
    OrthogonalModularity, LinearModularity,
)


# =====================================================================
# TDR: find regression axes
# =====================================================================
def extract_tdr_data(model, cfg, n_trials=5000, device="cuda"):
    """Run many trials and collect hidden states + task variables.

    Returns:
        H: (n_trials, n_units) hidden states at end of stimulus
        motion_coh: (n_trials,) signed motion coherence
        color_coh: (n_trials,) signed color coherence
        context: (n_trials,) +1 or -1
        choice: (n_trials,) 0 or 1 (network's actual choice)
    """
    model.eval()
    inputs, targets, meta = generate_trials(n_trials, cfg, device)

    with torch.no_grad():
        outputs, hidden = model(inputs, return_hidden=True)

    # Hidden state at end of stimulus
    H = hidden[:, -1, :].float().cpu().numpy()  # (n_trials, n_units)

    # Task variables
    motion_coh = meta["motion_coh"].cpu().numpy()
    color_coh = meta["color_coh"].cpu().numpy()
    context = meta["context"].cpu().numpy()
    choice = outputs[:, -1, :].argmax(dim=1).cpu().numpy()

    return H, motion_coh, color_coh, context, choice


def compute_tdr_axes(H, motion_coh, color_coh, context, choice):
    """Compute TDR axes via linear regression.

    For each task variable, find the direction in neural space that
    best predicts that variable via linear regression.

    Returns:
        axes: dict mapping variable name to unit-norm axis (n_units,)
        R2: dict mapping variable name to R^2 of regression
    """
    n_trials, n_units = H.shape

    # Center H
    H_centered = H - H.mean(axis=0, keepdims=True)

    axes = {}
    R2 = {}

    for name, y in [("motion", motion_coh),
                     ("color", color_coh),
                     ("context", context),
                     ("choice", choice.astype(float))]:
        # Center y
        y_centered = y - y.mean()

        # Linear regression: beta = (H^T H)^-1 H^T y
        # Use pseudoinverse for stability
        beta = np.linalg.lstsq(H_centered, y_centered, rcond=None)[0]

        # Normalize to unit vector
        norm = np.linalg.norm(beta)
        if norm > 1e-10:
            axis = beta / norm
        else:
            axis = np.zeros(n_units)

        # R^2
        y_pred = H_centered @ beta
        ss_res = np.sum((y_centered - y_pred) ** 2)
        ss_tot = np.sum(y_centered ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        axes[name] = axis
        R2[name] = float(r2)

    return axes, R2


def measure_orthogonality(axes):
    """Compute pairwise cosine similarity between TDR axes."""
    names = list(axes.keys())
    n = len(names)
    cos_sim = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cos_sim[i, j] = np.dot(axes[names[i]], axes[names[j]])

    return cos_sim, names


# =====================================================================
# Causal ablation along TDR axes
# =====================================================================
def ablate_tdr_axis(h, axis, n_dims=1):
    """Project out the TDR axis from hidden state.

    For n_dims=1, removes the component along `axis`.
    For n_dims>1, axis should be (n_dims, n_units) and we remove all components.
    """
    if axis.ndim == 1:
        axis = axis.reshape(1, -1)

    # Project out: h_new = h - sum_i (h . axis_i) * axis_i
    for i in range(axis.shape[0]):
        a = axis[i]
        proj = (h @ a).unsqueeze(1) * a.unsqueeze(0)  # (batch, n_units)
        h = h - proj

    return h


def evaluate_tdr_ablation(model, cfg, axes, n_trials=5000, device="cuda"):
    """Ablate along TDR axes and measure task performance.

    Tests:
    - No ablation (baseline)
    - Ablate motion axis
    - Ablate color axis
    - Ablate motion+color axes
    - Ablate context axis
    """
    model.eval()
    results = {}

    axis_configs = {
        "none": None,
        "ablate_motion_axis": ["motion"],
        "ablate_color_axis": ["color"],
        "ablate_both_axes": ["motion", "color"],
        "ablate_context_axis": ["context"],
    }

    for config_name, ablate_names in axis_configs.items():
        inputs, targets, meta = generate_trials(n_trials, cfg, device)
        T = inputs.shape[1]

        h = torch.zeros(inputs.shape[0], model.n_units, device=device)
        all_o = []

        # Convert axes to tensors
        if ablate_names is not None:
            ablate_axes = torch.tensor(
                np.stack([axes[name] for name in ablate_names]),
                dtype=torch.float32, device=device)

        for t in range(T):
            u = inputs[:, t, :]
            h = (1 - model.alpha) * h + model.alpha * model.activation(
                model.W_rec(h) + model.W_in(u))

            # Apply ablation during stimulus period
            if ablate_names is not None and t >= cfg.fixation_dur:
                with torch.no_grad():
                    h = ablate_tdr_axis(h.float(), ablate_axes)

            all_o.append(model.W_out(h))

        outputs = torch.stack(all_o, dim=1)
        preds = outputs[:, -1, :].argmax(dim=1)
        tgts = targets

        motion_mask = meta["context"] > 0
        color_mask = ~motion_mask

        # Nonzero coherence
        attended = torch.where(meta["context"] > 0,
                                meta["motion_coh"], meta["color_coh"])
        nonzero = attended.abs() > 0.05

        acc_motion_nz = 0.5
        m_nz = motion_mask & nonzero
        if m_nz.sum() > 0:
            acc_motion_nz = (preds[m_nz] == tgts[m_nz]).float().mean().item()

        acc_color_nz = 0.5
        c_nz = color_mask & nonzero
        if c_nz.sum() > 0:
            acc_color_nz = (preds[c_nz] == tgts[c_nz]).float().mean().item()

        results[config_name] = {
            "acc": (preds == tgts).float().mean().item(),
            "acc_motion_nonzero": acc_motion_nz,
            "acc_color_nonzero": acc_color_nz,
        }

    return results


# =====================================================================
# Compare TDR vs learned decomposition
# =====================================================================
def compare_tdr_vs_learned(model, cfg, axes, decomp_path, dim_A,
                           level="orthogonal", device="cuda", n_trials=5000):
    """Compare TDR ablation vs learned decomposition ablation."""
    # TDR ablation
    tdr_results = evaluate_tdr_ablation(model, cfg, axes, n_trials, device)

    # Load learned decomposition
    if decomp_path and os.path.exists(decomp_path):
        hidden_dim = model.n_units
        if level == "orthogonal":
            decomp = OrthogonalModularity(hidden_dim, dim_A, dim_A)
        else:
            decomp = LinearModularity(hidden_dim, dim_A, dim_A, kappa=2.0)
        decomp.load_state_dict(torch.load(decomp_path, map_location=device))
        decomp = decomp.to(device).eval()

        # Evaluate learned ablation
        from mante_modularity import evaluate_causal
        learned_results = evaluate_causal(model, decomp, cfg, n_trials, device)
    else:
        learned_results = None

    return tdr_results, learned_results


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./mante_paper")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--n_units", type=int, default=256)
    p.add_argument("--n_trials", type=int, default=5000)
    p.add_argument("--dim_A", type=int, default=50,
                   help="dim_A for loading learned decomposition")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrialConfig()

    all_results = {}

    for seed in range(args.n_seeds):
        seed_dir = os.path.join(args.data_dir, f"seed{seed}")
        model_path = os.path.join(seed_dir, "rnn_model.pt")
        if not os.path.exists(model_path):
            print(f"  Skip seed {seed}")
            continue

        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # Load model
        model = ContextRNN(n_units=args.n_units, dt=cfg.dt).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # ---- TDR ----
        print("  Computing TDR axes...")
        H, motion_coh, color_coh, context, choice = extract_tdr_data(
            model, cfg, args.n_trials, device)

        axes, R2 = compute_tdr_axes(H, motion_coh, color_coh, context, choice)

        print(f"    R² — motion: {R2['motion']:.3f}  color: {R2['color']:.3f}  "
              f"context: {R2['context']:.3f}  choice: {R2['choice']:.3f}")

        # Orthogonality
        cos_sim, names = measure_orthogonality(axes)
        print(f"    Cosine similarities:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"      {names[i]:>8} vs {names[j]:<8}: {cos_sim[i,j]:+.3f}")

        # ---- TDR causal ablation ----
        print("  Evaluating TDR ablation...")
        tdr_results = evaluate_tdr_ablation(
            model, cfg, axes, args.n_trials, device)

        bl = tdr_results["none"]
        print(f"    Baseline: motion={bl['acc_motion_nonzero']:.3f}  "
              f"color={bl['acc_color_nonzero']:.3f}")
        for abl_name in ["ablate_motion_axis", "ablate_color_axis"]:
            r = tdr_results[abl_name]
            dM = bl["acc_motion_nonzero"] - r["acc_motion_nonzero"]
            dC = bl["acc_color_nonzero"] - r["acc_color_nonzero"]
            print(f"    {abl_name}: motion={r['acc_motion_nonzero']:.3f}  "
                  f"color={r['acc_color_nonzero']:.3f}  "
                  f"(ΔM={dM:+.3f} ΔC={dC:+.3f})")

        # ---- Compare with learned decomposition ----
        print("  Comparing with learned decomposition...")
        for level in ["orthogonal", "linear_k2.0"]:
            level_short = level.replace("linear_k", "linear_k")
            decomp_path = os.path.join(seed_dir,
                                        f"decomp_{level_short}_d{args.dim_A}.pt")
            if not os.path.exists(decomp_path):
                # Try alternate naming
                decomp_path = os.path.join(seed_dir,
                                            f"decomp_{level}_d{args.dim_A}.pt")

            if os.path.exists(decomp_path):
                hidden_dim = args.n_units
                if "linear" in level:
                    decomp = LinearModularity(hidden_dim, args.dim_A, args.dim_A,
                                              kappa=2.0)
                else:
                    decomp = OrthogonalModularity(hidden_dim, args.dim_A, args.dim_A)
                decomp.load_state_dict(torch.load(decomp_path, map_location=device))
                decomp = decomp.to(device).eval()

                from mante_modularity import evaluate_causal
                learned = evaluate_causal(model, decomp, cfg, args.n_trials, device)

                bl_l = learned["none"]
                for abl in ["ablate_A", "ablate_B"]:
                    r = learned[abl]
                    dM = bl_l["acc_motion_nonzero"] - r["acc_motion_nonzero"]
                    dC = bl_l["acc_color_nonzero"] - r["acc_color_nonzero"]
                    print(f"    Learned {level} {abl}: "
                          f"ΔM={dM:+.3f} ΔC={dC:+.3f}")
            else:
                print(f"    No saved decomposition for {level} d={args.dim_A}")

        # Store results
        all_results[f"seed{seed}"] = {
            "R2": R2,
            "cosine_similarity": {
                f"{names[i]}_vs_{names[j]}": float(cos_sim[i, j])
                for i in range(len(names))
                for j in range(i + 1, len(names))
            },
            "tdr_ablation": tdr_results,
        }

    # ---- Aggregate across seeds ----
    print(f"\n{'='*60}")
    print(f"  AGGREGATE (n={len(all_results)} seeds)")
    print(f"{'='*60}")

    if all_results:
        # R2
        r2_vals = {var: [r["R2"][var] for r in all_results.values()]
                   for var in ["motion", "color", "context", "choice"]}
        print("  R² (mean ± std):")
        for var, vals in r2_vals.items():
            print(f"    {var:>8}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

        # Cosine similarities
        cos_keys = list(list(all_results.values())[0]["cosine_similarity"].keys())
        print("  Cosine similarity (mean ± std):")
        for key in cos_keys:
            vals = [r["cosine_similarity"][key] for r in all_results.values()]
            print(f"    {key:>20}: {np.mean(vals):+.3f} ± {np.std(vals):.3f}")

        # TDR ablation effects
        print("  TDR ablation (mean ± std):")
        for abl_name in ["ablate_motion_axis", "ablate_color_axis"]:
            dMs = []
            dCs = []
            for r in all_results.values():
                bl = r["tdr_ablation"]["none"]
                abl = r["tdr_ablation"][abl_name]
                dMs.append(bl["acc_motion_nonzero"] - abl["acc_motion_nonzero"])
                dCs.append(bl["acc_color_nonzero"] - abl["acc_color_nonzero"])
            print(f"    {abl_name}:")
            print(f"      ΔM = {np.mean(dMs):+.3f} ± {np.std(dMs):.3f}")
            print(f"      ΔC = {np.mean(dCs):+.3f} ± {np.std(dCs):.3f}")

    # Save
    out_path = os.path.join(args.data_dir, "tdr_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
