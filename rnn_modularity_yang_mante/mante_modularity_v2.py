"""
Context-dependent integration task (Mante et al., 2013, Nature).

The canonical neuroscience paradigm for studying modularity:
A network receives two noisy sensory streams (e.g., motion and color) and
must integrate one while ignoring the other, depending on a context signal.

TASK DESIGN:
  - Two input channels: motion coherence and color coherence (noisy signals)
  - One context input: +1 = attend motion, -1 = attend color
  - Output: binary decision based on the attended input
  - Each trial: fixation (200ms) → stimulus (800ms) → response
  - Time step: 20ms, so 50 steps per trial

The network must learn to:
  1. Represent both inputs
  2. Selectively integrate only the context-relevant input
  3. Produce a decision

MODULARITY QUESTION:
  Does the network separate motion and color processing into distinct subspaces?
  At what level of modularity?
    - Unit-level: do individual neurons specialize for motion vs color?
    - Orthogonal: are motion and color in orthogonal subspaces?
    - Linear: are they in linearly separable (but non-orthogonal) subspaces?

Usage:
    # Train the RNN
    python mante_modularity.py train --n_units 256 --n_epochs 2000 --out_dir ./mante_data

    # Analyze modularity (offline + causal)
    python mante_modularity.py analyze --out_dir ./mante_data --levels unit,orthogonal,linear

    # Sweep subspace dimensions
    python mante_modularity.py analyze --out_dir ./mante_data --dim_A 10,25,50,100

    # Sweep kappa for linear level
    python mante_modularity.py analyze --out_dir ./mante_data --levels linear --kappa 1.0,2.0,5.0,10.0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Task: Context-dependent integration
# =====================================================================
@dataclass
class TrialConfig:
    dt: float = 20.0          # time step (ms)
    fixation_dur: int = 10    # fixation steps (200ms)
    stimulus_dur: int = 40    # stimulus steps (800ms)
    response_dur: int = 0     # no explicit response period; decision read at end
    noise_std: float = 1.0    # noise on sensory inputs
    coherences: Tuple = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)

    @property
    def total_steps(self):
        return self.fixation_dur + self.stimulus_dur


def generate_trials(n_trials, cfg=None, device="cpu"):
    """Generate batch of context-dependent integration trials.

    Returns:
        inputs: (n_trials, T, 3) — [motion, color, context]
        targets: (n_trials,) — correct choice {0, 1}
        metadata: dict with coherences, contexts
    """
    if cfg is None:
        cfg = TrialConfig()

    T = cfg.total_steps
    inputs = torch.zeros(n_trials, T, 3, device=device)

    # Random coherences and contexts
    motion_coh = torch.tensor(
        [cfg.coherences[i] for i in torch.randint(len(cfg.coherences), (n_trials,))],
        device=device)
    color_coh = torch.tensor(
        [cfg.coherences[i] for i in torch.randint(len(cfg.coherences), (n_trials,))],
        device=device)
    context = torch.sign(torch.randn(n_trials, device=device))  # +1 or -1
    context[context == 0] = 1.0

    # Build input sequences
    for t in range(T):
        if t >= cfg.fixation_dur:
            # Stimulus period: coherence + noise
            inputs[:, t, 0] = motion_coh + cfg.noise_std * torch.randn(n_trials, device=device)
            inputs[:, t, 1] = color_coh + cfg.noise_std * torch.randn(n_trials, device=device)
        # Context is always on
        inputs[:, t, 2] = context

    # Target: sign of attended coherence
    attended = torch.where(context > 0, motion_coh, color_coh)
    targets = (attended >= 0).long()  # 0 or 1

    metadata = {
        "motion_coh": motion_coh,
        "color_coh": color_coh,
        "context": context,
    }

    return inputs, targets, metadata


# =====================================================================
# RNN Model
# =====================================================================
class ContextRNN(nn.Module):
    """Continuous-time RNN for context-dependent integration.

    tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b)
    output = W_out @ h

    Discretized with Euler method.
    """
    def __init__(self, n_inputs=3, n_units=256, n_outputs=2,
                 tau=100.0, dt=20.0, noise_std=0.05):
        super().__init__()
        self.n_units = n_units
        self.alpha = dt / tau  # integration constant
        self.noise_std = noise_std

        self.W_in = nn.Linear(n_inputs, n_units, bias=False)
        self.W_rec = nn.Linear(n_units, n_units, bias=True)
        self.W_out = nn.Linear(n_units, n_outputs, bias=False)

        # Initialize recurrent weights with small spectral radius
        with torch.no_grad():
            self.W_rec.weight.copy_(0.1 * torch.randn(n_units, n_units) / (n_units ** 0.5))

        self.activation = torch.tanh

    def forward(self, inputs, return_hidden=False):
        """
        inputs: (batch, T, n_inputs)
        Returns: outputs (batch, T, n_outputs), hidden (batch, T, n_units) if requested
        """
        batch, T, _ = inputs.shape
        device = inputs.device

        h = torch.zeros(batch, self.n_units, device=device)
        all_h = []
        all_o = []

        for t in range(T):
            u = inputs[:, t, :]
            noise = self.noise_std * torch.randn_like(h) if self.training else 0

            # Continuous-time RNN update
            h = (1 - self.alpha) * h + self.alpha * self.activation(
                self.W_rec(h) + self.W_in(u)) + noise

            all_h.append(h)
            all_o.append(self.W_out(h))

        outputs = torch.stack(all_o, dim=1)
        hidden = torch.stack(all_h, dim=1)

        if return_hidden:
            return outputs, hidden
        return outputs


# =====================================================================
# Training
# =====================================================================
def train_rnn(n_units=256, n_epochs=2000, batch_size=256, lr=1e-3,
              device="cuda", seed=42):
    """Train RNN on context-dependent integration task."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = TrialConfig()
    model = ContextRNN(n_units=n_units, dt=cfg.dt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(n_epochs):
        model.train()
        inputs, targets, meta = generate_trials(batch_size, cfg, device)
        outputs = model(inputs)

        # Loss on last 10 time steps (decision period)
        decision_outputs = outputs[:, -10:, :]  # (batch, 10, 2)
        decision_targets = targets.unsqueeze(1).expand(-1, 10)  # (batch, 10)
        loss = F.cross_entropy(decision_outputs.reshape(-1, 2),
                                decision_targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_inputs, test_targets, test_meta = generate_trials(
                    1000, cfg, device)
                test_outputs = model(test_inputs)
                preds = test_outputs[:, -1, :].argmax(dim=1)
                acc = (preds == test_targets).float().mean().item()

                # Accuracy by context
                motion_mask = test_meta["context"] > 0
                acc_motion = (preds[motion_mask] == test_targets[motion_mask]).float().mean().item()
                acc_color = (preds[~motion_mask] == test_targets[~motion_mask]).float().mean().item()

                # Accuracy excluding zero coherence
                nonzero = torch.where(
                    test_meta["context"] > 0,
                    test_meta["motion_coh"],
                    test_meta["color_coh"]).abs() > 0.05
                acc_nonzero = (preds[nonzero] == test_targets[nonzero]).float().mean().item()

            print(f"  Epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}  "
                  f"acc={acc:.3f}  motion={acc_motion:.3f}  color={acc_color:.3f}  "
                  f"nonzero={acc_nonzero:.3f}")
            history.append({
                "epoch": epoch + 1, "loss": loss.item(),
                "acc": acc, "acc_motion": acc_motion,
                "acc_color": acc_color, "acc_nonzero": acc_nonzero,
            })

    return model, cfg, history


# =====================================================================
# Modularity decomposition models (same as vision/LLM scripts)
# =====================================================================
class UnitModularity(nn.Module):
    def __init__(self, hidden_dim, dim_A, dim_B, n_sinkhorn=20, tau=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.log_alpha = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.n_sinkhorn = n_sinkhorn
        self.tau = tau

    def _sinkhorn(self, log_alpha):
        for _ in range(self.n_sinkhorn):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
        return torch.exp(log_alpha)

    def get_permutation(self):
        return self._sinkhorn(self.log_alpha / self.tau)

    def forward(self, z):
        P = self.get_permutation()
        z_perm = z @ P
        return (z_perm[:, :self.dim_A],
                z_perm[:, self.dim_A:self.dim_A + self.dim_B],
                z_perm[:, self.dim_A + self.dim_B:])

    def reconstruct(self, z_A, z_B, z_res):
        P = self.get_permutation()
        return torch.cat([z_A, z_B, z_res], -1) @ P.t()

    def ablate_A(self, z):
        a, b, r = self.forward(z)
        return self.reconstruct(torch.zeros_like(a), b, r)

    def ablate_B(self, z):
        a, b, r = self.forward(z)
        return self.reconstruct(a, torch.zeros_like(b), r)


class OrthogonalModularity(nn.Module):
    def __init__(self, hidden_dim, dim_A, dim_B):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.S_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

    def get_rotation(self):
        S = self.S_params - self.S_params.t()
        I = torch.eye(self.hidden_dim, device=S.device)
        return torch.linalg.solve(I + S, I - S)

    def forward(self, z):
        Q = self.get_rotation()
        z_rot = z @ Q
        return (z_rot[:, :self.dim_A],
                z_rot[:, self.dim_A:self.dim_A + self.dim_B],
                z_rot[:, self.dim_A + self.dim_B:])

    def reconstruct(self, z_A, z_B, z_res):
        Q = self.get_rotation()
        return torch.cat([z_A, z_B, z_res], -1) @ Q.t()

    def ablate_A(self, z):
        a, b, r = self.forward(z)
        return self.reconstruct(torch.zeros_like(a), b, r)

    def ablate_B(self, z):
        a, b, r = self.forward(z)
        return self.reconstruct(a, torch.zeros_like(b), r)


class LinearModularity(nn.Module):
    def __init__(self, hidden_dim, dim_A, dim_B, kappa=5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.kappa = kappa
        self.T = nn.Parameter(
            torch.eye(hidden_dim) + torch.randn(hidden_dim, hidden_dim) * 0.01)

    def _clamp_singular_values(self):
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(self.T.data)
            S = S.clamp(min=1.0 / self.kappa, max=self.kappa)
            self.T.data.copy_(U @ torch.diag(S) @ Vh)

    def forward(self, z):
        z_t = z @ self.T
        return (z_t[:, :self.dim_A],
                z_t[:, self.dim_A:self.dim_A + self.dim_B],
                z_t[:, self.dim_A + self.dim_B:])

    def reconstruct(self, z_A, z_B, z_res):
        T_inv = torch.linalg.inv(self.T)
        return torch.cat([z_A, z_B, z_res], -1) @ T_inv

    def ablate_A(self, z):
        a, b, r = self.forward(z)
        return self.reconstruct(torch.zeros_like(a), b, r)

    def ablate_B(self, z):
        a, b, r = self.forward(z)
        return self.reconstruct(a, torch.zeros_like(b), r)

    def condition_number(self):
        with torch.no_grad():
            s = torch.linalg.svdvals(self.T)
            return (s[0] / s[-1]).item()


# =====================================================================
# Extract hidden states organized by condition
# =====================================================================
def extract_condition_activations(model, cfg, n_trials=2000, device="cuda"):
    """Extract hidden states at end of stimulus period, organized by condition.

    Returns:
        activations_motion: hidden states from motion-context trials (n, hidden_dim)
        activations_color: hidden states from color-context trials (n, hidden_dim)
        losses_motion: per-trial loss for motion context
        losses_color: per-trial loss for color context
    """
    model.eval()
    inputs, targets, meta = generate_trials(n_trials, cfg, device)

    with torch.no_grad():
        outputs, hidden = model(inputs, return_hidden=True)

    # Use hidden state at end of stimulus (last time step)
    h_final = hidden[:, -1, :].float().cpu()  # (n_trials, n_units)

    # Per-trial loss
    logits = outputs[:, -1, :]  # (n_trials, 2)
    per_trial_loss = F.cross_entropy(logits.float(), targets, reduction="none").cpu()

    # Split by context
    motion_mask = meta["context"] > 0
    color_mask = ~motion_mask

    return {
        "activations_motion": h_final[motion_mask.cpu()],
        "activations_color": h_final[color_mask.cpu()],
        "losses_motion": per_trial_loss[motion_mask.cpu()],
        "losses_color": per_trial_loss[color_mask.cpu()],
        "motion_coh": meta["motion_coh"][motion_mask].cpu(),
        "color_coh": meta["color_coh"][color_mask].cpu(),
        "targets_motion": targets[motion_mask].cpu(),
        "targets_color": targets[color_mask].cpu(),
        "full_hidden": hidden.float().cpu(),
        "full_meta": {k: v.cpu() for k, v in meta.items()},
        "full_targets": targets.cpu(),
    }


# =====================================================================
# Train decomposition
# =====================================================================
def train_decomposition(acts_A, acts_B, targets_A, targets_B,
                        level="orthogonal", dim_A=25, dim_B=25,
                        n_epochs=1000, lr=1e-3, lambda_adv=1.0,
                        kappa=5.0, device="cuda"):
    """Train modularity decomposition with task-sufficiency objective.

    A = motion context, B = color context.
    Objective: subspace A sufficient for motion-context choice,
               subspace B sufficient for color-context choice.

    Args:
        acts_A: (n, hidden_dim) hidden states from motion-context trials
        acts_B: (n, hidden_dim) hidden states from color-context trials
        targets_A: (n,) integer targets (choice 0 or 1) for motion trials
        targets_B: (n,) integer targets for color trials
    """
    hidden_dim = acts_A.shape[1]
    n_classes = 2

    if level == "unit":
        decomp = UnitModularity(hidden_dim, dim_A, dim_B).to(device)
    elif level == "orthogonal":
        decomp = OrthogonalModularity(hidden_dim, dim_A, dim_B).to(device)
    elif level == "linear":
        decomp = LinearModularity(hidden_dim, dim_A, dim_B, kappa=kappa).to(device)
    else:
        raise ValueError(f"Unknown level: {level}")

    # Readout heads: predict choice from subspace
    head_A = nn.Linear(dim_A, n_classes).to(device)   # subspace A → motion choice
    head_B = nn.Linear(dim_B, n_classes).to(device)   # subspace B → color choice
    # Adversarial heads
    head_A_adv = nn.Linear(dim_B, n_classes).to(device)  # subspace B → motion choice (should fail)
    head_B_adv = nn.Linear(dim_A, n_classes).to(device)  # subspace A → color choice (should fail)

    opt_main = torch.optim.Adam(
        list(decomp.parameters()) + list(head_A.parameters()) +
        list(head_B.parameters()), lr=lr)
    opt_adv = torch.optim.Adam(
        list(head_A_adv.parameters()) + list(head_B_adv.parameters()), lr=lr)

    Z_A = acts_A.float().to(device)
    Z_B = acts_B.float().to(device)
    T_A = targets_A.long().to(device)
    T_B = targets_B.long().to(device)

    n_min = min(len(Z_A), len(Z_B))
    history = []

    for epoch in range(n_epochs):
        pA = torch.randperm(len(Z_A))[:n_min]
        pB = torch.randperm(len(Z_B))[:n_min]
        zA, tA = Z_A[pA], T_A[pA]
        zB, tB = Z_B[pB], T_B[pB]

        zA_a, zA_b, zA_r = decomp(zA)
        zB_a, zB_b, zB_r = decomp(zB)

        # Sufficiency: subspace A predicts motion choice, B predicts color choice
        loss_suf_A = F.cross_entropy(head_A(zA_a), tA)
        loss_suf_B = F.cross_entropy(head_B(zB_b), tB)

        # Adversarial: train adversary to predict from wrong subspace
        loss_adv = (F.cross_entropy(head_A_adv(zA_b.detach()), tA) +
                    F.cross_entropy(head_B_adv(zB_a.detach()), tB))
        opt_adv.zero_grad()
        loss_adv.backward()
        opt_adv.step()

        # Anti-adversarial
        loss_anti = -(F.cross_entropy(head_A_adv(zA_b), tA) +
                      F.cross_entropy(head_B_adv(zB_a), tB))

        # Reconstruction
        loss_recon = (F.mse_loss(decomp.reconstruct(zA_a, zA_b, zA_r), zA) +
                      F.mse_loss(decomp.reconstruct(zB_a, zB_b, zB_r), zB)) / 2

        loss_total = loss_suf_A + loss_suf_B + lambda_adv * loss_anti + loss_recon

        opt_main.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(decomp.parameters(), 1.0)
        opt_main.step()

        if level == "linear":
            decomp._clamp_singular_values()

        if (epoch + 1) % 200 == 0:
            # Compute sufficiency accuracy
            with torch.no_grad():
                acc_A = (head_A(zA_a).argmax(1) == tA).float().mean().item()
                acc_B = (head_B(zB_b).argmax(1) == tB).float().mean().item()
            cond_str = f"  cond={decomp.condition_number():.1f}" if level == "linear" else ""
            print(f"    [{level}] ep {epoch+1}  suf_A={acc_A:.3f}  "
                  f"suf_B={acc_B:.3f}  adv={loss_adv.item():.4f}  "
                  f"recon={loss_recon.item():.6f}{cond_str}")
            history.append({
                "epoch": epoch + 1,
                "suf_acc_A": acc_A, "suf_acc_B": acc_B,
                "adv": loss_adv.item(), "recon": loss_recon.item(),
            })

    return decomp, history


# =====================================================================
# Evaluation: both offline and causal
# =====================================================================
def evaluate_offline(decomp, acts_A, acts_B, device="cuda"):
    """Offline distortion-based evaluation."""
    decomp.eval()
    Z_A = acts_A.float().to(device)
    Z_B = acts_B.float().to(device)

    with torch.no_grad():
        zA_ablA = decomp.ablate_A(Z_A)
        zB_ablA = decomp.ablate_A(Z_B)
        zA_ablB = decomp.ablate_B(Z_A)
        zB_ablB = decomp.ablate_B(Z_B)

    return {
        "ablate_A": {
            "on_motion": (zA_ablA - Z_A).pow(2).sum().item() / Z_A.pow(2).sum().item(),
            "on_color": (zB_ablA - Z_B).pow(2).sum().item() / Z_B.pow(2).sum().item(),
        },
        "ablate_B": {
            "on_motion": (zA_ablB - Z_A).pow(2).sum().item() / Z_A.pow(2).sum().item(),
            "on_color": (zB_ablB - Z_B).pow(2).sum().item() / Z_B.pow(2).sum().item(),
        },
    }


def evaluate_causal(model, decomp, cfg, n_trials=2000, device="cuda"):
    """Causal evaluation: intervene on hidden states during RNN execution
    and measure task accuracy.

    We intervene at every time step during the stimulus period by
    projecting the hidden state through the decomposition, zeroing one
    component, and reconstructing.
    """
    model.eval()
    decomp.eval()

    results = {}

    for ablation_name in ["none", "ablate_A", "ablate_B"]:
        inputs, targets, meta = generate_trials(n_trials, cfg, device)
        T = inputs.shape[1]

        h = torch.zeros(inputs.shape[0], model.n_units, device=device)
        all_o = []

        for t in range(T):
            u = inputs[:, t, :]
            h = (1 - model.alpha) * h + model.alpha * model.activation(
                model.W_rec(h) + model.W_in(u))

            # Apply ablation during stimulus period
            if t >= cfg.fixation_dur and ablation_name != "none":
                with torch.no_grad():
                    if ablation_name == "ablate_A":
                        h = decomp.ablate_A(h.float()).to(h.dtype)
                    elif ablation_name == "ablate_B":
                        h = decomp.ablate_B(h.float()).to(h.dtype)

            all_o.append(model.W_out(h))

        outputs = torch.stack(all_o, dim=1)
        preds = outputs[:, -1, :].argmax(dim=1)

        motion_mask = meta["context"] > 0
        color_mask = ~motion_mask

        # Overall accuracy
        acc = (preds == targets).float().mean().item()

        # Accuracy by context
        acc_motion = (preds[motion_mask] == targets[motion_mask]).float().mean().item()
        acc_color = (preds[color_mask] == targets[color_mask]).float().mean().item()

        # Accuracy by context, excluding zero coherence
        attended_coh = torch.where(meta["context"] > 0,
                                    meta["motion_coh"], meta["color_coh"])
        nonzero = attended_coh.abs() > 0.05

        acc_motion_nz = 0.5
        motion_nz = motion_mask & nonzero
        if motion_nz.sum() > 0:
            acc_motion_nz = (preds[motion_nz] == targets[motion_nz]).float().mean().item()

        acc_color_nz = 0.5
        color_nz = color_mask & nonzero
        if color_nz.sum() > 0:
            acc_color_nz = (preds[color_nz] == targets[color_nz]).float().mean().item()

        results[ablation_name] = {
            "acc": acc,
            "acc_motion": acc_motion,
            "acc_color": acc_color,
            "acc_motion_nonzero": acc_motion_nz,
            "acc_color_nonzero": acc_color_nz,
        }

    # Compute selectivity
    bl = results["none"]
    for abl_name in ["ablate_A", "ablate_B"]:
        r = results[abl_name]
        dM = bl["acc_motion_nonzero"] - r["acc_motion_nonzero"]
        dC = bl["acc_color_nonzero"] - r["acc_color_nonzero"]
        total = abs(dM) + abs(dC)

        # Label-agnostic: how selective is this ablation?
        sel = max(abs(dM), abs(dC)) / total if total > 1e-6 else 0.5

        r["delta_motion"] = dM
        r["delta_color"] = dC
        r["selectivity"] = sel

    return results


# =====================================================================
# Unit-level analysis (direct, no optimization)
# =====================================================================
def analyze_unit_selectivity(acts_motion, acts_color, losses_motion, losses_color):
    """Direct analysis: for each neuron, compute correlation with motion vs color loss.
    No optimization needed — just statistics.
    """
    n_units = acts_motion.shape[1]

    # Combine and compute per-neuron correlations
    motion_selectivity = []
    color_selectivity = []

    for i in range(n_units):
        # Variance of unit i across motion trials vs color trials
        var_motion = acts_motion[:, i].var().item()
        var_color = acts_color[:, i].var().item()

        # Correlation with loss
        corr_with_motion_loss = torch.corrcoef(
            torch.stack([acts_motion[:, i], losses_motion]))[0, 1].item()
        corr_with_color_loss = torch.corrcoef(
            torch.stack([acts_color[:, i], losses_color]))[0, 1].item()

        motion_selectivity.append(abs(corr_with_motion_loss))
        color_selectivity.append(abs(corr_with_color_loss))

    motion_selectivity = np.array(motion_selectivity)
    color_selectivity = np.array(color_selectivity)

    # Selectivity index: how specialized is each neuron?
    total = motion_selectivity + color_selectivity + 1e-8
    unit_si = np.abs(motion_selectivity - color_selectivity) / total

    return {
        "mean_selectivity_index": float(unit_si.mean()),
        "median_selectivity_index": float(np.median(unit_si)),
        "frac_selective_0.5": float((unit_si > 0.5).mean()),
        "frac_selective_0.3": float((unit_si > 0.3).mean()),
        "motion_selectivity": motion_selectivity.tolist(),
        "color_selectivity": color_selectivity.tolist(),
    }


# =====================================================================
# Main
# =====================================================================
def cmd_train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training RNN: {args.n_units} units, {args.n_epochs} epochs")
    model, cfg, history = train_rnn(
        n_units=args.n_units, n_epochs=args.n_epochs,
        batch_size=args.batch_size, lr=args.lr,
        device=device, seed=args.seed)

    # Save model and config
    torch.save(model.state_dict(), os.path.join(args.out_dir, "rnn_model.pt"))
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Extract and save activations
    print("\nExtracting activations...")
    data = extract_condition_activations(model, cfg, n_trials=args.n_extract, device=device)
    torch.save(data, os.path.join(args.out_dir, "activations.pt"))

    print(f"\nSaved to {args.out_dir}")
    print(f"  Motion trials: {len(data['activations_motion'])}")
    print(f"  Color trials: {len(data['activations_color'])}")
    print(f"  Final accuracy: {history[-1]['acc']:.3f}")


def cmd_analyze(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    cfg = TrialConfig()
    model = ContextRNN(n_units=args.n_units, dt=cfg.dt).to(device)
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "rnn_model.pt"),
                                      map_location=device))
    model.eval()

    # Load activations
    data = torch.load(os.path.join(args.out_dir, "activations.pt"),
                       map_location="cpu")

    acts_M = data["activations_motion"]
    acts_C = data["activations_color"]
    losses_M = data["losses_motion"]
    losses_C = data["losses_color"]
    targets_M = data["targets_motion"]
    targets_C = data["targets_color"]

    levels = [l.strip() for l in args.levels.split(",")]
    dim_As = [int(d) for d in args.dim_A.split(",")]
    kappas = [float(k) for k in args.kappa.split(",")]

    all_results = {}

    # Direct unit-level analysis
    print("\n" + "=" * 60)
    print("  Direct unit-level selectivity analysis")
    print("=" * 60)
    unit_stats = analyze_unit_selectivity(acts_M, acts_C, losses_M, losses_C)
    print(f"  Mean selectivity index: {unit_stats['mean_selectivity_index']:.3f}")
    print(f"  Fraction selective >0.3: {unit_stats['frac_selective_0.3']:.3f}")
    print(f"  Fraction selective >0.5: {unit_stats['frac_selective_0.5']:.3f}")
    all_results["unit_direct"] = unit_stats

    for dim_A in dim_As:
        dim_B = dim_A  # symmetric

        print(f"\n{'='*60}")
        print(f"  dim_A = dim_B = {dim_A}")
        print(f"{'='*60}")

        dim_results = {}

        for level in levels:
            for kappa in (kappas if level == "linear" else [5.0]):
                label = f"{level}" if level != "linear" else f"linear_k{kappa}"
                print(f"\n  --- {label} ---")

                decomp, history = train_decomposition(
                    acts_M, acts_C, targets_M, targets_C,
                    level=level, dim_A=dim_A, dim_B=dim_B,
                    n_epochs=args.n_epochs_decomp, lr=args.lr_decomp,
                    lambda_adv=args.lambda_adv, kappa=kappa,
                    device=device)

                # Offline eval
                offline = evaluate_offline(decomp, acts_M, acts_C, device=device)
                print(f"    Offline distortion:")
                print(f"      ablate_A: motion={offline['ablate_A']['on_motion']:.4f}  "
                      f"color={offline['ablate_A']['on_color']:.4f}")
                print(f"      ablate_B: motion={offline['ablate_B']['on_motion']:.4f}  "
                      f"color={offline['ablate_B']['on_color']:.4f}")

                # Causal eval
                causal = evaluate_causal(model, decomp, cfg,
                                          n_trials=args.n_causal, device=device)
                bl = causal["none"]
                print(f"    Causal (baseline: motion={bl['acc_motion_nonzero']:.3f}  "
                      f"color={bl['acc_color_nonzero']:.3f}):")
                for abl in ["ablate_A", "ablate_B"]:
                    r = causal[abl]
                    print(f"      {abl}: motion={r['acc_motion_nonzero']:.3f}  "
                          f"color={r['acc_color_nonzero']:.3f}  "
                          f"sel={r['selectivity']:.3f}  "
                          f"(ΔM={r['delta_motion']:+.3f} ΔC={r['delta_color']:+.3f})")

                # Save decomposition
                torch.save(decomp.state_dict(),
                           os.path.join(args.out_dir, f"decomp_{label}_d{dim_A}.pt"))

                dim_results[label] = {
                    "training_history": history,
                    "offline": offline,
                    "causal": causal,
                }

        all_results[f"dim_{dim_A}"] = dim_results

    # Save all results
    results_path = os.path.join(args.out_dir, "modularity_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")


def main():
    p = argparse.ArgumentParser(description="Mante task modularity analysis")
    sub = p.add_subparsers(dest="command")

    # Train
    tr = sub.add_parser("train", help="Train RNN on context-dependent task")
    tr.add_argument("--n_units", type=int, default=256)
    tr.add_argument("--n_epochs", type=int, default=2000)
    tr.add_argument("--batch_size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--n_extract", type=int, default=5000,
                    help="Number of trials for activation extraction")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--out_dir", default="./mante_data")

    # Analyze
    an = sub.add_parser("analyze", help="Analyze modularity")
    an.add_argument("--out_dir", default="./mante_data")
    an.add_argument("--n_units", type=int, default=256)
    an.add_argument("--levels", default="unit,orthogonal,linear")
    an.add_argument("--dim_A", default="25",
                    help="Comma-separated subspace dims to sweep")
    an.add_argument("--kappa", default="2.0,5.0",
                    help="Comma-separated kappa values for linear level")
    an.add_argument("--n_epochs_decomp", type=int, default=1000)
    an.add_argument("--lr_decomp", type=float, default=1e-3)
    an.add_argument("--lambda_adv", type=float, default=1.0)
    an.add_argument("--n_causal", type=int, default=5000,
                    help="Number of trials for causal evaluation")

    args = p.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
