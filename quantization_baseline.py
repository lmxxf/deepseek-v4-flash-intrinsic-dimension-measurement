"""
quantization_baseline.py — Estimate FP4/FP8 quantization noise floor on eRank.

Creates random matrices of the same shapes as V4 Flash weights,
quantizes to FP4/FP8 and dequantizes, then computes SVD.
This gives the "noise floor" eRank that pure quantization artifacts produce.

Compare with the real model's eRank to see how much is signal vs noise.
"""

import torch
import numpy as np
import time
import json

FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def quantize_fp4(x: torch.Tensor, block_size: int = 16) -> tuple:
    """Simulate FP4 E2M1 block quantization."""
    M, K = x.shape
    x_blocks = x.reshape(M, K // block_size, block_size)
    scale = x_blocks.abs().amax(dim=-1) / 6.0  # max representable = 6.0
    scale = scale.clamp(min=1e-12)
    x_scaled = x_blocks / scale.unsqueeze(-1)

    table = FP4_TABLE[:8].to(x.device)  # positive values only
    # Quantize: find nearest FP4 value
    x_sign = x_scaled.sign()
    x_abs = x_scaled.abs()
    diffs = (x_abs.unsqueeze(-1) - table.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)
    x_quant = table[indices] * x_sign

    # Dequantize
    x_deq = (x_quant * scale.unsqueeze(-1)).reshape(M, K)
    return x_deq


def quantize_fp8(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Simulate FP8 E4M3 block quantization."""
    M, K = x.shape
    x_blocks = x.reshape(M, K // block_size, block_size)
    scale = x_blocks.abs().amax(dim=-1) / 448.0  # max FP8 E4M3 = 448
    scale = scale.clamp(min=1e-12)
    x_scaled = x_blocks / scale.unsqueeze(-1)
    x_quant = x_scaled.to(torch.float8_e4m3fn).float()
    x_deq = (x_quant * scale.unsqueeze(-1)).reshape(M, K)
    return x_deq


def compute_erank(W: torch.Tensor) -> dict:
    sv = torch.linalg.svdvals(W)
    sv_np = sv.cpu().numpy().astype(np.float64)
    sv_np = sv_np[sv_np > 0]
    p = sv_np / sv_np.sum()
    H = -np.sum(p * np.log(p))
    erank = float(np.exp(H))
    stable_rank = float((sv_np ** 2).sum() / (sv_np[0] ** 2))
    return {"erank": round(erank, 2), "stable_rank": round(stable_rank, 2),
            "n_sv": len(sv_np), "sigma_max": float(sv_np[0]), "sigma_min": float(sv_np[-1])}


def run_baseline(device="cuda"):
    results = {}

    configs = [
        # (name, shape, quant_type, distribution)
        ("expert_w1_uniform", (2048, 4096), "fp4", "uniform"),
        ("expert_w1_normal", (2048, 4096), "fp4", "normal"),
        ("expert_w2_uniform", (4096, 2048), "fp4", "uniform"),
        ("expert_w2_normal", (4096, 2048), "fp4", "normal"),
        ("attn_wq_a_fp8", (1024, 4096), "fp8", "normal"),
        ("attn_wo_a_fp8", (8192, 4096), "fp8", "normal"),
        ("attn_wq_b_fp8", (32768, 1024), "fp8", "normal"),
        # Full rank matrix (no quantization) for reference
        ("expert_w1_fullrank_noquant", (2048, 4096), "none", "normal"),
        # Low rank matrix
        ("expert_w1_rank100", (2048, 4096), "fp4", "lowrank_100"),
        ("expert_w1_rank500", (2048, 4096), "fp4", "lowrank_500"),
    ]

    n_trials = 5

    for name, shape, quant, dist in configs:
        print(f"\n{name}: shape={shape}, quant={quant}, dist={dist}", flush=True)
        trial_results = []

        for trial in range(n_trials):
            torch.manual_seed(trial * 1000 + 42)
            M, N = shape

            if dist == "uniform":
                W = torch.rand(M, N, device=device) * 2 - 1  # [-1, 1]
            elif dist == "normal":
                W = torch.randn(M, N, device=device)
            elif dist.startswith("lowrank_"):
                rank = int(dist.split("_")[1])
                A = torch.randn(M, rank, device=device) / np.sqrt(rank)
                B = torch.randn(rank, N, device=device) / np.sqrt(rank)
                W = A @ B
            else:
                raise ValueError(f"Unknown dist: {dist}")

            if quant == "fp4":
                W_q = quantize_fp4(W)
            elif quant == "fp8":
                W_q = quantize_fp8(W)
            else:
                W_q = W

            stats = compute_erank(W_q)
            trial_results.append(stats)
            print(f"  trial {trial}: eRank={stats['erank']:.1f}, stable={stats['stable_rank']:.1f}", flush=True)

        eranks = [t["erank"] for t in trial_results]
        stable_ranks = [t["stable_rank"] for t in trial_results]
        results[name] = {
            "shape": list(shape),
            "quant": quant,
            "dist": dist,
            "erank_mean": round(float(np.mean(eranks)), 2),
            "erank_std": round(float(np.std(eranks)), 2),
            "stable_rank_mean": round(float(np.mean(stable_ranks)), 2),
            "stable_rank_std": round(float(np.std(stable_ranks)), 2),
            "trials": trial_results,
        }

    print("\n" + "=" * 70)
    print("QUANTIZATION NOISE BASELINE SUMMARY")
    print("=" * 70)
    print(f"{'Name':35s} {'eRank':>14s} {'StableRank':>14s} {'min_dim':>8s}")
    print("-" * 75)
    for name, r in results.items():
        min_dim = min(r["shape"])
        print(f"{name:35s} {r['erank_mean']:7.1f}±{r['erank_std']:<5.1f} "
              f"{r['stable_rank_mean']:7.1f}±{r['stable_rank_std']:<5.1f} {min_dim:8d}")

    with open("results/quantization_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/quantization_baseline.json")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run_baseline()
