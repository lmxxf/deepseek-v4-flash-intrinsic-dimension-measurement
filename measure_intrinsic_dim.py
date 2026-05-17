"""
measure_intrinsic_dim.py — Measure intrinsic dimensionality of DeepSeek V4 Flash 280B
via per-layer weight matrix SVD.

For each weight matrix W (after dequantization to float32):
  1. Compute singular values σ_1 ≥ σ_2 ≥ ... ≥ σ_r
  2. Normalize to probability distribution: p_i = σ_i / Σσ_j
  3. Spectral entropy: H = -Σ p_i log(p_i)
  4. Effective rank (eRank) = exp(H)  [Roy & Vetterli, 2007]

Also records:
  - Stable rank = ||W||_F² / ||W||_2² = (Σσ²) / σ_max²
  - Top-k singular value ratios (energy concentration)
  - α exponent from power-law fit to ESD tail [Martin & Mahoney, 2019]

Usage (inside Docker container with safetensors ≥ 0.7.0):
  python measure_intrinsic_dim.py \
    --ckpt-path /work/deepseek-v4-flash-deployment/deepseek-v4-flash \
    --output-dir /work/dsv4-flash-intrinsic-dimenstion-measurement/results \
    --expert-samples 8

Run on DGX Spark single node (no TP needed, no forward pass).
"""

import argparse
import json
import os
import time
from glob import glob

import torch
import numpy as np
from safetensors import safe_open


FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight with E8M0 per-block scale to float32.
    weight: [M, K] float8_e4m3fn
    scale:  [M//128, K//128] float8_e8m0fnu (block size 128)
    """
    w = weight.float()
    s = scale.float()
    M, K = w.shape
    bs_m = M // s.shape[0]
    bs_k = K // s.shape[1]
    s_expanded = s.repeat_interleave(bs_m, dim=0).repeat_interleave(bs_k, dim=1)
    return w * s_expanded


def dequant_fp4(weight_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 (packed as int8) weight with E8M0 per-block scale to float32.
    weight_int8: [N, K//2] int8 — two FP4 values packed per byte
    scale:       [N, K//8] float8_e8m0fnu (block size 8 on the packed K dim = 16 on real K)
    """
    raw = weight_int8.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    table = FP4_TABLE.to(raw.device)
    w_low = table[low.long()]
    w_high = table[high.long()]
    w = torch.stack([w_low, w_high], dim=-1).reshape(raw.shape[0], -1)  # [N, K]

    s = scale.float()
    N, K = w.shape
    bs_n = N // s.shape[0] if s.shape[0] > 0 else 1
    bs_k = K // s.shape[1] if s.shape[1] > 0 else 1
    s_expanded = s.repeat_interleave(bs_n, dim=0).repeat_interleave(bs_k, dim=1)
    return w * s_expanded


def compute_svd_stats(W: torch.Tensor, name: str) -> dict:
    """Compute SVD-based statistics for a single weight matrix."""
    M, N = W.shape
    t0 = time.time()
    sv = torch.linalg.svdvals(W)
    svd_time = time.time() - t0

    sv_np = sv.cpu().numpy().astype(np.float64)
    sv_np = sv_np[sv_np > 0]  # drop exact zeros

    if len(sv_np) == 0:
        return {"name": name, "shape": [M, N], "error": "all singular values zero"}

    # eRank = exp(spectral entropy)
    p = sv_np / sv_np.sum()
    H = -np.sum(p * np.log(p))
    erank = float(np.exp(H))

    # Stable rank = ||W||_F^2 / ||W||_2^2
    stable_rank = float((sv_np ** 2).sum() / (sv_np[0] ** 2))

    # Energy concentration: fraction of total energy in top-k
    energy = (sv_np ** 2).cumsum()
    total_energy = energy[-1]
    energy_frac = energy / total_energy

    top_k_energy = {}
    for k in [10, 50, 100, 200, 500]:
        if k <= len(sv_np):
            top_k_energy[f"top{k}"] = float(energy_frac[k - 1])

    # Power-law exponent α: fit log(σ) vs log(rank) on the tail (top 20% to 80%)
    n_sv = len(sv_np)
    lo = max(1, int(n_sv * 0.2))
    hi = int(n_sv * 0.8)
    if hi > lo + 2:
        log_rank = np.log(np.arange(lo, hi) + 1)
        log_sv = np.log(sv_np[lo:hi])
        coeffs = np.polyfit(log_rank, log_sv, 1)
        alpha = -float(coeffs[0])
    else:
        alpha = float("nan")

    # Numerical rank: count σ > σ_max * 1e-5
    numerical_rank = int((sv_np > sv_np[0] * 1e-5).sum())

    return {
        "name": name,
        "shape": [M, N],
        "min_dim": min(M, N),
        "n_singular_values": len(sv_np),
        "erank": round(erank, 2),
        "stable_rank": round(stable_rank, 2),
        "numerical_rank": numerical_rank,
        "spectral_entropy": round(float(H), 4),
        "alpha_powerlaw": round(alpha, 4),
        "energy_concentration": top_k_energy,
        "sigma_max": float(sv_np[0]),
        "sigma_min": float(sv_np[-1]),
        "sigma_ratio": float(sv_np[0] / sv_np[-1]) if sv_np[-1] > 0 else float("inf"),
        "svd_time_s": round(svd_time, 3),
        "top20_singular_values": sv_np[:20].tolist(),
    }


def classify_weight(key: str):
    """Classify a weight key into category. Returns (layer_idx, category, sub_name) or None."""
    if "experts" in key and "shared" not in key:
        return None  # experts handled separately
    if not key.startswith("layers."):
        if key == "embed.weight":
            return (-1, "embed", "embed")
        if key == "head.weight":
            return (999, "head", "head")
        return None

    parts = key.split(".")
    layer_idx = int(parts[1])
    rest = ".".join(parts[2:])

    if "attn.wq_a" in rest:
        return (layer_idx, "attn", "wq_a")
    if "attn.wq_b" in rest:
        return (layer_idx, "attn", "wq_b")
    if "attn.wkv" in rest and "norm" not in rest:
        return (layer_idx, "attn", "wkv")
    if "attn.wo_a" in rest and "norm" not in rest:
        return (layer_idx, "attn", "wo_a")
    if "attn.wo_b" in rest:
        return (layer_idx, "attn", "wo_b")
    if "ffn.gate.weight" in rest:
        return (layer_idx, "routing", "gate")
    if "compressor.wkv" in rest:
        return (layer_idx, "compressor", "comp_wkv")
    if "compressor.wgate" in rest:
        return (layer_idx, "compressor", "comp_wgate")
    if "indexer.wq_b" in rest:
        return (layer_idx, "indexer", "idx_wq_b")
    if "indexer.weights_proj" in rest:
        return (layer_idx, "indexer", "idx_wproj")
    if "indexer.compressor" in rest:
        if "wkv" in rest:
            return (layer_idx, "indexer", "idx_comp_wkv")
        if "wgate" in rest:
            return (layer_idx, "indexer", "idx_comp_wgate")

    return None


def parse_expert_key(key: str):
    """Parse expert weight key. Returns (layer_idx, expert_idx, weight_name) or None."""
    if "experts" not in key or "shared" in key:
        return None
    if not key.startswith("layers."):
        return None  # skip MTP experts
    parts = key.split(".")
    try:
        layer_idx = int(parts[1])
        expert_idx = int(parts[4])  # layers.X.ffn.experts.Y.wZ.weight
        weight_name = parts[5]  # w1, w2, w3
        if parts[-1] == "weight":
            return (layer_idx, expert_idx, weight_name)
    except (IndexError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-samples", type=int, default=8,
                        help="Number of experts to sample per layer for SVD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices to process (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    target_layers = None
    if args.layers:
        target_layers = set(int(x) for x in args.layers.split(","))

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"Found {len(shard_files)} shards in {args.ckpt_path}", flush=True)

    results_shared = []      # non-expert weight matrices
    results_experts = {}     # {layer_idx: {weight_name: [stats_per_expert]}}
    expert_sample_set = {}   # {layer_idx: set of sampled expert indices}

    # Buffer for weight+scale pairs (they appear in the same shard)
    scale_buffer = {}

    total_matrices = 0
    t_start = time.time()

    for shard_idx, shard_file in enumerate(shard_files):
        shard_name = os.path.basename(shard_file)
        print(f"\n[shard {shard_idx+1}/{len(shard_files)}] {shard_name}", flush=True)

        weight_buffer = {}

        with safe_open(shard_file, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            for key in keys:
                # Skip non-weight tensors
                if key.endswith(".scale"):
                    scale_buffer[key] = f.get_tensor(key)
                    continue
                if not key.endswith(".weight"):
                    continue

                tensor = f.get_tensor(key)

                # --- Shared (non-expert) weights ---
                info = classify_weight(key)
                if info is not None:
                    layer_idx, category, sub_name = info
                    if target_layers is not None and layer_idx not in target_layers and layer_idx not in (-1, 999):
                        del tensor
                        continue

                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(device), scale.to(device))
                    elif tensor.dtype == torch.bfloat16:
                        W = tensor.to(device).float()
                    elif tensor.dtype == torch.float32:
                        W = tensor.to(device)
                    else:
                        print(f"  SKIP {key}: unsupported dtype {tensor.dtype}", flush=True)
                        del tensor
                        continue

                    full_name = f"L{layer_idx}.{sub_name}"
                    print(f"  SVD: {full_name} {list(W.shape)}", end="", flush=True)
                    stats = compute_svd_stats(W, full_name)
                    stats["layer"] = layer_idx
                    stats["category"] = category
                    results_shared.append(stats)
                    total_matrices += 1
                    print(f" → eRank={stats['erank']:.1f}, stable={stats['stable_rank']:.1f}, "
                          f"α={stats['alpha_powerlaw']:.2f} ({stats['svd_time_s']:.1f}s)", flush=True)
                    del W, tensor
                    if scale is not None:
                        del scale
                    torch.cuda.empty_cache()
                    continue

                # --- Expert weights ---
                expert_info = parse_expert_key(key)
                if expert_info is not None:
                    layer_idx, expert_idx, weight_name = expert_info
                    if target_layers is not None and layer_idx not in target_layers:
                        del tensor
                        continue

                    # Determine which experts to sample for this layer
                    if layer_idx not in expert_sample_set:
                        rng = np.random.RandomState(layer_idx * 1000 + 42)
                        expert_sample_set[layer_idx] = set(
                            rng.choice(256, size=min(args.expert_samples, 256), replace=False)
                        )

                    if expert_idx not in expert_sample_set[layer_idx]:
                        del tensor
                        continue

                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.int8 and scale is not None:
                        W = dequant_fp4(tensor.to(device), scale.to(device))
                    elif tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(device), scale.to(device))
                    else:
                        print(f"  SKIP expert {key}: dtype={tensor.dtype}, has_scale={scale is not None}", flush=True)
                        del tensor
                        continue

                    full_name = f"L{layer_idx}.expert{expert_idx}.{weight_name}"
                    print(f"  SVD: {full_name} {list(W.shape)}", end="", flush=True)
                    stats = compute_svd_stats(W, full_name)
                    stats["layer"] = layer_idx
                    stats["expert_idx"] = expert_idx

                    if layer_idx not in results_experts:
                        results_experts[layer_idx] = {}
                    if weight_name not in results_experts[layer_idx]:
                        results_experts[layer_idx][weight_name] = []
                    results_experts[layer_idx][weight_name].append(stats)
                    total_matrices += 1
                    print(f" → eRank={stats['erank']:.1f}, stable={stats['stable_rank']:.1f} ({stats['svd_time_s']:.1f}s)", flush=True)
                    del W, tensor
                    if scale is not None:
                        del scale
                    torch.cuda.empty_cache()
                    continue

                del tensor

        # Clear stale scales from this shard
        stale = [k for k in scale_buffer if shard_name in k]
        for k in stale:
            del scale_buffer[k]

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! {total_matrices} matrices processed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # --- Save results ---
    output = {
        "model": "DeepSeek-V4-Flash-280B",
        "quantization": "FP4 experts / FP8 attention",
        "expert_samples_per_layer": args.expert_samples,
        "total_matrices": total_matrices,
        "elapsed_s": round(elapsed, 1),
        "shared_weights": results_shared,
        "expert_weights": {},
    }

    # Summarize expert results per layer
    for layer_idx in sorted(results_experts.keys()):
        layer_data = {}
        for wname, stats_list in results_experts[layer_idx].items():
            eranks = [s["erank"] for s in stats_list]
            stable_ranks = [s["stable_rank"] for s in stats_list]
            alphas = [s["alpha_powerlaw"] for s in stats_list if not np.isnan(s["alpha_powerlaw"])]
            layer_data[wname] = {
                "n_sampled": len(stats_list),
                "erank_mean": round(float(np.mean(eranks)), 2),
                "erank_std": round(float(np.std(eranks)), 2),
                "erank_min": round(float(np.min(eranks)), 2),
                "erank_max": round(float(np.max(eranks)), 2),
                "stable_rank_mean": round(float(np.mean(stable_ranks)), 2),
                "stable_rank_std": round(float(np.std(stable_ranks)), 2),
                "alpha_mean": round(float(np.mean(alphas)), 4) if alphas else None,
                "alpha_std": round(float(np.std(alphas)), 4) if alphas else None,
                "per_expert": stats_list,
            }
        output["expert_weights"][f"layer_{layer_idx}"] = layer_data

    out_path = os.path.join(args.output_dir, "svd_results.json")
    with open(out_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"Results saved to {out_path}", flush=True)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("SUMMARY: Shared (non-expert) weights")
    print(f"{'Name':30s} {'Shape':20s} {'eRank':>8s} {'Stable':>8s} {'NumRank':>8s} {'α':>8s}")
    print("-" * 84)
    for s in results_shared:
        print(f"{s['name']:30s} {str(s['shape']):20s} {s['erank']:8.1f} {s['stable_rank']:8.1f} "
              f"{s['numerical_rank']:8d} {s['alpha_powerlaw']:8.2f}")

    print(f"\n{'='*60}")
    print("SUMMARY: Expert weights (mean ± std across sampled experts)")
    print(f"{'Layer':>6s} {'Weight':>6s} {'eRank':>14s} {'StableRank':>14s} {'α':>14s}")
    print("-" * 60)
    for layer_idx in sorted(results_experts.keys()):
        for wname in ["w1", "w2", "w3"]:
            if wname in results_experts[layer_idx]:
                d = output["expert_weights"][f"layer_{layer_idx}"][wname]
                print(f"{layer_idx:6d} {wname:>6s} {d['erank_mean']:7.1f}±{d['erank_std']:<5.1f} "
                      f"{d['stable_rank_mean']:7.1f}±{d['stable_rank_std']:<5.1f} "
                      f"{d['alpha_mean']:7.2f}±{d['alpha_std']:<5.2f}" if d['alpha_mean'] else "")


if __name__ == "__main__":
    main()
