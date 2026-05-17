"""
measure_fractal_dim.py — Estimate fractal/intrinsic dimension of weight matrices
using correlation dimension (Grassberger-Procaccia) and TwoNN estimator.

SVD eRank measures "how many linear directions" — like asking what box fits the tree.
Fractal dimension measures "how the tree fills space" — the actual structural complexity.

Methods:
  1. Correlation Dimension (Grassberger-Procaccia 1983):
     C(r) = #(pairs with dist < r) / #(total pairs)
     Plot log C(r) vs log r, slope = correlation dimension D2

  2. TwoNN (Facco et al. 2017, Science):
     For each point, compute μ = r2/r1 (ratio of 2nd to 1st nearest neighbor distance)
     The distribution of μ follows Pareto(d,1) where d = intrinsic dimension
     Estimator: d = N / Σ log(μ_i)

  3. Participation Ratio (PR):
     From singular value spectrum: PR = (Σ σ_i²)² / Σ σ_i⁴
     For uniform spectrum PR = n, for single spike PR = 1
     More robust to noise than eRank for measuring "effective number of active directions"

For weight matrices:
  - Rows are "data points" (each row is a vector in the column space)
  - We measure the intrinsic dimension of the set of row vectors
  - This tells us the structural complexity of what the matrix computes

Usage:
  python measure_fractal_dim.py \
    --ckpt-path /work/deepseek-v4-flash-deployment/deepseek-v4-flash \
    --output-dir /work/dsv4-flash-intrinsic-dimenstion-measurement/results \
    --layers 0,10,20,30,40,42
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


def dequant_fp8(weight, scale):
    w = weight.float()
    s = scale.float()
    M, K = w.shape
    bs_m = M // s.shape[0]
    bs_k = K // s.shape[1]
    s_expanded = s.repeat_interleave(bs_m, dim=0).repeat_interleave(bs_k, dim=1)
    return w * s_expanded


def dequant_fp4(weight_int8, scale):
    raw = weight_int8.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    table = FP4_TABLE.to(raw.device)
    w_low = table[low.long()]
    w_high = table[high.long()]
    w = torch.stack([w_low, w_high], dim=-1).reshape(raw.shape[0], -1)
    s = scale.float()
    N, K = w.shape
    bs_n = N // s.shape[0] if s.shape[0] > 0 else 1
    bs_k = K // s.shape[1] if s.shape[1] > 0 else 1
    s_expanded = s.repeat_interleave(bs_n, dim=0).repeat_interleave(bs_k, dim=1)
    return w * s_expanded


def participation_ratio(W: torch.Tensor) -> float:
    """PR = (Σσ²)² / Σσ⁴ — effective number of active singular directions."""
    sv = torch.linalg.svdvals(W)
    sv2 = (sv ** 2).double()
    return float((sv2.sum() ** 2) / (sv2 ** 2).sum())


def twonn_dimension(X: torch.Tensor, max_points: int = 5000) -> float:
    """TwoNN intrinsic dimension estimator (Facco et al. 2017).
    X: [N, D] — N points in D-dimensional space.
    Uses ratio of 2nd to 1st nearest neighbor distance.
    """
    N = X.shape[0]
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        X = X[idx]
        N = max_points

    # Compute pairwise distances in chunks to avoid OOM
    chunk_size = 512
    mus = []

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        Xi = X[i:end_i]  # [chunk, D]
        dists = torch.cdist(Xi, X)  # [chunk, N]
        # Set self-distance to inf
        for j in range(end_i - i):
            dists[j, i + j] = float("inf")
        # Get 2 nearest neighbors
        top2, _ = dists.topk(2, dim=1, largest=False)
        r1 = top2[:, 0]
        r2 = top2[:, 1]
        mu = r2 / r1.clamp(min=1e-30)
        mus.append(mu)

    mus = torch.cat(mus)
    mus = mus[mus > 1.0]  # filter degenerate cases
    if len(mus) < 10:
        return float("nan")

    # MLE estimator: d = N / Σ log(μ_i)
    d = float(len(mus) / torch.log(mus).sum())
    return d


def correlation_dimension(X: torch.Tensor, max_points: int = 3000, n_radii: int = 30) -> dict:
    """Grassberger-Procaccia correlation dimension.
    Returns D2 estimate and the log-log curve data.
    """
    N = X.shape[0]
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        X = X[idx]
        N = max_points

    # Compute all pairwise distances
    dists_list = []
    chunk_size = 512
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        Xi = X[i:end_i]
        d = torch.cdist(Xi, X)
        # Only keep upper triangle entries
        for j in range(end_i - i):
            d[j, i + j] = float("inf")  # exclude self
            d[j, :i + j] = float("inf")  # exclude lower triangle (already counted)
        dists_list.append(d.reshape(-1))

    all_dists = torch.cat(dists_list)
    all_dists = all_dists[all_dists < float("inf")]
    all_dists = all_dists[all_dists > 0]

    if len(all_dists) < 100:
        return {"D2": float("nan")}

    # Generate log-spaced radii
    d_min = all_dists.min().item()
    d_max = all_dists.max().item()
    radii = np.logspace(np.log10(d_min * 1.1), np.log10(d_max * 0.9), n_radii)

    n_pairs = len(all_dists)
    log_r = []
    log_C = []

    for r in radii:
        count = (all_dists < r).sum().item()
        if count > 0:
            C = count / n_pairs
            log_r.append(np.log(r))
            log_C.append(np.log(C))

    if len(log_r) < 5:
        return {"D2": float("nan")}

    log_r = np.array(log_r)
    log_C = np.array(log_C)

    # Fit slope in the linear region (middle 50%)
    n = len(log_r)
    lo = n // 4
    hi = 3 * n // 4
    if hi - lo < 3:
        lo, hi = 0, n

    coeffs = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)
    D2 = float(coeffs[0])

    return {
        "D2": round(D2, 2),
        "log_r": log_r.tolist(),
        "log_C": log_C.tolist(),
        "fit_range": [lo, hi],
    }


def analyze_matrix(W: torch.Tensor, name: str) -> dict:
    """Run all dimension estimators on a weight matrix."""
    M, N = W.shape
    t0 = time.time()

    # Participation ratio (from SVD)
    pr = participation_ratio(W)

    # TwoNN on rows (each row is a point in column-space)
    twonn_row = twonn_dimension(W, max_points=4000)

    # TwoNN on columns (each column is a point in row-space)
    twonn_col = twonn_dimension(W.T, max_points=4000)

    # Correlation dimension on rows
    corr = correlation_dimension(W, max_points=2000)

    elapsed = time.time() - t0

    result = {
        "name": name,
        "shape": [M, N],
        "participation_ratio": round(pr, 2),
        "twonn_row_dim": round(twonn_row, 2) if not np.isnan(twonn_row) else None,
        "twonn_col_dim": round(twonn_col, 2) if not np.isnan(twonn_col) else None,
        "correlation_dim_D2": corr["D2"],
        "time_s": round(elapsed, 1),
    }

    print(f"  {name} {[M,N]}: PR={pr:.1f}, TwoNN_row={twonn_row:.1f}, "
          f"TwoNN_col={twonn_col:.1f}, D2={corr['D2']:.1f} ({elapsed:.1f}s)", flush=True)

    return result


def classify_weight(key):
    if "experts" in key and "shared" not in key:
        return None
    if not key.startswith("layers."):
        if key == "embed.weight":
            return (-1, "embed", "embed")
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
    return None


def parse_expert_key(key):
    if "experts" not in key or "shared" in key:
        return None
    if not key.startswith("layers."):
        return None
    parts = key.split(".")
    try:
        layer_idx = int(parts[1])
        expert_idx = int(parts[4])
        weight_name = parts[5]
        if parts[-1] == "weight":
            return (layer_idx, expert_idx, weight_name)
    except (IndexError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", required=True,
                        help="Comma-separated layer indices")
    parser.add_argument("--expert-samples", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    target_layers = set(int(x) for x in args.layers.split(","))

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"Found {len(shard_files)} shards, target layers: {sorted(target_layers)}", flush=True)

    results = []
    scale_buffer = {}
    expert_sample_set = {}
    total = 0
    t_start = time.time()

    for shard_idx, shard_file in enumerate(shard_files):
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                if key.endswith(".scale"):
                    scale_buffer[key] = f.get_tensor(key)
                    continue
                if not key.endswith(".weight"):
                    continue

                tensor = f.get_tensor(key)

                # Shared weights
                info = classify_weight(key)
                if info is not None:
                    layer_idx, category, sub_name = info
                    if layer_idx not in target_layers and layer_idx != -1:
                        del tensor
                        continue

                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(args.device), scale.to(args.device))
                    elif tensor.dtype in (torch.bfloat16, torch.float32):
                        W = tensor.to(args.device).float()
                    else:
                        del tensor
                        continue

                    name = f"L{layer_idx}.{sub_name}"
                    r = analyze_matrix(W, name)
                    r["layer"] = layer_idx
                    r["category"] = category
                    results.append(r)
                    total += 1
                    del W, tensor
                    torch.cuda.empty_cache()
                    continue

                # Expert weights (sample 2 per layer)
                expert_info = parse_expert_key(key)
                if expert_info is not None:
                    layer_idx, expert_idx, weight_name = expert_info
                    if layer_idx not in target_layers:
                        del tensor
                        continue

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
                        W = dequant_fp4(tensor.to(args.device), scale.to(args.device))
                    else:
                        del tensor
                        continue

                    name = f"L{layer_idx}.expert{expert_idx}.{weight_name}"
                    r = analyze_matrix(W, name)
                    r["layer"] = layer_idx
                    r["expert_idx"] = expert_idx
                    results.append(r)
                    total += 1
                    del W, tensor
                    torch.cuda.empty_cache()
                    continue

                del tensor

    elapsed = time.time() - t_start
    print(f"\nDone! {total} matrices in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    output = {
        "model": "DeepSeek-V4-Flash-280B",
        "method": "fractal dimension estimation",
        "total_matrices": total,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }

    out_path = os.path.join(args.output_dir, "fractal_dim_results.json")
    with open(out_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"Saved to {out_path}", flush=True)

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Name':30s} {'PR':>8s} {'TwoNN_row':>10s} {'TwoNN_col':>10s} {'D2':>8s}")
    print("-" * 70)
    for r in results:
        pr = r["participation_ratio"]
        tr = r.get("twonn_row_dim") or float("nan")
        tc = r.get("twonn_col_dim") or float("nan")
        d2 = r.get("correlation_dim_D2", float("nan"))
        print(f"{r['name']:30s} {pr:8.1f} {tr:10.1f} {tc:10.1f} {d2:8.1f}")


if __name__ == "__main__":
    main()
