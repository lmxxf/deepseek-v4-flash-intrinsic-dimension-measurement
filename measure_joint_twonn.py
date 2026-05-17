"""
measure_joint_twonn.py — Cross-layer joint TwoNN intrinsic dimension estimation.

Instead of measuring each layer's weight matrix independently, we pool row vectors
from the SAME weight type across ALL layers into one point cloud, then run TwoNN once.

This measures how much the weight structure is shared vs independent across layers:
  - If joint TwoNN ≈ single-layer TwoNN: layers share the same subspace (high overlap)
  - If joint TwoNN >> single-layer TwoNN: layers occupy different directions (low overlap)

Usage:
  python measure_joint_twonn.py \
    --ckpt-path /work/deepseek-v4-flash-deployment/deepseek-v4-flash \
    --output-dir /work/dsv4-flash-intrinsic-dimenstion-measurement/results \
    --sample-rows 200 \
    --expert-samples 2
"""

import argparse
import json
import os
import time
from glob import glob
from collections import defaultdict

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


def twonn_dimension(X: torch.Tensor, max_points: int = 5000) -> float:
    N = X.shape[0]
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        X = X[idx]
        N = max_points

    chunk_size = 512
    mus = []

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        Xi = X[i:end_i]
        dists = torch.cdist(Xi, X)
        for j in range(end_i - i):
            dists[j, i + j] = float("inf")
        top2, _ = dists.topk(2, dim=1, largest=False)
        r1 = top2[:, 0]
        r2 = top2[:, 1]
        mu = r2 / r1.clamp(min=1e-30)
        mus.append(mu)

    mus = torch.cat(mus)
    mus = mus[mus > 1.0]
    if len(mus) < 10:
        return float("nan")

    d = float(len(mus) / torch.log(mus).sum())
    return d


def classify_weight(key):
    if "experts" in key and "shared" not in key:
        return None
    if not key.startswith("layers."):
        return None  # skip embed/head for joint analysis
    parts = key.split(".")
    layer_idx = int(parts[1])
    rest = ".".join(parts[2:])
    if "attn.wq_a" in rest:
        return (layer_idx, "wq_a")
    if "attn.wq_b" in rest:
        return (layer_idx, "wq_b")
    if "attn.wkv" in rest and "norm" not in rest:
        return (layer_idx, "wkv")
    if "attn.wo_a" in rest and "norm" not in rest:
        return (layer_idx, "wo_a")
    if "attn.wo_b" in rest:
        return (layer_idx, "wo_b")
    if "ffn.gate.weight" in rest:
        return (layer_idx, "gate")
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
    parser.add_argument("--sample-rows", type=int, default=200,
                        help="Number of rows to sample per layer per weight type")
    parser.add_argument("--expert-samples", type=int, default=2,
                        help="Number of experts to sample per layer")
    parser.add_argument("--max-twonn-points", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shard_files = sorted(glob(os.path.join(args.ckpt_path, "*.safetensors")))
    print(f"Found {len(shard_files)} shards", flush=True)

    # Pools: {weight_type: list of row vectors (CPU tensors)}
    pools = defaultdict(list)
    layer_count = defaultdict(int)
    scale_buffer = {}
    expert_sample_set = {}

    t_start = time.time()

    for shard_idx, shard_file in enumerate(shard_files):
        shard_name = os.path.basename(shard_file)
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
                    layer_idx, sub_name = info

                    scale_key = key.replace(".weight", ".scale")
                    scale = scale_buffer.pop(scale_key, None)

                    if tensor.dtype == torch.float8_e4m3fn and scale is not None:
                        W = dequant_fp8(tensor.to(args.device), scale.to(args.device))
                    elif tensor.dtype in (torch.bfloat16, torch.float32):
                        W = tensor.to(args.device).float()
                    else:
                        del tensor
                        continue

                    # Sample rows
                    n_rows = W.shape[0]
                    n_sample = min(args.sample_rows, n_rows)
                    idx = torch.randperm(n_rows)[:n_sample]
                    sampled = W[idx].cpu()
                    pools[sub_name].append(sampled)
                    layer_count[sub_name] += 1

                    del W, tensor
                    if scale is not None:
                        del scale
                    torch.cuda.empty_cache()
                    continue

                # Expert weights
                expert_info = parse_expert_key(key)
                if expert_info is not None:
                    layer_idx, expert_idx, weight_name = expert_info

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

                    n_rows = W.shape[0]
                    n_sample = min(args.sample_rows, n_rows)
                    idx = torch.randperm(n_rows)[:n_sample]
                    sampled = W[idx].cpu()

                    pool_key = f"expert_{weight_name}"
                    pools[pool_key].append(sampled)
                    layer_count[pool_key] += 1

                    del W, tensor
                    if scale is not None:
                        del scale
                    torch.cuda.empty_cache()
                    continue

                del tensor

        if (shard_idx + 1) % 10 == 0:
            print(f"  shard {shard_idx+1}/{len(shard_files)} done", flush=True)

    collect_time = time.time() - t_start
    print(f"\nCollection done in {collect_time:.0f}s", flush=True)

    # Run TwoNN on each pool
    results = {}
    for wtype in sorted(pools.keys()):
        all_rows = torch.cat(pools[wtype], dim=0)
        n_layers = layer_count[wtype]
        n_total = all_rows.shape[0]
        dim = all_rows.shape[1]

        print(f"\n{wtype}: {n_total} rows from {n_layers} layers, dim={dim}", flush=True)

        all_rows_gpu = all_rows.to(args.device)
        t0 = time.time()
        joint_dim = twonn_dimension(all_rows_gpu, max_points=args.max_twonn_points)
        twonn_time = time.time() - t0

        print(f"  Joint TwoNN = {joint_dim:.1f} ({twonn_time:.1f}s)", flush=True)

        results[wtype] = {
            "n_rows_total": n_total,
            "n_layers": n_layers,
            "embedding_dim": dim,
            "sample_rows_per_layer": args.sample_rows,
            "joint_twonn_dim": round(joint_dim, 2) if not np.isnan(joint_dim) else None,
            "twonn_time_s": round(twonn_time, 1),
        }

        del all_rows, all_rows_gpu
        torch.cuda.empty_cache()

    # Free pools
    del pools

    elapsed = time.time() - t_start
    print(f"\nDone! {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # Save
    output = {
        "model": "DeepSeek-V4-Flash-280B",
        "method": "cross-layer joint TwoNN",
        "sample_rows_per_layer": args.sample_rows,
        "expert_samples_per_layer": args.expert_samples,
        "max_twonn_points": args.max_twonn_points,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }

    out_path = os.path.join(args.output_dir, "joint_twonn_results.json")
    with open(out_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"Saved to {out_path}", flush=True)

    # Summary with comparison to single-layer results
    print(f"\n{'='*70}")
    print(f"{'Weight':15s} {'Layers':>7s} {'Rows':>7s} {'Joint TwoNN':>12s} {'Single TwoNN':>13s} {'Ratio':>7s}")
    print("-" * 65)

    # Single-layer reference values (from measure_fractal_dim.py results)
    single_ref = {
        "wkv": 98, "wq_a": 137, "wq_b": 75, "wo_a": 277, "wo_b": 195,
        "gate": 34, "expert_w1": 95, "expert_w2": 72, "expert_w3": 111,
    }

    for wtype in sorted(results.keys()):
        r = results[wtype]
        jd = r["joint_twonn_dim"]
        sd = single_ref.get(wtype, None)
        ratio = f"{jd/sd:.2f}x" if (jd and sd) else "—"
        sd_str = f"{sd}" if sd else "—"
        jd_str = f"{jd:.1f}" if jd else "nan"
        print(f"{wtype:15s} {r['n_layers']:7d} {r['n_rows_total']:7d} {jd_str:>12s} {sd_str:>13s} {ratio:>7s}")

    print(f"\nInterpretation:")
    print(f"  ratio ≈ 1.0 → layers share the same subspace (high overlap)")
    print(f"  ratio >> 1.0 → layers occupy different directions (low overlap)")


if __name__ == "__main__":
    main()
