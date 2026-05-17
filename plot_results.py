"""
plot_results.py — Visualize SVD intrinsic dimensionality results.

Reads svd_results.json and generates:
  1. Shared weights: eRank by layer (attention, compressor, indexer, routing)
  2. Expert weights: eRank by layer (w1/w2/w3 mean ± std)
  3. Combined: all eRank on one plot
  4. Power-law α by layer
  5. Stable rank by layer
  6. Energy concentration curves for a few representative matrices

Usage:
  python plot_results.py --input results/svd_results.json --output-dir results/plots
"""

import argparse
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_shared_erank(data, out_dir):
    """Plot eRank of shared (non-expert) weights by layer."""
    shared = data["shared_weights"]

    categories = {}
    for s in shared:
        layer = s["layer"]
        name = s["name"].split(".")[-1]
        cat = s.get("category", "other")
        if cat not in categories:
            categories[cat] = {"layers": [], "eranks": [], "names": []}
        categories[cat]["layers"].append(layer)
        categories[cat]["eranks"].append(s["erank"])
        categories[cat]["names"].append(name)

    fig, ax = plt.subplots(figsize=(16, 8))

    markers = {"attn": "o", "compressor": "s", "indexer": "^", "routing": "D",
               "embed": "*", "head": "*", "other": "x"}
    colors = {"attn": "#2196F3", "compressor": "#FF9800", "indexer": "#4CAF50",
              "routing": "#9C27B0", "embed": "#F44336", "head": "#F44336", "other": "#607D8B"}

    for cat, vals in sorted(categories.items()):
        ax.scatter(vals["layers"], vals["eranks"],
                   marker=markers.get(cat, "o"), color=colors.get(cat, "gray"),
                   label=cat, alpha=0.7, s=40)

    # Connect same sub_name across layers
    sub_names = {}
    for s in shared:
        sn = s["name"].split(".")[-1]
        if sn not in sub_names:
            sub_names[sn] = {"layers": [], "eranks": []}
        sub_names[sn]["layers"].append(s["layer"])
        sub_names[sn]["eranks"].append(s["erank"])

    for sn, vals in sub_names.items():
        if len(vals["layers"]) > 1:
            order = np.argsort(vals["layers"])
            ax.plot([vals["layers"][i] for i in order],
                    [vals["eranks"][i] for i in order],
                    alpha=0.3, linewidth=1, color="gray")

    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Effective Rank (eRank)", fontsize=14)
    ax.set_title("DeepSeek V4 Flash 280B — Shared Weight eRank by Layer", fontsize=16)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shared_erank_by_layer.png"), dpi=150)
    plt.close()


def plot_expert_erank(data, out_dir):
    """Plot expert eRank (mean ± std) by layer for w1/w2/w3."""
    expert_data = data.get("expert_weights", {})
    if not expert_data:
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    colors = {"w1": "#E91E63", "w2": "#00BCD4", "w3": "#8BC34A"}

    for wname in ["w1", "w2", "w3"]:
        layers = []
        means = []
        stds = []
        for layer_key in sorted(expert_data.keys(), key=lambda x: int(x.split("_")[1])):
            layer_idx = int(layer_key.split("_")[1])
            if wname in expert_data[layer_key]:
                d = expert_data[layer_key][wname]
                layers.append(layer_idx)
                means.append(d["erank_mean"])
                stds.append(d["erank_std"])

        if layers:
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(layers, means, "-o", label=wname, color=colors[wname], markersize=4)
            ax.fill_between(layers, means - stds, means + stds, alpha=0.15, color=colors[wname])

    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Effective Rank (eRank)", fontsize=14)
    ax.set_title("DeepSeek V4 Flash 280B — Expert Weight eRank by Layer (mean±std, n=8)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "expert_erank_by_layer.png"), dpi=150)
    plt.close()


def plot_alpha_by_layer(data, out_dir):
    """Plot power-law exponent α by layer for shared weights."""
    shared = data["shared_weights"]

    sub_names = {}
    for s in shared:
        sn = s["name"].split(".")[-1]
        alpha = s.get("alpha_powerlaw", float("nan"))
        if np.isnan(alpha):
            continue
        if sn not in sub_names:
            sub_names[sn] = {"layers": [], "alphas": []}
        sub_names[sn]["layers"].append(s["layer"])
        sub_names[sn]["alphas"].append(alpha)

    fig, ax = plt.subplots(figsize=(16, 8))

    for sn, vals in sorted(sub_names.items()):
        if len(vals["layers"]) > 2:
            order = np.argsort(vals["layers"])
            ax.plot([vals["layers"][i] for i in order],
                    [vals["alphas"][i] for i in order],
                    "-o", label=sn, markersize=4, alpha=0.7)

    # Martin & Mahoney reference lines
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="α=2 (well-trained, M&M)")
    ax.axhline(y=4.0, color="orange", linestyle="--", alpha=0.5, label="α=4 (over-regularized)")

    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Power-law α", fontsize=14)
    ax.set_title("DeepSeek V4 Flash 280B — ESD Power-law Exponent α by Layer", fontsize=16)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alpha_by_layer.png"), dpi=150)
    plt.close()


def plot_stable_rank(data, out_dir):
    """Plot stable rank by layer for shared weights."""
    shared = data["shared_weights"]

    sub_names = {}
    for s in shared:
        sn = s["name"].split(".")[-1]
        if sn not in sub_names:
            sub_names[sn] = {"layers": [], "stable_ranks": []}
        sub_names[sn]["layers"].append(s["layer"])
        sub_names[sn]["stable_ranks"].append(s["stable_rank"])

    fig, ax = plt.subplots(figsize=(16, 8))

    for sn, vals in sorted(sub_names.items()):
        if len(vals["layers"]) > 2:
            order = np.argsort(vals["layers"])
            ax.plot([vals["layers"][i] for i in order],
                    [vals["stable_ranks"][i] for i in order],
                    "-o", label=sn, markersize=4, alpha=0.7)

    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Stable Rank (||W||_F² / ||W||_2²)", fontsize=14)
    ax.set_title("DeepSeek V4 Flash 280B — Stable Rank by Layer", fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stable_rank_by_layer.png"), dpi=150)
    plt.close()


def plot_combined(data, out_dir):
    """Combined plot: shared + expert eRank, all on one figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    # Top: shared
    shared = data["shared_weights"]
    sub_names = {}
    for s in shared:
        sn = s["name"].split(".")[-1]
        if sn not in sub_names:
            sub_names[sn] = {"layers": [], "eranks": []}
        sub_names[sn]["layers"].append(s["layer"])
        sub_names[sn]["eranks"].append(s["erank"])

    for sn, vals in sorted(sub_names.items()):
        if len(vals["layers"]) > 2:
            order = np.argsort(vals["layers"])
            ax1.plot([vals["layers"][i] for i in order],
                     [vals["eranks"][i] for i in order],
                     "-o", label=sn, markersize=4, alpha=0.7)

    ax1.set_ylabel("eRank", fontsize=14)
    ax1.set_title("Shared (Attention + Routing) Weights", fontsize=14)
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Bottom: experts
    expert_data = data.get("expert_weights", {})
    colors = {"w1": "#E91E63", "w2": "#00BCD4", "w3": "#8BC34A"}
    for wname in ["w1", "w2", "w3"]:
        layers = []
        means = []
        stds = []
        for layer_key in sorted(expert_data.keys(), key=lambda x: int(x.split("_")[1])):
            layer_idx = int(layer_key.split("_")[1])
            if wname in expert_data[layer_key]:
                d = expert_data[layer_key][wname]
                layers.append(layer_idx)
                means.append(d["erank_mean"])
                stds.append(d["erank_std"])
        if layers:
            means = np.array(means)
            stds = np.array(stds)
            ax2.plot(layers, means, "-o", label=wname, color=colors[wname], markersize=4)
            ax2.fill_between(layers, means - stds, means + stds, alpha=0.15, color=colors[wname])

    ax2.set_xlabel("Layer", fontsize=14)
    ax2.set_ylabel("eRank", fontsize=14)
    ax2.set_title("Expert (MoE) Weights — mean ± std", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("DeepSeek V4 Flash 280B — Intrinsic Dimensionality (eRank) by Layer", fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_erank.png"), dpi=150)
    plt.close()


def print_global_summary(data):
    """Print overall statistics."""
    shared = data["shared_weights"]
    expert_data = data.get("expert_weights", {})

    # Attention weights across layers
    attn_keys = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]
    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)

    for key in attn_keys:
        eranks = [s["erank"] for s in shared if s["name"].endswith(f".{key}")]
        if eranks:
            print(f"  {key:12s}: eRank {np.mean(eranks):7.1f} ± {np.std(eranks):5.1f} "
                  f"(min={np.min(eranks):.1f}, max={np.max(eranks):.1f}, n={len(eranks)})")

    # Expert aggregation
    all_expert_eranks = {"w1": [], "w2": [], "w3": []}
    for layer_key, layer_data in expert_data.items():
        for wname in ["w1", "w2", "w3"]:
            if wname in layer_data:
                all_expert_eranks[wname].append(layer_data[wname]["erank_mean"])

    print()
    for wname in ["w1", "w2", "w3"]:
        vals = all_expert_eranks[wname]
        if vals:
            print(f"  expert.{wname:4s}: eRank {np.mean(vals):7.1f} ± {np.std(vals):5.1f} "
                  f"(min={np.min(vals):.1f}, max={np.max(vals):.1f}, across {len(vals)} layers)")

    # Overall "model intrinsic dimension" estimate
    all_eranks = [s["erank"] for s in shared if s["layer"] >= 0 and s["layer"] < 900]
    for wname in ["w1", "w2", "w3"]:
        all_eranks.extend(all_expert_eranks[wname])

    print(f"\n  All weight matrices (excl embed/head):")
    print(f"    median eRank = {np.median(all_eranks):.1f}")
    print(f"    mean   eRank = {np.mean(all_eranks):.1f} ± {np.std(all_eranks):.1f}")
    print(f"    range: [{np.min(all_eranks):.1f}, {np.max(all_eranks):.1f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/svd_results.json")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_results(args.input)

    print(f"Loaded {data['total_matrices']} matrices from {data['model']}")

    plot_shared_erank(data, args.output_dir)
    plot_expert_erank(data, args.output_dir)
    plot_alpha_by_layer(data, args.output_dir)
    plot_stable_rank(data, args.output_dir)
    plot_combined(data, args.output_dir)

    print_global_summary(data)

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
