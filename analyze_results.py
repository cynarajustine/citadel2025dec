#!/usr/bin/env python3
"""
Analysis and Visualization for CITADEL-FL v2 (with Robustness Score)
-------------------------------------------------------------------
Reads experiment results, computes summary metrics, and generates plots including robustness.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = "results_citadel_v2"
OUT_DIR = "citadel_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["accuracy", "f1", "precision", "recall", "ASR"]

# --------------------------
# Helper Functions
# --------------------------
def load_results():
    """Load results from all experiments."""
    records = []
    
    for root, _, files in os.walk(BASE_DIR):
        if "history.json" not in files:
            continue
            
        path = os.path.join(root, "history.json")
        
        try:
            with open(path) as f:
                hist = json.load(f)
            
            if not isinstance(hist, list) or len(hist) == 0:
                continue
            
            # Extract dataset, attack, mode from path
            parts = Path(path).parts
            if len(parts) < 4 or "tuning" in parts:
                continue
                
            dataset, attack, mode = parts[-4], parts[-3], parts[-2]
            
            # Average over last 3 rounds
            last_rounds = hist[-3:] if len(hist) >= 3 else hist
            
            clean_avgs = {}
            attack_avgs = {}
            
            for m in METRICS:
                if m == "ASR":
                    attack_avgs[m] = float(np.mean([r.get("asr", 0.0) for r in last_rounds]))
                else:
                    clean_vals = [float(r.get("clean", {}).get(m, 0.0)) for r in last_rounds]
                    attack_vals = [float(r.get("attack", {}).get(m, 0.0)) for r in last_rounds]
                    
                    clean_avgs[m] = float(np.mean(clean_vals)) if clean_vals else 0.0
                    attack_avgs[m] = float(np.mean(attack_vals)) if attack_vals else 0.0
            
            robustness_avg = float(np.mean([float(r.get("robustness_score", 0.0)) for r in last_rounds]))
            
            record = {
                "dataset": dataset,
                "attack": attack,
                "mode": mode,
                **{f"clean_{m}": clean_avgs.get(m, 0.0) for m in METRICS if m != "ASR"},
                **{f"attack_{m}": attack_avgs.get(m, 0.0) for m in METRICS},
                "robustness_score": robustness_avg
            }
            
            records.append(record)
            
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")
            continue
    
    df = pd.DataFrame(records)
    # Ensure numeric dtypes for plotting and ranking
    for col in df.columns:
        if col.startswith("clean_") or col.startswith("attack_") or col == "robustness_score":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def plot_metric(df, metric, save_dir=OUT_DIR):
    """Plot metric across modes and attacks."""
    try:
        if df.empty:
            print(f"⚠️ No data to plot for {metric}")
            return

        plt.figure(figsize=(12, 6))
        
        attacks = sorted(df["attack"].unique())
        modes = sorted(df["mode"].unique())
        
        for attack in attacks:
            subset = df[df["attack"] == attack].set_index("mode").reindex(modes)
            
            if metric == "robustness_score":
                plt.plot(modes, subset[metric], marker='o', linestyle='-', label=attack, linewidth=2)
            elif metric == "ASR":
                # ASR exists only as attack_ASR
                plt.plot(modes, subset["attack_ASR"], marker='x', linestyle='--',
                         label=f"{attack}-ASR", linewidth=2)
            else:
                plt.plot(modes, subset[f"clean_{metric}"],
                         marker='o', linestyle='-', label=f"{attack}-clean", linewidth=2)
                plt.plot(modes, subset[f"attack_{metric}"], marker='x', linestyle='--',
                         label=f"{attack}-adversarial", linewidth=2)
        
        title_metric = "ASR" if metric == "ASR" else metric.capitalize()
        plt.title(f"CITADEL-FL: {title_metric} vs. Ablation Mode")
        plt.xlabel("Ablation Mode")
        plt.ylabel(title_metric)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"{metric}_comparison.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        
        print(f"✅ Saved plot: {save_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to plot {metric}: {e}")


def plot_summary_heatmap(df, metric, save_dir=OUT_DIR):
    """Generate heatmap of metric across modes and attacks."""
    try:
        if df.empty:
            print(f"⚠️ No data for heatmap: {metric}")
            return
        
        if metric == "robustness_score":
            pivot = df.pivot_table(index="attack", columns="mode", values=metric, aggfunc="mean")
            label = "Robustness score"
        elif metric == "ASR":
            pivot = df.pivot_table(index="attack", columns="mode", values="attack_ASR", aggfunc="mean")
            label = "ASR"
        else:
            pivot = df.pivot_table(index="attack", columns="mode", values=f"attack_{metric}", aggfunc="mean")
            label = metric.capitalize()
        
        if pivot.empty:
            print(f"⚠️ No data for heatmap: {metric}")
            return
        
        plt.figure(figsize=(10, 6))
        plt.imshow(pivot.values, cmap="viridis", aspect="auto")
        plt.colorbar(label=label)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"CITADEL-FL: {label} Heatmap")
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"heatmap_{metric}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        
        print(f"✅ Saved heatmap: {save_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to create heatmap {metric}: {e}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print("📊 Loading results...")
    df = load_results()
    
    if df.empty:
        print("⚠️ No results found. Please run experiments first.")
        exit(0)
    
    print(f"✅ Loaded {len(df)} experiment results\n")
    
    # Save summary table
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved summary metrics: {csv_path}\n")
    
    # Generate plots
    print("📈 Generating plots...")
    for m in METRICS:
        plot_metric(df, m)
        plot_summary_heatmap(df, m)
    
    # Robustness score plots
    plot_metric(df, "robustness_score")
    plot_summary_heatmap(df, "robustness_score")
    
    # Summary statistics
    print("\n🏆 Top-performing modes (by Robustness Score):")
    top_robustness = df.nlargest(10, "robustness_score")[
        ["dataset", "attack", "mode", "robustness_score", "clean_accuracy", "attack_accuracy", "attack_ASR"]
    ]
    print(top_robustness.to_string())
    
    print(f"\n📂 Analysis results saved in: {OUT_DIR}")