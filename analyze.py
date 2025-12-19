#!/usr/bin/env python3
"""
CITADEL-FL v4: Unified Experiment Analyzer
------------------------------------------------
- Scans all experiment results in a given directory.
- Compiles all metrics (accuracy, F1, ASR, robustness_score) into a single CSV.
- Generates plots and heatmaps for key metrics.
- Prints comprehensive summary statistics to the console and a text file.
- Replaces analyze_results.py and analyze_ablation_results.py.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse

def collect_results(base_dir):
    """Collect results from all completed experiments."""
    rows = []
    skipped = []
    
    print(f"🔍 Scanning for results in: {base_dir}")
    for dataset_dir in Path(base_dir).glob("*"):
        if not dataset_dir.is_dir() or dataset_dir.name in ["tuning", "pretrained", "analysis"]:
            continue
        
        dataset_name = dataset_dir.name
        
        for attack_dir in dataset_dir.glob("*"):
            if not attack_dir.is_dir():
                continue
            
            attack_name = attack_dir.name
            
            for mode_dir in attack_dir.glob("*"):
                if not mode_dir.is_dir():
                    continue
                
                mode_name = mode_dir.name
                hist_path = mode_dir / "history.json"
                
                if not hist_path.exists():
                    skipped.append(f"{dataset_name}/{attack_name}/{mode_name} (no history.json)")
                    continue
                
                try:
                    if hist_path.stat().st_size < 20: # Basic check for non-empty JSON
                        skipped.append(f"{dataset_name}/{attack_name}/{mode_name} (empty history.json)")
                        continue
                    
                    with open(hist_path, 'r') as f:
                        hist = json.load(f)
                    
                    if not isinstance(hist, list) or len(hist) == 0:
                        skipped.append(f"{dataset_name}/{attack_name}/{mode_name} (invalid history format)")
                        continue
                    
                    # Average over last 3 rounds for stability
                    last_rounds = hist[-3:] if len(hist) >= 3 else hist
                    
                    row = {
                        "dataset": dataset_name,
                        "attack": attack_name,
                        "mode": mode_name,
                        "clean_accuracy": np.mean([r.get("clean", {}).get("accuracy", 0) for r in last_rounds]),
                        "clean_f1": np.mean([r.get("clean", {}).get("f1", 0) for r in last_rounds]),
                        "attack_accuracy": np.mean([r.get("attack", {}).get("accuracy", 0) for r in last_rounds]),
                        "attack_f1": np.mean([r.get("attack", {}).get("f1", 0) for r in last_rounds]),
                        "ASR": np.mean([r.get("asr", 0) for r in last_rounds]),
                        "robustness_score": np.mean([r.get("robustness_score", 0) for r in last_rounds]),
                        "rounds": len(hist),
                    }
                    rows.append(row)
                    
                except Exception as e:
                    skipped.append(f"{dataset_name}/{attack_name}/{mode_name} ({type(e).__name__})")
    
    print(f"\n✅ Collection complete. Found {len(rows)} valid results. Skipped {len(skipped)}.")
    if skipped:
        print("🔍 Skipped entries (first 5):")
        for item in skipped[:5]:
            print(f"  - {item}")
    
    return pd.DataFrame(rows)


def generate_plots(df, out_dir):
    """Generate and save plots for key metrics."""
    if df.empty:
        return
    
    plot_metrics = ["robustness_score", "clean_accuracy", "attack_f1", "ASR"]
    
    for metric in plot_metrics:
        try:
            plt.figure(figsize=(14, 7))
            
            # Pivot data for plotting
            pivot = df.pivot_table(index="mode", columns=["dataset", "attack"], values=metric, aggfunc="mean")
            pivot = pivot.reindex(sorted(pivot.index)) # Sort modes alphabetically
            
            pivot.plot(kind="bar", ax=plt.gca(), width=0.8)
            
            plt.title(f"Performance by Mode: {metric.replace('_', ' ').title()}", fontsize=16)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.xlabel("Ablation Mode", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title="Dataset-Attack", fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            
            save_path = os.path.join(out_dir, f"plot_{metric}.png")
            plt.savefig(save_path, dpi=120)
            plt.close()
            print(f"🖼️ Saved plot: {save_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to plot {metric}: {e}")

def save_summaries(df, out_dir):
    """Save CSV and text summaries."""
    if df.empty:
        return

    # Save full data to CSV
    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"💾 Saved full results to: {csv_path}")

    # Save text summary
    stats_path = os.path.join(out_dir, "ablation_statistics.txt")
    with open(stats_path, "w") as f:
        f.write("CITADEL-FL Ablation Study Statistics\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("🏆 Top 10 Experiments by Robustness Score\n")
        f.write("-" * 80 + "\n")
        top_10 = df.nlargest(10, "robustness_score")[
            ["dataset", "attack", "mode", "robustness_score", "clean_accuracy", "attack_f1", "ASR"]
        ]
        f.write(top_10.to_string(index=False))
        f.write("\n\n")

        f.write("📊 Per-Mode Average Performance (across all datasets/attacks)\n")
        f.write("-" * 80 + "\n")
        mode_summary = df.groupby("mode")[
            ["robustness_score", "clean_accuracy", "attack_f1", "ASR"]
        ].mean().sort_values("robustness_score", ascending=False)
        f.write(mode_summary.to_string())
        f.write("\n\n")
        
        f.write("🎯 Best Mode per Attack (based on highest mean robustness_score)\n")
        f.write("-" * 80 + "\n")
        best_per_attack = df.loc[df.groupby(["dataset", "attack"])["robustness_score"].idxmax()]
        f.write(best_per_attack[["dataset", "attack", "mode", "robustness_score"]].to_string(index=False))
        
    print(f"📄 Saved text summary to: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CITADEL-FL Unified Analyzer")
    parser.add_argument("--results_dir", type=str, default="results_citadel_v2", help="Directory containing experiment results.")
    parser.add_argument("--output_dir", type=str, default="citadel_analysis", help="Directory to save analysis files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    df = collect_results(args.results_dir)
    
    if df.empty:
        print("❌ No valid results found. Exiting.")
    else:
        generate_plots(df, args.output_dir)
        save_summaries(df, args.output_dir)
        print("\n🎉 Analysis complete!")