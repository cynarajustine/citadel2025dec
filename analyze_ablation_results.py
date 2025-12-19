#!/usr/bin/env python3
"""
CITADEL-FL: Ablation Results Analyzer (with Robustness Score)
-------------------------------------------------------------
Scans all ablation results and compiles metrics into summary CSV including robustness score.
"""

import os
import json
import pandas as pd
from pathlib import Path
import time

def collect_results(base_dir="results_citadel_v2"):
    """Collect results from all completed experiments."""
    rows = []
    
    for dataset_dir in Path(base_dir).glob("*"):
        if not dataset_dir.is_dir() or dataset_dir.name == "tuning":
            continue
            
        for attack_dir in dataset_dir.glob("*"):
            if not attack_dir.is_dir():
                continue
                
            for mode_dir in attack_dir.glob("*"):
                if not mode_dir.is_dir():
                    continue
                    
                hist_path = mode_dir / "history.json"
                
                if not hist_path.exists():
                    continue
                
                try:
                    if hist_path.stat().st_size > 0:
                        with open(hist_path) as f:
                            hist = json.load(f)
                    else:
                        continue
                    
                    if not isinstance(hist, list) or len(hist) == 0:
                        continue
                    
                    # Get last round metrics
                    last_round = hist[-1]
                    clean = last_round.get("clean", {})
                    attack = last_round.get("attack", {})
                    asr = last_round.get("asr", 0)
                    robustness_score = last_round.get("robustness_score", 0)
                    
                    row = {
                        "dataset": dataset_dir.name,
                        "attack": attack_dir.name,
                        "mode": mode_dir.name,
                        "clean_accuracy": clean.get("accuracy", 0),
                        "clean_f1": clean.get("f1", 0),
                        "clean_precision": clean.get("precision", 0),
                        "clean_recall": clean.get("recall", 0),
                        "attack_accuracy": attack.get("accuracy", 0),
                        "attack_f1": attack.get("f1", 0),
                        "attack_precision": attack.get("precision", 0),
                        "attack_recall": attack.get("recall", 0),
                        "ASR": asr,
                        "robustness_score": robustness_score,
                        "rounds": len(hist)
                    }
                    
                    rows.append(row)
                    
                except Exception as e:
                    print(f"⚠️ Failed to parse {hist_path}: {e}")
                    continue
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("🔍 Collecting ablation results...")
    df = collect_results()
    
    if df.empty:
        print("⚠️ No valid history.json files found.")
        exit(0)
    
    # Save summary
    out_path = "citadel_ablation_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved ablation summary → {out_path}")
    
    # Print statistics
    print("\n📊 Summary Statistics:")
    print(f"Total experiments: {len(df)}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Attacks: {df['attack'].unique().tolist()}")
    print(f"Modes: {df['mode'].unique().tolist()}")
    
    print("\n🏆 Average metrics by attack:")
    summary = df.groupby(["dataset", "attack"])[["clean_accuracy", "attack_f1", "ASR", "robustness_score"]].mean().round(4)
    print(summary)
    
    print("\n📈 Best robustness score per dataset/attack:")
    best_robustness = df.loc[df.groupby(["dataset", "attack"])["robustness_score"].idxmax()][["dataset", "attack", "mode", "robustness_score", "clean_accuracy", "attack_accuracy", "ASR"]]
    print(best_robustness)
    
    print(f"\n📁 Detailed results saved in: {out_path}")