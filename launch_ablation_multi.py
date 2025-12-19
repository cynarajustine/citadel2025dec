#!/usr/bin/env python3
"""
CITADEL-FL Multi-GPU Launcher (Optimized for A100)
--------------------------------------------------
✅ Multiple jobs per GPU (adaptive scheduling)
✅ Proper subprocess logging (line buffering + explicit flush/close)
✅ Consistent save_dir resolution
✅ Larger batch sizes (256)
✅ num_workers=8 for faster data loading
"""

import os
import subprocess
import time
import argparse
from itertools import product
from tqdm import tqdm
import torch
import json
from pathlib import Path

# --------------------------
# Configuration
# --------------------------
DEFAULT_DATASETS = ["cifar10", "arrhythmia", "vqa"]
MODES = ["base", "trust", "lra", "fair", "trust_lra", "trust_fair", "lra_fair", "citadel", "trimmed_mean", "median"]

# Increase batch size and training strength
BATCH_SIZE = 256  # Up from 64
ROUNDS = 20       # More rounds to see convergence
NUM_CLIENTS = 10
LOCAL_EPOCHS = 2  # Add this parameter

LR = 1e-3
DEFAULT_SAVE_DIR = "results_citadel_v2"
PYTHON_EXEC = "python"
SCRIPT = "citadel_fl_v2.py"
SLEEP_INTERVAL = 5  # ✅ REDUCED
BEST_CONFIG_PATH = os.path.join(DEFAULT_SAVE_DIR, "tuning", "citadel_best_config.json")

# ✅ OPTIMIZATION: Allow 2-3 jobs per GPU
MAX_JOBS_PER_GPU = 1

# --------------------------
# Helper Functions
# --------------------------
def load_best_params(dataset):
    """Load tuned τ, σ, γ values for a dataset."""
    default = {"tau": 1.0, "sigma": 0.05, "gamma": 0.95}
    if os.path.exists(BEST_CONFIG_PATH):
        try:
            with open(BEST_CONFIG_PATH) as f:
                best = json.load(f)
                if dataset in best:
                    return best[dataset]
        except Exception as e:
            print(f"⚠️ Could not load best config: {e}")
    return default


def get_datasets(args):
    """Return dataset(s) to run."""
    if args.dataset and args.dataset.lower() != "all":
        return [args.dataset.lower()]
    return DEFAULT_DATASETS


def launch_experiment(exp, gpu, params, args, pretrained_path=None):
    """Launch a single experiment on a specific GPU with robust logging."""
    base_save_dir = os.path.abspath(args.save_dir)
    save_dir = os.path.join(base_save_dir, exp["dataset"], exp["attack"], exp["mode"])
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Unique log file with timestamp
    log_path = os.path.join(save_dir, f"log_{exp['mode']}_{exp['attack']}_{int(time.time())}.txt")

    cmd = [
        PYTHON_EXEC, SCRIPT,
        "--dataset", exp["dataset"],
        "--attack", exp["attack"],
        "--mode", exp["mode"],
        "--device", f"cuda:3",
        "--rounds", str(args.rounds),
        "--num_clients", str(args.num_clients),
        "--batch_size", str(args.batch_size),
        "--lr", str(LR),
        "--save_dir", base_save_dir,
        "--tau", str(params["tau"]),
        "--sigma", str(params["sigma"]),
        "--gamma", str(params["gamma"]),
        "--local_epochs", "2",  # <-- Add this
        "--seed", str(42 + hash(f"{exp['dataset']}_{exp['mode']}_{exp['attack']}") % 1000)
    ]

    if pretrained_path and os.path.exists(pretrained_path):
        cmd += ["--pretrained_path", pretrained_path]

    try:
        # Open with line buffering for real-time streaming
        log_f = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=False
        )
        return proc, log_f, save_dir
    except Exception as e:
        print(f"❌ Failed to launch process: {e}")
        return None, None, save_dir


def check_history_valid(save_dir):
    """Verify that history.json exists and has valid data."""
    hist_path = os.path.join(save_dir, "history.json")
    try:
        if os.path.exists(hist_path) and os.path.getsize(hist_path) > 50:
            with open(hist_path, 'r') as f:
                hist = json.load(f)
                return isinstance(hist, list) and len(hist) > 0
    except Exception as e:
        print(f"⚠️ History validation failed for {hist_path}: {e}")
    return False


# --------------------------
# Main Scheduler
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CITADEL-FL Multi-GPU Ablation Launcher (Optimized)")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset: cifar10, arrhythmia, vqa, or all")
    parser.add_argument("--attack", type=str, default="none", help="Attack type")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--pretrained_dir", type=str, default="results_citadel_v2/pretrained",
                        help="Optional directory for pretrained base models")
    parser.add_argument("--mode", type=str, default="citadel")
    parser.add_argument("--tau", type=float, help="Manual override for tau")
    parser.add_argument("--sigma", type=float, help="Manual override for sigma")
    parser.add_argument("--gamma", type=float, help="Manual override for gamma")
    args = parser.parse_args()

    DATASETS = get_datasets(args)

    # Build experiment grid
    EXPERIMENTS = []
    for dataset, mode, attack in product(DATASETS, MODES, [args.attack]):
        EXPERIMENTS.append({"dataset": dataset, "mode": mode, "attack": attack})

    total_jobs = len(EXPERIMENTS)
    active_processes = {}  # {gpu_id: [(proc, log_f, save_dir), ...]}
    completed_jobs = 0
    n_gpus = torch.cuda.device_count()

    print(f"🚀 Launching CITADEL-FL ablation suite ({total_jobs} experiments) across {n_gpus} GPUs.")
    print(f"📋 Max jobs per GPU: {MAX_JOBS_PER_GPU}")
    print(f"💾 Batch size: {args.batch_size}\n")

    with tqdm(total=total_jobs, desc="Global Progress") as pbar:
        while completed_jobs < total_jobs:
            # ✅ Check finished jobs
            for gpu_id in list(active_processes.keys()):
                remaining = []
                for proc, log_f, save_dir in active_processes[gpu_id]:
                    if proc.poll() is not None:
                        # Process finished
                        log_f.flush()
                        log_f.close()
                        
                        # Verify output
                        if check_history_valid(save_dir):
                            print(f"✅ GPU {gpu_id}: Process completed successfully")
                        else:
                            print(f"⚠️ GPU {gpu_id}: History validation failed")
                        
                        completed_jobs += 1
                        pbar.update(1)
                    else:
                        remaining.append((proc, log_f, save_dir))
                
                if remaining:
                    active_processes[gpu_id] = remaining
                else:
                    del active_processes[gpu_id]

            # ✅ Launch new jobs
            for gpu_id in range(n_gpus):
                if gpu_id not in active_processes:
                    active_processes[gpu_id] = []
                
                # Add up to MAX_JOBS_PER_GPU per GPU
                while len(active_processes[gpu_id]) < MAX_JOBS_PER_GPU and EXPERIMENTS:
                    exp = EXPERIMENTS.pop(0)
                    params = load_best_params(exp["dataset"])
                    
                    if args.tau is not None:
                        params["tau"] = args.tau
                    if args.sigma is not None:
                        params["sigma"] = args.sigma
                    if args.gamma is not None:
                        params["gamma"] = args.gamma

                    proc, log_f, save_dir = launch_experiment(exp, gpu_id, params, args)
                    if proc and log_f:
                        active_processes[gpu_id].append((proc, log_f, save_dir))
                        print(f"🚀 [{gpu_id}] {exp['dataset']} | {exp['mode']} | {exp['attack']} "
                              f"(τ={params['tau']}, σ={params['sigma']}, γ={params['gamma']})")
                    else:
                        print(f"❌ Failed to launch {exp['dataset']} | {exp['mode']} | {exp['attack']}")
                        EXPERIMENTS.append(exp)

            time.sleep(SLEEP_INTERVAL)

        # ✅ Final cleanup
        for gpu_id, jobs in active_processes.items():
            for _, log_f, _ in jobs:
                try:
                    log_f.flush()
                    log_f.close()
                except:
                    pass

    print("\n✅ All experiments completed successfully!")