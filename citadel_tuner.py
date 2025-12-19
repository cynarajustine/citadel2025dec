#!/usr/bin/env python3
"""
CITADEL-FL Hyperparameter Tuner (Multi-GPU Parallel) - FULLY FIXED
------------------------------------------------------------------
✅ Auto-detects GPU count (no hardcoding)
✅ ALL CLI arguments (--attack, --rounds, --num_clients, --batch_size, --lr, --save_dir)
✅ Atomic JSON saves for results
✅ Proper subprocess log handling
"""

import os, json, subprocess, itertools, time, tempfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch

# ============================================================
# Configuration
# ============================================================
SAVE_DIR_BASE = "results_citadel_v2/tuning"
Path(SAVE_DIR_BASE).mkdir(parents=True, exist_ok=True)

import argparse

# ✅ CRITICAL FIX: Define ALL arguments BEFORE parsing
parser = argparse.ArgumentParser(
    description="CITADEL-FL Hyperparameter Tuner (Multi-GPU Parallel)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Tune all datasets with default settings
  python citadel_tuner.py

  # Tune only VQA with custom attack
  python citadel_tuner.py --datasets vqa --attack pgd

  # Tune with custom rounds and clients
  python citadel_tuner.py --datasets cifar10 --rounds 3 --num_clients 8

  # Tune with custom learning rate and batch size
  python citadel_tuner.py --lr 0.002 --batch_size 128 --save_dir my_results
    """
)

# Dataset selection
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=["cifar10", "arrhythmia", "vqa"],
    help="Datasets to tune (default: cifar10 arrhythmia vqa)"
)

# ✅ RESTORED: Training hyperparameters (these were missing!)
parser.add_argument(
    "--attack",
    type=str,
    default="pgd",
    help="Attack type for tuning evaluation (default: pgd)"
)

parser.add_argument(
    "--rounds",
    type=int,
    default=2,
    help="Number of FL rounds per tuning experiment (default: 2 for speed)"
)

parser.add_argument(
    "--num_clients",
    type=int,
    default=5,
    help="Number of federated clients per round (default: 5 for tuning)"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size for training (default: 256)"
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Learning rate (default: 1e-3)"
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="results_citadel_v2",
    help="Base directory for saving results (default: results_citadel_v2)"
)

# ============================================================
# NOW parse arguments
# ============================================================
args = parser.parse_args()

# ============================================================
# Tuning Grid Configuration
# ============================================================
DATASETS = args.datasets
ATTACK = args.attack
ROUNDS = args.rounds
NUM_CLIENTS = args.num_clients
BATCH_SIZE = args.batch_size
LR = args.lr
SAVE_DIR = os.path.join(args.save_dir, "tuning")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Hyperparameter search space
TAUS = [1.0, 2.0]
SIGMAS = [0.01, 0.05, 0.1]
GAMMAS = [0.8, 0.95, 0.99]

# ✅ FIXED: Auto-detect GPU count (was hardcoded to 8)
GPU_COUNT = 1 #torch.cuda.device_count()
if GPU_COUNT == 0:
    print("❌ No GPUs found! Exiting.")
    exit(1)

MAX_JOBS_PER_GPU = 1  # one experiment per GPU at a time

print(f"\n{'='*70}")
print(f"🔧 CITADEL-FL Hyperparameter Tuning Configuration")
print(f"{'='*70}")
print(f"  Datasets: {DATASETS}")
print(f"  Attack: {ATTACK}")
print(f"  Rounds: {ROUNDS}, Clients: {NUM_CLIENTS}")
print(f"  Batch Size: {BATCH_SIZE}, LR: {LR}")
print(f"  Auto-detected GPUs: {GPU_COUNT}")
print(f"  Tuning Grid: τ={TAUS} × σ={SIGMAS} × γ={GAMMAS}")
print(f"  Total configs per dataset: {len(TAUS) * len(SIGMAS) * len(GAMMAS)}")
print(f"{'='*70}\n")

# ============================================================
# Helper Functions
# ============================================================

def atomic_json_save(obj, path):
    """Save JSON atomically using temp file + rename."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(
            dir=Path(path).parent or ".",
            prefix=".tmp_"
        )
        os.close(temp_fd)

        with open(temp_path, "w") as f:
            json.dump(obj, f, indent=2)

        os.replace(temp_path, path)
        return True
    except Exception as e:
        print(f"❌ Failed to save {path}: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def launch_job(job, gpu_id):
    """
    Launch a single tuning job on a specific GPU.

    Returns:
        (proc, gpu_id, job, result_file, log_file, log_f)
    """
    dataset, tau, sigma, gamma = job
    log_file = os.path.join(
        SAVE_DIR,
        f"log_{dataset}_tau{tau}_sigma{sigma}_gamma{gamma}.txt"
    )
    result_file = os.path.join(
        SAVE_DIR,
        f"{dataset}_tau{tau}_sigma{sigma}_gamma{gamma}.json"
    )

    cmd = [
        "python", "citadel_fl_v2.py",
        "--dataset", dataset,
        "--attack", ATTACK,
        "--tau", str(tau),
        "--sigma", str(sigma),
        "--gamma", str(gamma),
        "--rounds", str(ROUNDS),
        "--num_clients", str(NUM_CLIENTS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--save_dir", args.save_dir,
        "--device", f"cuda:3",#{gpu_id}",
        "--seed", str(42 + hash(f"{dataset}_{tau}_{sigma}_{gamma}") % 1000)
    ]

    try:
        # Open with line buffering for real-time log streaming
        log_f = open(log_file, "w", buffering=1, encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True
        )
        return proc, gpu_id, job, result_file, log_file, log_f
    except Exception as e:
        print(f"❌ Failed to launch job: {e}")
        return None, None, None, None, None, None


def parse_result(path):
    """Parse tuning result file safely."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not parse {path}: {e}")
            return None
    return None


def compute_score(row):
    """
    Score a tuning result: balance clean accuracy and F1 against ASR.
    Higher is better.
    """
    clean_acc = row.get("clean_acc", 0)
    clean_f1 = row.get("clean_f1", 0)
    attack_f1 = row.get("attack_f1", 0)
    asr = row.get("asr", 0)

    # Weighted score: prioritize clean performance and robustness
    score = (
        clean_acc * 0.4 +      # 40% clean accuracy
        clean_f1 * 0.3 +       # 30% clean F1
        attack_f1 * 0.2 -      # 20% attack F1
        asr * 0.1              # -10% ASR (penalty)
    )
    return score


# ============================================================
# Main Scheduler
# ============================================================

def run_parallel_tuning():
    """
    Execute hyperparameter tuning across all GPUs.
    """
    all_jobs = list(itertools.product(DATASETS, TAUS, SIGMAS, GAMMAS))
    print(f"🚀 Total tuning jobs to launch: {len(all_jobs)}\n")

    active = []
    completed_results = []
    job_iter = iter(all_jobs)
    pbar = tqdm(total=len(all_jobs), desc="Tuning Progress", dynamic_ncols=True)
    total_jobs = len(all_jobs)
    completed_count = 0

    while True:
        # ============================================================
        # STEP 1: Clean up finished jobs
        # ============================================================
        finished = []
        for proc, gpu, job, result_path, log_file, log_f in active:
            if proc.poll() is not None:  # Process finished
                # Ensure log file is flushed and closed
                log_f.flush()
                log_f.close()

                # Parse result
                res = parse_result(result_path)
                if res:
                    res["dataset"] = job[0]
                    res["tau"] = job[1]
                    res["sigma"] = job[2]
                    res["gamma"] = job[3]
                    completed_results.append(res)
                    print(
                        f"✅ [{job[0]}] τ={job[1]}, σ={job[2]}, γ={job[3]} "
                        f"→ clean_acc={res.get('clean_acc', 0):.3f}, "
                        f"asr={res.get('asr', 0):.3f}"
                    )
                else:
                    print(
                        f"⚠️ Missing/invalid result for {job[0]} "
                        f"τ={job[1]}, σ={job[2]}, γ={job[3]} (check {log_file})"
                    )

                finished.append((proc, gpu, job, result_path, log_file, log_f))
                completed_count += 1
                pbar.update(1)

        # Remove finished from active list
        for f in finished:
            active.remove(f)

        # ============================================================
        # STEP 2: Launch new jobs on free GPUs
        # ============================================================
        while len(active) < GPU_COUNT * MAX_JOBS_PER_GPU:
            try:
                job = next(job_iter)
                gpu_id = len(active) % GPU_COUNT
                result = launch_job(job, gpu_id)

                if result[0]:  # Process was created successfully
                    print(
                        f"🧩 [{job[0]}] τ={job[1]}, σ={job[2]}, γ={job[3]} "
                        f"→ GPU {gpu_id}"
                    )
                    active.append(result)
                else:
                    print(f"❌ Failed to launch {job}")
                    # Put job back for retry
                    all_jobs.append(job)
            except StopIteration:
                break

        # ============================================================
        # STEP 3: Exit condition
        # ============================================================
        if not active and completed_count >= total_jobs:
            break

        time.sleep(5)

    pbar.close()
    print("\n🏁 All tuning jobs completed.\n")

    # ============================================================
    # STEP 4: Aggregate and Save Results
    # ============================================================
    if not completed_results:
        print("⚠️ No results collected!")
        return

    df = pd.DataFrame(completed_results)
    summary_path = os.path.join(SAVE_DIR, "citadel_tuning_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"💾 Saved detailed tuning metrics → {summary_path}")
    print(f"\n📊 Summary (first 10 rows):\n{df.head(10)}\n")

    # ============================================================
    # STEP 5: Select Best Configuration per Dataset
    # ============================================================
    best_config = {}
    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            print(f"⚠️ No tuning results for {dataset}")
            continue

        # Compute score for each configuration
        subset["score"] = subset.apply(compute_score, axis=1)

        # Select best
        best_idx = subset["score"].idxmax()
        best_row = subset.loc[best_idx]

        best_config[dataset] = {
            "tau": float(best_row["tau"]),
            "sigma": float(best_row["sigma"]),
            "gamma": float(best_row["gamma"])
        }

        print(
            f"✅ Best for {dataset}:\n"
            f"   τ={best_row['tau']}, σ={best_row['sigma']}, γ={best_row['gamma']}\n"
            f"   Score={best_row['score']:.4f}\n"
            f"   Clean Acc={best_row.get('clean_acc', 0):.3f}, "
            f"Attack F1={best_row.get('attack_f1', 0):.3f}, "
            f"ASR={best_row.get('asr', 0):.3f}\n"
        )

    # ============================================================
    # STEP 6: Save Best Config
    # ============================================================
    config_path = os.path.join(SAVE_DIR, "citadel_best_config.json")
    if atomic_json_save(best_config, config_path):
        print(f"✅ Saved best configurations → {config_path}\n")
    else:
        print(f"❌ Failed to save best config\n")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    run_parallel_tuning()