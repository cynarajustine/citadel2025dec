# CITADEL-FL v4: Federated Learning with Trust-Weighted Consensus

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+
- 8× A100 GPUs (or adapt GPU_COUNT in launchers)

### Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers timm scikit-learn pandas numpy matplotlib tqdm filelock

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## 🚀 Quick Start

### Option 1: Hyperparameter Tuning (Recommended)

Tune τ, σ, γ for each dataset across all GPUs:

```bash
python citadel_tuner.py \
  --datasets cifar10 arrhythmia vqa
```

**Output**: `results_citadel_v2/tuning/citadel_best_config.json`

### Option 2: Run Ablation Studies

Run all 8 modes × 1 attack × 3 datasets = 24 experiments:

```bash
# All datasets, all attacks
python launch_ablation_multi.py \
  --dataset all \
  --attack pgd \
  --rounds 5 \
  --num_clients 10 \
  --batch_size 64 \
  --save_dir results_citadel_v2

# Single dataset
python launch_ablation_multi.py \
  --dataset cifar10 \
  --attack pgd

# Override tuned hyperparameters
python launch_ablation_multi.py \
  --dataset cifar10 \
  --attack pgd \
  --tau 1.5 \
  --sigma 0.1 \
  --gamma 0.9
```

**Output**: 
- Per-experiment logs: `results_citadel_v2/{dataset}/{attack}/{mode}/log_*.txt`
- History: `results_citadel_v2/{dataset}/{attack}/{mode}/history.json`
- Checkpoints: `results_citadel_v2/{dataset}/{attack}/{mode}/checkpoints/round_*.pt`

### Option 3: Analyze Results

After experiments complete:

```bash
# Collect and summarize results
python analyze_ablation_results.py
# Output: citadel_ablation_summary.csv

# Generate plots
python analyze_results.py
# Output: citadel_analysis/summary.csv, plots, heatmaps
```

## 📊 File Structure

```
results_citadel_v2/
├── tuning/
│   ├── citadel_best_config.json          # Best τ, σ, γ per dataset
│   ├── citadel_tuning_summary.csv        # All tuning results
│   └── log_*.txt                         # Tuning logs
├── cifar10/
│   ├── pgd/
│   │   ├── citadel/
│   │   │   ├── history.json              # Final metrics
│   │   │   ├── log_citadel_pgd.txt       # Console output
│   │   │   └── checkpoints/
│   │   │       ├── round_0.pt
│   │   │       └── round_4.pt            # Latest checkpoint
│   │   └── ...other modes...
│   └── ...other attacks...
├── arrhythmia/
└── vqa/

citadel_ablation_summary.csv             # Aggregated results
citadel_analysis/
├── summary.csv                          # Analysis summary
├── accuracy_comparison.png
├── f1_comparison.png
├── heatmap_f1.png
└── ...plots...
```

## 🧪 Command Examples

### Smoke test (quick validation)
```bash
python citadel_fl_v2.py \
  --dataset cifar10 \
  --attack none \
  --mode citadel \
  --rounds 1 \
  --num_clients 2 \
  --batch_size 32 \