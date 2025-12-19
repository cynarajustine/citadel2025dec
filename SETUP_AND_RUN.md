# CITADEL-FL v2: Complete Setup & Execution Guide

## Environment Setup

```bash
# 1. Create shared dataset directory (one-time)
mkdir -p /raid/datasets
chmod 777 /raid/datasets

# 2. Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers timm scikit-learn pandas numpy matplotlib tqdm filelock

# 3. Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"