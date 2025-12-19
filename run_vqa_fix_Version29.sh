#!/usr/bin/env bash
set -euo pipefail

# Fast, pragmatic steps to get VQA learning with the MobileNet fast path.
# Assumes your citadel_fl_v2.py already supports VQA_FAST and FastVQAModel.

export VQA_FAST=1
export VQA_TRAIN_SUBSET=${VQA_TRAIN_SUBSET:-8000}
export VQA_TEST_SUBSET=${VQA_TEST_SUBSET:-2000}
export NUM_WORKERS=${NUM_WORKERS:-8}

# 1) Sanity check: single client, base mode, no attack.
#    Use higher LR and unfreeze the last MobileNet block to ensure the head/backbone can adapt.
#    Expect accuracy to jump well above random (>0.5 often) after one local epoch on the subset.
export VQA_UNFREEZE=1   # requires your FastVQAModel to honor this env; if unsupported, ignore
python citadel_fl_v2.py \
  --dataset vqa \
  --mode base \
  --attack none \
  --rounds 1 \
  --num_clients 1 \
  --batch_size 256 \
  --device cuda:0 \
  --lr 0.01

# 2) Now run CITADEL tuned for this fast VQA setup.
#    Keep the higher LR and unfreeze flag. Start with slightly lower gamma for faster movement.
python citadel_fl_v2.py \
  --dataset vqa \
  --mode citadel \
  --attack fgsm \
  --rounds 12 \
  --num_clients 10 \
  --batch_size 128 \
  --device cuda:3 \
  --lr 0.01 \
  --tau 3 \
  --sigma 0.01 \
  --gamma 0.8 \
  --save_dir results_citadel_v2

# 3) Analyze after it completes (optional).
python analyze_results.py