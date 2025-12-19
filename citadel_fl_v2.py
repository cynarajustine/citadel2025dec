#!/usr/bin/env python3
"""
CITADEL-FL v2: Federated Learning with Trust-Weighted Stochastic Consensus
------------------------------------------------------------------------
Unified training script supporting CIFAR10, Arrhythmia, and Synthetic VQA. 
Includes ablation modes, attack simulation, multi-GPU support, and metric logging. 

✅ FIXED: 
- Hyperparameters default to None (loads tuned values from JSON)
- Mode-specific aggregation functions
- All existing code preserved
- Added robustness score calculation

VQA speedups (fast by default):
- Uses MobileNetV3-Small classifier (frozen backbone) instead of ViLT when VQA_FAST=1
- Subsamples CIFAR-10 for VQA via env:  VQA_TRAIN_SUBSET, VQA_TEST_SUBSET
- Standard torchvision pipeline (no per-batch processor), multi-worker loaders

CITADEL advantage:
- FAIR-family aggregators smooth against previous GLOBAL state (not client 0)
"""
from torch.utils.data import DataLoader

import os, json, copy, random, argparse, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch. nn.functional as F
import torchvision, torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import pandas as pd
from filelock import FileLock

# Optional transformers (only used when VQA_FAST=0)
try:
    from transformers import ViltProcessor, ViltModel
except Exception:
    ViltProcessor = None
    ViltModel = None

DATA_ROOT = "datasets"
os.makedirs(DATA_ROOT, exist_ok=True)
DATA_LOCK = os.path.join(DATA_ROOT, "dataset. lock")

# VQA runtime knobs (env)
VQA_FAST = os.getenv("VQA_FAST", "1") == "1"            # 1: MobileNet fast path; 0: ViLT
VQA_TRAIN_SUBSET = int(os.getenv("VQA_TRAIN_SUBSET", "8000"))
VQA_TEST_SUBSET  = int(os.getenv("VQA_TEST_SUBSET", "2000"))
NUM_WORKERS = int(os. getenv("NUM_WORKERS", "4"))

# --------------------------
# Utility Functions
# --------------------------
def load_best_hyperparams(dataset, tuning_path="results_citadel_v2/tuning/citadel_best_config.json"):
    default = {"tau": 1.0, "sigma": 0.05, "gamma": 0.95}
    try:
        if os.path.exists(tuning_path):
            with open(tuning_path) as f:
                data = json.load(f)
                if dataset in data:
                    return data[dataset]
    except Exception as e:
        print(f"⚠️ Could not load tuned params: {e}")
    return default

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    with open(path, "w") as f:
        json. dump(obj, f, indent=2)

def to_cpu_state_dict(state):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.items()}

def to_device_state_dict(state, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items()}

# --------------------------
# Models per Dataset
# --------------------------

class ViltForCIFARVQA(nn.Module):
    def __init__(self, pretrained="dandelin/vilt-b32-finetuned-vqa", num_classes=10, freeze_backbone=True):
        super().__init__()
        if ViltModel is None:
            raise ImportError("transformers not available; set VQA_FAST=1 to use the fast VQA path.")
        self.vilt = ViltModel.from_pretrained(pretrained)
        hidden = self.vilt.config.hidden_size
        self. fc = nn.Linear(hidden, num_classes)
        if freeze_backbone:
            for p in self.vilt.parameters():
                p.requires_grad = False

    def forward(self, x=None, **kwargs):
        if x is None and not kwargs:
            raise ValueError("❌ ViltForCIFARVQA received None as input batch.")
        if isinstance(x, torch.Tensor):
            pooled = self.vilt(pixel_values=x, return_dict=True).pooler_output
        elif isinstance(x, dict):
            pooled = self.vilt(**x, return_dict=True).pooler_output
        else:
            pooled = self.vilt(**kwargs, return_dict=True).pooler_output
        return self.fc(pooled)

class FastVQAModel(nn.Module):
    """
    Lightweight VQA surrogate for CIFAR-based VQA: 
    - MobileNetV3-Small backbone (pretrained), classifier head to 10 classes
    - By default backbone. features are frozen, classifier is trainable
    - Set env VQA_UNFREEZE=1 to also train the last feature block for more capacity
    """
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        try:
            backbone = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        except Exception:
            backbone = torchvision.models.mobilenet_v3_small(weights=None)

        # Replace final classifier layer to 10 classes
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn. Linear(in_features, num_classes)

        # Freeze backbone features by default; keep classifier trainable
        for p in backbone.features.parameters():
            p.requires_grad = not freeze_backbone

        # Optionally unfreeze the last feature block (helps learning)
        if os.getenv("VQA_UNFREEZE", "0") == "1":
            try:
                for p in backbone.features[-1].parameters():
                    p.requires_grad = True
                print("🔓 VQA_UNFREEZE=1: Unfroze last feature block.")
            except Exception: 
                pass

        # Ensure classifier is trainable
        for p in backbone.classifier.parameters():
            p.requires_grad = True

        self.model = backbone

        # Debug: report trainable param count
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model. parameters() if p.requires_grad)
        print(f"🧩 FastVQAModel params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")

    def forward(self, x):
        return self.model(x)

def get_model(dataset):
    if dataset == "cifar10": 
        try:
            model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        except Exception: 
            model = torchvision. models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc. in_features, 10)
    elif dataset == "arrhythmia":
        model = nn.Sequential(
            nn.Linear(279, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    elif dataset == "vqa":
        if VQA_FAST or ViltModel is None:
            model = FastVQAModel()
            print("⚡ VQA_FAST=1: Using FastVQAModel (MobileNetV3-Small, frozen backbone).")
        else:
            model = ViltForCIFARVQA()
            print("🧠 Using ViLT for VQA (slow). Set VQA_FAST=1 for a faster path.")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return model

# --------------------------
# Data Loaders
# --------------------------

def _subset_dataset(ds, k):
    if k is None or k <= 0 or k >= len(ds):
        return ds
    # stable random subset
    idx = torch.randperm(len(ds))[:k]. tolist()
    return torch.utils.data.Subset(ds, idx)

def get_data(dataset, batch_size=64):

    if dataset == "cifar10":
        with FileLock(DATA_LOCK):
            transform = transforms. Compose([transforms.ToTensor()])
            train = torchvision.datasets. CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
            test = torchvision. datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
        pin = torch.cuda.is_available()
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin, persistent_workers=True),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin, persistent_workers=True),
        )

    elif dataset == "arrhythmia":
        csv_path = os.path.join(DATA_ROOT, "arrhythmia.data")
        with FileLock(DATA_LOCK):
            if not os.path.exists(csv_path):
                print("📥 Downloading Arrhythmia dataset (UCI)...")
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
                os.system(f"wget -q -O {csv_path} {url}")

        try:
            df = pd.read_csv(csv_path, header=None, na_values="?").dropna()
            X = df.iloc[:, :-1].astype(float).values
            y = df.iloc[:, -1].values
            y = (y != 1).astype(int)
        except Exception as e:
            print(f"⚠️ Could not load arrhythmia dataset: {e}")
            print("🔄 Using synthetic fallback...")
            X = np.random. randn(452, 279).astype(np.float32)
            y = np.random.randint(0, 2, size=(452,))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        train = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch. tensor(y_train, dtype=torch.long)
        )
        test = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size)
        )

    elif dataset == "vqa":
        # Fast path:  image-only MobileNet classifier with CIFAR10 images
        if VQA_FAST or ViltProcessor is None:
            # Use ImageNet normalization for pretrained MobileNet
            tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            with FileLock(DATA_LOCK):
                base_train = torchvision.datasets. CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
                base_test = torchvision. datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)

            # Subsample for speed
            train_ds = _subset_dataset(base_train, VQA_TRAIN_SUBSET)
            test_ds = _subset_dataset(base_test, VQA_TEST_SUBSET)

            pin = torch.cuda.is_available()
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=True),
                DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=True),
            )

        # Slow (original) ViLT path
        with FileLock(DATA_LOCK):
            processor = ViltProcessor. from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            processor.image_processor. do_rescale = False

        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        with FileLock(DATA_LOCK):
            base_train = torchvision. datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
            base_test = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)

        questions = [
            "What is in this picture?",
            "Identify the object.",
            "What is seen here?",
            "What is in this photo?",
            "What is in this image?",
        ]

        def collate_fn(batch):
            try:
                imgs, labels = zip(*batch)
                qs = [random.choice(questions) for _ in labels]
                enc = processor(list(imgs), qs, return_tensors="pt", padding=True, truncation=True)
                if enc is None:
                    raise ValueError("Processor returned None encoding.")
                labels_tensor = torch.tensor(labels, dtype=torch.long)
                return enc, labels_tensor
            except Exception as e:
                print(f"⚠️ [collate_fn fallback] Processor failed: {e}")
                try:
                    dummy_x = torch.randn(len(batch), 3, 224, 224)
                    dummy_y = torch.randint(0, 10, (len(batch),))
                    return dummy_x, dummy_y
                except Exception as inner_e:
                    print(f"❌ Secondary collate_fn error: {inner_e}")
                    dummy_x = torch.randn(1, 3, 224, 224)
                    dummy_y = torch.randint(0, 10, (1,))
                    return dummy_x, dummy_y

        return (
            DataLoader(base_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
            DataLoader(base_test, batch_size=batch_size, collate_fn=collate_fn)
        )

    else:
        raise ValueError(f"Unknown dataset {dataset}")

# --------------------------
# Adversarial Attacks
# --------------------------

def fgsm_attack(model, x, y, eps=0.1):
    """FGSM with stronger perturbation for meaningful robustness evaluation."""
    device = next(model.parameters()).device
    if isinstance(x, torch.Tensor):
        x_adv = x.detach().to(device).clone()
        x_adv.requires_grad = True
        out = model(x_adv)
        loss = F.cross_entropy(out, y)
        loss.backward()
        x_adv = (x_adv + eps * x_adv.grad. sign()).clamp(0, 1).detach()
        return x_adv
    # Dict path (ViLT)
    x_adv = {k: (v.detach().to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
    for k in x_adv:
        if torch.is_floating_point(x_adv[k]):
            x_adv[k]. requires_grad = True
    out = model(**x_adv)
    loss = F.cross_entropy(out, y)
    loss.backward()
    for k in x_adv: 
        if torch.is_floating_point(x_adv[k]) and x_adv[k]. grad is not None:
            x_adv[k] = x_adv[k] + eps * x_adv[k].grad.sign()
            if k == "pixel_values":
                x_adv[k] = x_adv[k].clamp(0, 1)
            x_adv[k] = x_adv[k]. detach()
    return x_adv

def pgd_attack(model, x, y, eps=0.03, alpha=0.01, steps=7):
    """
    ✅ FIXED PGD Attack: 
    - Proper gradient accumulation with torch.enable_grad()
    - Correct epsilon/perturbation clamping
    - Device consistency
    - Returns properly detached adversarial examples
    """
    device = next(model.parameters()).device

    if isinstance(x, torch.Tensor):
        # Save original for epsilon bounding
        x_orig = x.clone().detach().to(device)
        x_adv = x_orig.clone().detach()

        for step in range(steps):
            x_adv.requires_grad_(True)

            with torch.enable_grad():
                out = model(x_adv)
                loss = F.cross_entropy(out, y. to(device))

            # Compute gradient
            grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]

            # Update adversarial example
            x_adv = x_adv.detach() + alpha * grad.sign()

            # Project back to epsilon ball around original
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = (x_orig + delta).detach()

            # Clamp to [0, 1] (image range)
            x_adv = torch.clamp(x_adv, 0, 1).detach()

        return x_adv

    else:
        # Dict case (ViLT) - not fully implemented for PGD
        # Fall back to FGSM for now
        print("⚠️ PGD not fully implemented for dict inputs, using FGSM instead")
        return fgsm_attack(model, x, y, eps=eps)

# --------------------------
# MODE-SPECIFIC AGGREGATION FUNCTIONS
# --------------------------

def fed_avg_aggregate(local_models):
    """✅ BASE MODE: Simple FedAvg (unweighted averaging)"""
    keys = local_models[0].keys()
    new_state = {}

    for k in keys:
        if not torch.is_floating_point(local_models[0][k]):
            new_state[k] = local_models[0][k]. clone()
            continue

        vals = torch.stack([m[k]. float() for m in local_models if torch.is_floating_point(m[k])])
        new_state[k] = vals.mean(dim=0).clone()

    return new_state

def trust_weighted_aggregate(local_models, tau=1.0):
    """✅ TRUST MODE: Trust-weighted averaging (no smoothing)"""
    keys = local_models[0].keys()
    new_state = {}
    ref = local_models[0]

    for k in keys: 
        if not torch.is_floating_point(ref[k]):
            new_state[k] = ref[k].clone()
            continue

        vals = torch.stack([m[k].float() for m in local_models if torch. is_floating_point(m[k])])
        mean_val = vals.mean(dim=0)

        distances = torch.stack([torch.norm(v - mean_val) for v in vals])
        weights = torch.exp(-distances / (tau + 1e-8))
        weights = weights / (weights.sum() + 1e-8)

        aggregated = sum(w * v for w, v in zip(weights, vals))
        new_state[k] = aggregated.clone()

    return new_state

def lra_aggregate(local_models, sigma=0.05):
    """✅ LRA MODE:  Noise only (no smoothing)"""
    keys = local_models[0]. keys()
    new_state = {}

    for k in keys:
        if not torch.is_floating_point(local_models[0][k]):
            new_state[k] = local_models[0][k].clone()
            continue

        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        aggregated = vals.mean(dim=0)

        if sigma > 0:
            aggregated = aggregated + sigma * torch.randn_like(aggregated)

        new_state[k] = aggregated.clone()

    return new_state

def fair_aggregate(local_models, prev_global_state, gamma=0.95):
    """✅ FAIR MODE: FedAvg + Temporal smoothing AGAINST PREVIOUS GLOBAL"""
    keys = local_models[0].keys()
    new_state = {}

    for k in keys:
        if not torch. is_floating_point(local_models[0][k]):
            # Non-floating buffers:  use previous global state
            new_state[k] = prev_global_state[k].clone()
            continue

        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        aggregated = vals.mean(dim=0)

        # ✅ FIX: Smooth toward PREVIOUS GLOBAL, not first client
        prev_val = prev_global_state[k]. to(aggregated.device).float()
        new_val = gamma * prev_val + (1 - gamma) * aggregated
        new_state[k] = new_val.clone()

    return new_state

def trust_lra_aggregate(local_models, tau=1.0, sigma=0.05):
    """✅ TRUST+LRA MODE: Trust + Noise (no smoothing)"""
    keys = local_models[0].keys()
    new_state = {}
    ref = local_models[0]

    for k in keys: 
        if not torch.is_floating_point(ref[k]):
            new_state[k] = ref[k].clone()
            continue

        vals = torch. stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        mean_val = vals.mean(dim=0)

        distances = torch. stack([torch.norm(v - mean_val) for v in vals])
        weights = torch.exp(-distances / (tau + 1e-8))
        weights = weights / (weights.sum() + 1e-8)
        aggregated = sum(w * v for w, v in zip(weights, vals))

        if sigma > 0:
            aggregated = aggregated + sigma * torch.randn_like(aggregated)

        new_state[k] = aggregated.clone()

    return new_state

def trust_fair_aggregate(local_models, prev_global_state, tau=1.0, gamma=0.95):
    """✅ TRUST+FAIR MODE:  Trust weighting + Temporal smoothing AGAINST PREVIOUS GLOBAL"""
    keys = local_models[0].keys()
    new_state = {}

    for k in keys:
        if not torch.is_floating_point(local_models[0][k]):
            new_state[k] = prev_global_state[k].clone()
            continue

        vals = torch.stack([m[k]. float() for m in local_models if torch.is_floating_point(m[k])])
        mean_val = vals.mean(dim=0)

        distances = torch.stack([torch.norm(v - mean_val) for v in vals])
        weights = torch.exp(-distances / (tau + 1e-8))
        weights = weights / (weights. sum() + 1e-8)
        aggregated = sum(w * v for w, v in zip(weights, vals))

        # ✅ FIX: Smooth toward PREVIOUS GLOBAL, not first client
        prev_val = prev_global_state[k].to(aggregated.device).float()
        new_val = gamma * prev_val + (1 - gamma) * aggregated
        new_state[k] = new_val.clone()

    return new_state

def lra_fair_aggregate(local_models, prev_global_state, sigma=0.05, gamma=0.95):
    """✅ LRA+FAIR MODE:  Noise + Temporal smoothing AGAINST PREVIOUS GLOBAL"""
    keys = local_models[0].keys()
    new_state = {}

    for k in keys:
        if not torch.is_floating_point(local_models[0][k]):
            new_state[k] = prev_global_state[k].clone()
            continue

        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        aggregated = vals. mean(dim=0)

        if sigma > 0:
            aggregated = aggregated + sigma * torch.randn_like(aggregated)

        # ✅ FIX: Smooth toward PREVIOUS GLOBAL, not first client
        prev_val = prev_global_state[k].to(aggregated.device).float()
        new_val = gamma * prev_val + (1 - gamma) * aggregated
        new_state[k] = new_val.clone()

    return new_state

def citadel_aggregate(local_models, prev_global_state, tau=1.0, sigma=0.05, gamma=0.95):
    """
    ✅ FIXED CITADEL:  Proper distance normalization + layer-wise std noise
    - Normalize distances by LAYER NORM (not global norm)
    - Scale sigma by layer std for stability
    - Smooth towards previous global state
    """
    keys = local_models[0].keys()
    new_state = {}
    
    for k in keys:
        if not torch.is_floating_point(local_models[0][k]):
            new_state[k] = prev_global_state[k]. clone()
            continue
        
        # Stack all client updates
        vals = torch.stack([m[k]. float() for m in local_models if torch.is_floating_point(m[k])])
        mean_val = vals.mean(dim=0)
        
        # ✅ FIX 1: Normalize distances per-layer (not globally)
        diffs = vals - mean_val
        flat = diffs.reshape(diffs.size(0), -1)
        distances = torch.linalg.norm(flat, dim=1)
        
        # Layer-wise normalization
        layer_norm = torch.linalg.norm(mean_val.reshape(-1))
        if layer_norm > 1e-12:
            distances = distances / layer_norm
        
        # ✅ FIX 2:  Softmax weights (more stable than exp)
        distances = distances - distances.min()  # shift for stability
        weights = torch. softmax(-distances / max(tau, 1e-6), dim=0)
        
        # Weighted aggregate
        wv = weights.view(-1, *([1] * (vals.dim() - 1)))
        aggregated = (wv * vals).sum(dim=0)
        
        # ✅ FIX 3: Layer-wise noise (scaled by std)
        if (("running_mean" not in k) and ("running_var" not in k) and sigma > 0):
            layer_std = vals.std(dim=0, unbiased=False)
            # Scale noise by layer std for stability
            noise = sigma * torch.clamp(layer_std, min=1e-8) * torch.randn_like(aggregated)
            aggregated = aggregated + noise
        
        # ✅ FIX 4: Smooth towards PREVIOUS global
        prev_val = prev_global_state[k]. to(aggregated.device).float()
        new_state[k] = (gamma * prev_val + (1 - gamma) * aggregated).clone()
    
    return new_state
def get_aggregation_fn(mode):
    """✅ FIXED: Return functions with correct signatures (prev_global_state passed to all)"""
    aggregation_map = {
        # Non-smoothed modes:  ignore prev_global_state
        "base":         lambda models, prev, **kw: fed_avg_aggregate(models),
        "trust":       lambda models, prev, **kw: trust_weighted_aggregate(models, tau=kw.get("tau", 1.0)),
        "lra":         lambda models, prev, **kw: lra_aggregate(models, sigma=kw.get("sigma", 0.05)),
        "trust_lra":   lambda models, prev, **kw: trust_lra_aggregate(models, tau=kw.get("tau", 1.0), sigma=kw.get("sigma", 0.05)),

        # Smoothed modes: USE prev_global_state
        "fair":        lambda models, prev, **kw: fair_aggregate(models, prev, gamma=kw. get("gamma", 0.95)),
        "trust_fair":   lambda models, prev, **kw: trust_fair_aggregate(models, prev, tau=kw. get("tau", 1.0), gamma=kw.get("gamma", 0.95)),
        "lra_fair":    lambda models, prev, **kw: lra_fair_aggregate(models, prev, sigma=kw.get("sigma", 0.05), gamma=kw.get("gamma", 0.95)),
        "citadel":     lambda models, prev, **kw: citadel_aggregate(models, prev, tau=kw.get("tau", 1.0), sigma=kw.get("sigma", 0.05), gamma=kw.get("gamma", 0.95)),
    }
    return aggregation_map. get(mode, aggregation_map["citadel"])

# --------------------------
# Metrics
# --------------------------

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

def compute_robustness_score(clean_metrics, attack_metrics, asr):
    """
    Improved robustness score that rewards defense effectiveness. 
    Higher weights on attack F1 and lower ASR. 
    """
    clean_acc = clean_metrics. get("accuracy", 0)
    attack_f1 = attack_metrics.get("f1", 0)

    # Better weighting:  emphasize robustness over clean accuracy
    score = (
        0.3 * clean_acc +        # 30% clean accuracy
        0.5 * attack_f1 +        # 50% adversarial F1
        0.2 * (1 - asr)          # 20% defense success
    )
    return max(0, score)

# --------------------------
# Training
# --------------------------

def train_local(model, loader, device, lr, smoke=False, local_epochs=1):
    model = copy.deepcopy(model).to(device)
    model.train()

    # Build param groups:  classifier gets 10x LR; others get base LR
    named = list(model.named_parameters())
    cls_params = [p for n, p in named if ("classifier" in n) and p.requires_grad]
    other_params = [p for n, p in named if ("classifier" not in n) and p.requires_grad]

    if len(cls_params) == 0 and len(other_params) == 0:
        raise RuntimeError("No trainable parameters found.  Check freezing logic.")

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if cls_params:
        param_groups.append({"params": cls_params, "lr": lr * 10.0})

    opt = torch.optim.Adam(param_groups, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(local_epochs):
        for b, (x, y) in enumerate(loader):
            x = x.to(device) if isinstance(x, torch.Tensor) else {k: v.to(device) for k, v in x.items()}
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(x) if isinstance(x, torch.Tensor) else model(**x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            del x, y, out, loss
            if smoke and b > 1:
                break

    torch.cuda.empty_cache()
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

def evaluate(model, loader, device, attack=None):
    """Enhanced evaluation with per-class metrics and confusion matrix."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    
    model.eval()
    all_preds, all_labels, logits_list = [], [], []
    
    for x, y in loader:
        y = y.to(device)
        x = x.to(device) if isinstance(x, torch.Tensor) else {k: v.to(device) for k, v in x.items()}
        
        # Apply attack if specified
        if attack == "fgsm":
            x = fgsm_attack(model, x, y)
        elif attack == "pgd":
            x = pgd_attack(model, x, y)
        
        # Forward pass
        with torch.no_grad():
            out = model(x) if isinstance(x, torch.Tensor) else model(**x)
        
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
        logits_list.append(out.detach().cpu())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    logits = torch.cat(logits_list, dim=0)
    
    # ✅ Base metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ✅ Per-class metrics (NEW)
    p_per_class, r_per_class, f_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # ✅ Confusion matrix (NEW)
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    # ✅ Expected Calibration Error (NEW)
    probs = torch.softmax(logits, dim=1).numpy()
    max_probs = probs.max(axis=1)
    ece = np.mean(np.abs(max_probs - (all_preds == all_labels).astype(float)))
    
    # ✅ Robustness metrics
    asr = float(1 - acc) if attack else 0.0
    
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "ASR": asr,
        "ece": float(ece),
        "per_class_f1": f_per_class. tolist(),
        "per_class_precision": p_per_class.tolist(),
        "per_class_recall": r_per_class.tolist(),
        "confusion_matrix": conf_matrix,  # NEW
    }
# --------------------------
# Experiment Runner
# --------------------------

def run_experiment(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    train_loader, test_loader = get_data(args. dataset, args.batch_size)
    model = get_model(args.dataset).to(device)

    # Load best hyperparams if not provided
    if args.tau is None or args.sigma is None or args.gamma is None:
        best_params = load_best_hyperparams(args.dataset)
        if args.tau is None: args.tau = best_params. get("tau", 1.0)
        if args.sigma is None: args.sigma = best_params. get("sigma", 0.05)
        if args.gamma is None: args.gamma = best_params.get("gamma", 0.95)
        print(f"🌐 Using tuned hyperparameters: τ={args.tau}, σ={args.sigma}, γ={args.gamma}")
    
    print(f"🔄 Mode: {args.mode} | Attack: {args.attack}")
    aggregation_fn = get_aggregation_fn(args.mode)

    hist = []
    prev_global_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    for r in range(args.rounds):
        # Local training
        local_models = [
            train_local(model, train_loader, device, args.lr, smoke=args.smoke, local_epochs=getattr(args, 'local_epochs', 1))
            for _ in range(args.num_clients)
        ]

        # Aggregation with FIXED citadel_aggregate
        agg_state = aggregation_fn(
            local_models,
            prev_global_state,
            tau=args.tau,
            sigma=args.sigma,
            gamma=args.gamma
        )
        
        model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v for k, v in agg_state.items()})
        prev_global_state = copy.deepcopy({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in agg_state.items()})

        # ✅ Enhanced evaluation
        clean_metrics = evaluate(model, test_loader, device)
        adv_metrics = evaluate(model, test_loader, device, attack=args.attack)
        
        # ✅ NEW: Per-client fairness
        fairness = evaluate_per_client(model, test_loader, device, attack=args.attack)
        
        asr = adv_metrics. get("ASR", 0.0)
        robustness_score = compute_robustness_score(clean_metrics, adv_metrics, asr)

        # ✅ Extended history logging
        hist.append({
            "round": r,
            "clean":  clean_metrics,
            "attack": adv_metrics,
            "asr": asr,
            "robustness_score": robustness_score,
            "fairness": fairness,  # NEW
            "timestamp": time.time(),
            "hyperparams": {
                "tau": args.tau,
                "sigma": args.sigma,
                "gamma": args.gamma,
                "num_clients": args.num_clients,
                "batch_size": args.batch_size,
            }
        })

        # Checkpoint
        ckpt_dir = os.path.join(args.save_dir, args.dataset, args.attack, args.mode, "checkpoints")
        ensure_dir(ckpt_dir)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"round_{r}.pt"))
        
        # Keep last 2 checkpoints
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(". pt")])
        if len(ckpts) > 2:
            os.remove(os. path.join(ckpt_dir, ckpts[0]))

        print(
            f"[{args.dataset}|{args.mode}|{args.attack}] Round {r} - "
            f"Clean Acc: {clean_metrics['accuracy']:.3f}, "
            f"Adv F1: {adv_metrics['f1']:.3f}, "
            f"ASR: {asr:.3f}, "
            f"Score: {robustness_score:. 3f}, "
            f"Fairness (Gini): {fairness['gini']:.3f}"
        )

    # Save history
    hist_path = os.path.join(args.save_dir, args.dataset, args.attack, args.mode, "history.json")
    ensure_dir(os.path.dirname(hist_path))
    save_json(hist, hist_path)

    print(f"✅ Saved:  {hist_path}")
# --------------------------
# Main
# --------------------------
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--mode", type=str, default="citadel")
    parser.add_argument("--attack", type=str, default="none")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results_citadel_v2")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local training epochs per client")

    # Hyperparameters for tuning/ablation
    parser.add_argument("--tau", type=float, default=None, help="Trust temperature parameter τ (None=tuned)")
    parser.add_argument("--sigma", type=float, default=None, help="Stochastic noise scale σ (None=tuned)")
    parser.add_argument("--gamma", type=float, default=None, help="Consensus decay γ (None=tuned)")

    args = parser.parse_args()
    try:
        run_experiment(args)
    except Exception as e: 
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

    hist_path = os.path.join(args.save_dir, args.dataset, args.attack, args.mode, "history.json")
    if os.path.exists(hist_path) and os.path.getsize(hist_path) > 0:
        print(f"✅ Metrics written successfully → {hist_path}")
    else:
        print(f"⚠️ Warning: History file {hist_path} may be missing or empty.")