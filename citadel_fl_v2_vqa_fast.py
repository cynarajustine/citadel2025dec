#!/usr/bin/env python3
"""
CITADEL-FL v2 (VQA-fast focused)
- Fast VQA path using MobileNetV3-Small with ImageNet normalization
- Optional unfreeze of last feature block via env VQA_UNFREEZE=1
- Classifier head gets 10x learning rate (param groups)
- Supports local_epochs for meaningful per-client progress
- CIFAR-10 and Arrhythmia remain supported

Env knobs (optional):
- VQA_FAST=1            # enable fast VQA path (always on in this script)
- VQA_UNFREEZE=1        # unfreeze last MobileNet block
- VQA_TRAIN_SUBSET=8000 # subsample train set for speed
- VQA_TEST_SUBSET=2000  # subsample test set for speed
- NUM_WORKERS=8         # dataloader workers
"""
import os, copy, argparse, json
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from filelock import FileLock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

DATA_ROOT = "datasets"
os.makedirs(DATA_ROOT, exist_ok=True)

# ---------- Env knobs ----------
VQA_UNFREEZE = os.getenv("VQA_UNFREEZE", "0") == "1"
VQA_TRAIN_SUBSET = int(os.getenv("VQA_TRAIN_SUBSET", "8000"))
VQA_TEST_SUBSET  = int(os.getenv("VQA_TEST_SUBSET", "2000"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))

# ---------- Utils ----------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def save_json(obj, path): 
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ---------- Models ----------
class FastVQAModel(nn.Module):
    """
    MobileNetV3-Small head for 10-class CIFAR surrogate VQA.
    - ImageNet weights if available
    - Freeze backbone.features by default
    - Optionally unfreeze last feature block with VQA_UNFREEZE=1
    - Classifier is always trainable
    """
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        try:
            backbone = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        except Exception:
            backbone = torchvision.models.mobilenet_v3_small(weights=None)

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        # Freeze all features by default
        for p in backbone.features.parameters():
            p.requires_grad = not freeze_backbone

        # Optionally unfreeze the last block for capacity
        if VQA_UNFREEZE:
            try:
                for p in backbone.features[-1].parameters():
                    p.requires_grad = True
                print("🔓 VQA_UNFREEZE=1: Unfroze last feature block.")
            except Exception:
                pass

        # Classifier trainable
        for p in backbone.classifier.parameters():
            p.requires_grad = True

        self.model = backbone

        # Report param counts
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"🧩 FastVQAModel params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")

    def forward(self, x):
        return self.model(x)

def get_model(dataset):
    if dataset == "cifar10":
        try:
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        except Exception:
            m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 10)
        return m
    elif dataset == "arrhythmia":
        return nn.Sequential(
            nn.Linear(279, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    elif dataset == "vqa":
        print("⚡ VQA_FAST: Using FastVQAModel (MobileNetV3-Small).")
        return FastVQAModel(num_classes=10, freeze_backbone=True)
    else:
        raise ValueError(dataset)

# ---------- Data ----------
def _subset(ds, k, seed=42):
    if k <= 0 or k >= len(ds): return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:k].tolist()
    return Subset(ds, idx)

def get_data(dataset, batch_size):
    if dataset == "cifar10":
        tfm = T.Compose([T.ToTensor()])
        with FileLock(os.path.join(DATA_ROOT, "c10.lock")):
            train = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
            test  = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)
        pin = torch.cuda.is_available()
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin, persistent_workers=True),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin, persistent_workers=True),
        )
    elif dataset == "arrhythmia":
        csv_path = os.path.join(DATA_ROOT, "arrhythmia.data")
        with FileLock(csv_path + ".lock"):
            if not os.path.exists(csv_path):
                os.system(f"wget -q -O {csv_path} https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data")
        try:
            df = pd.read_csv(csv_path, header=None, na_values="?").dropna()
            X = df.iloc[:, :-1].astype(float).values
            y = (df.iloc[:, -1].values != 1).astype(int)
        except Exception:
            X = np.random.randn(452, 279).astype(np.float32); y = np.random.randint(0,2,(452,))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        tr = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
        te = torch.utils.data.TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
        return DataLoader(tr, batch_size=batch_size, shuffle=True), DataLoader(te, batch_size=batch_size, shuffle=False)
    elif dataset == "vqa":
        # ImageNet normalization is critical for MobileNet features
        tfm = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with FileLock(os.path.join(DATA_ROOT, "c10.lock")):
            base_train = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
            base_test  = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)
        train_ds = _subset(base_train, VQA_TRAIN_SUBSET)
        test_ds  = _subset(base_test, VQA_TEST_SUBSET)
        pin = torch.cuda.is_available()
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=True),
        )
    else:
        raise ValueError(dataset)

# ---------- Attacks (eval only) ----------
def fgsm_attack(model, x, y, eps=0.03):
    """
    FGSM in evaluation. For the VQA-fast path inputs are normalized; we perturb in that space.
    Returns a detached adversarial tensor.
    """
    device = next(model.parameters()).device
    x_adv = x.detach().to(device).clone()
    x_adv.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out = model(x_adv)
    loss = F.cross_entropy(out, y)
    loss.backward()
    x_adv = (x_adv + eps * x_adv.grad.sign()).detach()
    return x_adv

# ---------- Aggregation ----------
def fed_avg_aggregate(local_models):
    keys = local_models[0].keys(); new = {}
    for k in keys:
        if not torch.is_floating_point(local_models[0][k]):
            new[k] = local_models[0][k].clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        new[k] = vals.mean(0).clone()
    return new

def trust_weighted_aggregate(local_models, tau=1.0):
    keys = local_models[0].keys(); new = {}
    ref = local_models[0]
    for k in keys:
        if not torch.is_floating_point(ref[k]):
            new[k] = ref[k].clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        mean = vals.mean(0)
        d = torch.stack([torch.norm(v - mean) for v in vals])
        w = torch.exp(-d / (tau + 1e-8)); w = w / (w.sum() + 1e-8)
        agg = sum(wi * vi for wi, vi in zip(w, vals))
        new[k] = agg.clone()
    return new

def lra_aggregate(local_models, sigma=0.05):
    keys = local_models[0].keys(); new = {}
    for k in keys:
        if not torch.is_floating_point(local_models[0][k]):
            new[k] = local_models[0][k].clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        agg = vals.mean(0)
        if sigma > 0: agg = agg + sigma * torch.randn_like(agg)
        new[k] = agg.clone()
    return new

def fair_aggregate(local_models, gamma=0.95):
    keys = local_models[0].keys(); new = {}; ref = local_models[0]
    for k in keys:
        if not torch.is_floating_point(ref[k]):
            new[k] = ref[k].clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        agg = vals.mean(0)
        prev = ref[k].float()
        new[k] = (gamma * prev + (1 - gamma) * agg).clone()
    return new

def trust_lra_aggregate(local_models, tau=1.0, sigma=0.05):
    keys = local_models[0].keys(); new = {}; ref = local_models[0]
    for k in keys:
        if not torch.is_floating_point(ref[k]):
            new[k] = ref[k].clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        mean = vals.mean(0)
        d = torch.stack([torch.norm(v - mean) for v in vals])
        w = torch.exp(-d / (tau + 1e-8)); w = w / (w.sum() + 1e-8)
        agg = sum(wi * vi for wi, vi in zip(w, vals))
        if sigma > 0: agg = agg + sigma * torch.randn_like(agg)
        new[k] = agg.clone()
    return new

def trust_fair_aggregate(local_models, tau=1.0, gamma=0.95):
    keys = local_models[0].keys(); new = {}; ref = local_models[0]
    for k in keys:
        if not torch.is_floating_point(ref[k]):
            new[k] = ref[k].clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        mean = vals.mean(0)
        d = torch.stack([torch.norm(v - mean) for v in vals])
        w = torch.exp(-d / (tau + 1e-8)); w = w / (w.sum() + 1e-8)
        agg = sum(wi * vi for wi, vi in zip(w, vals))
        prev = ref[k].float()
        new[k] = (gamma * prev + (1 - gamma) * agg).clone()
    return new

def lra_fair_aggregate(local_models, sigma=0.05, gamma=0.95):
    keys = local_models[0].keys(); new = {}; ref = local_models[0]
    for k in keys:
        if not torch.is_floating_point(ref[k]):
            new[k] = ref[k].clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        agg = vals.mean(0)
        if sigma > 0: agg = agg + sigma * torch.randn_like(agg)
        prev = ref[k].float()
        new[k] = (gamma * prev + (1 - gamma) * agg).clone()
    return new

def citadel_aggregate(local_models, tau=1.0, sigma=0.01, gamma=0.8):
    """
    Stabilized CITADEL for VQA-fast:
    - Softmax trust on normalized distances
    - Small noise scaled to layer std (if available)
    - Temporal smoothing
    """
    keys = local_models[0].keys(); new = {}; ref = local_models[0]
    for k in keys:
        ref_t = ref[k]
        if not torch.is_floating_point(ref_t):
            new[k] = ref_t.clone(); continue
        vals = torch.stack([m[k].float() for m in local_models if torch.is_floating_point(m[k])])
        mean = vals.mean(0)
        flat = (vals - mean).reshape(vals.size(0), -1)
        d = torch.linalg.norm(flat, dim=1)
        scale = mean.norm().clamp_min(1e-12)
        d = d / scale
        d = d - d.min()
        w = torch.softmax(-d / max(tau, 1e-6), dim=0)
        wv = w.view(-1, *([1] * (vals.dim() - 1)))
        agg = (wv * vals).sum(0)
        if ("running_mean" in k) or ("running_var" in k):
            agg = vals.mean(0)
        elif sigma > 0:
            try:
                layer_std = vals.std(dim=0, unbiased=False)
                agg = agg + sigma * layer_std * torch.randn_like(agg)
            except Exception:
                agg = agg + sigma * torch.randn_like(agg)
        prev = ref_t.float()
        new[k] = (gamma * prev + (1 - gamma) * agg).clone()
    return new

def get_aggregation_fn(mode):
    return {
        "base":       lambda ms, **kw: fed_avg_aggregate(ms),
        "trust":      lambda ms, **kw: trust_weighted_aggregate(ms, tau=kw.get("tau", 1.0)),
        "lra":        lambda ms, **kw: lra_aggregate(ms, sigma=kw.get("sigma", 0.05)),
        "fair":       lambda ms, **kw: fair_aggregate(ms, gamma=kw.get("gamma", 0.95)),
        "trust_lra":  lambda ms, **kw: trust_lra_aggregate(ms, tau=kw.get("tau", 1.0), sigma=kw.get("sigma", 0.05)),
        "trust_fair": lambda ms, **kw: trust_fair_aggregate(ms, tau=kw.get("tau", 1.0), gamma=kw.get("gamma", 0.95)),
        "lra_fair":   lambda ms, **kw: lra_fair_aggregate(ms, sigma=kw.get("sigma", 0.05), gamma=kw.get("gamma", 0.95)),
        "citadel":    lambda ms, **kw: citadel_aggregate(ms, tau=kw.get("tau", 3.0), sigma=kw.get("sigma", 0.01), gamma=kw.get("gamma", 0.8)),
    }[mode]

# ---------- Metrics ----------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

def compute_robustness_score(clean, attack, asr):
    return max(0.0, (0.5 * clean.get("accuracy", 0)) + (0.5 * attack.get("f1", 0)) - 0.25 * asr)

# ---------- Training / Eval ----------
def train_local(model, loader, device, lr, local_epochs=1, smoke=False):
    # Copy global -> client
    model = copy.deepcopy(model).to(device)
    model.train()

    # Param groups: classifier 10x LR, others base LR (if trainable)
    named = list(model.named_parameters())
    cls_params = [p for n, p in named if ("classifier" in n) and p.requires_grad]
    other_params = [p for n, p in named if ("classifier" not in n) and p.requires_grad]

    param_groups = []
    if other_params: param_groups.append({"params": other_params, "lr": lr})
    if cls_params:   param_groups.append({"params": cls_params, "lr": lr * 10.0})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for client.")

    opt = torch.optim.Adam(param_groups)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for b, (x, y) in enumerate(loader):
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x); loss = loss_fn(out, y)
            loss.backward(); opt.step()
            if smoke and b > 1: break

    # Return CPU state for aggregation
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}

def evaluate(model, loader, device, attack=None):
    """
    Evaluation with optional attack. IMPORTANT:
    - DO NOT disable autograd globally if we need to craft adversarial examples.
    - Use grads only during adversarial crafting, then no_grad() for prediction.
    """
    model.eval().to(device)
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)

        if attack == "fgsm":
            # Enable grads to generate adversarial examples
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                x_adv = fgsm_attack(model, x, y)
            # Inference on adversarial inputs without grads
            with torch.no_grad():
                out = model(x_adv)
        else:
            with torch.no_grad():
                out = model(x)

        ps.extend(out.argmax(1).cpu().numpy())
        ys.extend(y.cpu().numpy())

    m = compute_metrics(ys, ps)
    m["ASR"] = 1.0 - m["accuracy"] if attack else 0.0
    return m

# ---------- Runner ----------
def run(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    train_loader, test_loader = get_data(args.dataset, args.batch_size)
    model = get_model(args.dataset).to(device)

    print(f"🌐 Using provided hyperparameters: τ={args.tau}, σ={args.sigma}, γ={args.gamma}")
    print(f"🔄 Mode: {args.mode} | Attack: {args.attack}")

    agg_fn = get_aggregation_fn(args.mode)
    hist = []

    for r in range(args.rounds):
        locals_states = [
            train_local(model, train_loader, device, lr=args.lr, local_epochs=args.local_epochs, smoke=args.smoke)
            for _ in range(args.num_clients)
        ]
        new_state = agg_fn(locals_states, tau=args.tau, sigma=args.sigma, gamma=args.gamma)
        model.load_state_dict(new_state)

        clean = evaluate(model, test_loader, device)
        adv = evaluate(model, test_loader, device, attack=args.attack if args.attack != "none" else None)
        asr = adv.get("ASR", 0.0)
        score = compute_robustness_score(clean, adv, asr)

        hist.append({"round": r, "clean": clean, "attack": adv, "asr": asr, "robustness_score": score})
        print(f"[{args.dataset}|{args.mode}|{args.attack}] Round {r} - Acc: {clean['accuracy']:.3f}, F1: {clean['f1']:.3f}, Adv F1: {adv['f1']:.3f}, ASR: {asr:.3f}, Score: {score:.3f}")

    out_dir = os.path.join(args.save_dir, args.dataset, args.attack, args.mode)
    ensure_dir(out_dir)
    save_json(hist, os.path.join(out_dir, "history.json"))
    print(f"✅ Metrics written → {os.path.join(out_dir, 'history.json')}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="vqa", choices=["vqa", "cifar10", "arrhythmia"])
    p.add_argument("--mode", type=str, default="citadel",
                   choices=["base", "trust", "lra", "fair", "trust_lra", "trust_fair", "lra_fair", "citadel"])
    p.add_argument("--attack", type=str, default="fgsm", choices=["none", "fgsm"])
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--lr", type=float, default=0.01)           # higher LR helps head/backbone adapt
    p.add_argument("--local_epochs", type=int, default=1)      # you can set 2 for faster gains
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--save_dir", type=str, default="results_citadel_v2")
    args = p.parse_args()
    run(args)