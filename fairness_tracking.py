def evaluate_per_client(model, loader, device, attack=None):
    """Evaluate accuracy per client (for fairness analysis)."""
    model.eval()
    client_accs = []
    
    for x, y in loader: 
        y = y.to(device)
        x = x.to(device) if isinstance(x, torch.Tensor) else {k: v.to(device) for k, v in x.items()}
        
        if attack == "pgd":
            x = pgd_attack(model, x, y)
        elif attack == "fgsm": 
            x = fgsm_attack(model, x, y)
        
        with torch.no_grad():
            out = model(x) if isinstance(x, torch.Tensor) else model(**x)
        
        preds = out.argmax(dim=1)
        batch_acc = (preds == y).float().mean().item()
        client_accs.append(batch_acc)
    
    return {
        "mean_acc": np.mean(client_accs),
        "std_acc": np.std(client_accs),
        "min_acc": np.min(client_accs),
        "max_acc": np.max(client_accs),
        "gini": compute_gini(np.array(client_accs)),
    }

def compute_gini(x):
    """Gini coefficient for fairness (0=fair, 1=unfair)."""
    x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x)
    return float(2 * np.sum((np.arange(1, n+1)) * x) / (n * cumsum[-1]) - (n + 1) / n)