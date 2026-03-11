"""Run experiment 17: periodic k=16 + SWA from epoch 15, lr=1e-4, 35 epochs."""
import sys
sys.path.insert(0, '.')
import json
import time
import copy
import numpy as np
import torch
from pathlib import Path
from run_experiment import *
from tabula.data.datasets import build_dataloaders
from tabula.models.transformer import TabularTransformer
from tabula.training.engine import (
    _move_batch, _make_criterion, _run_epoch, _compute_loss, _autocast_dtype, _use_amp, 
    EpochResult
)
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed

def run_swa_experiment(exp_id, desc, config, swa_start, total_epochs):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"DESC: {desc}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    torch.cuda.empty_cache()
    
    set_seed(config.seed)
    device = torch.device(config.training.device)
    train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(config)
    effective_output_dim = 1
    
    model = TabularTransformer(config, num_numeric, num_categorical, num_text, effective_output_dim).to(device)
    criterion = _make_criterion(config.task.problem_type, effective_output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))
    
    state_dicts = []
    best_val_loss = float("inf")
    
    for epoch in range(1, total_epochs + 1):
        train_result = _run_epoch(model, train_loader, device, criterion, optimizer, config, effective_output_dim, scaler)
        val_result = _run_epoch(model, val_loader, device, criterion, None, config, effective_output_dim, None)
        print(f"epoch={epoch} train_loss={train_result.loss:.4f} val_loss={val_result.loss:.4f} val_metrics={val_result.metrics}")
        
        if epoch >= swa_start:
            state_dicts.append(copy.deepcopy(model.state_dict()))
        
        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
    
    print(f"\nAveraging {len(state_dicts)} state dicts from epochs {swa_start}-{total_epochs}...")
    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
    model.load_state_dict(avg_state_dict)
    
    model.eval()
    all_y, all_logits = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            all_y.append(batch.y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    
    y_true = np.concatenate(all_y)
    logits_arr = np.concatenate(all_logits)
    swa_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits_arr.reshape(-1)), torch.tensor(y_true.astype(np.float32))
    ).item())
    metrics = compute_metrics(config.task.problem_type, y_true, logits_arr)
    accuracy = metrics.get('accuracy')
    roc_auc = metrics.get('roc_auc')
    duration = time.time() - start_time
    
    print(f"\nSWA results: val_loss={swa_loss:.6f} accuracy={accuracy:.4f} roc_auc={roc_auc:.4f}")
    
    output_dir = Path(config.artifacts_root) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config,
                 "num_numeric": num_numeric, "num_categorical": num_categorical,
                 "num_text": num_text, "output_dim": effective_output_dim}, output_dir / "best.pt")
    
    log_experiment(exp_id, desc, 'success', swa_loss, accuracy, roc_auc, total_epochs, duration)
    prev_best = get_best_val_loss()
    if swa_loss < prev_best:
        update_best(exp_id, swa_loss, roc_auc)
        print(f"\n*** NEW BEST: val_loss={swa_loss:.6f} (prev={prev_best:.6f}) ***\n")
    else:
        print(f"\nNo improvement: val_loss={swa_loss:.6f} vs best={prev_best:.6f}\n")
    return {"val_loss": swa_loss, "accuracy": accuracy, "roc_auc": roc_auc, "duration": duration}

# Exp 17: SWA from epoch 15, 40 epochs, lr=1e-4
config17 = make_adult_config(
    'exp_017_swa_longer',
    numeric_embedding='periodic',
    numeric_periodic_features=16,
    lr=1e-4,
    max_epochs=40,
    early_stopping_patience=40,
)
r17 = run_swa_experiment('exp_017_swa_longer', 'Periodic k=16 SWA avg epochs 15-40 lr=1e-4', config17, 15, 40)
print(json.dumps(r17, indent=2, default=str))
