import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

def train_hca(model, optimizer, train_loader, device: str = "cuda", phase: str = "warmup"):
    """
    One training epoch for HCA-ProtoNet.
    
    Phase behavior:
    - warmup   : update only shared prototypes via momentum
    - joint    : update both shared + rare prototypes via momentum  
    - finetune : no prototype updates (only optimize W_shared_to_class)
    """
    
    if phase not in ["warmup", "joint", "finetune"]:
        raise ValueError(f"Invalid phase: {phase}. Must be 'warmup', 'joint', or 'finetune'")
    
    model.train()
    
    epoch_loss = 0.0
    epoch_metrics = {
        'ce': 0.0,
        'diversity': 0.0,
        'separation': 0.0,
        'coverage': 0.0,
        'entropy': 0.0,
    }
    
    num_batches = 0
    
    for batch_idx, data in enumerate(train_loader):
        # Handle both tensor and non-tensor labels
        x = data['bmode'].to(device)
        y = data['primus_label']
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        y = y.to(device)
        
        # Forward pass
        logits, features, _, _ = model(x, y)
        loss, loss_dict = model.compute_loss(logits, features, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Prototype updates based on phase
        if phase in ["warmup", "joint"]:
            with torch.no_grad():
                model.update_prototypes(features, y)
        
        # elif phase == "finetune": no prototype updates
        
        # Accumulate metrics
        epoch_loss += loss.item()
        for k in epoch_metrics:
            epoch_metrics[k] += float(loss_dict[k])
        
        num_batches += 1
        
        # Progress logging
        if batch_idx % 10 == 0:
            print(
                f"  [{phase.upper()}] Batch {batch_idx}/{len(train_loader)} "
                f"Loss={loss.item():.4f}"
            )
    
    # Average metrics over epoch
    epoch_loss /= max(1, num_batches)
    epoch_metrics = {
        k: v / max(1, num_batches) for k, v in epoch_metrics.items()
    }
    
    return epoch_loss, epoch_metrics


