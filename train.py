import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

def train_hca(model, optimizer, train_loader, device: str = "cuda", phase: str="warmup"):
    """
    One training epoch for HCA-ProtoNet.
    Phase behavior:
    - warmup   : no rare prototype updates
    - joint    : momentum updates for shared + rare prototypes
    - finetune : NO prototype updates
    """

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
        x = data['bmode'].to(device)
        y = data['primus_label'].to(device)

        logits, features, _, _ = model(x, y)
        loss, loss_dict = model.compute_loss(logits, features, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if phase == "warmup":
            # Update ONLY shared prototypes
            with torch.no_grad():
                shared_sim = F.cosine_similarity(
                    features.unsqueeze(1), 
                    model.shared_prototypes.unsqueeze(0), 
                    dim=2
                )
                assign_shared_idx = shared_sim.argmax(dim=-1)
                
                for i in range(model.K_shared):
                    mask = (assign_shared_idx == i)
                    if mask.sum() > 0:
                        centroid = features[mask].mean(dim=0)
                        model.shared_prototypes[i].data = (
                            model.m_shared * model.shared_prototypes[i].data + 
                            (1 - model.m_shared) * centroid
                        )

        # Momentum prototype update
        elif phase == "joint":
            with torch.no_grad():
                model.update_prototypes(features, y)

        epoch_loss += loss.item()
        for k in epoch_metrics:
            epoch_metrics[k] += float(loss_dict[k])

        num_batches += 1

        if batch_idx % 10 == 0:
            print(
                f"  Batch {batch_idx}/{len(train_loader)} "
                f"Loss={loss.item():.4f}"
            )

    epoch_loss /= max(1, num_batches)
    epoch_metrics = {
        k: v / max(1, num_batches) for k, v in epoch_metrics.items()
    }

    return epoch_loss, epoch_metrics



