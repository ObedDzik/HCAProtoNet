import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import os

def validate(model, val_loader, rare_classes, device):
    """Validation loop with comprehensive metrics"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, _, _, _ = model(x, target=None)
            preds = logits.argmax(dim=-1)
            
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Overall metrics
    accuracy = (all_preds == all_targets).float().mean().item()
    f1_macro = f1_score(all_targets.numpy(), all_preds.numpy(), average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets.numpy(), all_preds.numpy(), average='weighted', zero_division=0)
    
    # Rare class metrics
    rare_mask = torch.tensor([t.item() in rare_classes for t in all_targets])
    if rare_mask.sum() > 0:
        rare_preds = all_preds[rare_mask]
        rare_targets = all_targets[rare_mask]
        rare_accuracy = (rare_preds == rare_targets).float().mean().item()
        rare_f1 = f1_score(rare_targets.numpy(), rare_preds.numpy(), average='macro', zero_division=0)
        rare_precision = precision_score(rare_targets.numpy(), rare_preds.numpy(), 
                                        average='macro', zero_division=0)
        rare_recall = recall_score(rare_targets.numpy(), rare_preds.numpy(), 
                                  average='macro', zero_division=0)
    else:
        rare_accuracy = rare_f1 = rare_precision = rare_recall = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'rare_accuracy': rare_accuracy,
        'rare_f1': rare_f1,
        'rare_precision': rare_precision,
        'rare_recall': rare_recall,
    }
    
    model.train()
    return metrics


def visualize_prototypes(model, dataset, rare_classes, save_path, device, samples_per_proto=5):
    """Visualize prototypes by finding nearest training patches"""
    model.eval()
    
    # Get all features
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_features = []
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            features = model.backbone(x)
            all_features.append(features.cpu())
            all_images.append(x.cpu())
            all_labels.append(y)
    
    all_features = torch.cat(all_features)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    
    # Visualize shared prototypes
    n_shared = min(model.K_shared, 10)  # Limit to 10 for visualization
    fig, axes = plt.subplots(n_shared, samples_per_proto + 1, 
                             figsize=(3*(samples_per_proto+1), 3*n_shared))
    
    with torch.no_grad():
        for i in range(n_shared):
            proto = model.shared_prototypes[i].cpu()
            
            # Find nearest patches
            similarities = F.cosine_similarity(all_features, proto.unsqueeze(0), dim=1)
            top_indices = similarities.topk(samples_per_proto).indices
            
            # Plot prototype label
            ax = axes[i, 0] if n_shared > 1 else axes[0]
            ax.text(0.5, 0.5, f'Shared\nProto {i}', 
                   ha='center', va='center', fontsize=12, weight='bold')
            ax.axis('off')
            ax.set_title(f'Shared {i}')
            
            # Plot nearest patches
            for j, idx in enumerate(top_indices):
                ax = axes[i, j+1] if n_shared > 1 else axes[j+1]
                img = all_images[idx].permute(1, 2, 0).numpy()
                if img.shape[2] == 1:
                    img = img.squeeze()
                    ax.imshow(img, cmap='gray')
                else:
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'C{all_labels[idx].item()}\n{similarities[idx]:.2f}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(save_path).replace('.png', '_shared.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Visualize rare prototypes
    if len(rare_classes) > 0:
        total_rare = len(rare_classes) * model.K_rare_c
        fig, axes = plt.subplots(total_rare, samples_per_proto + 1,
                                figsize=(3*(samples_per_proto+1), 2*total_rare))
        
        with torch.no_grad():
            row = 0
            for c in rare_classes:
                rare_proto_dict = model.rare_prototypes[str(c)].cpu()
                
                for i in range(model.K_rare_c):
                    proto = rare_proto_dict[i]
                    
                    # Find nearest patches from this class
                    class_mask = (all_labels == c)
                    if class_mask.sum() > 0:
                        class_features = all_features[class_mask]
                        class_images = all_images[class_mask]
                        
                        similarities = F.cosine_similarity(class_features, proto.unsqueeze(0), dim=1)
                        k = min(samples_per_proto, len(similarities))
                        top_indices = similarities.topk(k).indices
                        
                        # Plot prototype
                        ax = axes[row, 0] if total_rare > 1 else axes[0]
                        ax.text(0.5, 0.5, f'Rare\nC{c}-P{i}',
                               ha='center', va='center', fontsize=10, weight='bold')
                        ax.axis('off')
                        ax.set_title(f'C{c} P{i}', fontsize=8)
                        
                        # Plot nearest patches
                        for j, idx in enumerate(top_indices):
                            ax = axes[row, j+1] if total_rare > 1 else axes[j+1]
                            img = class_images[idx].permute(1, 2, 0).numpy()
                            if img.shape[2] == 1:
                                img = img.squeeze()
                                ax.imshow(img, cmap='gray')
                            else:
                                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                                ax.imshow(img)
                            ax.axis('off')
                            ax.set_title(f'{similarities[idx]:.2f}', fontsize=8)
                        
                        # Fill remaining
                        for j in range(k, samples_per_proto):
                            ax = axes[row, j+1] if total_rare > 1 else axes[j+1]
                            ax.axis('off')
                    
                    row += 1
        
        plt.tight_layout()
        plt.savefig(str(save_path).replace('.png', '_rare.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()


def hca_inference(model, x, device='cuda'):
    """Simplified inference returning predictions and probabilities"""
    model.eval()
    x = x.to(device)
    
    with torch.no_grad():
        logits, _, _, _ = model(x, target=None)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
    
    return preds, probs, logits


def interpret_sample(model, x_sample, target_sample, rare_classes, device='cuda', 
                    topk_shared=3, topk_rare=2):
    """Interpret a single sample with detailed breakdown"""
    model.eval()
    x_sample = x_sample.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, features, shared_sim, logits_rare = model(
            x_sample, 
            target_sample.unsqueeze(0).to(device) if target_sample is not None else None
        )
        predicted_class = logits.argmax(dim=-1).item()
        probs = F.softmax(logits, dim=-1)

        print(f"\n{'='*50}")
        print(f"Prediction: Class {predicted_class} (confidence: {probs[0, predicted_class]:.3f})")
        if target_sample is not None:
            print(f"Ground Truth: Class {target_sample.item()}")
            print(f"Correct: {predicted_class == target_sample.item()}")
        print(f"{'='*50}\n")
        
        # Top-k shared
        top_shared_vals, top_shared_idx = shared_sim[0].topk(topk_shared)
        print(f"Top {topk_shared} Shared Prototypes:")
        for i, (idx, val) in enumerate(zip(top_shared_idx, top_shared_vals)):
            print(f"  {i+1}. Prototype {idx.item()}: similarity={val.item():.3f}")
        print()

        # Rare prototypes
        if predicted_class in rare_classes:
            rare_proto = model.rare_prototypes[str(predicted_class)]
            rare_sim_pred = F.cosine_similarity(
                features.unsqueeze(1), 
                rare_proto.unsqueeze(0), 
                dim=2
            )[0]
            top_rare_vals, top_rare_idx = rare_sim_pred.topk(min(topk_rare, len(rare_sim_pred)))
            
            print(f"Top {topk_rare} Rare Prototypes (Class {predicted_class}):")
            for i, (idx, val) in enumerate(zip(top_rare_idx, top_rare_vals)):
                print(f"  {i+1}. Prototype {idx.item()}: similarity={val.item():.3f}")
            print()

        # Contribution breakdown
        shared_contrib = (logits[0, predicted_class] - logits_rare[0, predicted_class]).item()
        rare_contrib = logits_rare[0, predicted_class].item()
        total = shared_contrib + rare_contrib
        
        print("Contribution Breakdown:")
        if total != 0:
            print(f"  Shared: {shared_contrib:.3f} ({shared_contrib/total*100:.1f}%)")
            print(f"  Rare:   {rare_contrib:.3f} ({rare_contrib/total*100:.1f}%)")
        else:
            print(f"  Shared: {shared_contrib:.3f}")
            print(f"  Rare:   {rare_contrib:.3f}")
        
        # All probabilities
        print("\nAll Class Probabilities:")
        for c in range(model.num_classes):
            marker = "★" if c == predicted_class else " "
            rare_marker = "[RARE]" if c in rare_classes else ""
            print(f"  {marker} Class {c}: {probs[0, c]:.3f} {rare_marker}")
        
        print(f"{'='*50}\n")
        
        return {
            'prediction': predicted_class,
            'probabilities': probs[0].cpu().numpy(),
            'top_shared_prototypes': top_shared_idx.cpu().numpy(),
            'shared_similarities': top_shared_vals.cpu().numpy(),
            'shared_contribution': shared_contrib,
            'rare_contribution': rare_contrib
        }


def save_prototype_activations(model, dataset, rare_classes, save_dir, device='cuda'):
    """Save individual prototype images with activation overlays"""
    save_dir = Path(save_dir)
    (save_dir / 'prototype_originals').mkdir(parents=True, exist_ok=True)
    (save_dir / 'prototype_overlays').mkdir(parents=True, exist_ok=True)
    
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Collect features
    all_features = []
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            features = model.backbone(x)
            all_features.append(features.cpu())
            all_images.append(x.cpu())
            all_labels.append(y)
    
    all_features = torch.cat(all_features)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    
    # Save shared prototypes
    with torch.no_grad():
        for i in range(model.K_shared):
            proto = model.shared_prototypes[i].cpu()
            similarities = F.cosine_similarity(all_features, proto.unsqueeze(0), dim=1)
            best_idx = similarities.argmax().item()
            best_img = all_images[best_idx]
            best_label = all_labels[best_idx].item()
            best_sim = similarities[best_idx].item()
            
            # Save original
            img_np = best_img.permute(1, 2, 0).numpy()
            img_np = np.clip((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8), 0, 1)
            
            plt.figure(figsize=(4, 4))
            if img_np.shape[2] == 1:
                plt.imshow(img_np.squeeze(), cmap='gray')
            else:
                plt.imshow(img_np)
            plt.axis('off')
            plt.title(f'Shared Proto {i}\nClass {best_label}, Sim: {best_sim:.3f}')
            plt.savefig(save_dir / 'prototype_originals' / f'shared_proto_{i}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save overlay
            plt.figure(figsize=(4, 4))
            if img_np.shape[2] == 1:
                plt.imshow(img_np.squeeze(), cmap='gray')
            else:
                plt.imshow(img_np)
            plt.imshow(np.ones_like(img_np[:,:,0]) * best_sim, 
                      alpha=0.3, cmap='hot', vmin=0, vmax=1)
            plt.axis('off')
            plt.title(f'Shared Proto {i} (Overlay)\nClass {best_label}, Sim: {best_sim:.3f}')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.savefig(save_dir / 'prototype_overlays' / f'shared_proto_{i}_overlay.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    # Save rare prototypes
    with torch.no_grad():
        for c in rare_classes:
            rare_proto_dict = model.rare_prototypes[str(c)].cpu()
            class_mask = (all_labels == c)
            
            if class_mask.sum() > 0:
                class_features = all_features[class_mask]
                class_images = all_images[class_mask]
                
                for i in range(model.K_rare_c):
                    proto = rare_proto_dict[i]
                    similarities = F.cosine_similarity(class_features, proto.unsqueeze(0), dim=1)
                    best_idx = similarities.argmax().item()
                    best_img = class_images[best_idx]
                    best_sim = similarities[best_idx].item()
                    
                    # Save original
                    img_np = best_img.permute(1, 2, 0).numpy()
                    img_np = np.clip((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8), 0, 1)
                    
                    plt.figure(figsize=(4, 4))
                    if img_np.shape[2] == 1:
                        plt.imshow(img_np.squeeze(), cmap='gray')
                    else:
                        plt.imshow(img_np)
                    plt.axis('off')
                    plt.title(f'Rare Proto C{c} P{i}\nSim: {best_sim:.3f}')
                    plt.savefig(save_dir / 'prototype_originals' / f'rare_class_{c}_proto_{i}.png',
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Save overlay
                    plt.figure(figsize=(4, 4))
                    if img_np.shape[2] == 1:
                        plt.imshow(img_np.squeeze(), cmap='gray')
                    else:
                        plt.imshow(img_np)
                    plt.imshow(np.ones_like(img_np[:,:,0]) * best_sim,
                              alpha=0.3, cmap='hot', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.title(f'Rare Proto C{c} P{i} (Overlay)\nSim: {best_sim:.3f}')
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.savefig(save_dir / 'prototype_overlays' / f'rare_class_{c}_proto_{i}_overlay.png',
                               dpi=150, bbox_inches='tight')
                    plt.close()
    
    print(f"✓ Prototype activations saved to {save_dir}")
    model.train()