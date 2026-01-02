import torch
import wandb
from omegaconf import OmegaConf
import argparse
import os
import numpy as np
import random
from pathlib import Path
from collections import Counter
from pathlib import Path

from network import HCAProtoNet
from train import train_hca
from inference import validate, visualize_prototypes
from as_dataset import get_datasets
from as_dataloader import make_loader
from backbone import create_backbone
from log import create_logger


def main(cfg):

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model_dir = Path(cfg.train.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    img_dir = model_dir / "img"
    img_dir.mkdir(exist_ok=True)

    proto_dir = model_dir / "prototypes"
    proto_dir.mkdir(exist_ok=True)

    log, logclose = create_logger(log_filename=str(model_dir / "train.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    backbone = create_backbone(cfg.architecture)
    model = HCAProtoNet(
        backbone=backbone,
        num_classes=cfg.train.num_classes,
        K_shared=cfg.train.num_shared_prototypes,
        rare_classes=cfg.train.rare_classes,
        K_rare_c=cfg.train.prototypes_per_class,
        m_shared=cfg.train.momentum_shared_prototypes,
        m_rare=cfg.train.momentum_rare_class,
        lambda_div=cfg.train.coefs.diversity,
        lambda_sep=cfg.train.coefs.sep,
        lambda_cov=cfg.train.coefs.coverage,
        lambda_ent=cfg.train.coefs.entropy,
        threshold_freq=cfg.train.threshold_freq,
        min_active=cfg.train.min_active_prototypes,
        temperature=cfg.train.uncertainty_temp,
    ).to(device)

    log(f"Model initialized with {cfg.train.num_classes} classes")
    log(f"Rare classes: {cfg.train.rare_classes}")

    # Data
    train_dataset, val_dataset = get_datasets(cfg.data)
    train_loader = make_loader(train_dataset, cfg.data, shuffle=True, weighted=True, pin_memory=True, drop_last=True)
    val_loader = make_loader(val_dataset, cfg.data, shuffle=False, weighted=False)

    # Training Frequencies
    label_counts = Counter(
        label.item() if torch.is_tensor(label) else label
        for _, label in train_dataset
    )
    model.training_frequencies = {
        c: label_counts.get(c, 1) for c in range(model.num_classes)
    }
    model.N_max = max(model.training_frequencies.values())

    log(f"Training frequencies: {model.training_frequencies}")

    # Optimizer
    rare_params = [model.rare_prototypes[str(c)] for c in cfg.train.rare_classes]
    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters()},
            {"params": [model.shared_prototypes, model.W_shared_to_class]},
            {"params": rare_params},
        ],
        lr=cfg.train.lr,
    )

    best_val_f1_weighted = 0.0
    best_epoch = 0

    warmup_epochs = cfg.train.get('warmup_epochs', 20)
    joint_epochs = cfg.train.get('joint_epochs', 80)

    for epoch in range(cfg.train.num_epochs):
        if epoch < warmup_epochs:
            phase = "warmup"
            for c in cfg.train.rare_classes:
                model.rare_prototypes[str(c)].requires_grad_(False)

        elif epoch < joint_epochs:
            phase = "joint"
            for c in cfg.train.rare_classes:
                model.rare_prototypes[str(c)].requires_grad_(True)

        else:
            phase = "finetune"
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            model.shared_prototypes.requires_grad_(False)
            for c in cfg.train.rare_classes:
                model.rare_prototypes[str(c)].requires_grad_(False)
            optimizer = torch.optim.Adam([model.W_shared_to_class],lr=cfg.train.finetune_lr)

        log(f"\nEpoch {epoch+1}/{cfg.train.num_epochs} — Phase: {phase}")
        wandb.log({"epoch": epoch, "phase": phase})

        # Train
        model.train()
        train_loss, train_metrics = train_hca(
            model, optimizer, train_loader, device, phase=phase
        )
        train_loss /= len(train_loader)
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        log(f"Train Loss: {train_loss:.4f}")
        wandb.log({"train/loss": train_loss, **{
            f"train/{k}": v for k, v in train_metrics.items()
        }})

        # Validate
        val_metrics = validate(model, val_loader, cfg.train.rare_classes, device)

        log(
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"F1-macro: {val_metrics['f1_macro']:.4f} | "
            f"F1-weighted: {val_metrics['f1_weighted']:.4f} | "
            f"Rare F1: {val_metrics['rare_f1']:.4f}"
        )
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

        if val_metrics["f1_weighted"] > best_val_f1_weighted:
            best_val_f1_weighted = val_metrics["f1_weighted"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                model_dir / "best_model.pth")
            log("Best model saved (F1-weighted improved)")

    best_ckpt_path = model_dir / "best_model.pth"
    assert best_ckpt_path.exists(), "Best model checkpoint not found!"

    checkpoint = torch.load(best_ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    log(f"Loaded best model from epoch {checkpoint['epoch'] + 1} "
        f"(F1-weighted={checkpoint['val_metrics']['f1_weighted']:.4f})")
    visualize_prototypes(
        model=model,
        dataset=train_dataset,
        rare_classes=cfg.train.rare_classes,
        save_path=proto_dir / "best_model.png",
        device=device,
    )
    log("Prototype visualization completed using best model")

    log("=" * 60)
    log(f"Training complete")
    log(f"Best epoch: {best_epoch}")
    log(f"Best weighted F1: {best_val_f1_weighted:.4f}")
    log("=" * 60)

    wandb.finish()
    logclose()

def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config','-c',help='Path to config file')
    parser.add_agument('-gpuid', nargs=1, type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    cfg = load_config(args.config)
    main(cfg)

