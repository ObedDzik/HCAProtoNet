import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
from typing import Literal
import numpy as np
import pandas as pd
from prostate_transform import ProstateTransform
import os
import sklearn.model_selection

def strip_padding_pil(img: Image.Image) -> Image.Image:
    """
    Removes rows and columns that are entirely zero (black) from a PIL Image.

    Supports grayscale ('L') and RGB ('RGB') images.
    Safe against fully-black images.
    """
    img_np = np.array(img)
    # Handle grayscale vs RGB
    if img_np.ndim == 2:
        # (H, W) → (H, W, 1)
        img_np = img_np[:, :, None]
    H, W, C = img_np.shape
    # Identify non-zero rows and columns
    row_mask = np.any(img_np != 0, axis=(1, 2))  # shape (H,)
    col_mask = np.any(img_np != 0, axis=(0, 2))  # shape (W,)
    # If everything is zero, return original image
    if not row_mask.any() or not col_mask.any():
        return img
    img_np = img_np[row_mask][:, col_mask]
    # Convert back to grayscale if needed
    if C == 1:
        img_np = img_np[:, :, 0]
    return Image.fromarray(img_np)


class ProstateDataset(Dataset):

    def __init__(
        self,
        root_dir,
        case_ids=None,
        transform=None,
        needle_mask_fname="needle_mask.png",
        out_fmt: Literal["pil", "np"] = "pil",
        strip_padding=False,
        tqdm_kw=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.out_fmt = out_fmt
        self.strip_padding = strip_padding
        self.tqdm_kw = tqdm_kw or {}

        self.case_ids = set(case_ids) if case_ids is not None else None
        self.data = []
        frame_paths = sorted(self.root_dir.rglob("frames/*.png"))

        for frame_path in tqdm(frame_paths, desc="Indexing frames", **self.tqdm_kw):
            frame_path = Path(frame_path)

            try:
                center, case, _, _ = frame_path.relative_to(self.root_dir).parts
            except ValueError:
                # Unexpected directory depth
                continue

            if self.case_ids is not None and center not in self.case_ids:
                continue

            case_dir = self.root_dir / center / case
            info_path = case_dir / "info.json"
            needle_mask_path = case_dir / needle_mask_fname

            if not info_path.exists():
                continue

            with open(info_path, "r") as f:
                info = json.load(f)

            self.data.append(
                {
                    "image_path": frame_path,
                    "needle_mask_path": needle_mask_path if needle_mask_path.exists() else None,
                    "info": info,
                    "center": center,
                    "case": case,
                }
            )

        self.metadata = [
            {
                "primus": d["info"]["PRI-MUS"],
                "diagnosis": d["info"]["Diagnosis"],
                "grade_group": d["info"]["GG"],
                "center": d["info"]["center"],
                "pct_cancer": d["info"]["% Cancer"],
                "inv": d["info"]["P Inv"],
            }
            for d in self.data
        ]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        if self.strip_padding:
            image = strip_padding_pil(image)

        needle_mask = None
        if sample["needle_mask_path"] is not None:
            needle_mask = Image.open(sample["needle_mask_path"]).convert("L")
            if self.strip_padding:
                needle_mask = strip_padding_pil(needle_mask)

        if self.out_fmt == "np":
            image = np.array(image)
            if needle_mask is not None:
                needle_mask = np.array(needle_mask)

        out = {
            "image": image,
            "needle_mask": needle_mask,
            "info": sample["info"], 
            "path": str(sample["image_path"]),
        }

        if self.transform is not None:
            out = self.transform(out)

        return out

    def list_indices_by_patient_ids(self):
        """
        Groups dataset indices by case ID.
        """
        outputs = {}
        for idx, sample in enumerate(self.data):
            outputs.setdefault(sample["case"], []).append(idx)
        return outputs

    def get_weighted_sampler(self):
        labels = []

        for item in self.data:
            primus_score = item["info"]["PRI-MUS"]
            try:
                primus_val = int(float(primus_score))
            except (TypeError, ValueError):
                primus_val = 1
            labels.append(primus_val-1)
        labels = np.array(labels)

        class_count = np.bincount(labels)
        class_weights = 1.0/class_count
        class_weights[1]*=0.4
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            num_samples = len(sample_weights),
            weights = sample_weights,
            replacement=True,
        )
        return sampler
    
    def set_aug_strength(self, strength):
        self.transform.set_aug_strength(strength)

    def get_metadata(self):
        return self.metadata


def get_datasets(*, cfg, **kwargs):
    cohort_mode = kwargs.get("cohort_selection_mode", getattr(cfg, "cohort_selection_mode", "splits_file"))
    augment = kwargs.get("augment", getattr(cfg, "augmentations", "none"))
    augment_strength = cfg.get('augment_strength', 'light')
    return_raw_images = kwargs.get("return_raw_images", getattr(cfg, "return_raw_images", False))
    data_type = kwargs.get("data_type", getattr(cfg, "data_type", None))
    normalize = kwargs.get("normalize", getattr(cfg, "normalize", True))

    train_transform = ProstateTransform(
        augment=augment,
        augment_strength=augment_strength,
        image_size=cfg.image_size,
        mask_size=cfg.mask_size,
        mean=cfg.mean,
        std=cfg.std,
        normalize=normalize,
        crop_to_prostate=cfg.crop_to_prostate,
        first_downsample_size=cfg.first_downsample_size,
        return_raw_images=return_raw_images,
        grade_group_for_positive_label=getattr(
            cfg, "grade_group_for_positive_label", 1
        ),
        flip_ud=cfg.flip_ud,
    )

    val_transform = ProstateTransform(
        augment="none",
        augment_strength = 'none',
        image_size=cfg.image_size,
        mask_size=cfg.mask_size,
        mean=cfg.mean,
        std=cfg.std,
        normalize=normalize,
        crop_to_prostate=cfg.crop_to_prostate,
        first_downsample_size=cfg.first_downsample_size,
        return_raw_images=return_raw_images,
        grade_group_for_positive_label=getattr(
            cfg, "grade_group_for_positive_label", 1
        ),
        flip_ud=cfg.flip_ud,
    )
    mode=None
    needle_mask_fname = (
        "needle_mask_full.png" if mode == "heatmap" else "needle_mask.png"
    )

    def make_dataset(case_ids, transform):
        return ProstateDataset(
            root_dir=cfg.data_path,
            case_ids=case_ids,
            transform=transform,
            needle_mask_fname=needle_mask_fname,
        )
    
    # use all cases
    if cohort_mode in (None, "all", "train_only"):
        train_dataset = make_dataset(None, train_transform)
        val_dataset = make_dataset(None, val_transform)

    # Train / val split
    elif cohort_mode == "train_val":
        cases = [
            p for p in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, p))
        ]
        train_cases, val_cases = sklearn.model_selection.train_test_split(
            cases,
            test_size=0.2,
            random_state=cfg.train_subsample_seed,
        )
        train_dataset = make_dataset(train_cases, train_transform)
        val_dataset = make_dataset(val_cases, val_transform)

    # K-fold
    elif cohort_mode == "kfold":
        cases = [
            p for p in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, p))
        ]
        skf = sklearn.model_selection.KFold(
            n_splits=cfg.n_folds,
            shuffle=True,
            random_state=cfg.train_subsample_seed,
        )
        train_cases, val_cases = [], []
        for fold, (train_idx, val_idx) in enumerate(skf.split(cases)):
            if fold == cfg.fold:
                train_cases = [cases[i] for i in train_idx]
                val_cases = [cases[i] for i in val_idx]
                break
        train_dataset = make_dataset(train_cases, train_transform)
        val_dataset = make_dataset(val_cases, val_transform)

    # From splits.json
    elif cohort_mode == "splits_file":
        with open(cfg.splits_file, "r") as f:
            splits = json.load(f)
        fold = cfg.fold
        data_of_interest = splits[f'kfold_cv_fold-{fold}']
        train_dataset = make_dataset(data_of_interest.get("train"), train_transform)
        val_dataset = make_dataset(data_of_interest.get("val"), val_transform)

    else:
        raise ValueError(f"Unknown cohort selection mode: {cohort_mode}")
    
    if data_type == 'push':
        return make_dataset(data_of_interest.get("train"), val_transform)
    else:
        return train_dataset, val_dataset



