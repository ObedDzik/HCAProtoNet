import json
import torch
from torch.utils.data import DataLoader
import os


def make_loader(*, dataset, cfg, shuffle, weighted=False, **kwargs):

    batch_size=kwargs.get("batch_size", getattr(cfg, "batch_size", 16))
    num_workers=kwargs.get("num_workers", getattr(cfg, "num_workers", 4))
    pin_memory=kwargs.get("pin_memory", getattr(cfg, "pin_memory", True))
    drop_last=kwargs.get("drop_last", getattr(cfg, "drop_last", False))
    weighted=kwargs.get("weighted", getattr(cfg, "weighted", False))

    if weighted:
        sampler = dataset.get_weighted_sampler()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )



