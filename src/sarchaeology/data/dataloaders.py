
from pathlib import Path

import torch

from torch.utils.data import DataLoader

from sarchaeology.data.sar_dataset import (
    SARDataset,
)


def verify_disjoint_datasets(*datasets):
    """
    Ensure no geographic patch appears in
    more than one dataset split.
    """

    seen = set()

    for dataset in datasets:
        patch_ids = set(
            dataset.manifest["patch_id"]
        )

        intersection = seen & patch_ids

        if intersection:
            raise ValueError(
                "Geographic data leakage detected: "
                f"{len(intersection)} overlapping patches"
            )

        seen.update(patch_ids)


def create_sar_dataloaders(
    project_root,
    config,
    include_test=False,
):
    """
    Construct SAR Dataset and DataLoader objects.

    The test loader is optional and should not
    be used during model development.
    """

    root = Path(project_root)

    dataset_config = config["dataset"]
    loader_config = config["dataloader"]

    dataset_id = dataset_config["dataset_id"]

    def make_dataset(split):
        return SARDataset(
            project_root=root,
            dataset_id=dataset_id,
            split=split,
            min_valid_fraction=dataset_config[
                "min_valid_fraction"
            ],
            cache_size=dataset_config[
                "cache_size"
            ],
        )

    train_dataset = make_dataset("train")
    val_dataset = make_dataset("val")

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
    }

    if include_test:
        datasets["test"] = make_dataset(
            "test"
        )

    verify_disjoint_datasets(
        *datasets.values()
    )

    num_workers = int(
        loader_config["num_workers"]
    )

    persistent_workers = (
        bool(loader_config["persistent_workers"])
        and num_workers > 0
    )

    use_pin_memory = (
        bool(loader_config["pin_memory"])
        and torch.cuda.is_available()
    )

    def make_loader(split, shuffle):
        return DataLoader(
            datasets[split],
            batch_size=int(
                loader_config["batch_size"]
            ),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            drop_last=(
                bool(loader_config["drop_last_train"])
                if split == "train"
                else False
            ),
        )

    loaders = {
        "train": make_loader(
            "train",
            bool(loader_config["shuffle_train"]),
        ),
        "val": make_loader(
            "val",
            False,
        ),
    }

    if include_test:
        loaders["test"] = make_loader(
            "test",
            False,
        )

    return datasets, loaders
