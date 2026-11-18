
import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sarchaeology.models.sar_cae import SARCAE
from sarchaeology.training.losses import (
    MaskedSARReconstructionLoss,
)
from sarchaeology.training.trainer import (
    load_checkpoint,
    make_optimizer,
    make_scaler,
    make_scheduler,
    run_epoch,
    train_model,
)


class ToySARDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(
            index + 42
        )
        image = torch.randn(
            2, 32, 32,
            generator=generator,
        )
        mask = torch.ones(1, 32, 32)
        mask[:, :4, :] = 0
        image[:, :4, :] = 0

        return {
            "image": image,
            "valid_mask": mask,
        }


@pytest.fixture
def setup_components():
    model = SARCAE(
        base_channels=8,
        latent_channels=16,
        embedding_dim=16,
        group_norm_groups=8,
    )

    loader = DataLoader(
        ToySARDataset(),
        batch_size=2,
        shuffle=False,
    )

    criterion = MaskedSARReconstructionLoss(
        loss_type="huber"
    )

    cfg = {
        "seed": 42,
        "epochs": 2,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "min_learning_rate": 1e-6,
        "gradient_clip_norm": 1.0,
        "log_every_batches": 0,
    }

    device = torch.device("cpu")

    amp = {
        "enabled": False,
        "dtype": torch.float16,
    }

    return model, loader, criterion, cfg, device, amp


def test_optimizer_and_scheduler(setup_components):
    model, _, _, cfg, _, _ = setup_components

    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(
        optimizer, cfg, 2
    )

    assert isinstance(
        optimizer, torch.optim.AdamW
    )
    assert scheduler is not None


def test_train_epoch(setup_components):
    model, loader, criterion, cfg, device, amp = (
        setup_components
    )

    optimizer = make_optimizer(model, cfg)
    scaler = make_scaler(device, amp)

    result = run_epoch(
        model,
        loader,
        criterion,
        device,
        amp,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=1.0,
    )

    assert result["batches"] == 2
    assert result["samples"] == 4
    assert result["loss"] >= 0
    assert torch.isfinite(
        torch.tensor(result["loss"])
    )


def test_validation_does_not_update_weights(
    setup_components,
):
    model, loader, criterion, _, device, amp = (
        setup_components
    )

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    result = run_epoch(
        model,
        loader,
        criterion,
        device,
        amp,
    )

    assert result["batches"] == 2

    for name, param in model.named_parameters():
        assert torch.equal(
            before[name], param
        )


def test_checkpoint_and_history(
    setup_components,
    tmp_path,
):
    model, loader, criterion, cfg, device, amp = (
        setup_components
    )

    result = train_model(
        model=model,
        criterion=criterion,
        train_loader=loader,
        val_loader=loader,
        device=device,
        amp=amp,
        training_cfg=cfg,
        checkpoint_dir=tmp_path / "checkpoints",
        run_dir=tmp_path / "run",
        run_id="test_run",
        normalization_sha256="test_hash",
        epochs=1,
    )

    best = tmp_path / "checkpoints" / "best.pt"
    last = tmp_path / "checkpoints" / "last.pt"
    history = tmp_path / "run" / "history.jsonl"

    assert best.exists()
    assert last.exists()
    assert history.exists()

    checkpoint = load_checkpoint(
        last, device
    )

    assert checkpoint["epoch"] == 1
    assert checkpoint["run_id"] == "test_run"

    records = [
        json.loads(line)
        for line in history.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 1
    assert records[0]["epoch"] == 1
    assert result["best_val_loss"] >= 0


def test_invalid_resume_hash(
    setup_components,
    tmp_path,
):
    model, loader, criterion, cfg, device, amp = (
        setup_components
    )

    ckpt_dir = tmp_path / "checkpoints"
    run_dir = tmp_path / "run"

    train_model(
        model=model,
        criterion=criterion,
        train_loader=loader,
        val_loader=loader,
        device=device,
        amp=amp,
        training_cfg=cfg,
        checkpoint_dir=ckpt_dir,
        run_dir=run_dir,
        run_id="test_run",
        normalization_sha256="original_hash",
        epochs=2,
        max_train_batches=1,
        max_val_batches=1,
    )

    fresh_model = SARCAE(
        base_channels=8,
        latent_channels=16,
        embedding_dim=16,
        group_norm_groups=8,
    )

    with pytest.raises(
        RuntimeError,
        match="normalization changed",
    ):
        train_model(
            model=fresh_model,
            criterion=criterion,
            train_loader=loader,
            val_loader=loader,
            device=device,
            amp=amp,
            training_cfg=cfg,
            checkpoint_dir=ckpt_dir,
            run_dir=run_dir,
            run_id="test_run",
            normalization_sha256="different_hash",
            epochs=2,
            resume=ckpt_dir / "last.pt",
        )
