
"""Shared SAR experiment preparation utilities."""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml


def set_global_seed(seed: int, deterministic=True):
    """Seed supported RNGs and configure PyTorch."""
    seed = int(seed)

    if seed < 0:
        raise ValueError("Seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # May reduce performance, but helps reproducibility.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def resolve_device(requested="cuda"):
    """Require the requested accelerator; never silently fall back."""
    requested = str(requested).lower()

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. "
                "Check your PyTorch installation and GPU driver."
            )

        return torch.device("cuda:0")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {requested}. "
        "Choose 'cuda' or 'cpu'."
    )


def resolve_amp(device, enabled=True, dtype="float16"):
    """Determine whether CUDA autocasting can be used."""
    supported = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype not in supported:
        raise ValueError(
            f"Unsupported AMP dtype: {dtype}"
        )

    enabled = bool(enabled) and device.type == "cuda"

    return {
        "enabled": enabled,
        "dtype": supported[dtype],
        "dtype_name": dtype,
    }


def load_training_config(config_path):
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")

    required = [
        "dataset",
        "dataloader",
        "training",
        "normalization",
        "experiment",
    ]

    for section in required:
        if not isinstance(config.get(section), dict):
            raise ValueError(
                f"Missing or invalid section: {section}"
            )

    return config


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit(project_root):
    """Return commit ID when Git is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def create_experiment(
    project_root,
    config,
    *,
    normalization_path,
    train_count,
    validation_count,
):
    """
    Create a unique experiment folder and
    persist configuration and provenance.

    Does not train a model.
    """
    root = Path(project_root).resolve()

    experiment_cfg = config["experiment"]
    model_name = experiment_cfg["model_name"]
    dataset_id = config["dataset"]["dataset_id"]

    if not model_name or not all(
        c.isalnum() or c in "_-" for c in model_name
    ):
        raise ValueError("Invalid model_name")

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    run_id = (
        f"{model_name}_{timestamp}_"
        f"{uuid4().hex[:8]}"
    )

    output_base = (
        root / experiment_cfg["output_directory"]
    ).resolve()

    checkpoint_base = (
        root / experiment_cfg["checkpoint_directory"]
    ).resolve()

    run_dir = output_base / model_name / run_id
    checkpoint_dir = checkpoint_base / model_name / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    normalization_path = Path(
        normalization_path
    ).resolve()

    normalization = json.loads(
        normalization_path.read_text(
            encoding="utf-8"
        )
    )

    metadata = {
        "run_id": run_id,
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "project": experiment_cfg["project"],
        "model_name": model_name,
        "dataset_id": dataset_id,
        "train_observations": int(train_count),
        "validation_observations": int(
            validation_count
        ),
        "normalization_path": str(
            normalization_path
        ),
        "normalization_sha256": sha256_file(
            normalization_path
        ),
        "train_manifest_sha256": normalization.get(
            "source_manifest_sha256"
        ),
        "normalization_method": normalization.get(
            "method"
        ),
        "random_seed": config["training"]["seed"],
        "git_commit": git_commit(root),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "operating_system": platform.platform(),
        "run_directory": str(run_dir),
        "checkpoint_directory": str(
            checkpoint_dir
        ),
        "status": "prepared",
    }

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    (run_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Preserve the exact statistics used by this run.
    (run_dir / "sar_normalization.json").write_text(
        normalization_path.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    return metadata
