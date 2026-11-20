"""SARchaeology AI — Phase 2.12: frozen SARMAE feature extraction.

Windows-compatible download directly into checkpoints/, avoiding the
Hugging Face shared snapshot cache's symlinks.

Examples:
    uv run python scripts/phase2_sar/extract_sarmae_embeddings.py --split val --batch-size 2 --max-batches 1
    uv run python scripts/phase2_sar/extract_sarmae_embeddings.py --split train --batch-size 4
    uv run python scripts/phase2_sar/extract_sarmae_embeddings.py --split val --batch-size 4

SECURITY: the converted model needs trust_remote_code=True. Review its
modeling_*.py / configuration_*.py files and local imports before allowing
execution. The repository revision is pinned by the Phase 2.10 audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sarchaeology.data.dataloaders import create_sar_dataloaders
from sarchaeology.data.sarmae_adapter import SARMAEAdapterConfig, SARMAEInputAdapter
from sarchaeology.preprocessing.sar_normalization import SARNormalizer

RUN_ID = "sar_cae_20261118T192046Z_2dbe6bd2"
REPO = "BiliSakura/SARMAE-transformers"
SUBFOLDER = "vit-base-patch16-pretrain"


def get_pinned_revision() -> str:
    path = ROOT / "outputs/inspections/sarmae/compatibility_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run inspect_sarmae.py first: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("repository") != REPO or audit.get("subfolder") != SUBFOLDER:
        raise RuntimeError("Phase 2.10 audit refers to a different model/subfolder")
    revision = audit.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or not all(
        c in "0123456789abcdef" for c in revision.lower()
    ):
        raise RuntimeError("Audit must contain a pinned 40-character commit SHA")
    return revision


def download_pinned_model(revision: str) -> Path:
    """Use local_dir and one worker to avoid Windows symlink-cache failures."""
    local_root = ROOT / "checkpoints/pretrained/sarmae" / revision
    model_dir = local_root / SUBFOLDER
    required = ("config.json", "model.safetensors", "modeling_sarmae.py")

    if not all((model_dir / name).is_file() for name in required):
        print("Downloading pinned SARMAE files to:", local_root)
        snapshot_download(
            repo_id=REPO,
            revision=revision,
            allow_patterns=[f"{SUBFOLDER}/*"],
            local_dir=str(local_root),
            max_workers=1,
        )

    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete SARMAE download in {model_dir}: missing {missing}"
        )
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("num_channels") != 3 or config.get("image_size") != 224:
        raise ValueError("Expected a three-channel 224x224 SARMAE checkpoint")
    print("Local model directory:", model_dir)
    return model_dir


def make_loader(config: dict, normalizer: SARNormalizer, split: str, batch_size: int):
    copied = dict(config)
    copied["dataloader"] = dict(config["dataloader"])
    copied["dataloader"]["batch_size"] = batch_size
    copied["dataloader"]["num_workers"] = 0
    copied["dataloader"]["shuffle_train"] = False
    _, loaders = create_sar_dataloaders(
        project_root=ROOT,
        config=copied,
        include_test=False,
        normalizer=normalizer,
    )
    return loaders[split]


def extract_spatial_tokens(output: object, batch_size: int) -> torch.Tensor:
    """Reject unsupported layouts rather than silently pooling the wrong tensor."""
    if hasattr(output, "last_hidden_state"):
        tokens = output.last_hidden_state
    elif isinstance(output, dict):
        tokens = output.get("last_hidden_state")
    elif isinstance(output, (tuple, list)) and output:
        tokens = output[0]
    else:
        raise TypeError(f"Unexpected SARMAE output type: {type(output)!r}")

    if not isinstance(tokens, torch.Tensor):
        raise TypeError("Model did not return a tensor last_hidden_state")
    if tokens.ndim != 3 or tokens.shape[0] != batch_size or tokens.shape[2] != 768:
        raise ValueError(f"Unexpected SARMAE sequence shape: {tuple(tokens.shape)}")
    if tokens.shape[1] == 197:
        return tokens[:, 1:, :]
    if tokens.shape[1] == 196:
        return tokens
    raise ValueError(
        f"Expected 196 patch tokens (plus optional CLS), got {tokens.shape[1]}. "
        "Inspect the converted model's forward output before proceeding."
    )


def masked_pool(tokens: torch.Tensor, pixel_mask: torch.Tensor):
    """Pool only completely valid 16x16 patches."""
    coverage = F.avg_pool2d(pixel_mask.float(), kernel_size=16, stride=16).flatten(1)
    keep = coverage >= 0.999
    if keep.shape != tokens.shape[:2]:
        raise ValueError("Patch-mask and token-grid mismatch")
    if not keep.any(dim=1).all():
        raise ValueError("One or more observations have no fully valid tokens")
    pooled = (tokens.float() * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True)
    if not torch.isfinite(pooled).all():
        raise ValueError("SARMAE returned non-finite embeddings")
    return pooled, keep.sum(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--allow-custom-code",
        action="store_true",
        help="Acknowledge that local pretrained-model custom Python code has been reviewed",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("Batch size and max-batches (if provided) must be positive")

    revision = get_pinned_revision()
    model_dir = download_pinned_model(revision)
    if not args.allow_custom_code:
        parser.error(
            "Pretrained-model files are downloaded. Review modeling_sarmae.py, "
            "configuration_*.py and their local imports in the printed model directory, "
            "then rerun with --allow-custom-code to execute them."
        )

    run_dir = ROOT / "outputs/experiments/sar/sar_cae" / args.run_id
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    normalizer = SARNormalizer.from_json(run_dir / "sar_normalization.json")
    loader = make_loader(config, normalizer, args.split, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SARMAE from local directory:", model_dir)
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    ).to(device)
    model.eval().requires_grad_(False)

    adapter = SARMAEInputAdapter(
        SARMAEAdapterConfig(input_mode="normalized_proxy")
    )

    embeddings_parts = []
    metadata = {"patch_id": [], "acquisition_date": [], "polarization": [], "valid_tokens": []}
    observations_seen = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            adapted = adapter(batch["image"], batch["valid_mask"])
            pixels = adapted["pixel_values"].to(device, non_blocking=True)
            masks = adapted["pixel_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(pixel_values=pixels, return_dict=True)
            tokens = extract_spatial_tokens(output, pixels.shape[0])
            pooled, token_counts = masked_pool(tokens, masks)

            b = int(batch["image"].shape[0])
            observations_seen += b
            embeddings_parts.append(pooled.cpu().numpy().astype(np.float32))
            metadata["valid_tokens"].extend(token_counts.cpu().tolist())
            for polarization in ("VV", "VH"):
                metadata["polarization"].extend([polarization] * b)
                metadata["patch_id"].extend(str(x) for x in batch["patch_id"])
                metadata["acquisition_date"].extend(
                    str(x) for x in batch["acquisition_date"]
                )
            if (batch_index + 1) % 25 == 0:
                print(f"{args.split}: processed {observations_seen} observations")

    if not embeddings_parts:
        raise RuntimeError("No observations processed")
    embeddings = np.concatenate(embeddings_parts, axis=0)
    lengths = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not (lengths > 1e-12).all():
        raise RuntimeError("Invalid, non-finite, or zero-length embeddings")
    if len(embeddings) != 2 * observations_seen:
        raise RuntimeError("Unexpected polarization embedding count")

    out_dir = ROOT / "outputs/embeddings/sarmae" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.max_batches is not None else ""
    out_path = out_dir / f"{args.split}{suffix}.npz"
    np.savez_compressed(
        out_path,
        embedding=embeddings,
        patch_id=np.asarray(metadata["patch_id"], dtype=str),
        acquisition_date=np.asarray(metadata["acquisition_date"], dtype=str),
        polarization=np.asarray(metadata["polarization"], dtype=str),
        valid_tokens=np.asarray(metadata["valid_tokens"], dtype=np.int32),
        source_revision=np.asarray(revision),
        input_mode=np.asarray("normalized_proxy"),
    )
    print("\nPhase 2.12 extraction complete")
    print("Device:", device)
    print("Observations:", observations_seen)
    print("Embeddings:", embeddings.shape)
    print("Valid tokens min/max:", min(metadata["valid_tokens"]), max(metadata["valid_tokens"]))
    print("Input mode: normalized_proxy")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
