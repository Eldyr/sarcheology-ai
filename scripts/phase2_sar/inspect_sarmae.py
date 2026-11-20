
"""Phase 2.10: SARMAE pretrained-model compatibility audit."""

import argparse
import json
from pathlib import Path

from huggingface_hub import (
    HfApi,
    hf_hub_download,
)

REPO = "BiliSakura/SARMAE-transformers"
FOLDER = "vit-base-patch16-pretrain"
OUT = Path("outputs/inspections/sarmae")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inspect-weights",
        action="store_true",
        help="Also download the ~343 MB safetensors checkpoint",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(
        repo_id=REPO,
        revision="main",
    )
    revision = info.sha

    def fetch(filename):
        path = hf_hub_download(
            repo_id=REPO,
            filename=f"{FOLDER}/{filename}",
            revision=revision,
        )
        return Path(path)

    config = json.loads(
        fetch("config.json").read_text("utf-8")
    )
    processor = json.loads(
        fetch("preprocessor_config.json").read_text(
            "utf-8"
        )
    )

    audit = {
        "repository": REPO,
        "subfolder": FOLDER,
        "revision": revision,
        "architecture": config.get("architectures"),
        "image_size": config.get("image_size"),
        "num_channels": config.get("num_channels"),
        "patch_size": config.get("patch_size"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get(
            "num_hidden_layers"
        ),
        "num_attention_heads": config.get(
            "num_attention_heads"
        ),
        "repeat_grayscale_channels": config.get(
            "repeat_grayscale_channels"
        ),
        "processor": processor,
        "local_input_channels": 2,
        "recommended_adapter": (
            "separate_VV_VH_grayscale_to_RGB"
        ),
        "checkpoint_shapes": None,
    }

    assert audit["num_channels"] == 3
    assert audit["image_size"] == 224
    assert audit["patch_size"] == 16
    assert audit["hidden_size"] == 768

    if args.inspect_weights:
        from safetensors import safe_open

        path = fetch("model.safetensors")

        with safe_open(
            str(path), framework="pt", device="cpu"
        ) as file:
            shapes = {
                key: list(file.get_slice(key).get_shape())
                for key in file.keys()
            }

        audit["checkpoint_shapes"] = shapes

        patch_embeddings = {
            key: shape
            for key, shape in shapes.items()
            if "patch_embed" in key
            and key.endswith("weight")
        }

        print("Patch embedding weights:")
        print(json.dumps(patch_embeddings, indent=2))

        if not any(
            len(shape) == 4
            and shape[1] == 3
            and shape[2:] == [16, 16]
            for shape in patch_embeddings.values()
        ):
            raise RuntimeError(
                "Could not confirm a 3-channel "
                "16x16 patch embedding weight. "
                "Inspect checkpoint keys manually."
            )

    path = OUT / "compatibility_audit.json"
    path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\nSARMAE COMPATIBILITY AUDIT")
    for key in (
        "revision",
        "architecture",
        "image_size",
        "num_channels",
        "patch_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "recommended_adapter",
    ):
        print(f"{key}: {audit[key]}")

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
