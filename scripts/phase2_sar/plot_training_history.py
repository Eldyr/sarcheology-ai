
"""Plot the learning curves for a SAR CAE experiment."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    history_path = run_dir / "history.jsonl"

    if not history_path.is_file():
        raise FileNotFoundError(history_path)

    records = [
        json.loads(line)
        for line in history_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not records:
        raise RuntimeError("No training history found")

    epochs = [r["epoch"] for r in records]
    train = [r["train_loss"] for r in records]
    val = [r["val_loss"] for r in records]

    best_index = min(
        range(len(records)),
        key=lambda i: val[i],
    )

    print("Epochs completed:", len(records))
    print("Best epoch:", epochs[best_index])
    print("Best validation loss:", val[best_index])
    print("Final training loss:", train[-1])
    print("Final validation loss:", val[-1])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train, label="Training")
    ax.plot(epochs, val, label="Validation")

    ax.axvline(
        epochs[best_index],
        linestyle="--",
        alpha=0.6,
        label=f"Best epoch: {epochs[best_index]}",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked Huber loss")
    ax.set_title("SAR CAE — Training history")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output = run_dir / "learning_curves.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print("Saved:", output)


if __name__ == "__main__":
    main()
