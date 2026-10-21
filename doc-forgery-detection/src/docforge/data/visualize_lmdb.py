import io
import sys
from pathlib import Path

import lmdb
from PIL import Image
import matplotlib.pyplot as plt


def visualize_lmdb(lmdb_dir: str, index: int = 0):
    path = Path(lmdb_dir)

    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
    )

    image_key = f"image-{index:09d}".encode()
    label_key = f"label-{index:09d}".encode()

    with env.begin() as txn:
        image_bytes = txn.get(image_key)
        label_bytes = txn.get(label_key)

    if image_bytes is None:
        print(f"Could not find {image_key!r}")
        return

    print(f"Found image: {image_key!r}")
    print(f"Image bytes: {len(image_bytes):,}")

    image = Image.open(io.BytesIO(image_bytes))

    print(f"Image format: {image.format}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")

    if label_bytes is not None:
        print(f"Found label: {label_key!r}")
        print(f"Label bytes: {len(label_bytes):,}")

        label = Image.open(io.BytesIO(label_bytes))

        print(f"Label format: {label.format}")
        print(f"Label size: {label.size}")
        print(f"Label mode: {label.mode}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(image)
        axes[0].set_title("Document Image")
        axes[0].axis("off")

        axes[1].imshow(label, cmap="gray")
        axes[1].set_title("Ground Truth Label")
        axes[1].axis("off")

    else:
        print(f"No label found for {image_key!r}")

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("Document Image")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: poetry run python "
            "src/docforge/data/visualize_lmdb.py <lmdb_dir> [index]"
        )
        sys.exit(1)

    lmdb_dir = sys.argv[1]

    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    visualize_lmdb(lmdb_dir, index)