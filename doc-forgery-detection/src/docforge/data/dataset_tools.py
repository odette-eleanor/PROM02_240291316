
import argparse
import csv
import io
import random
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image


# --------------------------------------------------
# 1. Dataset configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASETS = {
    "small": {
        "path": PROJECT_ROOT / "data/raw/small",
        "num_samples": 2_000,
        "split_csv": PROJECT_ROOT / "data/splits/small_dataset_split.csv",
    },
    "large": {
        "path": PROJECT_ROOT / "data/raw/large",
        "num_samples": 120_000,
        "split_csv": PROJECT_ROOT / "data/splits/large_dataset_split.csv",
    },
}


def open_lmdb(dataset_name):
    """Open the selected LMDB dataset in read-only mode."""
    path = DATASETS[dataset_name]["path"]

    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )


def get_sample(txn, index):
    """Retrieve one image and its corresponding mask."""
    image_key = f"image-{index:09d}".encode()
    label_key = f"label-{index:09d}".encode()

    image_bytes = txn.get(image_key)
    label_bytes = txn.get(label_key)

    return image_bytes, label_bytes


# --------------------------------------------------
# 2. INSPECT
# --------------------------------------------------

def inspect_dataset(dataset_name):
    config = DATASETS[dataset_name]
    env = open_lmdb(dataset_name)

    image_count = 0
    label_count = 0
    other_keys = []

    print(f"\nInspecting: {dataset_name.upper()} dataset")
    print(f"Location: {config['path']}")

    with env.begin() as txn:
        print(f"\nLMDB metadata: {txn.get(b'num-samples')}")

        cursor = txn.cursor()

        for key, value in cursor:
            if key.startswith(b"image-"):
                image_count += 1
            elif key.startswith(b"label-"):
                label_count += 1
            else:
                other_keys.append(key)

        print("\nKey summary")
        print("------------------------")
        print(f"Images: {image_count:,}")
        print(f"Labels: {label_count:,}")
        print(f"Other keys: {len(other_keys):,}")

        print("\nSample inspection")
        print("------------------------")

        indexes = [0, 1, 10, 100, config["num_samples"] - 1]

        for index in indexes:
            image_bytes, label_bytes = get_sample(txn, index)

            if image_bytes is None or label_bytes is None:
                print(f"Sample {index}: missing image or label")
                continue

            with Image.open(io.BytesIO(image_bytes)) as image:
                print(
                    f"Sample {index} image: "
                    f"size={image.size}, mode={image.mode}, "
                    f"format={image.format}"
                )

            with Image.open(io.BytesIO(label_bytes)) as mask:
                mask.load()
                mask_array = np.array(mask)

                print(
                    f"Sample {index} mask: "
                    f"size={mask.size}, mode={mask.mode}, "
                    f"values={np.unique(mask_array).tolist()}, "
                    f"nonzero={np.count_nonzero(mask_array):,}"
                )

    env.close()


# --------------------------------------------------
# 3. VERIFY
# --------------------------------------------------

def verify_dataset(dataset_name):
    config = DATASETS[dataset_name]
    total = config["num_samples"]

    env = open_lmdb(dataset_name)

    errors = []
    empty_masks = 0
    image_dimensions = {}
    mask_dimensions = {}
    mask_value_counts = {}

    print(f"\nVerifying {dataset_name.upper()} dataset")
    print(f"Expected samples: {total:,}\n")

    with env.begin() as txn:
        for index in range(total):
            image_bytes, label_bytes = get_sample(txn, index)

            if image_bytes is None or label_bytes is None:
                errors.append(f"Sample {index}: missing image or label")
                continue

            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    image_size = image.size

                with Image.open(io.BytesIO(label_bytes)) as mask:
                    mask.load()
                    mask_array = np.array(mask)
                    mask_size = mask.size

                if image_size != (512, 512):
                    errors.append(
                        f"Sample {index}: image size {image_size}"
                    )

                if mask_size != (512, 512):
                    errors.append(
                        f"Sample {index}: mask size {mask_size}"
                    )

                values = np.unique(mask_array)

                # The currently inspected masks use 0, 1 or 255.
                if not np.isin(values, [0, 1, 255]).all():
                    errors.append(
                        f"Sample {index}: unexpected mask values "
                        f"{values.tolist()}"
                    )

                if np.count_nonzero(mask_array) == 0:
                    empty_masks += 1

                image_dimensions[image_size] = (
                    image_dimensions.get(image_size, 0) + 1
                )

                mask_dimensions[mask_size] = (
                    mask_dimensions.get(mask_size, 0) + 1
                )

                for value in values:
                    value = int(value)
                    mask_value_counts[value] = (
                        mask_value_counts.get(value, 0) + 1
                    )

            except Exception as error:
                errors.append(f"Sample {index}: {error}")

            if (index + 1) % 10_000 == 0:
                print(f"Checked {index + 1:,}/{total:,}")

    env.close()

    print("\nVerification summary")
    print("------------------------")
    print(f"Samples expected: {total:,}")
    print(f"Errors: {len(errors):,}")
    print(f"Empty masks: {empty_masks:,}")
    print(f"Image dimensions: {image_dimensions}")
    print(f"Mask dimensions: {mask_dimensions}")
    print(f"Mask values encountered: {sorted(mask_value_counts)}")

    if errors:
        print("\nFirst 20 errors:")
        for error in errors[:20]:
            print(error)
        print("\nFAIL: Please investigate the errors.")
    else:
        print("\nPASS: All image-mask pairs passed verification.")


# --------------------------------------------------
# 4. VISUALIZE
# --------------------------------------------------

def visualize_dataset(dataset_name, index):
    import matplotlib.pyplot as plt

    config = DATASETS[dataset_name]

    if index < 0 or index >= config["num_samples"]:
        raise ValueError(
            f"Index must be between 0 and {config['num_samples'] - 1}"
        )

    env = open_lmdb(dataset_name)

    with env.begin() as txn:
        image_bytes, label_bytes = get_sample(txn, index)

        if image_bytes is None or label_bytes is None:
            env.close()
            raise ValueError(f"Missing image or mask at index {index}")

        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")

        with Image.open(io.BytesIO(label_bytes)) as mask:
            mask = mask.convert("L")

    env.close()

    image_array = np.array(image)
    mask_array = np.array(mask)

    # Normalise foreground values (1 or 255) to binary 1.
    binary_mask = (mask_array > 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_array)
    axes[0].set_title("Original document")

    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(image_array)
    axes[2].imshow(binary_mask, cmap="Reds", alpha=0.45)
    axes[2].set_title("Manipulation overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(f"{dataset_name.upper()} dataset — sample {index}")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 5. SPLIT
# --------------------------------------------------

def split_dataset(dataset_name, seed=42):
    config = DATASETS[dataset_name]
    total = config["num_samples"]
    output_path = config["split_csv"]

    if output_path.exists():
        raise FileExistsError(
            f"Split file already exists: {output_path}\n"
            "It will not be overwritten."
        )

    print(f"\nPreparing split for {dataset_name.upper()} dataset")
    print(f"Samples: {total:,}")
    print(f"Output: {output_path}")

    env = open_lmdb(dataset_name)

    # Check that every expected image and label key exists.
    with env.begin() as txn:
        for index in range(total):
            image_bytes, label_bytes = get_sample(txn, index)

            if image_bytes is None or label_bytes is None:
                env.close()
                raise ValueError(
                    f"Missing image or label at sample {index}. "
                    "No split CSV was created."
                )

            if (index + 1) % 10_000 == 0:
                print(f"Checked keys: {index + 1:,}/{total:,}")

    env.close()

    # Reproducible random split: 80% train, 10% validation, 10% test.
    indexes = list(range(total))
    random.Random(seed).shuffle(indexes)

    train_end = int(total * 0.8)
    validation_end = int(total * 0.9)

    assignments = {}

    for index in indexes[:train_end]:
        assignments[index] = "train"

    for index in indexes[train_end:validation_end]:
        assignments[index] = "validation"

    for index in indexes[validation_end:]:
        assignments[index] = "test"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "sample_id",
            "image_key",
            "label_key",
            "split",
        ])

        for index in range(total):
            writer.writerow([
                index,
                f"image-{index:09d}",
                f"label-{index:09d}",
                assignments[index],
            ])

    print("\nSplit summary")
    print("------------------------")
    print(f"Training: {train_end:,}")
    print(f"Validation: {validation_end - train_end:,}")
    print(f"Testing: {total - validation_end:,}")
    print(f"Saved: {output_path}")


# --------------------------------------------------
# 6. COMMAND-LINE INTERFACE
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect, verify, visualise and split LMDB datasets."
    )

    parser.add_argument(
        "action",
        choices=["inspect", "verify", "visualize", "split"],
    )

    parser.add_argument(
        "--dataset",
        choices=["small", "large"],
        required=True,
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index for visualisation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )

    args = parser.parse_args()

    if args.action == "inspect":
        inspect_dataset(args.dataset)

    elif args.action == "verify":
        verify_dataset(args.dataset)

    elif args.action == "visualize":
        visualize_dataset(args.dataset, args.index)

    elif args.action == "split":
        split_dataset(args.dataset, args.seed)


if __name__ == "__main__":
    main()
