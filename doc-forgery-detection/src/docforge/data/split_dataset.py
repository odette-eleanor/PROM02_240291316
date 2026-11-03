import csv
import random 
from pathlib import Path

TOTAL_SAMPLES = 2000
SEED = 42

OUTPUT_PATH = Path("data/dataset_split.csv")

def main():
    sample_ids = list(range(TOTAL_SAMPLES))

    rng = random.Random(SEED)
    rng.shuffle(sample_ids)

    # allocate 80% for training, 10% for validation, 10% for testing
    train_end = int(TOTAL_SAMPLES * 0.8)
    val_end = int(TOTAL_SAMPLES * 0.9)

    split_assignments = {}

    for sample_id in sample_ids[:train_end]:
        split_assignments[sample_id] = "train"
    
    for sample_id in sample_ids[train_end:val_end]:
        split_assignments[sample_id] = "validation"

    for sample_id in sample_ids[val_end:]:
        split_assignments[sample_id] = "test"

    # Save the Assignments to CSV.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["sample_id", "image_key", "label_key", "split"],
        )
        writer.writeheader()

        for sample_id in range(TOTAL_SAMPLES):
            writer.writerow({
                "sample_id": sample_id,
                "image_key": f"image-{sample_id:09d}",
                "label_key": f"label-{sample_id:09d}",
                "split": split_assignments[sample_id],
            })

    print("Dataset split completed.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Training: {train_end}")
    print(f"Validation: {val_end - train_end}")
    print(f"Test: {TOTAL_SAMPLES - val_end}")
    print(f"Random seed: {SEED}")

if __name__ == "__main__":
    main()