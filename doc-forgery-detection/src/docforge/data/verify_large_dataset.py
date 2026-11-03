import io
import lmdb
import numpy as np

from pathlib import Path
from PIL import Image


DATASET_PATH = Path("data/data_large")
NUM_SAMPLES = 120_000


def verify_dataset():
    env = lmdb.open(
        str(DATASET_PATH),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )

    errors = []
    empty_masks = 0
    total_mask_pixels = 0

    with env.begin() as txn:
        for index in range(NUM_SAMPLES):
            image_key = f"image-{index:09d}".encode()
            label_key = f"label-{index:09d}".encode()

            image_bytes = txn.get(image_key)
            label_bytes = txn.get(label_key)

            if image_bytes is None or label_bytes is None:
                errors.append(f"Sample {index}: missing image or label")
                continue

            try:
                image = Image.open(io.BytesIO(image_bytes))
                mask = Image.open(io.BytesIO(label_bytes))

                image.load()
                mask.load()

                if image.size != (512, 512):
                    errors.append(
                        f"Sample {index}: unexpected image size {image.size}"
                    )

                if mask.size != (512, 512):
                    errors.append(
                        f"Sample {index}: unexpected mask size {mask.size}"
                    )

                mask_array = np.array(mask)

                if not np.isin(np.unique(mask_array), [0, 1, 255]).all():
                    errors.append(
                        f"Sample {index}: unexpected mask values"
                    )

                if np.count_nonzero(mask_array) == 0:
                    empty_masks += 1

                total_mask_pixels += np.count_nonzero(mask_array)

            except Exception as error:
                errors.append(f"Sample {index}: {error}")

            if (index + 1) % 10_000 == 0:
                print(f"Checked {index + 1:,}/{NUM_SAMPLES:,}")

    env.close()

    print("\nVerification summary")
    print("--------------------")
    print(f"Total samples checked: {NUM_SAMPLES:,}")
    print(f"Errors: {len(errors)}")
    print(f"Empty masks: {empty_masks:,}")
    print(f"Total nonzero mask pixels: {total_mask_pixels:,}")

    if errors:
        print("\nFirst 20 errors:")
        for error in errors[:20]:
            print(error)
    else:
        print("\nPASS: All image-mask pairs passed verification.")


if __name__ == "__main__":
    verify_dataset()