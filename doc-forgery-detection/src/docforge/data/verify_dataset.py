
import lmdb
import numpy as np
from io import BytesIO
from PIL import Image

def main():
    env = lmdb.open(
        "data",
        readonly=True,
        lock=False,
        readahead=False,
    )

    checked = 0
    errors = []
    mask_values = set()
    empty_masks = 0

    try:
        with env.begin() as txn:
            for i in range(2000):
                image_key = f"image-{i:09d}".encode()
                label_key = f"label-{i:09d}".encode()

                image_bytes = txn.get(image_key)
                label_bytes = txn.get(label_key)

                if image_bytes is None or label_bytes is None:
                    errors.append(f"Sample {i}: missing image or label")
                    continue

                try:
                    with Image.open(BytesIO(image_bytes)) as img:
                        image = img.convert("RGB")
                        image_size = image.size

                    with Image.open(BytesIO(label_bytes)) as lbl:
                        mask = np.array(lbl.convert("L"))
                        mask_size = lbl.size

                    if image_size != mask_size:
                        errors.append(
                            f"Sample {i}: image {image_size}, "
                            f"mask {mask_size}"
                        )
                        continue

                    values = set(np.unique(mask).tolist())
                    mask_values.update(values)

                    if not np.any(mask):
                        empty_masks += 1

                    checked += 1

                except Exception as exc:
                    errors.append(f"Sample {i}: {exc}")

    finally:
        env.close()

    print("\n--- Dataset Verification ---")
    print(f"Valid image-mask pairs: {checked}/2000")
    print(f"Errors: {len(errors)}")
    print(f"Mask pixel values: {sorted(mask_values)}")
    print(f"Completely empty masks: {empty_masks}")

    if errors:
        print("\nFirst 20 errors:")
        for error in errors[:20]:
            print("-", error)

    if checked == 2000 and not errors:
        print("\nPASS: All image-mask pairs passed basic checks.")
    else:
        print("\nCHECK REQUIRED: Fix the issues before training.")

if __name__ == "__main__":
    main()
