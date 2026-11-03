
import lmdb
import numpy as np
from io import BytesIO
from PIL import Image

def main():
    env = lmdb.open("data", readonly=True, lock=False)

    try:
        with env.begin() as txn:
            for i in [0, 112, 1999]:
                key = f"label-{i:09d}".encode()
                raw = txn.get(key)

                if raw is None:
                    print(f"\n{key}: MISSING")
                    continue

                mask = np.array(Image.open(BytesIO(raw)))
                values, counts = np.unique(mask, return_counts=True)

                print(f"\nSample {i}")
                print("Shape:", mask.shape)
                print("Image mode:", Image.open(BytesIO(raw)).mode)
                print("Pixel values and counts:")
                print(dict(zip(values.tolist(), counts.tolist())))
                print("Min/max:", mask.min(), mask.max())
    finally:
        env.close()

if __name__ == "__main__":
    main()
