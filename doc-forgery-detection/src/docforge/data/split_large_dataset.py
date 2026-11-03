import lmdb
import io
import numpy as np
from PIL import Image

env = lmdb.open(
    "data/data_large",
    readonly=True,
    lock=False,
    readahead=False,
)

with env.begin() as txn:
    print("Metadata keys:")
    for key, value in txn.cursor():
        if not key.startswith((b"image-", b"label-")):
            print(key, value[:200])

    print("\nSample label inspection:")

    for i in [0, 1, 2, 10, 100, 1000, 10000, 50000, 100000, 119999]:
        key = f"label-{i:09d}".encode()
        value = txn.get(key)

        if value is None:
            print(i, "MISSING LABEL")
            continue

        mask = np.array(Image.open(io.BytesIO(value)))

        print(
            f"Sample {i}:",
            "shape =", mask.shape,
            "values =", np.unique(mask)[:20],
            "nonzero pixels =", np.count_nonzero(mask),
        )

env.close()
