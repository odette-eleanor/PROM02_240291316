import io
import sys
from pathlib import Path

import lmdb
from PIL import Image


def inspect_lmdb(lmdb_dir: str):
    path = Path(lmdb_dir)

    print(f"LMDB dir: {path}")

    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
    )

    with env.begin() as txn:
        stat = txn.stat()
        print(f"Total entries: {stat['entries']}")
        print("-" * 50)

        cursor = txn.cursor()

        image_count = 0
        label_count = 0
        other_count = 0

        for key, value in cursor:
            key_str = key.decode("utf-8", errors="replace")

            if key_str.startswith("image-"):
                image_count += 1

            elif key_str.startswith("label-"):
                label_count += 1

            else:
                other_count += 1

            # Show the first 20 entries
            if image_count + label_count + other_count <= 20:
                print(f"{key!r} -> {len(value):,} bytes")

        print("-" * 50)
        print(f"Images : {image_count}")
        print(f"Labels : {label_count}")
        print(f"Other  : {other_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: poetry run python src/docforge/data/lmdb_reader.py <lmdb_dir>")
        sys.exit(1)

    inspect_lmdb(sys.argv[1])