
import lmdb

def main():
    env = lmdb.open("data", readonly=True, lock=False)

    try:
        with env.begin() as txn:
            for key, value in txn.cursor():
                name = key.decode("utf-8", errors="replace")

                if not name.startswith(("image-", "label-")):
                    print("Metadata key:", name)
                    print("Value size:", len(value))
                    print("Value preview:", repr(value[:500]))
    finally:
        env.close()

if __name__ == "__main__":
    main()
