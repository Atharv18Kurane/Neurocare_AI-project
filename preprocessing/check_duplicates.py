import os
import hashlib
from collections import defaultdict


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# SPLITS
# ============================================================

SPLITS = {
    "train": os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "train"
    ),

    "validation": os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "validation"
    ),

    "test": os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "test"
    )
}


# ============================================================
# HASH FUNCTION
# ============================================================

def file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# COLLECT HASHES
# ============================================================

hashes = defaultdict(list)


print("=" * 70)
print("NeuroCareAI 2.0 DATASET DUPLICATE CHECK")
print("=" * 70)


for split_name, split_dir in SPLITS.items():

    print(
        f"\nScanning: {split_name}"
    )

    if not os.path.exists(split_dir):

        print(
            f"WARNING: Directory not found: {split_dir}"
        )

        continue


    for root, dirs, files in os.walk(split_dir):

        for filename in files:

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                continue


            path = os.path.join(
                root,
                filename
            )


            try:

                h = file_hash(path)

                hashes[h].append(
                    (
                        split_name,
                        path
                    )
                )

            except Exception as e:

                print(
                    f"Could not hash {path}: {e}"
                )


# ============================================================
# FIND CROSS-SPLIT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("CROSS-SPLIT DUPLICATES")
print("=" * 70)


cross_split_duplicates = []


for h, entries in hashes.items():

    split_names = set(
        entry[0]
        for entry in entries
    )


    if len(split_names) > 1:

        cross_split_duplicates.append(
            (
                h,
                entries
            )
        )


print(
    f"\nDuplicate groups across splits: "
    f"{len(cross_split_duplicates)}"
)


for h, entries in cross_split_duplicates[:20]:

    print("\nHash:")

    print(h)

    for split_name, path in entries:

        print(
            f"  {split_name}: {path}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)


if len(cross_split_duplicates) == 0:

    print(
        "RESULT: No exact image duplicates found "
        "across train/validation/test."
    )

else:

    print(
        "RESULT: Cross-split duplicates were found."
    )

    print(
        "Do NOT finalize the 96.26% research result yet."
    )


print("=" * 70)