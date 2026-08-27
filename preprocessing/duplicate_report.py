import os
import hashlib
from collections import defaultdict


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed"
)


SPLITS = [
    "train",
    "validation",
    "test"
]


def file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


print("=" * 70)
print("NeuroCareAI 2.0 DUPLICATE DATASET REPORT")
print("=" * 70)


all_hashes = defaultdict(list)

split_counts = {}


for split in SPLITS:

    split_dir = os.path.join(
        DATASET_DIR,
        split
    )

    count = 0

    print(f"\nScanning {split}...")

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

            h = file_hash(path)

            all_hashes[h].append(
                (
                    split,
                    path
                )
            )

            count += 1

    split_counts[split] = count


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("IMAGE COUNTS")
print("=" * 70)

for split, count in split_counts.items():

    print(
        f"{split:<12}: {count}"
    )


total_images = sum(
    split_counts.values()
)


unique_images = len(
    all_hashes
)


duplicate_images = (
    total_images - unique_images
)


print(
    f"\nTotal images : {total_images}"
)

print(
    f"Unique images: {unique_images}"
)

print(
    f"Duplicate copies: {duplicate_images}"
)


# ============================================================
# CROSS-SPLIT
# ============================================================

cross_split_groups = 0
cross_split_images = 0


for h, entries in all_hashes.items():

    splits = set(
        split
        for split, path in entries
    )

    if len(splits) > 1:

        cross_split_groups += 1

        cross_split_images += len(
            entries
        )


print("\n" + "=" * 70)
print("CROSS-SPLIT DUPLICATES")
print("=" * 70)

print(
    f"Duplicate hash groups: "
    f"{cross_split_groups}"
)

print(
    f"Images involved: "
    f"{cross_split_images}"
)


print("\n" + "=" * 70)
print("REPORT COMPLETE")
print("=" * 70)