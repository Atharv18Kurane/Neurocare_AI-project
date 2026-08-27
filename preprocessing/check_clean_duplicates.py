import os
import hashlib
from collections import defaultdict


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CLEAN_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "clean"
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
print("CLEAN DATASET DUPLICATE CHECK")
print("=" * 70)


hashes = defaultdict(list)

total_images = 0


for split in SPLITS:

    split_dir = os.path.join(
        CLEAN_DIR,
        split
    )

    print(
        f"\nScanning {split}..."
    )

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

            image_hash = file_hash(path)

            hashes[image_hash].append(
                (
                    split,
                    path
                )
            )

            total_images += 1


# ============================================================
# CHECK CROSS-SPLIT DUPLICATES
# ============================================================

cross_split_groups = []

for image_hash, entries in hashes.items():

    split_names = set(
        split
        for split, path in entries
    )

    if len(split_names) > 1:

        cross_split_groups.append(
            (
                image_hash,
                entries
            )
        )


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Total clean images: {total_images}"
)

print(
    f"Unique hashes: {len(hashes)}"
)

print(
    f"Cross-split duplicate groups: "
    f"{len(cross_split_groups)}"
)


if len(cross_split_groups) == 0:

    print(
        "\nSUCCESS!"
    )

    print(
        "ZERO cross-split exact duplicates found."
    )

else:

    print(
        "\nWARNING!"
    )

    print(
        "Cross-split duplicates still exist."
    )

    for image_hash, entries in cross_split_groups[:20]:

        print(
            "\nHash:",
            image_hash
        )

        for split, path in entries:

            print(
                f"  {split}: {path}"
            )


print("=" * 70)