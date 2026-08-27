import os
import shutil
import hashlib
import random
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
# PATHS
# ============================================================

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw"
)

CLEAN_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "clean"
)


# ============================================================
# SETTINGS
# ============================================================

CLASSES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# HASH FUNCTION
# ============================================================

def file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            data = f.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    for split in [
        "train",
        "validation",
        "test"
    ]:

        for class_name in CLASSES:

            directory = os.path.join(
                CLEAN_DIR,
                split,
                class_name
            )

            os.makedirs(
                directory,
                exist_ok=True
            )


# ============================================================
# COLLECT UNIQUE IMAGES
# ============================================================

def collect_unique_images():

    print("=" * 70)
    print("COLLECTING UNIQUE IMAGES")
    print("=" * 70)

    unique_images = {}

    total_images = 0
    duplicate_images = 0

    for class_name in CLASSES:

        class_dir = os.path.join(
            RAW_DIR,
            class_name
        )

        print(
            f"\nScanning {class_name}..."
        )

        class_count = 0
        class_duplicates = 0

        hashes_seen_in_class = set()

        for filename in os.listdir(
            class_dir
        ):

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            path = os.path.join(
                class_dir,
                filename
            )

            total_images += 1
            class_count += 1

            try:

                image_hash = file_hash(
                    path
                )

            except Exception as e:

                print(
                    f"Could not read: {path}"
                )

                print(e)

                continue


            # Exact duplicate
            if image_hash in unique_images:

                duplicate_images += 1
                class_duplicates += 1

                continue


            unique_images[image_hash] = (
                class_name,
                path
            )

            hashes_seen_in_class.add(
                image_hash
            )


        print(
            f"Images found : {class_count}"
        )

        print(
            f"Duplicates   : {class_duplicates}"
        )

    print("\n" + "=" * 70)

    print(
        f"Total raw images    : {total_images}"
    )

    print(
        f"Duplicate copies    : {duplicate_images}"
    )

    print(
        f"Unique images       : {len(unique_images)}"
    )

    print("=" * 70)

    return unique_images


# ============================================================
# GROUP BY CLASS
# ============================================================

def group_by_class(unique_images):

    class_images = defaultdict(list)

    for image_hash, (
        class_name,
        path
    ) in unique_images.items():

        class_images[
            class_name
        ].append(
            (
                image_hash,
                path
            )
        )

    return class_images


# ============================================================
# SPLIT DATA
# ============================================================

def split_images(class_images):

    random.seed(
        RANDOM_SEED
    )

    split_data = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list)
    }


    for class_name in CLASSES:

        images = class_images[
            class_name
        ]

        random.shuffle(
            images
        )

        total = len(images)

        train_end = int(
            total * TRAIN_RATIO
        )

        validation_end = (
            train_end
            + int(total * VALIDATION_RATIO)
        )


        train_images = images[
            :train_end
        ]

        validation_images = images[
            train_end:validation_end
        ]

        test_images = images[
            validation_end:
        ]


        split_data[
            "train"
        ][class_name] = train_images

        split_data[
            "validation"
        ][class_name] = validation_images

        split_data[
            "test"
        ][class_name] = test_images


        print(
            f"\n{class_name}"
        )

        print(
            f"  Total      : {total}"
        )

        print(
            f"  Train      : {len(train_images)}"
        )

        print(
            f"  Validation : {len(validation_images)}"
        )

        print(
            f"  Test       : {len(test_images)}"
        )


    return split_data


# ============================================================
# COPY FILES
# ============================================================

def copy_files(split_data):

    print("\n" + "=" * 70)
    print("COPYING CLEAN DATASET")
    print("=" * 70)


    total_copied = 0


    for split in [
        "train",
        "validation",
        "test"
    ]:

        print(
            f"\nCreating {split}..."
        )


        for class_name in CLASSES:

            images = split_data[
                split
            ][class_name]


            destination_dir = os.path.join(
                CLEAN_DIR,
                split,
                class_name
            )


            for index, (
                image_hash,
                source_path
            ) in enumerate(
                images
            ):

                # Keep original extension
                extension = os.path.splitext(
                    source_path
                )[1].lower()


                # Give every clean image
                # a unique deterministic name
                destination_name = (
                    f"{index:06d}_"
                    f"{image_hash[:16]}"
                    f"{extension}"
                )


                destination_path = os.path.join(
                    destination_dir,
                    destination_name
                )


                shutil.copy2(
                    source_path,
                    destination_path
                )


                total_copied += 1


            print(
                f"  {class_name}: "
                f"{len(images)} images"
            )


    print("\n" + "=" * 70)

    print(
        f"Total copied: {total_copied}"
    )

    print(
        f"Clean dataset: {CLEAN_DIR}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NeuroCareAI 2.0")
    print("CLEAN DATASET CREATION")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The existing dataset\\processed directory "
        "will NOT be modified."
    )

    print(
        "The existing model will NOT be modified."
    )


    # Safety check

    if os.path.exists(CLEAN_DIR):

        print(
            "\nWARNING:"
        )

        print(
            f"{CLEAN_DIR} already exists."
        )

        answer = input(
            "\nDelete and recreate it? "
            "(yes/no): "
        )

        if answer.lower() != "yes":

            print(
                "\nCancelled."
            )

            raise SystemExit

        shutil.rmtree(
            CLEAN_DIR
        )


    create_directories()


    unique_images = (
        collect_unique_images()
    )


    class_images = (
        group_by_class(
            unique_images
        )
    )


    split_data = (
        split_images(
            class_images
        )
    )


    copy_files(
        split_data
    )


    print(
        "\nCLEAN DATASET CREATION COMPLETE."
    )

    print(
        "\nNext step:"
    )

    print(
        "We will verify that there are "
        "ZERO cross-split duplicates."
    )