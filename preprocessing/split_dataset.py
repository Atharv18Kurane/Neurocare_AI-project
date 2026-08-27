import os
import shutil
import random

SOURCE_DIR = r"dataset\raw"
OUTPUT_DIR = r"dataset\processed"

CLASSES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def split_class(class_name):

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    if not os.path.exists(source_class_dir):
        print(f"ERROR: Folder not found: {source_class_dir}")
        return

    images = [
        file for file in os.listdir(source_class_dir)
        if file.lower().endswith(VALID_EXTENSIONS)
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images
    }

    print(f"\n{class_name}")
    print("-" * 50)
    print(f"Total      : {total}")
    print(f"Training   : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Testing    : {len(test_images)}")

    for split_name, split_images in splits.items():

        destination_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        create_directory(destination_dir)

        for image_name in split_images:

            source_path = os.path.join(
                source_class_dir,
                image_name
            )

            destination_path = os.path.join(
                destination_dir,
                image_name
            )

            shutil.copy2(
                source_path,
                destination_path
            )


def main():

    print("=" * 60)
    print("       NeuroCareAI 2.0 Dataset Splitter")
    print("=" * 60)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nSplit ratio:")
    print("Training   : 70%")
    print("Validation : 15%")
    print("Testing    : 15%")

    random.seed(SEED)

    create_directory(OUTPUT_DIR)

    for class_name in CLASSES:
        split_class(class_name)

    print("\n" + "=" * 60)
    print("Dataset splitting completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()