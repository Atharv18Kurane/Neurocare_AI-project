import os
from PIL import Image

DATASET_PATH = r"dataset\raw"

CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def check_dataset():

    print("=" * 60)
    print("        NeuroCareAI 2.0 Dataset Checker")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print("\nERROR: Dataset folder not found:")
        print(DATASET_PATH)
        return

    total_images = 0
    corrupted_images = []

    print("\nClass Distribution")
    print("-" * 60)

    for class_name in CLASS_NAMES:

        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(class_path):
            print(f"{class_name:20s}: FOLDER NOT FOUND")
            continue

        images = [
            file for file in os.listdir(class_path)
            if file.lower().endswith(VALID_EXTENSIONS)
        ]

        print(f"{class_name:20s}: {len(images)} images")

        total_images += len(images)

        # Check corrupted images
        for image_name in images:

            image_path = os.path.join(class_path, image_name)

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except Exception:
                corrupted_images.append(image_path)

    print("-" * 60)
    print(f"Total images          : {total_images}")
    print(f"Corrupted images      : {len(corrupted_images)}")

    if corrupted_images:

        print("\nCorrupted files:")

        for image in corrupted_images:
            print(image)

    else:
        print("\nNo corrupted images found.")

    print("\nDataset checking completed.")
    print("=" * 60)


if __name__ == "__main__":
    check_dataset()