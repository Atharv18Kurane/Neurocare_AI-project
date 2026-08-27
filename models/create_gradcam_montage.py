import os
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

GRADCAM_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "gradcam_errors"
)

OUTPUT_PATH = os.path.join(
    GRADCAM_DIR,
    "gradcam_error_montage.png"
)

files = sorted(
    [
        f for f in os.listdir(GRADCAM_DIR)
        if f.lower().endswith(".png")
        and "gradcam" in f.lower()
        and "montage" not in f.lower()
    ]
)

if not files:
    raise FileNotFoundError(
        "No Grad-CAM images found."
    )

images = []

for filename in files:

    path = os.path.join(
        GRADCAM_DIR,
        filename
    )

    image = Image.open(path).convert("RGB")

    image.thumbnail(
        (700, 500)
    )

    canvas = Image.new(
        "RGB",
        (720, 550),
        "white"
    )

    x = (720 - image.width) // 2
    y = 35

    canvas.paste(
        image,
        (x, y)
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 10),
        filename,
        fill="black"
    )

    images.append(canvas)


columns = 2
rows = (len(images) + columns - 1) // columns

montage = Image.new(
    "RGB",
    (
        columns * 720,
        rows * 550
    ),
    "white"
)

for index, image in enumerate(images):

    x = (index % columns) * 720
    y = (index // columns) * 550

    montage.paste(
        image,
        (x, y)
    )

montage.save(
    OUTPUT_PATH,
    quality=95
)

print("=" * 70)
print("GRAD-CAM MONTAGE CREATED")
print("=" * 70)
print(f"Images: {len(images)}")
print(f"Saved: {OUTPUT_PATH}")