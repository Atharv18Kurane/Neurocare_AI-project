import os
import sys
import csv

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# GRAD-CAM
# ============================================================

from explainability.gradcam import save_gradcam


# ============================================================
# PATHS
# ============================================================

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

CSV_PATH = os.path.join(
    REPORT_DIR,
    "error_analysis.csv"
)

GRADCAM_DIR = os.path.join(
    REPORT_DIR,
    "gradcam_errors"
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NeuroCareAI 2.0")
print("GRAD-CAM ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD ERROR CSV
# ============================================================

if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(
        f"Error report not found:\n{CSV_PATH}"
    )


errors = []

with open(
    CSV_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        row["confidence"] = float(
            row["confidence"]
        )

        row["actual_probability"] = float(
            row["actual_probability"]
        )

        errors.append(row)


print(
    f"\nTotal misclassified images: "
    f"{len(errors)}"
)


# ============================================================
# SELECT IMPORTANT ERROR GROUPS
# ============================================================

group_1 = [
    error
    for error in errors
    if (
        error["actual"] == "VeryMildDemented"
        and
        error["predicted"] == "NonDemented"
    )
]

group_2 = [
    error
    for error in errors
    if (
        error["actual"] == "NonDemented"
        and
        error["predicted"] == "VeryMildDemented"
    )
]


# ============================================================
# SORT BY MODEL CONFIDENCE
# ============================================================

group_1.sort(
    key=lambda x: x["confidence"],
    reverse=True
)

group_2.sort(
    key=lambda x: x["confidence"],
    reverse=True
)


# ============================================================
# TAKE TOP 5
# ============================================================

selected = []

for error in group_1[:5]:

    selected.append(
        (
            "VeryMild_to_Non",
            error
        )
    )


for error in group_2[:5]:

    selected.append(
        (
            "Non_to_VeryMild",
            error
        )
    )


print()
print(
    f"VeryMildDemented -> NonDemented: "
    f"{len(group_1)} errors"
)

print(
    f"NonDemented -> VeryMildDemented: "
    f"{len(group_2)} errors"
)

print(
    f"\nSelected for Grad-CAM: "
    f"{len(selected)}"
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

results = []


for number, (
    group_name,
    error
) in enumerate(
    selected,
    start=1
):

    image_path = error[
        "image_path"
    ]

    print()
    print("-" * 70)

    print(
        f"Case {number}/"
        f"{len(selected)}"
    )

    print(
        f"Actual     : "
        f"{error['actual']}"
    )

    print(
        f"Prediction : "
        f"{error['predicted']}"
    )

    print(
        f"Confidence : "
        f"{error['confidence'] * 100:.2f}%"
    )

    print(
        f"Image      : "
        f"{image_path}"
    )


    if not os.path.exists(
        image_path
    ):

        print(
            "ERROR: Image not found."
        )

        continue


    try:

        (
            prediction,
            confidence,
            probabilities,
            output_path
        ) = save_gradcam(
            image_path
        )


        # ----------------------------------------------------
        # Copy/reference output into error folder
        # ----------------------------------------------------

        base_name = os.path.splitext(
            os.path.basename(
                image_path
            )
        )[0]


        error_output_name = (
            f"{number:02d}_"
            f"{group_name}_"
            f"{base_name}_gradcam.png"
        )


        error_output_path = os.path.join(
            GRADCAM_DIR,
            error_output_name
        )


        # Copy generated Grad-CAM image

        import shutil

        shutil.copy2(
            output_path,
            error_output_path
        )


        results.append(
            {
                "case":
                    number,

                "group":
                    group_name,

                "filename":
                    os.path.basename(
                        image_path
                    ),

                "actual":
                    error["actual"],

                "prediction":
                    prediction,

                "confidence":
                    confidence,

                "original_confidence":
                    error["confidence"],

                "gradcam_path":
                    error_output_path
            }
        )


        print(
            "Grad-CAM generated successfully."
        )

        print(
            f"Saved: "
            f"{error_output_path}"
        )


    except Exception as e:

        print(
            "Grad-CAM FAILED:"
        )

        print(e)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    GRADCAM_DIR,
    "gradcam_error_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "NeuroCareAI 2.0 "
        "GRAD-CAM ERROR ANALYSIS\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    for result in results:

        file.write(
            f"Case {result['case']}\n"
        )

        file.write(
            f"Group      : "
            f"{result['group']}\n"
        )

        file.write(
            f"Filename   : "
            f"{result['filename']}\n"
        )

        file.write(
            f"Actual     : "
            f"{result['actual']}\n"
        )

        file.write(
            f"Prediction : "
            f"{result['prediction']}\n"
        )

        file.write(
            f"Confidence : "
            f"{result['confidence'] * 100:.2f}%\n"
        )

        file.write(
            f"Grad-CAM   : "
            f"{result['gradcam_path']}\n"
        )

        file.write(
            "\n"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("GRAD-CAM ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nGrad-CAM folder:"
)

print(
    GRADCAM_DIR
)

print(
    f"\nSuccessful Grad-CAM cases: "
    f"{len(results)}/{len(selected)}"
)

print(
    f"\nSummary:"
)

print(
    summary_path
)

print("=" * 70)