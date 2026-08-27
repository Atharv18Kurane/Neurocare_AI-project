# NeuroCareAI 2.0

NeuroCareAI 2.0 is a Streamlit application for Alzheimer MRI image analysis. It provides an authenticated workspace for doctors to manage patients, run MRI predictions, inspect Grad-CAM explanations, review prediction history, and view model performance.

The prediction model uses EfficientNetV2-B3 with CORAL ordinal classification and supports four ordered classes:

- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

This project is for research and educational use. It is not a medical device and must not be used as a substitute for qualified clinical assessment.

## Requirements

- Python 3.10 or newer
- TensorFlow-compatible hardware and installation
- The dependencies listed in `requirements.txt`
- A trained model at `saved_models/best_model.keras` to use prediction and Grad-CAM features

## Installation

From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Training scripts expect the cleaned dataset in this structure:

```text
dataset/
	clean/
		train/
			NonDemented/
			VeryMildDemented/
			MildDemented/
			ModerateDemented/
		validation/
			...
		test/
			...
```

The raw dataset and generated datasets are intentionally excluded from Git by `.gitignore`. Use the preprocessing scripts in `preprocessing/` to create or split local datasets when needed.

## Run the application

Start the Streamlit app from the project root:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The app creates its SQLite authentication and clinical data database files under `database/` on first use.

### Create a doctor account

There is currently no signup page in the Streamlit navigation. Create the first account from a Python prompt:

```powershell
python -c "from auth.authentication import create_doctor; print(create_doctor('Example Doctor', 'doctor@example.com', 'change-this-password'))"
```

Use the resulting email and password on the login page. Do not commit real credentials or generated database files.

## Train and evaluate

To train using the cleaned dataset:

```powershell
python models/train_clean.py
```

The training configuration uses EfficientNetV2-B3, 300x300 RGB images, CORAL loss, and two training stages. Models are written to `saved_models/` and training reports to `reports/`.

Additional scripts are available for evaluation and analysis:

```powershell
python models/evaluate.py
python models/evaluate_visuals.py
python models/error_analysis.py
python models/plot_results.py
```

Run Grad-CAM utilities only after a compatible trained model is available:

```powershell
python explainability/test_gradcam.py
```

## Main application areas

- **Home:** application overview and navigation
- **Patients:** patient records and clinical information
- **Dashboard:** summary metrics
- **Predict:** upload an MRI image and view class probabilities
- **GradCAM:** inspect model attention for an image
- **Performance:** review evaluation results
- **History:** review stored predictions
- **About:** project information

## Project structure

```text
app.py                 Streamlit entry point
login.py               Doctor login page
auth/                  Authentication and account helpers
config/                Paths and model/training configuration
database/              SQLite clinical data helpers
models/                Model, training, prediction, and evaluation code
pages/                 Streamlit application pages
preprocessing/         Dataset preparation scripts
explainability/        Grad-CAM implementation and checks
data/                  Local MRI and Grad-CAM artifacts
dataset/               Local raw, clean, and processed datasets
saved_models/          Local trained model files
reports/               Local generated reports
```

## Security and data handling

- Keep patient data, MRI images, credentials, model files, and generated reports out of version control.
- Use strong, unique passwords for doctor accounts.
- Review local database and upload storage permissions before using real patient data.
- The current authentication helper uses SHA-256 password hashing; production deployments should use a password hashing scheme designed for credentials, such as Argon2 or bcrypt.
