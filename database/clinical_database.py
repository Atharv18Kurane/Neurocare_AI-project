import os
import sqlite3
from datetime import datetime


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_PATH = os.path.join(
    BASE_DIR,
    "database",
    "neurocare.db"
)


# ============================================================
# CONNECTION
# ============================================================

def get_connection():

    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# INITIALIZE TABLES
# ============================================================

def initialize_clinical_database():

    connection = get_connection()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Patients
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_code TEXT UNIQUE NOT NULL,

            name TEXT NOT NULL,

            age INTEGER,

            gender TEXT,

            phone TEXT,

            created_at TEXT NOT NULL

        )
    """)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_id INTEGER NOT NULL,

            doctor_name TEXT,

            image_name TEXT,

            prediction TEXT NOT NULL,

            confidence REAL NOT NULL,

            non_demented REAL,

            very_mild_demented REAL,

            mild_demented REAL,

            moderate_demented REAL,

            created_at TEXT NOT NULL,

            FOREIGN KEY (
                patient_id
            )
            REFERENCES patients(id)
            ON DELETE CASCADE

        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# ADD PATIENT
# ============================================================

def add_patient(
    patient_code,
    name,
    age,
    gender,
    phone
):

    initialize_clinical_database()

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO patients
            (
                patient_code,
                name,
                age,
                gender,
                phone,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patient_code,
                name,
                age,
                gender,
                phone,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        connection.commit()

        patient_id = cursor.lastrowid

        return True, patient_id, "Patient added successfully."

    except sqlite3.IntegrityError:

        return False, None, "Patient code already exists."

    finally:

        connection.close()


# ============================================================
# GET ALL PATIENTS
# ============================================================

def get_patients():

    initialize_clinical_database()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            patient_code,
            name,
            age,
            gender,
            phone,
            created_at

        FROM patients

        ORDER BY id DESC
        """
    )

    patients = cursor.fetchall()

    connection.close()

    return patients


# ============================================================
# GET PATIENT
# ============================================================

def get_patient(
    patient_id
):

    initialize_clinical_database()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            patient_code,
            name,
            age,
            gender,
            phone,
            created_at

        FROM patients

        WHERE id = ?
        """,
        (
            patient_id,
        )
    )

    patient = cursor.fetchone()

    connection.close()

    return patient

# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    patient_id,
    doctor_name,
    image_name,
    prediction,
    confidence,
    probabilities,
    image_path=None,
    gradcam_path=None
):

    # Make sure new columns exist
    add_prediction_columns()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (
            patient_id,
            doctor_name,
            image_name,
            prediction,
            confidence,
            non_demented,
            very_mild_demented,
            mild_demented,
            moderate_demented,
            image_path,
            gradcam_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            doctor_name,
            image_name,
            prediction,
            confidence,

            probabilities.get(
                "NonDemented",
                0
            ),

            probabilities.get(
                "VeryMildDemented",
                0
            ),

            probabilities.get(
                "MildDemented",
                0
            ),

            probabilities.get(
                "ModerateDemented",
                0
            ),

            image_path,

            gradcam_path,

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id
# ============================================================
# GET PREDICTION HISTORY
# ============================================================

def get_prediction_history():

    initialize_clinical_database()

    add_prediction_columns()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT

            p.id,

            patients.patient_code,

            patients.name,

            p.doctor_name,

            p.image_name,

            p.prediction,

            p.confidence,

            p.non_demented,

            p.very_mild_demented,

            p.mild_demented,

            p.moderate_demented,

            p.image_path,

            p.gradcam_path,

            p.created_at

        FROM predictions p

        INNER JOIN patients

        ON p.patient_id = patients.id

        ORDER BY p.id DESC
        """
    )

    history = cursor.fetchall()

    connection.close()

    return history

    
def get_patient_statistics(patient_id):

    initialize_clinical_database()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*),
            MAX(created_at),
            MAX(prediction),
            MAX(confidence)

        FROM predictions

        WHERE patient_id = ?
        """,
        (patient_id,)
    )

    result = cursor.fetchone()

    connection.close()

    return result
def get_patient_predictions(patient_id):

    initialize_clinical_database()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            image_name,
            prediction,
            confidence,
            created_at

        FROM predictions

        WHERE patient_id = ?

        ORDER BY id DESC
        """,
        (patient_id,)
    )

    results = cursor.fetchall()

    connection.close()

    return results
def get_total_patients():

    initialize_clinical_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM patients"
    )

    result = cursor.fetchone()[0]

    connection.close()

    return result
def get_total_predictions():

    initialize_clinical_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    result = cursor.fetchone()[0]

    connection.close()

    return result
def get_today_predictions():

    initialize_clinical_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE DATE(created_at) = DATE('now')
        """
    )

    result = cursor.fetchone()[0]

    connection.close()

    return result
def get_recent_predictions(limit=10):

    initialize_clinical_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            p.patient_code,
            p.name,
            pr.prediction,
            pr.confidence,
            pr.created_at

        FROM predictions pr

        JOIN patients p
        ON pr.patient_id = p.id

        ORDER BY pr.id DESC

        LIMIT ?
        """,
        (limit,)
    )

    results = cursor.fetchall()

    connection.close()

    return results
def get_all_predictions():

    initialize_clinical_database()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            pr.id,
            p.patient_code,
            p.name,
            pr.image_name,
            pr.prediction,
            pr.confidence,
            pr.created_at

        FROM predictions pr

        JOIN patients p
        ON pr.patient_id = p.id

        ORDER BY pr.id DESC
        """
    )

    results = cursor.fetchall()

    connection.close()

    return results
def add_prediction_columns():

    initialize_clinical_database()

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # Add image_path
    # --------------------------------------------------------

    try:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN image_path TEXT
            """
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # Add gradcam_path
    # --------------------------------------------------------

    try:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN gradcam_path TEXT
            """
        )

    except Exception:

        pass


    connection.commit()
    connection.close()