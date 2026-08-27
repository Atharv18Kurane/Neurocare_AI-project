import hashlib
import sqlite3
import os


# Project root
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# SQLite database
DB_PATH = os.path.join(
    BASE_DIR,
    "database",
    "neurocare.db"
)


# ============================================================
# Password hashing
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# Create doctors table
# ============================================================

def initialize_auth_database():

    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


# ============================================================
# Create doctor account
# ============================================================

def create_doctor(
    name,
    email,
    password
):

    initialize_auth_database()

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO doctors
            (name, email, password)
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                hash_password(password)
            )
        )

        connection.commit()

        return True, "Doctor account created successfully."

    except sqlite3.IntegrityError:

        return False, "Email already exists."

    finally:

        connection.close()


# ============================================================
# Doctor Login
# ============================================================

def authenticate_doctor(
    email,
    password
):

    initialize_auth_database()

    connection = sqlite3.connect(
        DB_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, name, email
        FROM doctors
        WHERE email = ?
        AND password = ?
        """,
        (
            email,
            hash_password(password)
        )
    )

    doctor = cursor.fetchone()

    connection.close()

    return doctor