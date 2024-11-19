import sqlite3
import pickle

def initialize_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS attendance")
    cursor.execute('''
        CREATE TABLE attendance (
            student_id TEXT PRIMARY KEY,
            status TEXT,
            time TEXT,
            last_seen_time TEXT,
            presence_status TEXT,
            recent_focus REAL,
            cumulative_focus REAL
        )
    ''')
    conn.commit()
    conn.close()

def load_registered_students(registered_faces_path):
    with open(registered_faces_path, 'rb') as f:
        registered_faces = pickle.load(f)
    return registered_faces.keys()
