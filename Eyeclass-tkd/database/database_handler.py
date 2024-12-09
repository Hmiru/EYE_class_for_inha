import sqlite3
from datetime import datetime
import time

class DatabaseHandler:
    def __init__(self, db_path="attendance.db"):
        self.db_path = db_path
        self._initialize_attendance_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_attendance_db(self):
        """데이터베이스에 모든 학생을 결석으로 초기화"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                student_id TEXT PRIMARY KEY,
                status TEXT,
                time TEXT,
                last_seen_time TEXT,
                presence_status TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def initialize_students(self, registered_students):
        """모든 학생을 결석으로 초기화"""
        conn = self._get_connection()
        cursor = conn.cursor()
        for student_id in registered_students:
            cursor.execute('''
                INSERT OR REPLACE INTO attendance (student_id, status, time, last_seen_time, presence_status)
                VALUES (?, 'Absent', NULL, NULL, 'Absent')
            ''', (student_id,))
        conn.commit()
        conn.close()

    def mark_student_absent(self, student_id):
        """학생을 결석으로 마킹"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE attendance SET presence_status = 'Absent' WHERE student_id = ?
        ''', (student_id,))
        conn.commit()
        conn.close()

    def update_attendance_status(self, student_id, start_time):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_since_start = time.time() - start_time
        status = "Late" if time_since_start > 300 else "Present"
        presence_status = "Present"

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO attendance (student_id, status, time, last_seen_time, presence_status)
                VALUES (?, ?, COALESCE((SELECT time FROM attendance WHERE student_id = ?), ?), ?, ?)
            ''', (student_id, status, student_id, current_time, current_time, presence_status))
            conn.commit()
            conn.close()

            print(f"DB Updated: {student_id}, {status}")
        except sqlite3.Error as e:
            print(f"DB Error: {e}")
