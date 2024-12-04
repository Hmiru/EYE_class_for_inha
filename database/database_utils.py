import sqlite3
import pickle
import time
from datetime import datetime
import pandas as pd

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
            presence_status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_registered_students(registered_faces_path):
    with open(registered_faces_path, 'rb') as f:
        registered_faces = pickle.load(f)
    return registered_faces.keys()
def fetch_data():
    """출석 데이터를 SQLite에서 읽어오기"""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # 데이터 가져오기
    cursor.execute("SELECT * FROM attendance")
    rows = cursor.fetchall()

    # 컬럼 이름 가져오기
    columns = [description[0] for description in cursor.description]

    # 데이터베이스 연결 닫기
    conn.close()

    # pandas DataFrame으로 변환
    df = pd.DataFrame(rows, columns=columns)
    return df

def export_db_to_excel():
    # 현재 날짜와 시간을 파일 이름에 포함
    now = datetime.now()
    file_name = now.strftime("attendance_%Y%m%d_%H%M%S.xlsx")

    # SQLite 데이터베이스 연결
    conn = sqlite3.connect("attendance.db")

    # 데이터베이스 테이블을 pandas DataFrame으로 변환
    df = pd.read_sql_query("SELECT * FROM attendance", conn)

    # 엑셀 파일로 저장
    df.to_excel(file_name, index=False)

    # 데이터베이스 연결 닫기
    conn.close()

    print(f"출석 정보가 엑셀 파일로 저장되었습니다: {file_name}")

