import sqlite3
import pickle
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

# 얼굴 정보 불러오는 함수
def load_registered_students(registered_faces_path):
    with open(registered_faces_path, 'rb') as f:
        registered_faces = pickle.load(f)
    return registered_faces.keys()

# 엑셀 파일로 저장하는 함수
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

