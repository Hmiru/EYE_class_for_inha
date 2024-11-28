# video_processor.py
import sqlite3
from datetime import datetime

import cv2
from calculation.focus_tracker import FocusTracker
from dlib import correlation_tracker, rectangle
from monitoring.prevent_to_go_out import AbsencePrevention
import time
import os


class VideoProcessor:

    def __init__(self, video_capture_handler,
                 face_detector, mouth_detector, yawn_detector,
                 absence_prevention, registered_students, db_path, skip_frames=1):
        self.video_capture_handler = video_capture_handler
        self.face_detector = face_detector
        self.mouth_detector = mouth_detector
        self.yawn_detector = yawn_detector

        self.absence_prevention = absence_prevention
        self.trackers = {}
        self.people_status = {}
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.focus_trackers = {}

        self.last_seen = {}
        self.registered_students = registered_students
        self.start_time = time.time()  # 출석 시작 시간 기록
        self.db_path = os.path.abspath(db_path or "attendance.db")
        if self.db_path:  # db_path가 있는 경우에만 DB 초기화
            self._initialize_database()

    def _initialize_database(self):
        """DB를 초기화하고 학생 정보를 설정"""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        for student_id in self.registered_students:
            cursor.execute('''
                INSERT OR REPLACE INTO attendance (student_id, status, time, last_seen_time, presence_status)
                VALUES (?, '결석', NULL, NULL, 'Absent')
            ''', (student_id,))
        conn.commit()
        conn.close()

    def process(self):
        """비디오 프레임을 처리하고, 얼굴과 하품을 탐지하는 메인 루프"""
        while True:
            frame = self.video_capture_handler.get_frame()
            if frame is None:
                break
            if self.frame_count % self.skip_frames != 0 and self.frame_count % 5 != 0:
                self.frame_count += 1
                continue
            frame = cv2.resize(frame, (640, 480))  # 프레임 리사이즈
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
            self.absence_prevention.absence_prevention_live(frame)
            self._process_frame_for_yawns(frame, gray)

            # 처리된 프레임 표시
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        print("Video processing complete.")

        self.video_capture_handler.release()

        cv2.destroyAllWindows()

    def _process_frame_for_yawns(self, frame, gray):
        faces = self.face_detector.detect_faces(frame)  # 얼굴 바운딩 박스 탐지

        # 이전 기록 유지: 얼굴이 없는 경우에도 사용
        if 'frame_status' in self.people_status:
            total_yawn_count = self.people_status['frame_status']['yawn_counter']
        else:
            # 초기값 설정
            total_yawn_count = 0

        if 'frame_focus' in self.focus_trackers:
            cumulative_focus = self.focus_trackers['frame_focus'].cumulative
            recent_focus = self.focus_trackers['frame_focus'].recent
        else:
            # 초기값 설정
            cumulative_focus = 100.0
            recent_focus = 100.0

        if faces:  # 얼굴이 감지된 경우
            for idx, bbox in enumerate(faces):
                x, y, x2, y2 = bbox
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                mouth_landmarks = self.mouth_detector.detect_mouth(gray, bbox)
                if mouth_landmarks is not None:
                    mar = self.yawn_detector.calculate_mar(mouth_landmarks, frame)

                    stats = self._update_student_status(mar)  # 통계 정보 반환

                    total_yawn_count = stats["total_yawns"]
                    cumulative_focus = stats["cumulative"]
                    recent_focus = stats["recent"]

        self.display_student_info(frame, total_yawn_count, cumulative_focus, recent_focus)

    def _update_student_status(self, mar):
        if 'frame_status' not in self.people_status:
            self.people_status['frame_status'] = {
                'is_yawning': False, 'yawn_frame_count': 0, 'yawn_counter': 0
            }
        if 'frame_focus' not in self.focus_trackers:
            self.focus_trackers['frame_focus'] = FocusTracker(k_minutes=1, wY=1.0)

        status = self.people_status['frame_status']

        new_yawn_detected = False
        condition = mar > self.yawn_detector.yawn_threshold

        if condition:
            if not status['is_yawning']:
                status['yawn_frame_count'] += 1
                if status['yawn_frame_count'] >= self.yawn_detector.consecutive_frames:
                    status['is_yawning'] = True
                    status['yawn_counter'] += 1
                    status['yawn_frame_count'] = 0
                    new_yawn_detected = True
            else:
                status['yawn_frame_count'] = 0
        else:
            status['yawn_frame_count'] = 0
            status['is_yawning'] = False

        self.focus_trackers['frame_focus'].update_focus(new_yawn_detected)

        # 프레임 내 전체 통계 반환
        yawn_count = status['yawn_counter']
        focus_scores = self.focus_trackers['frame_focus'].get_focus()
        cumulative_focus = focus_scores["cumulative"]
        recent_focus = focus_scores["recent"]
        return {
            "total_yawns": yawn_count,
            "cumulative": cumulative_focus,
            "recent": recent_focus
        }

    def _absent_monitor(self, frame, gray, student_data):
        conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()

        current_time = time.time()

        for student_id in self.registered_students:
            if student_id not in self.last_seen or current_time - self.last_seen[student_id] > 5:
                cursor.execute('''
                    UPDATE attendance SET presence_status = 'Absent' WHERE student_id = ?
                ''', (student_id,))

                conn.commit()

        for face_info in student_data:
            student_id = face_info["student_id"]
            bbox = face_info["bbox"]
            x, y, x2, y2 = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)


                # Status update based on presence tracking
            self.last_seen[student_id] = current_time
            first_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_since_start = current_time - self.start_time

            if time_since_start > 300:  # 5분이 지난 후 얼굴 인식 시 지각 처리
                status = "지각"
            else:
                status = "출석"

            last_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            presence_status = "Present"

            cursor.execute('''
                INSERT OR REPLACE INTO attendance (student_id, status, time, last_seen_time, presence_status)
                VALUES (?, ?, COALESCE((SELECT time FROM attendance WHERE student_id = ?), ?), ?, ?)
            ''', (student_id, status, student_id, first_seen_time, last_seen_time, presence_status))

            conn.commit()
            print(f"Updating DB for student {student_id}: {status}, {last_seen_time}")
        conn.close()

    def display_student_info(self, frame, total_yawns, cumulative_focus, recent_focus):
        """학생의 ID와 상태 정보를 바운딩 박스 위에 출력하는 함수"""
        text_y_position = 30
        # 텍스트 출력
        cv2.putText(frame, f"Total Yawns: {total_yawns}", (10, text_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Cumulative Focus: {cumulative_focus:.1f}%", (10, text_y_position + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Recent Focus: {recent_focus:.1f}%", (10, text_y_position + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)