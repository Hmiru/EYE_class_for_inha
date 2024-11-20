import sqlite3
from datetime import datetime

import cv2
from calculation.focus_tracker import FocusTracker
from dlib import correlation_tracker, rectangle
from monitoring.prevent_to_go_out import AbsencePrevention
import time
class VideoProcessor:

    def __init__(self, video_capture_handler, 
                 face_detector, mouth_detector, eye_detector, yawn_detector, eye_closed_detector, 
                 absence_prevention, registered_students, skip_frames=40):
        self.video_capture_handler = video_capture_handler
        self.face_detector = face_detector
        self.mouth_detector = mouth_detector
        self.eye_detector = eye_detector
        self.yawn_detector = yawn_detector
        self.eye_closed_detector = eye_closed_detector
        self.absence_prevention = absence_prevention
        self.trackers = {}
        self.people_status = {}
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.focus_trackers = {}

        self.last_seen = {}
        self.registered_students = registered_students
        self.start_time = time.time()  # 출석 시작 시간 기록


        # Initialize all students as absent in the database
        # conn = sqlite3.connect("attendance.db")
        # cursor = conn.cursor()
        # for student_id in self.registered_students:
        #     cursor.execute('''
        #         INSERT OR REPLACE INTO attendance (student_id, status, time, last_seen_time, presence_status, recent_focus, cumulative_focus)
        #         VALUES (?, '결석', NULL, NULL, 'Absent', NULL, NULL)
        #     ''', (student_id,))
        # conn.commit()
        # conn.close()
        
    def process(self):
        """비디오 프레임을 처리하고, 얼굴과 하품을 탐지하는 메인 루프"""
        while True:
            frame = self.video_capture_handler.get_frame()
            if frame is None:
                break
            if self.frame_count % self.skip_frames != 0:
                self.frame_count += 1
                continue

            frame = cv2.resize(frame, (400, 300))  # 프레임 리사이즈
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환

            # 얼굴 바운딩 박스 탐지
            recognized_faces_info,absence_detected  = self.absence_prevention.absence_prevention_live(frame)
            print(recognized_faces_info)
            # 각 얼굴에 대해 하품 감지 및 상태 처리
            self._process_students(frame, gray, recognized_faces_info)

            # 처리된 프레임 표시
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture_handler.release()
        cv2.destroyAllWindows()


    def _update_status(self, status, metric_value, ratio_threshold, frame_threshold,
                       is_active_key, frame_count, action_count,  above_threshold=True):
        new_event_detected = False
        condition = metric_value > ratio_threshold if above_threshold else metric_value < ratio_threshold

        if condition:
            if not status[is_active_key]:
                status[frame_count] += 1
                if status[frame_count] >= frame_threshold:
                    status[is_active_key] = True
                    status[action_count] += 1
                    status[frame_count] = 0
                    new_event_detected = True
            else:
                status[frame_count] = 0
        else:
            status[frame_count] = 0
            status[is_active_key] = False
        return new_event_detected


    def _update_student_status(self, student_id, mar,ear,frame):
        if student_id not in self.people_status:
            self.people_status[student_id] = {
                'is_yawning': False, 'yawn_frame_count': 0, 'yawn_counter': 0,
                'is_eye_closed': False, 'eye_frame_count': 0, 'eye_closed_counter': 0
            }
        if student_id not in self.focus_trackers:
            self.focus_trackers[student_id] = FocusTracker(k_minutes=1, wY=1.0, wE=1.0)

        status = self.people_status[student_id]

        new_yawn_detected = self._update_status(
            status, mar, self.yawn_detector.yawn_threshold, self.yawn_detector.consecutive_frames,
            'is_yawning', 'yawn_frame_count', 'yawn_counter', above_threshold=True
        )

        # 눈 감김 상태 업데이트 (EAR은 임계값 이하 시 활성화)
        new_eye_closed_detected = self._update_status    (
            status, ear, self.eye_closed_detector.eye_threshold, self.eye_closed_detector.consecutive_frames,
            'is_eye_closed', 'eye_frame_count', 'eye_closed_counter', above_threshold=False
        )

        self.focus_trackers[student_id].update_focus(student_id, new_yawn_detected, new_eye_closed_detected)

        yawn_count=status['yawn_counter']
        eye_count=status['eye_closed_counter']

        focus_scores = self.focus_trackers[student_id].get_focus(student_id)
    def _process_students(self, frame, gray, student_data):
        # conn = sqlite3.connect("attendance.db")
        # cursor = conn.cursor()

        # current_time = time.time()

        # # Mark all students as absent initially
        # for student_id in self.registered_students:
        #     if student_id not in self.last_seen or current_time - self.last_seen[student_id] > 5:
        #         cursor.execute('''
        #             UPDATE attendance SET presence_status = 'Absent' WHERE student_id = ?
        #         ''', (student_id,))
                
        #         conn.commit()

        for face_info in student_data:
            student_id = face_info ["student_id"]
            bbox = face_info ["bbox"]
            x, y, x2, y2 = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            mouth_landmarks = self.mouth_detector.detect_mouth(gray, bbox)
            if mouth_landmarks is not None:
                mar = self.yawn_detector.calculate_mar(mouth_landmarks, frame)
                print(mar, student_id)

                
            eye_landmarks = self.eye_detector.detect_eyes(gray, bbox)
            if eye_landmarks is not None:
                left_eye, right_eye = eye_landmarks  # 좌, 우 눈 분리

                # 왼쪽 눈 EAR 계산 및 상태 업데이트
                left_ear = self.eye_closed_detector.calculate_ear(left_eye, frame)
                right_ear = self.eye_closed_detector.calculate_ear(right_eye, frame)

                average_ear = (left_ear + right_ear) / 2.0
                

            self._update_student_status(student_id, mar,average_ear, frame)

            yawn_count = self.people_status[student_id]['yawn_counter']
            eye_count = self.people_status[student_id]['eye_closed_counter']
            focus_scores = self.focus_trackers[student_id].get_focus(student_id)
            cumulative_focus = focus_scores["cumulative"]
            recent_focus = focus_scores["recent"]
            
            # Status update based on presence tracking
            # self.last_seen[student_id] = current_time
            # first_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # time_since_start = current_time - self.start_time

            # if time_since_start > 300:  # 5분이 지난 후 얼굴 인식 시 지각 처리
            #     status = "지각"
            # else:
            #     status = "출석"
  
            # last_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # presence_status = "Present"

            # cursor.execute('''
            #     INSERT OR REPLACE INTO attendance (student_id, status, time, last_seen_time, presence_status, recent_focus, cumulative_focus)
            #     VALUES (?, ?, COALESCE((SELECT time FROM attendance WHERE student_id = ?), ?), ?, ?, ?, ?)
            # ''', (student_id, status, student_id, first_seen_time, last_seen_time, presence_status, recent_focus,
            #       cumulative_focus))

            # conn.commit()
            # 학생 정보 출력
            self.display_student_info(frame, student_id, yawn_count, eye_count, cumulative_focus, recent_focus, bbox)
        #conn.close()
    def display_student_info(self, frame, student_id, yawn_count, eye_count, cumulative_focus, recent_focus, bbox):
        """학생의 ID와 상태 정보를 바운딩 박스 위에 출력하는 함수"""
        x, y, x2, y2 = bbox
        text_y_position = y - 60  # 바운딩 박스 위에 텍스트를 표시할 y 좌표를 충분히 올림

        # 텍스트 출력
        cv2.putText(frame, f"ID: {student_id}", (x, text_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Yawns: {yawn_count}", (x, text_y_position + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Eyes Closed: {eye_count}", (x, text_y_position + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Cumulative: {cumulative_focus:.1f} Recent: {recent_focus:.1f}",
                    (x, text_y_position + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
