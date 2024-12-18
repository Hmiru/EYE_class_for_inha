# video_processor.py
import sqlite3
from datetime import datetime

import cv2
from calculation.focus_tracker import FocusTracker
from dlib import correlation_tracker, rectangle
from monitoring.prevent_to_go_out import AbsencePrevention
import time
import os
import numpy as np
import queue
import threading
from video_processing.preprocessor import Preprocessor

class VideoProcessor:

    def __init__(self, video_capture_handler,
                 face_detector, mouth_detector, yawn_detector,
                 absence_prevention, registered_students, db_path, skip_frames=3):
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
        self.last_frame_time = None
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

    # def process(self):
    #     """비디오 프레임을 처리하고, 얼굴과 하품을 탐지하는 메인 루프"""
    #
    #     fixed_size = (480, 480)
    #     while True:
    #         frame = self.video_capture_handler.get_frame()
    #         if frame is None:
    #             break
    #         if self.frame_count % self.skip_frames != 0 and self.frame_count % 10 != 0:
    #             self.frame_count += 1
    #             continue
    #         frame = cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)  # 프레임 리사이즈
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# 그레이스케일 변환
    #         recognized_faces_info, _ = self.absence_prevention.absence_prevention_live(frame)
    #         self._absent_monitor(frame, gray, recognized_faces_info)
    #         self.absence_prevention.absence_prevention_live(frame)
    #         self._process_frame_for_yawns(frame, gray)
    #
    #         fps = self._calculate_fps()
    #
    #         # FPS 화면에 표시
    #         self.display_fps(frame, fps)
    #
    #         # 처리된 프레임 표시
    #         cv2.imshow("Face Detection", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #         self.frame_count += 1
    #
    #     print("Video processing complete.")
    #
    #     self.video_capture_handler.release()
    #
    #     cv2.destroyAllWindows()

    def process(self):
        """비디오 프레임을 처리하고, 얼굴과 하품을 탐지하며 DB와 연동"""
        fixed_size = (480, 480)  # YOLO 모델과 일치하는 입력 크기

        while True:
            frame = self.video_capture_handler.get_frame()
            if frame is None:
                break

            # 프레임 크기 조정
            frame_resized = cv2.resize(frame, fixed_size, interpolation=cv2.INTER_LINEAR)

            # YOLO를 통한 얼굴 탐지 및 식별 수행
            recognized_faces_info, detection_results = self.absence_prevention.absence_prevention_live(frame_resized)

            # DB 업데이트 (출석 및 상태 모니터링)
            self._absent_monitor(frame, None, recognized_faces_info)

            # ROI 영역에 대해 하품 탐지 수행
            if detection_results and hasattr(detection_results, 'boxes'):
                roi_boxes = detection_results.boxes.xyxy.cpu().numpy()
                self._process_roi(frame, roi_boxes, fixed_size)

            if self.frame_count % self.skip_frames == 0:  # skip_frames에 따라 주기적으로 호출
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._process_frame_for_yawns(frame, gray_frame)

            # FPS 계산 및 표시
            if self.frame_count % 40 == 0:  # 10프레임마다 FPS 계산
                fps = self._calculate_fps()
                self.display_fps(frame, fps)

            # 처리된 프레임 표시
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture_handler.release()
        cv2.destroyAllWindows()

    def _process_roi(self, frame, roi_boxes, input_size):
        """
        ROI 영역에서 탐지 및 상태 업데이트
        :param frame: 원본 프레임
        :param roi_boxes: YOLO 출력 바운딩 박스 (모델 입력 크기 기준)
        :param input_size: YOLO 모델 입력 크기 (예: 320x320 또는 480x480)
        """
        orig_h, orig_w = frame.shape[:2]  # 원본 프레임 크기
        scale_w, scale_h = orig_w / input_size[0], orig_h / input_size[1]  # 크기 비율 계산

        for bbox in roi_boxes:
            # YOLO 바운딩 박스를 원본 프레임 크기로 변환
            x1, y1, x2, y2 = (bbox[0] * scale_w, bbox[1] * scale_h,
                              bbox[2] * scale_w, bbox[3] * scale_h)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 정수로 변환

            # ROI 추출
            roi = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 디버깅용: 바운딩 박스 시각화
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 하품 탐지 수행
            mouth_landmarks = self.mouth_detector.detect_mouth(gray_roi, (0, 0, x2 - x1, y2 - y1))
            if mouth_landmarks is not None:
                mar = self.yawn_detector.calculate_mar(mouth_landmarks, roi)
                self._update_student_status(mar)

    def _calculate_fps(self):
        """FPS를 계산"""
        current_time = time.time()
        if self.last_frame_time is None:
            fps = 0
        else:
            fps = 1 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        return fps

    def display_fps(self, frame, fps):
        """FPS를 화면에 표시"""
        frame_height, frame_width = frame.shape[:2]
        position = (frame_width - 150, 30)  # 우측 상단 위치 (x: 너비 - 150, y: 30)
        cv2.putText(frame, f"FPS: {fps:.2f}", position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def _process_frame_for_yawns(self, frame, gray):
        # YOLO로 얼굴 바운딩 박스 탐지
        faces = self.face_detector.detect_faces(frame)

        # 이전 기록 유지: 얼굴이 없는 경우에도 사용
        if 'frame_status' in self.people_status:
            total_yawn_count = self.people_status['frame_status']['yawn_counter']
        else:
            total_yawn_count = 0

        if 'frame_focus' in self.focus_trackers:
            cumulative_focus = self.focus_trackers['frame_focus'].cumulative
            recent_focus = self.focus_trackers['frame_focus'].recent
        else:
            cumulative_focus = 100.0
            recent_focus = 100.0

        # 얼굴이 감지된 경우 ROI에서만 처리
        if faces:
            for idx, bbox in enumerate(faces):
                x1, y1, x2, y2 = bbox
                face_roi = frame[y1:y2, x1:x2]  # ROI: 얼굴 바운딩 박스 영역
                gray_roi = gray[y1:y2, x1:x2]  # ROI: 얼굴의 그레이스케일 영역

                # 바운딩 박스 그리기 (디버깅용)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ROI에서 landmarks 추출 및 MAR 계산
                mouth_landmarks = self.mouth_detector.detect_mouth(gray_roi, (0, 0, x2 - x1, y2 - y1))
                if mouth_landmarks is not None:
                    mar = self.yawn_detector.calculate_mar(mouth_landmarks, face_roi)

                    # 하품 상태 업데이트
                    stats = self._update_student_status(mar)
                    total_yawn_count = stats["total_yawns"]
                    cumulative_focus = stats["cumulative"]
                    recent_focus = stats["recent"]

        # 결과 텍스트를 화면에 표시
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

    def _absent_monitor(self, frame, gray, student_data):#출석, 지각, 결석 db
        conn = sqlite3.connect(self.db_path)  # DB 연결
        cursor = conn.cursor()

        current_time = time.time()

        # 얼굴이 감지되지 않은 학생은 Absent로 설정
        for student_id in self.registered_students:
            if student_id not in self.last_seen or current_time - self.last_seen[student_id] > 5:
                cursor.execute('''
                    UPDATE attendance SET presence_status = 'Absent' WHERE student_id = ?
                ''', (student_id,))
                conn.commit()

        # 얼굴이 감지된 학생은 출석으로 설정
        for face_info in student_data:
            student_id = face_info["student_id"]
            bbox = face_info["bbox"]

            # 마지막으로 얼굴이 감지된 시간 갱신
            self.last_seen[student_id] = current_time

            # 출석 정보 업데이트
            cursor.execute('''
                SELECT status, time FROM attendance WHERE student_id = ?
            ''', (student_id,))
            existing_record = cursor.fetchone()

            if existing_record is None or existing_record[0] == '결석':
                first_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                time_since_start = current_time - self.start_time

                # 5분 이후 얼굴이 감지된 경우 지각 처리
                if time_since_start <= 10:
                    status = "출석"
                else:
                    status = "지각"
            else:
                status = existing_record[0]
                first_seen_time = existing_record[1]  # 기존의 처음 인식된 시간 유지

            last_seen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            presence_status = "Present"

            cursor.execute('''
                INSERT OR REPLACE INTO attendance (student_id, status, time, last_seen_time, presence_status)
                VALUES (?, ?, ?, ?, ?)
            ''', (student_id, status, first_seen_time, last_seen_time, presence_status))
            conn.commit()

            # 디버깅용 출력
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