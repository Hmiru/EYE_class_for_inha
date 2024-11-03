import cv2
from EYE_class_for_inha.calculation.focus_tracker import FocusTracker
from dlib import correlation_tracker, rectangle

class VideoProcessor:

    def __init__(self, video_capture_handler, face_detector, mouth_detector, eye_detector, yawn_detector, eye_closed_detector, face_box_provider=None, skip_frames=2):
        self.video_capture_handler = video_capture_handler
        self.face_detector = face_detector
        self.mouth_detector = mouth_detector
        self.eye_detector = eye_detector
        self.yawn_detector = yawn_detector
        self.eye_closed_detector = eye_closed_detector
        self.face_box_provider = face_box_provider
        self.trackers = {}

        # 각 사람별로 하품 상태와 프레임 수를 저장하는 리스트
        self.people_status = {} # 사람별로 하품 상태와 프레임 수를 저장하는 dictionary

        self.skip_frames = skip_frames  # 프레임 스킵 주기
        self.frame_count = 0  # 현재 프레임 카운트
        self.focus_trackers = {}  # 사람별로 하품 상태를 추적하는 FocusTracker 객체
    def assign_id(self, faces_bounding_boxes, frame):
        """얼굴에 ID를 할당하고 correlation_tracker를 업데이트"""
        students_data = []
        assigned_ids = set()

        # 기존의 트래커로 현재 얼굴 위치와 일치하는지 확인
        for student_id, tracker in list(self.trackers.items()):
            position = tracker.get_position()
            t_x1, t_y1, t_x2, t_y2 = map(int, (position.left(), position.top(), position.right(), position.bottom()))

            for bbox in faces_bounding_boxes:
                x1, y1, x2, y2 = bbox
                if abs(t_x1 - x1) < 20 and abs(t_y1 - y1) < 20:
                    tracker.update(frame)  # 트래커 업데이트
                    students_data.append((student_id, bbox))
                    assigned_ids.add(student_id)
                    break

        # 새로운 얼굴 감지 시 ID 할당 및 트래커 추가
        next_id = max(self.trackers.keys(), default=0) + 1
        for bbox in faces_bounding_boxes:
            if len([data for data in students_data if data[1] == bbox]) == 0:
                x1, y1, x2, y2 = bbox
                tracker = correlation_tracker()
                tracker.start_track(frame, rectangle(x1, y1, x2, y2))
                self.trackers[next_id] = tracker
                students_data.append((next_id, bbox))
                next_id += 1

        # 사라진 얼굴 제거
        for student_id in list(self.trackers.keys()):
            if student_id not in assigned_ids:
                del self.trackers[student_id]

        return students_data

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
            if self.face_box_provider:
                faces_bounding_boxes = self.face_box_provider.get_face_boxes()
            else:
                faces_bounding_boxes = self.face_detector.detect_faces(gray)

            students_data=self.assign_id(faces_bounding_boxes, gray)
        
            # 현재 탐지된 얼굴 수와 상태 리스트를 동기화
            #self.sync_people_status(faces_bounding_boxes)

            # 각 얼굴에 대해 하품 감지 및 상태 처리
            self._process_students(frame, gray, students_data)

            # 처리된 프레임 표시
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture_handler.release()

    def sync_people_status(self, faces_bounding_boxes):
        """탐지된 얼굴 수에 맞게 상태 리스트를 동기화"""
        if len(self.people_status) != len(faces_bounding_boxes):
            # 사람 수에 맞게 상태 초기화
            self.people_status = [{'is_yawning': False, 'frame_count': 0, 'yawn_counter':0} for _ in faces_bounding_boxes]

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

        cumulative_focus = focus_scores["cumulative"]
        recent_focus = focus_scores["recent"]

        cv2.putText(frame, f"ID: {student_id} Yawn Count: {yawn_count} Eye Count: {eye_count}",
                    (10, 120 + (student_id % 5) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"ID: {student_id} Cumulative: {cumulative_focus:.1f} Recent: {recent_focus:.1f}",
                    (10, 180 + (student_id % 5) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def _process_students(self, frame, gray, student_data):
        for student_id, bbox in student_data:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            mouth_landmarks = self.mouth_detector.detect_mouth(gray, bbox)

            if mouth_landmarks is not None:

                mar = self.yawn_detector.calculate_mar(mouth_landmarks, frame)


            eye_landmarks = self.eye_detector.detect_eyes(gray, bbox)

            if eye_landmarks is not None:
                left_eye, right_eye = eye_landmarks  # 좌, 우 눈 분리

                # 왼쪽 눈 EAR 계산 및 상태 업데이트
                left_ear = self.eye_closed_detector.calculate_ear(left_eye, frame)
                right_ear = self.eye_closed_detector.calculate_ear(right_eye, frame)

                average_ear = (left_ear + right_ear) / 2.0
                print(average_ear)

            self._update_student_status(student_id, mar,average_ear, frame)

