import cv2
from EYE_class_for_inha.calculation.focus_tracker import FocusTracker



class VideoProcessor:
    def __init__(self, video_capture_handler, face_detector, mouth_detector, yawn_detector, yawn_counter,
                 face_box_provider=None, skip_frames=2):
        self.video_capture_handler = video_capture_handler
        self.face_detector = face_detector
        self.mouth_detector = mouth_detector
        self.yawn_detector = yawn_detector
        self.yawn_counter = yawn_counter
        self.face_box_provider = face_box_provider

        # 각 사람별로 하품 상태와 프레임 수를 저장하는 리스트
        self.people_status = {} # 사람별로 하품 상태와 프레임 수를 저장하는 dictionary

        self.skip_frames = skip_frames  # 프레임 스킵 주기
        self.frame_count = 0  # 현재 프레임 카운트
        self.focus_tracker = FocusTracker(k_minutes=1, wY=1.0, wE=1.0)  # FocusTracker 초기화, k_minutes는 최근 집중도 기준 시간

    def assign_id(self, faces_bounding_boxes):
        """얼굴에 ID를 할당"""
        stuents_data = []
        for bbox in faces_bounding_boxes:
            x1, y1, x2, y2 = bbox
            face_center_x=(x1+x2)/2
            if face_center_x <400:
                student_id=1
            else:
                student_id=2
            stuents_data.append((student_id, bbox))
        return stuents_data

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

            students_data=self.assign_id(faces_bounding_boxes)
        
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

    # def process_faces(self, frame, gray, faces_bounding_boxes):
    #     """각 얼굴에 대해 하품을 감지하고 상태를 업데이트"""
    #     for i, rect in enumerate(faces_bounding_boxes):
    #         x1, y1, x2, y2 = rect
    #
    #         # 얼굴 바운딩 박스 그리기
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #         # 입 검출 및 하품 감지
    #         mouth_box = self.mouth_detector.detect_mouth(gray, rect)
    #         if mouth_box is not None and len(mouth_box) > 0:
    #             yawn_detected, mar = self.yawn_detector.detect_yawn(gray, mouth_box, frame)
    #
    #             # MAR 값 표시
    #             mar_text = f"MAR: {mar:.2f}"
    #             cv2.putText(frame, mar_text, tuple(mouth_box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #
    #             # 사람별 하품 상태 업데이트
    #             self.update_person_status(i, yawn_detected,frame)

            # 하품 횟수 화면에 표시
    #
    # def update_person_status(self, i, yawn_detected,frame):
    #     """사람별 하품 상태를 업데이트"""
    #     person_status = self.people_status[i]
    #
    #     if yawn_detected and not person_status['is_yawning']:
    #         self.yawn_counter.increment()  # 하품 카운트 증가
    #         person_status['is_yawning'] = True  # 하품 상태로 변경
    #         person_status['frame_count'] = 0  # 프레임 카운트 초기화
    #         person_status['yawn_counter'] += 1
    #     elif not yawn_detected and person_status['is_yawning']:
    #         person_status['is_yawning'] = False  # 하품 종료
    #
    #     yawn_count = person_status['yawn_counter']
    #     cv2.putText(frame, f"Yawn Count: {yawn_count}", (10, 60 + i * 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    def _update_student_status(self, student_id, mar,frame):
        if student_id not in self.people_status:
            self.people_status[student_id] = {'is_yawning': False, 'frame_count': 0, 'yawn_counter': 0}

        status = self.people_status[student_id]

        if mar > self.yawn_detector.yawn_threshold:
            if not status['is_yawning']:
                # 연속된 프레임 기준 충족 시 하품 상태로 전환
                status['frame_count'] += 1
                if status['frame_count'] >= self.yawn_detector.consecutive_frames:
                    self.yawn_counter.increment()
                    status['is_yawning'] = True
                    status['yawn_counter'] += 1
                    status['frame_count'] = 0  # 초기화
            else:
                # 입을 계속 벌리고 있는 상태에서는 frame_count 증가하지 않음
                status['frame_count'] = 0
        else:
            # 입을 다물면 하품 상태 해제
            status['frame_count'] = 0
            status['is_yawning'] = False

        yawn_count=status['yawn_counter']
        self.focus_tracker.update_focus(student_id, yawn_count, 0)  # 눈 감음 데이터는 0으로 가정
        focus_scores = self.focus_tracker.get_focus(student_id)

        cumulative_focus = focus_scores["cumulative"]
        recent_focus = focus_scores["recent"]

        cv2.putText(frame, f"ID: {student_id} Yawn Count: {yawn_count}",
                    (10, 60 + (student_id % 5) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {student_id} Cumulative: {cumulative_focus:.1f} Recent: {recent_focus:.1f}",
                    (10, 90 + (student_id % 5) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def _process_students(self, frame, gray, student_data):
        for student_id, bbox in student_data:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mouth_landmarks = self.mouth_detector.detect_mouth(gray, bbox)

            if mouth_landmarks is not None:
                mar = self.yawn_detector.calculate_mar(mouth_landmarks, frame)
                self._update_student_status(student_id, mar, frame)
