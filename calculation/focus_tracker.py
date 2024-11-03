from .recent_calc import RecentCalculator
from .cumulative_calc import CumulativeCalculator

class FocusTracker:
    def __init__(self, k_minutes, wY=1.0, wE=1.0):
        self.k_minutes = k_minutes
        self.wY = wY
        self.wE = wE
        self.student_data = {}

    def update_focus(self, student_id, new_yawn_detected, new_eye_closed_detected):
        """
        Update focus scores based on yawning and eye-closing events for each student.
        """
        # 학생별로 cumulative_calculator와 recent_calculator 초기화
        if student_id not in self.student_data:
            self.student_data[student_id] = {
                "cumulative_calculator": CumulativeCalculator(w1=self.wY, w2=self.wE),
                "recent_calculator": RecentCalculator(k_minutes=self.k_minutes, w1=self.wY, w2=self.wE),
                "cumulative": 100,
                "recent": 100
            }

        yawn_count = 1 if new_yawn_detected else 0
        eye_close_count = 1 if new_eye_closed_detected else 0
        student = self.student_data[student_id]

        # 학생별로 집중도 점수 업데이트
        cumulative_focus = student["cumulative_calculator"].calculate_score(yawn_count, eye_close_count)
        student["recent_calculator"].update_events(yawn_count, eye_close_count)
        recent_focus = student["recent_calculator"].calculate_score()

        # 업데이트된 점수를 저장
        student["cumulative"] = cumulative_focus
        student["recent"] = recent_focus

    def get_focus(self, student_id):
        """
        Retrieve the cumulative and recent focus scores for the specified student.
        """
        return self.student_data.get(student_id, {"cumulative": 100, "recent": 100})
