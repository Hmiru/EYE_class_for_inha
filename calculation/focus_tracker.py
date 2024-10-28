
from .recent_calc import RecentCalculator
from .cumulative_calc import CumulativeCalculator

class FocusTracker:
    def __init__(self, k_minutes, wY=1.0, wE=1.0):
        self.cumulative_calculator = CumulativeCalculator(w1=wY, w2=wE)
        self.recent_calculator = RecentCalculator(k_minutes=k_minutes, w1=wY, w2=wE)
        self.student_data = {}

    def update_focus(self, student_id, yawn_count, eye_close_count):
        """
        각 학생의 하품 및 눈 감음 횟수를 받아와 집중도를 업데이트합니다.
        """
        if student_id not in self.student_data:
            self.student_data[student_id] = {"cumulative": 100, "recent": 100}

        # 누적 집중도 업데이트
        cumulative_focus = self.cumulative_calculator.calculate_score(yawn_count, eye_close_count)
        # 최근 집중도 업데이트
        self.recent_calculator.update_events(yawn_count, eye_close_count)
        recent_focus = self.recent_calculator.calculate_score()

        self.student_data[student_id]["cumulative"] = cumulative_focus
        self.student_data[student_id]["recent"] = recent_focus

        def get_focus(self, student_id):
            """
            Retrieves the focus scores for a given student.
            """
            return self.student_data.get(student_id, {"cumulative": 100, "recent": 100})

