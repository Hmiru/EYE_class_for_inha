from .recent_calc import RecentCalculator
from .cumulative_calc import CumulativeCalculator
from datetime import datetime, timedelta

class FocusTracker:
    def __init__(self, k_minutes, wY=1.0, wE=1.0):
        self.cumulative_calculator = CumulativeCalculator(w1=wY, w2=wE)
        self.recent_calculator = RecentCalculator(k_minutes=k_minutes, w1=wY, w2=wE)
        self.student_data = {}

    def update_focus(self, student_id, new_yawn_detected, eye_close_count):
        """
        Update focus scores based on yawning and eye-closing events for each student.
        """
        if student_id not in self.student_data:
            self.student_data[student_id] = {"cumulative": 100, "recent": 100}

        if new_yawn_detected:
            yawn_count = 1  # 새 하품을 탐지한 경우에만 1
        else:
            yawn_count = 0


        cumulative_focus = self.cumulative_calculator.calculate_score(yawn_count, eye_close_count)

        # Update recent events in RecentCalculator
        self.recent_calculator.update_events(yawn_count, eye_close_count)
        recent_focus = self.recent_calculator.calculate_score()

        # Update cumulative and recent focus in student data
        self.student_data[student_id]["cumulative"] = cumulative_focus
        self.student_data[student_id]["recent"] = recent_focus

    def get_focus(self, student_id):
        """
        Retrieve the cumulative and recent focus scores for the specified student.
        """
        return self.student_data.get(student_id, {"cumulative": 100, "recent": 100})
