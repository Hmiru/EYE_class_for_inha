from .recent_calc import RecentCalculator
from .cumulative_calc import CumulativeCalculator

class FocusTracker:
    def __init__(self, k_minutes, wY=1.0):
        """
        Initialize FocusTracker for frame-wide analysis.
        """
        self.cumulative_calculator = CumulativeCalculator(w1=wY)
        self.recent_calculator = RecentCalculator(k_minutes=k_minutes, w1=wY)

        # Default scores for frame-level focus
        self.cumulative = 100
        self.recent = 100

    def update_focus(self, new_yawn_detected):
        yawn_count = 1 if new_yawn_detected else 0
  
        self.cumulative = self.cumulative_calculator.calculate_score(yawn_count)
  
        self.recent_calculator.update_events(yawn_count)
        self.recent = self.recent_calculator.calculate_score()
  


    def get_focus(self):
        """
        Retrieve the cumulative and recent focus scores for the entire frame.
        """
        return {
            "cumulative": self.cumulative,
            "recent": self.recent
        }
