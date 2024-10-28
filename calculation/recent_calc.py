from .calculator import Calculator

class RecentCalculator(Calculator):

    def __init__(self, k_minutes, w1=1.0, w2=1.0):
        super().__init__()
        self.k_minutes = k_minutes
        self.weight_yawn = w1
        self.weight_closed_eyes = w2
        self.recent_yawn_events = []
        self.recent_closed_eye_events = []

    def update_events(self, yawn_count, closed_eye_count):
        """
        Updates recent events by adding the latest counts and keeping only the last K minutes of data.

        :param yawn_count: Yawn events in the current minute.
        :param closed_eye_count: Closed-eye events in the current minute.
        """
        self.recent_yawn_events.append(yawn_count)
        self.recent_closed_eye_events.append(closed_eye_count)

        # Remove the oldest entry if it exceeds the K-minute window
        if len(self.recent_yawn_events) > self.k_minutes:
            self.recent_yawn_events.pop(0)
        if len(self.recent_closed_eye_events) > self.k_minutes:
            self.recent_closed_eye_events.pop(0)

    def calculate_score(self):
        """
        Calculate recent focus score based on recent yawn and closed-eye events within the last K minutes.

        :return: Recent focus score.
        """
        total_recent_yawns = sum(self.recent_yawn_events)
        total_recent_closed_eyes = sum(self.recent_closed_eye_events)

        weighted_sum = self._apply_weights(total_recent_yawns, total_recent_closed_eyes)
        score = max(0, 100 - weighted_sum)  # Ensure score does not drop below 0
        return score
