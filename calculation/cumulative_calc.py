from .calculator import Calculator

class CumulativeCalculator(Calculator):
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.w1 = w1
        self.w2 = w2

    def calculate_score(self, yawn_count, eye_close_count):
        weighted_sum = self.w1 * yawn_count + self.w2 * eye_close_count
        return max(0, 100 - weighted_sum)

