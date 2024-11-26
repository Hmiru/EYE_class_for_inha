from .calculator import Calculator

class CumulativeCalculator(Calculator):
    def __init__(self, w1=1.0):
        super().__init__()
        self.w1 = w1
        # 내부 누적 카운터 추가
        self.total_yawn_count = 0

    def calculate_score(self, yawn_count):
        # 전달된 값들을 내부 카운터에 누적
        self.total_yawn_count += yawn_count

        # 누적된 카운터를 이용해 가중 합 계산
        weighted_sum = self.w1 * self.total_yawn_count
        return max(0, 100 - weighted_sum)
