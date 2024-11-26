from collections import deque
from datetime import datetime, timedelta
from .calculator import Calculator  # 추상 클래스 Calculator를 가져옴

class RecentCalculator(Calculator):  # Calculator 추상 클래스 상속
    def __init__(self, k_minutes, w1=1.0):
        super().__init__(w1)  # 부모 클래스의 초기화 호출
        self.k_minutes = k_minutes
        self.event_queue = deque()  # 제한 없이 deque 생성
        self.time_window = timedelta(minutes=k_minutes)  # 최근 K분 설정

    def update_events(self, yawn_count):
        """
        Updates recent events by adding the latest counts and keeping only the last K minutes of data.

        :param yawn_count: Yawn events in the current time period.
        :param closed_eye_count: Closed-eye events in the current time period.
        """
        now = datetime.now()
        self.event_queue.append({'yawn_count': yawn_count, 'timestamp': now})

        # K분이 넘는 이벤트 제거
        while self.event_queue and (now - self.event_queue[0]['timestamp']) > self.time_window:
            self.event_queue.popleft()

    def calculate_score(self):
        """
        Calculate recent focus score based on recent yawn and closed-eye events within the last K minutes.

        :return: Recent focus score.
        """
        total_recent_yawns = sum(event['yawn_count'] for event in self.event_queue)
        
        weighted_sum = self._apply_weights(total_recent_yawns)
        score = max(0, 100 - weighted_sum)  # Ensure score does not drop below 0
        return score
