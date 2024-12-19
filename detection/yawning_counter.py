from detection.counter import Counter

class yawningCounter:
    def __init__(self):
        self.yawn_counter = Counter()

    def increment(self):
        self.yawn_counter.increment()

    def get_count(self):
        return self.yawn_counter.get_count()

    def reset(self):
        self.yawn_counter = Counter()