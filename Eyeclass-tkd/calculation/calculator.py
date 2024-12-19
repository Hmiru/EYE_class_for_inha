from abc import ABC, abstractmethod

class Calculator(ABC):
    """
    Base class for all calculators.
    Provides a template for calculating scores based on certain parameters.
    """

    def __init__(self, weight_yawn=1.0):
        """
        Initialize with weights for yawn and closed eyes events.

        :param weight_yawn: Weight for yawn events.
        :param weight_closed_eyes: Weight for closed eyes events.
        """
        self.weight_yawn = weight_yawn


    @abstractmethod
    def calculate_score(self, yawn_count):
        """
        Abstract method to calculate score. Must be implemented by subclasses.

        :param yawn_count: Number of yawns detected.
        :param closed_eyes_count: Number of times eyes were closed.
        :return: Calculated score.
        """
        pass

    def _apply_weights(self, yawn_count):
        """
        Apply weights to the counts.

        :param yawn_count: Number of yawns.
        :param closed_eyes_count: Number of times eyes were closed.
        :return: Weighted sum of yawn and closed eyes counts.
        """
        return (self.weight_yawn * yawn_count)
