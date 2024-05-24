from abc import ABC, abstractmethod
from typing import Any

# this is the class that all algorithms need to inherit
class BaseAlgorithm(ABC):
    # two attributes that each algorithm must have
    def __init__(self):
        self.y_true = None
        self.y_score = None

    #a method that each algorithm must have
    @abstractmethod
    def predict(self):
        pass