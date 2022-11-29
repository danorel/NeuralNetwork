import math

from typing import List


def sigmoid(x_vector: List[float], weight_vector: List[float]) -> List[float]:
    return [1 / (1 + math.exp(-x * weight)) for x, weight in zip(x_vector, weight_vector)]


def softmax(vector: List[float]) -> int:
    return []
