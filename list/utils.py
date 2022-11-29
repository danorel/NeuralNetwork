import math


def sigmoid(x: float, weight: float) -> float:
    return 1 / (1 + math.exp(-x * weight))


def softmax(z: float) -> int:
    return 0
