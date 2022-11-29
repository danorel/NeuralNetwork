from random import random


class Neuron:
    def __init__(self):
        self._weight: float = random()
        self._last_active_x = None

    def activate(self, x: float):
        self._last_active_x = x
        return self._weight * x

    def set_weight(self, weight: float):
        self._weight = weight

    def get_weight(self):
        return self._weight

    def get_last_active_x(self):
        return self._last_active_x

    def __repr__(self):
        return f"Neuron=[{self._weight}]"
