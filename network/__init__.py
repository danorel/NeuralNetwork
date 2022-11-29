import math

from typing import List
from common.neuron import Neuron


def performance_function(d_vector: List[float], o_vector: List[float]):
    if len(d_vector) != len(o_vector):
        pass
    diff_vector = [-math.pow(d_vector[i] - o_vector[i], 2) for i, _ in enumerate(d_vector)]
    diff_sum = 0
    for diff_value in diff_vector:
        diff_sum += diff_value
    P = - diff_sum / len(diff_vector)
    return P


class NeuralLayer:
    def __init__(self, size: int):
        self._neurons = [Neuron() for _ in range(size)]

    def activate(self, vector: List[float]) -> List[float]:
        if len(vector) != len(self._neurons):
            raise RuntimeError("Dimension error")
        return [neuron.activate(x=vector[i]) for i, neuron in enumerate(self._neurons)]

    def weights(self):
        return [neuron.get_weight() for neuron in self._neurons]

    def x(self):
        return [neuron.get_last_active_x() for neuron in self._neurons]

    def neurons(self):
        return self._neurons

    def __repr__(self):
        return f"NeuralLayer=[{[neuron for neuron in self._neurons]}]"


class NeuralNetwork:
    def __init__(self,
                 hidden_layers,
                 activation_function,
                 step_rate=0.15):
        self._hidden_layers: int = hidden_layers
        self._activation_function = activation_function
        self._step_rate = step_rate
        self._neural_layers: List[NeuralLayer] = []

    def compile(self, size: int):
        for _ in range(self._hidden_layers):
            neural_layer = NeuralLayer(size=size)
            self._neural_layers.append(neural_layer)
        pass

    def fit(self, input_vectors: List[List[float]], desired_vector: List[float]):
        for input_vector in input_vectors:
            output_vector = self._fit_vector(input_vector=input_vector)
            self.__back_propagation(d_vector=desired_vector, o_vector=output_vector)
        pass

    def _fit_vector(self, input_vector: List[float]):
        output_vector = input_vector
        for neural_layer in self._neural_layers:
            x_vector, weight_vector = neural_layer.activate(vector=output_vector), neural_layer.weights()
            output_vector = self._activation_function(x_vector=x_vector, weight_vector=weight_vector)
        return output_vector

    def predict(self, test_vector: List[float]):
        return self._fit_vector(input_vector=test_vector)

    def __back_propagation(self, d_vector: List[float], o_vector: List[float]):
        d = 0
        for d_value in d_vector:
            d += d_value

        o = 0
        for o_value in o_vector:
            o += o_value

        memory = d - o

        for neural_layer in reversed(self._neural_layers):
            neurons = neural_layer.neurons()

            x_vector = neural_layer.x()
            x = 0
            for x_value in x_vector:
                x += x_value

            dOdX = o * (1 - o)
            dXdW = x
            dOdW = dOdX * dXdW
            dPdW = memory * dOdW
            memory *= dOdW

            print(dOdX, dXdW, dOdW)
            deltaW = self._step_rate * dPdW

            for neuron in neurons:
                neuron.set_weight(neuron.get_weight() * deltaW)

        pass

    def __repr__(self):
        return f"NeuralNetwork=[{[neural_layer for neural_layer in self._neural_layers]}]"


class ClassificationNeuralNetwork(NeuralNetwork):
    def __init__(self,
                 hidden_layers,
                 activation_function,
                 classification_function,
                 step_rate=0.15):
        super(ClassificationNeuralNetwork, self).__init__(
            hidden_layers=hidden_layers,
            activation_function=activation_function,
            step_rate=step_rate
        )
        self._classification_function = classification_function

    def predict(self, test_vector: List[float]):
        output_vector = self._fit_vector(input_vector=test_vector)
        output_accumulation = 0
        for output_value in output_vector:
            output_accumulation += output_value
        print(output_accumulation)
        return self._classification_function(vector=output_vector)

    def __repr__(self):
        return f"ClassificationNeuralNetwork=[{[neural_layer for neural_layer in self._neural_layers]}]"
