from typing import List

from common.neuron import Neuron


class NeuralList:
    def __init__(self,
                 hidden_neurons,
                 activation_function,
                 step_rate=1e3):
        self._hidden_neurons: int = hidden_neurons
        self._activation_function = activation_function
        self._step_rate = step_rate
        self._neurons: List[Neuron] = []

    def compile(self):
        for _ in range(self._hidden_neurons):
            neuron = Neuron()
            self._neurons.append(neuron)
        pass

    def fit(self, input_values: List[float], desired_values: List[float]):
        for i, input_value in enumerate(input_values):
            desired_value = desired_values[i]
            output_value = self._fit_value(input_value=input_value)
            self.__back_propagation(d=desired_value, o=output_value)
        pass

    def _fit_value(self, input_value: float):
        output_value = input_value
        for neuron in self._neurons:
            x, weight = neuron.activate(x=output_value), neuron.get_weight()
            output_value = self._activation_function(x=x, weight=weight)
        return output_value

    def predict(self, test_value: float):
        return self._fit_value(input_value=test_value)

    def __back_propagation(self, d: float, o: float):
        memory = d - o
        for neuron in reversed(self._neurons):
            x = neuron.get_last_active_x()
            dOdX = o * (1 - o)
            dXdW = x
            dOdW = dOdX * dXdW
            dPdW = memory * dOdW
            memory *= dOdW
            deltaW = self._step_rate * dPdW
            neuron.set_weight(neuron.get_weight() * deltaW)
        pass

    def __repr__(self):
        return f"NeuralList=[{[neuron for neuron in self._neurons]}]"


class ClassificationNeuralList(NeuralList):
    def __init__(self,
                 hidden_neurons,
                 activation_function,
                 classification_function,
                 step_rate=0.15):
        super(ClassificationNeuralList, self).__init__(
            hidden_neurons=hidden_neurons,
            activation_function=activation_function,
            step_rate=step_rate
        )
        self._classification_function = classification_function

    def predict(self, test_value: float):
        output_value = self._fit_value(input_value=test_value)
        return self._classification_function(z=output_value)

    def __repr__(self):
        return f"ClassificationNeuralList=[{[neuron for neuron in self._neurons]}]"
