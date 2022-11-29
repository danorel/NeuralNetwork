from common.domain import processed_data
from list import NeuralList, ClassificationNeuralList
from list.utils import sigmoid, softmax


def main():
    input_values = list[float](map(lambda processed_record: processed_record[0], processed_data))
    desired_values = list[float](map(lambda processed_record: processed_record[1], processed_data))
    run_NeuralList(input_values,
                   desired_values,
                   hidden_neurons=1)
    run_ClassificationNeuralList(input_values,
                                 desired_values,
                                 hidden_neurons=1)
    pass


def run_NeuralList(
        input_values: list[float],
        desired_values: list[float],
        hidden_neurons: int
):
    nn = NeuralList(
        hidden_neurons=hidden_neurons,
        activation_function=sigmoid
    )
    nn.compile()
    print(f"Initialized neural list with {hidden_neurons} hidden neurons: {nn}")
    nn.fit(
        input_values=input_values,
        desired_values=desired_values
    )
    print(f"Trained neural list with {hidden_neurons} hidden neurons: {nn}")
    print(f"Prediction #1: {nn.predict(test_value=sum([158, 89, 52]))}")
    print(f"Prediction #2: {nn.predict(test_value=sum([89, 152, 192]))}")
    pass


def run_ClassificationNeuralList(
        input_values: list[float],
        desired_values: list[float],
        hidden_neurons: int
):
    nn = ClassificationNeuralList(
        hidden_neurons=hidden_neurons,
        activation_function=sigmoid,
        classification_function=softmax
    )
    nn.compile()
    print(f"Initialized classification neural list with {hidden_neurons} hidden neurons: {nn}")
    nn.fit(
        input_values=input_values,
        desired_values=desired_values
    )
    print(f"Trained classification neural list with {hidden_neurons} hidden neurons: {nn}")
    print(f"Prediction #1: {nn.predict(test_value=sum([158, 89, 52]))}")
    print(f"Prediction #2: {nn.predict(test_value=sum([89, 152, 192]))}")
    pass


if __name__ == '__main__':
    main()
