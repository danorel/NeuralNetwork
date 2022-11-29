from common.domain import normalised_data
from network import NeuralNetwork, ClassificationNeuralNetwork
from network.utils import sigmoid, softmax


def main():
    input_vectors = list[list[float]](map(lambda normalised_record: normalised_record[0], normalised_data))
    desired_vector = list[float](map(lambda normalised_record: normalised_record[1], normalised_data))
    run_NeuralNetwork(input_vectors,
                      desired_vector,
                      hidden_layers=1)
    run_ClassificationNeuralNetwork(input_vectors,
                                    desired_vector,
                                    hidden_layers=1)
    pass


def run_NeuralNetwork(
        input_vectors: list[list[float]],
        desired_vector: list[float],
        hidden_layers: int
):
    nn = NeuralNetwork(
        hidden_layers=hidden_layers,
        activation_function=sigmoid
    )
    nn.compile(size=len(input_vectors[0]))
    print(f"Initialized neural network with {hidden_layers} hidden layers: {nn}")
    nn.fit(
        input_vectors=input_vectors,
        desired_vector=desired_vector
    )
    print(f"Trained neural network with {hidden_layers} hidden layers: {nn}")
    print(f"Prediction #1: {[round(v * 255) for v in nn.predict(test_vector=[158, 89, 52])]}")
    print(f"Prediction #2: {[round(v * 255) for v in nn.predict(test_vector=[89, 152, 192])]}")
    pass


def run_ClassificationNeuralNetwork(
        input_vectors: list[list[float]],
        desired_vector: list[float],
        hidden_layers: int
):
    nn = ClassificationNeuralNetwork(
        hidden_layers=hidden_layers,
        activation_function=sigmoid,
        classification_function=softmax
    )
    nn.compile(size=len(input_vectors[0]))
    print(f"Initialized classification neural network with {hidden_layers} hidden layers: {nn}")
    nn.fit(
        input_vectors=input_vectors,
        desired_vector=desired_vector
    )
    print(f"Trained classification neural network with {hidden_layers} hidden layers: {nn}")
    print(f"Prediction #1: {nn.predict(test_vector=[158, 89, 52])}")
    print(f"Prediction #2: {nn.predict(test_vector=[89, 152, 192])}")
    pass


if __name__ == '__main__':
    main()
