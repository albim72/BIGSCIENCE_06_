import numpy as np
from simplenn import SimpleNeuralNetwork

if __name__ == "__main__":
    nn = SimpleNeuralNetwork()
    print(nn)

    train_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0],[1, 0, 0]])
    train_outputs = np.array([[0, 0, 0, 0, 1, 1, 1]]).T
    train_iterators = 100_000


    nn.train(train_inputs, train_outputs, train_iterators)
    print(nn.weights)

    print("ocena modelu")
    test_data = np.array([[0, 0, 1],[1,1,1],[1,0,1],[0,1,1],[1,1,0],[0,0,0],[1,0,0]])
    test_labels = np.array([[0, 0, 0, 0, 1, 1, 1]]).T
    for data in test_data:
        print(f"dane: {data} -> {nn.propagation(data)}")

    predictions = np.array([nn.propagation(data) for data in test_data])

    #ocena modelu: accuracy, loss
    #zamiana wyników 0,1

    predictions_binary = (predictions > 0.5).astype(int)

    #accuracy
    accuracy = np.mean(predictions_binary == test_labels)
    print(f"accuracy: {accuracy*100:.2f}%")

    #loss
    loss = np.mean(np.square(predictions - test_labels))
    print(f"loss(MSE): {loss:.2f}")


    for data,pred,true in zip(test_data,predictions_binary,test_labels):
        print(f"dane: {data} -> pred: {pred} -> true: {true}")
