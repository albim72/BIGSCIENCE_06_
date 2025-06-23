import numpy as np

class SimpleNeuralNetwork:
    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(SimpleNeuralNetwork)

    def __init__(self):
        np.random.seed(1)
        self.weights = 2*np.random.random((3,1))-1

    def __repr__(self):
        return f"SimpleNeuralNetwork\n({self.weights})"
