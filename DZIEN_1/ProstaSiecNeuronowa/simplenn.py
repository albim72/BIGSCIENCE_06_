import numpy as np

class SimpleNeuralNetwork:
    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(SimpleNeuralNetwork)

    def __init__(self):
        np.random.seed(1)
        self.weights = 2*np.random.random((3,1))-1

    def __repr__(self):
        return f"SimpleNeuralNetwork\n({self.weights})"


    #funkcja aktywacji - sigmoid:
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    #różniczka funkcji aktywacji - sigmoid:
    def sigmoid_d(self, x):
        return x*(1-x)
    
    #propagacja
    def propagation(self, inputs):
        return self.sigmoid(np.dot(inputs.astype(float), self.weights))
    
    #propagacja wsteczna
    def propagation_back(self, propagation_result, train_input,train_output):       
        error = train_output - propagation_result
        self.weights += np.dot(train_input.T, error * self.sigmoid_d(propagation_result))
        
    #trening
    def train(self, train_input, train_output, train_iters):
        for _ in range(train_iters):
            propagation_result = self.propagation(train_input)
            self.propagation_back(propagation_result, train_input,train_output) 
        
    

    

