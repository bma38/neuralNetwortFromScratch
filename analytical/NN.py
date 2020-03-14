import numpy as np

class neuralNet:

    def __init__(self, dim1):
        self.size = [dim1]
        self.weights = []
        self.biases = []
        self.layerValueWithAct = []
        self.layerValueWithoutAct = []
        self.gradient_weights = []
        self.gradient_biases = []


        
    def addLayer(self, dim, activationFunction = "sig"):

        np.random.seed(3)

        weight = np.random.randn(dim, self.size[-1])
        bias = np.zeros((dim,1))
        self.weights.append(weight)
        self.biases.append(bias)
        self.size.append(dim)

    def propagate(self, inputVal):
        tempVal = inputVal
        self.layerValueWithAct.append(inputVal)
        for i in range(len(self.weights)):
            tempVal = self.biases[i] + np.matmul(self.weights[i], tempVal)
            self.layerValueWithoutAct.append(tempVal)
            tempVal = self.sigmoid(tempVal)
            self.layerValueWithAct.append(tempVal)
        return tempVal.T

    def backpropagate(self, Y):
        tempValue = self.loss(Y, True) * self.derivate_sigmoid(self.layerValueWithoutAct[-1])
        self.gradient_weights = [np.array([]) for i in range(len(self.weights))]
        self.gradient_biases = [np.array([]) for i in range(len(self.biases))]
        for i in range(2, len(self.size)):
            grad_W = tempValue * self.layerValueWithAct[-1 * i].T
            grad_B = tempValue
            self.gradient_weights[-1 * i + 1] = grad_W
            self.gradient_biases[-1 * i + 1] = grad_B

            tempValue =  np.matmul(self.weights[-1 * i + 1].T, tempValue)
            tempValue =  tempValue * self.derivate_sigmoid(self.layerValueWithoutAct[ i * -1])
            
        grad_W = tempValue * self.layerValueWithAct[0].T 
        grad_B = tempValue
        self.gradient_weights[0] = grad_W
        self.gradient_biases[0] = grad_B

    def fit(self, X, Y, learning_rate):
        self.propagate(X)
        self.backpropagate(Y)

        for i in range(len(self.gradient_weights)):
            self.weights[i] = self.weights[i] - self.gradient_weights[i] * learning_rate
            self.biases[i] = self.biases[i] - self.gradient_biases[i] * learning_rate
        
        print(self.loss(Y))
        self.setzero()

    def setzero(self):
        self.layerValueWithAct = []
        self.layerValueWithoutAct = []
        self.gradient_weights = []
        self.gradient_biases = []

    def sigmoid(self, X):
        return 1/(1+np.exp(-1 * X))

    def derivate_sigmoid(self, X):
        return self.sigmoid(X) * (1-self.sigmoid(X))

    def loss(self,Y, derive = False):
        if derive:
            return 2 * (self.layerValueWithAct[-1] - Y)
        else:
            return np.sum(np.power(self.layerValueWithAct[-1]-Y, 2))

    


