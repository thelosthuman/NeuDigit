import os
import random
import gzip as gz
import numpy as np
import _pickle as cp

DATA_FILE = 'TRAINING_DATA_FILE'
EPOCHS    = 100
RATE      = 10
MB_SIZE   = 250

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class AllData(object):
    def __init__(self, filetoload):
        self.filename = filetoload
        
        f = gz.open(self.filename, 'rb')
        tr_data, val_data, t_data = cp.load(f, encoding='latin1')
        f.close()
    
        # Create lists of tuples like below for convenient processing later
        # [ (TestData1, Result1),
        #    (TestData2, Result2)
        #    ...
        #    (TestData_N, Result_N)
        # ]
        self.training_data = [(np.reshape(x, (784,1)), self.labelasvector(y)) for x, y in zip(tr_data[0], tr_data[1])]
        self.validation_data = [(np.reshape(x, (784,1)), self.labelasvector(y)) for x, y in zip(val_data[0], val_data[1])]
        self.test_data = [(np.reshape(x, (784,1)), self.labelasvector(y)) for x, y in zip(t_data[0], t_data[1])]
        
    
    def labelasvector(self, label):
        
        zeroarray = np.zeros((10, 1))
        zeroarray[label] = 1
        vec = np.reshape(zeroarray, (10, 1))
        
        return vec
        

class Network(object):
    
    def __init__(self, sizes):
        
        self.num_of_layers = len(sizes)
        self.layers = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
            for x,y in zip(sizes[:-1], sizes[1:])]
        self.zs = []
        self.activations = []
        
    def __str__(self):
        print('Number of layers: {}'.format(self.num_of_layers))
        print('Layers: {}'.format(self.layers))
        print('Biases: {}'.format(self.biases))
        print('Weights: {}'.format(self.weights))
        
        return ''
    
    def feedforward(self, a):
        activation = a
        self.activations.append(a)
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            self.zs.append(z)
            
            activation = sigmoid(z)
            self.activations.append(activation)
        
        return activation
        
    def train(self):
        #First layer activations are input data

        for i in range(EPOCHS):
            sigma = 0.0
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            random.shuffle(d.training_data)
            mini_batches = [mb for mb in d.training_data[0:MB_SIZE]]

            for activation in mini_batches:
                self.activations.clear()
                self.zs.clear()

                fx = self.feedforward(activation[0])

                sigma += (fx - activation[1]) ** 2

                #Backpropagation here
                # Get the error of the last layer and use it to calculate
                # bias and weigth derivative
                ########
                error = fx - activation[1]
                delta = error * sigmoid_prime(self.zs[-1])

                nabla_b[-1] += delta
                nabla_w[-1] += np.dot(delta, self.activations[-2].transpose())

                for layer in range(2, self.num_of_layers):
                    layer *= -1
                    z = self.zs[layer]

                    delta = np.dot(self.weights[layer+1].transpose(), delta) * sigmoid_prime(z)
                    nabla_b[layer] += delta
                    nabla_w[layer] += np.dot(delta, self.activations[layer-1].transpose())

                ########

            self.weights = [w - ( RATE /len(d.training_data)) * nbw for w, nbw in zip(self.weights, nabla_w)]
            self.biases = [b - (RATE / len(d.training_data)) * nbb for b, nbb in zip(self.biases, nabla_b)]

            cost = sigma/(2 * len(d.training_data))
        
        return cost

    
if __name__ == '__main__':

    filename = input('Enter training data file: ')

    if not filename or filename == '':
        print('Error: Filename required. Try again')
        exit(1)
    elif not os.path.isfile(filename):
        print('Error: Valid filename required')
        exit(2)

    d = AllData(DATA_FILE)
    
    net = Network([784, 16, 10])
    netcost = net.train()
    print('RESULT WEIGHTS: {}'.format(net.weights))
    print('RESULT BIASES: {}'.format(net.biases))
