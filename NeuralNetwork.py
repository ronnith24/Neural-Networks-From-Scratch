import numpy as np

class NeuralNetwork(object):
    def __init__(self, topology, epsilon, numLabels):
        self.theta = []
        self.topology = topology
        self.numLabels = numLabels
        # setting the parameters randomly according to the topology provided
        for layer in range(len(self.topology)):
            if layer == 0:
                continue
            self.theta.append(np.random.rand(self.topology[layer], self.topology[layer - 1] + 1) * 2 * epsilon - epsilon)
    
    
    def gradientDescent(self, iters, alpha, lamda, X, Y):
        """
        Performs the gradient descent algorithm with iters iterations, alpha learning rate, lambda as regularization factor.
        """
        self.X = X
        self.Y = Y
        for i in range(iters):
            (J, thetaGrad) = self.getCostAndGradient(lamda)
            for layer in range(len(self.topology) - 1):
                self.theta[layer] -= thetaGrad[layer] * alpha
            print("Iter " + str(i) + ": " + str(J))
    
    
    def predict(self, x):
        """
        Predicts the output when the input is x using the learned parameters, that is, using theta.
        """
        x = x.reshape((x.shape[0], 1))
        x = np.concatenate(([[1]], x))
        # Forward Propogation
        for layer in range(1, len(self.topology)):
            x = np.matmul(self.theta[layer - 1], x)
            for i in range(x.shape[0]):
                x[i, 0] = self.sigmoid(x[i, 0])
            if layer != len(self.topology) - 1:
                x = np.concatenate(([[1]], x))
        
        prediction = -1
        predictionSurety = -1
        for i in range(self.numLabels):
            if x[i, 0] > predictionSurety:
                prediction = i
                predictionSurety = x[i, 0]
        
        return prediction
    
    
    def getCostAndGradient(self, lamda):
        """
        Calculates the gradient and the cost on the given data using lambda as the regularization factor
        """
        J = 0
        thetaGrad = []
        for layer in range(len(self.topology)):
            if layer == 0:
                continue
            thetaGrad.append(np.zeros((self.topology[layer], self.topology[layer - 1] + 1)))
        
        m = self.X.shape[0]
        for example in range(m):
            x = self.X[example].copy()
            x = x.reshape((x.shape[0], 1))
            y = np.zeros(self.numLabels)
            y[self.Y[example]] = 1
            y = y.reshape((y.shape[0], 1))
            a = []
            z = []
            delta = []
            
            for layer in range(len(self.topology)):
                if layer == 0:
                    a.append(np.concatenate(([[1]], x)))
                    z.append(np.concatenate(([[1]], x)))
                    delta.append(0)
                    continue
                z.append(np.matmul(self.theta[layer - 1], a[layer - 1]))
                a.append(z[layer].copy())
                for i in range(self.topology[layer]):
                    a[layer][i, 0] = self.sigmoid(a[layer][i, 0])
                if layer != len(self.topology) - 1:
                    a[layer] = np.concatenate(([[1]], a[layer]))
                    z[layer] = np.concatenate(([[1]], z[layer]))
                delta.append(0)
                
            for layer in range(len(self.topology) - 1, 0, -1):
                if layer == len(self.topology) - 1:
                    delta[layer] = a[layer] - y
                    thetaGrad[layer - 1] += np.matmul(delta[layer], a[layer - 1].transpose())
                    continue
                
                sigDerZ = z[layer].copy()
                for i in range(self.topology[layer] + 1):
                    sigDerZ[i] = self.sigmoidDerivative(sigDerZ[i])
                
                if layer >= len(self.topology) - 2:
                    delta[layer] = np.matmul(self.theta[layer].transpose(), delta[layer + 1]) * sigDerZ
                else:
                    delta[layer] = np.matmul(self.theta[layer].transpose(), delta[layer + 1][1:, :]) * sigDerZ
                
                thetaGrad[layer - 1] += np.matmul(delta[layer][1:, :], a[layer - 1].transpose())
            
            J += np.sum(-(1 - y) * np.log(1 - a[len(self.topology) - 1])) - np.sum(y * np.log(a[len(self.topology) - 1]))
        
        J /= m
        
        for layer in range(len(self.topology) - 1):
            thetaGrad[layer] *= (1 / m)
        
        for i in range(len(self.topology) - 1):
            for j in range(self.topology[i + 1]):
                for k in range(1, self.topology[i]):
                    J += (lamda / (2 * m)) * self.theta[i][j, k] ** 2
                    thetaGrad[i][j, k] += (lamda / m) * self.theta[i][j, k]
        
        return (J, thetaGrad)
    
    
    def sigmoid(self, x):
        """
        Input: x is a floating point number
        Returns: sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoidDerivative(self, x):
        """
        Returns the result of applying the derivate of the sigmoid to x
        """
        sig = self.sigmoid(x)
        return sig * (1 - sig)
