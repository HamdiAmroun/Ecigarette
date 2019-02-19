import numpy as np
import sklearn as sk
import sklearn.datasets

from sklearn.neural_network import __all__
# ---- Acticvation function tanh


def tanh(x):
    return np.tanh(x)


# ---- derivation de  la fonction tanh
def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


# ---- Acticvation function logistic
def logistic(x):
    return 1 / (1 + np.exp(-x))


# ---- derivation de  la fonction  logistic
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

# ---- Reseau de neuron
class NeuralNetwork:
    # constructeur
    # layers: nombre de neurons dans la couche
    # activation: la fonction d'activation
    def __init__(self, layers, activation):

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # initialiser les weight entre 0.25 et -0.25 aleatoirement
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.uniform(-1,1,(layers[i - 1] + 1, layers[i] )) - 1) * 0.25)
        self.weights.append((2 * np.random.uniform(-1,1,(layers[i] + 1, layers[i +
                                                                         1])) - 1) * 0.25)

        print(self.weights)

    # ---- entrainnement du reseau
    # x: Le data set
    # y: expected out put
    # learning rate: le taux d'apprenttisage(le changement à effectuer sur les poids
    # epochs: le nombre d'teration
    # choisir aleatoirement un ensemble de données et entrainer le reseau avec
    def fit(self, X, y, learning_rate=0.2, epochs=100):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)
        # répéter N fois
        for k in range(epochs):
            # choisir un elément aleatoirement de l'ensemble d'enrtrer
           for i in range(len(X)):
                a = [X[i]]

                #
                for l in range(len(self.weights)):
                    # recuperer les poids des hidden lyers
                    hidden_inputs = np.ones([self.weights[l].shape[1] + 1])

                    # activer les neurones et propager le resultats
                    hidden_inputs[0:-1] = self.activation(np.dot(a[l], self.weights[l]))
                    a.append(hidden_inputs)
                # calculer les erreurs et les deltas
                error = y[i] - a[-1][:-1]
                deltas = [error * self.activation_deriv(a[-1][:-1])]
                l = len(a) - 2

                # Telle est traiter séparément car pas de bias
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

                for l in range(len(a) - 3, 0, -1):  # we need to begin at the second to last layer
                    deltas.append(deltas[-1][:-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))

                deltas.reverse()
                for i in range(len(self.weights) - 1):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta[:, :-1])
                # Handle last layer separately because it doesn't have a bias unit
                i += 1
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

        #prediction

    def predict(self, x):
                a = np.array(x)
                for l in range(0, len(self.weights)):
                    temp = np.ones(a.shape[0] + 1)
                    temp[0:-1] = a
                    a = self.activation(np.dot(temp, self.weights[l]))
                return a

    def pso(self):
        return 0;
'''
nn = NeuralNetwork([2,4,1], 'tanh')
X = np.array([[0,1], [ 1,0], [ 0,0],
              [1,1],[5,4],[0,5] ])
y = np.array([1, 1, 0, 0,0,1])
nn.fit(X, y)
for i in [[1,1], [1,0], [0,1], [0,0], [2,0]]:
 print(i,nn.predict(i))

data, target= sk.load_iris ([ True ])
print(data)
print(target)

'''
X,y  = sklearn.datasets.load_iris([True])
nn = NeuralNetwork([4,8,8,8,1], 'tanh')
nn.fit(X, y)
for i in X:
 print(i,nn.predict(i))
"""
clf = sk.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1), random_state=1)
clf.fit(X, y)
clf.predict(X)
"""