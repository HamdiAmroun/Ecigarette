import copy
import math
import numpy as np
import random
import sklearn.datasets
import sklearn.metrics
import sys
import time


# ---- Acticvation function tanh


def tanh(x):
    return np.tanh(x)


# ---- derivation of Acticvation function tanh
def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


# ---- Acticvation function logistic
def logistic(x):
    return 1 / (1 + np.exp(-x))


# ---- derivation of Acticvation function logistic
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


# classe des particules
class Particle:
    # equivalent to NN weights


    # optional used to determine death-birth

    def __init__(self, position, error, velocity, bestPosition, bestError):
        self.position = position

        self.error = error;
        self.velocity = velocity;
        self.bestPosition = bestPosition;
        self.bestError = bestError;


# ---- Reseau de neuron
class NeuralNetwork:
    # constructeur
    # layers: nombre de neurons dans la couche
    # activation: la fonction d'activation
    def __init__(self, layers, activation):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        self.layers=layers
        self.numInput = layers[0];
        self.numHidden = np.sum(layers) - layers[0] - layers[len(layers) - 1];
        self.numOutput = layers[len(layers) - 1];
        self.layers = layers

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # initialiser les weight entre 0.25 et -0.25 aleatoirement
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.uniform(-1, 1, (layers[i - 1] + 1, layers[i])) - 1) * 0.25)
        self.weights.append((2 * np.random.uniform(-1, 1, (layers[i] + 1, layers[i +
                                                                                 1])) - 1) * 0.25)

    #calculer la taille de la matrice weights
    def weightSize(self):
        self.numWeight=0

        for i in range (len(self.layers)-1):
            self.numWeight=self.numWeight+((self.layers[i]+1)*self.layers[i+1])


    #modifier la matrice weight du model
    def setWeight(self, weights):
        '''
        numWeights = (self.numInput * self.numHidden) + (
        self.numHidden * self.numOutput) + self.numHidden + self.numOutput;'''
        self.weightSize()
        numWeights=self.numWeight
        if (len(weights) != numWeights):
            print ("erreur: !len(weights) != numWeights):")
        else:
            self.ihWeight = weights[0: self.numInput * self.layers[1]]
            self.ihWeight = self.ihWeight.reshape((self.numInput, self.layers[1]))
            self.hBiases  =( weights[self.numInput * self.layers[1]: self.layers[1] + self.numInput * self.layers[1]])
            nd=self.layers[1] + self.numInput * self.layers[1];
            nf=self.layers[1] + self.numInput * self.layers[1]
            self.hhWeight=[]
            self.hhBiases=[]
            for i in range (2,len(self.layers)-2,1):
                nf=nd+(self.layers[i]*self.layers[i-1])
                self.hhWeight.append( weights[nd:nf])
                x=np.array(self.hhWeight[i - 2])
                x=x.reshape(self.layers[i],self.layers[i-1] )
                self.hhWeight[i - 2]=x
                nd=nf
                nf=nf+self.layers[i]+1
                self.hhBiases.append(weights[nd: nf])



                nd=nf

            nf=nd+self.layers[len(self.layers)-2]*self.layers[len(self.layers)-1]
            self.hoWeight = weights[nd: nf]
            self.hoWeight = self.hoWeight.reshape((self.layers[len(self.layers)-2], self.layers[len(self.layers)-1]))
            self.oBiases = weights[nf:nf+self.layers[len(self.layers)-1]]
        # ---- entrainnement du reseau
        # x: Le data set
        # y: expected out put
        # learning rate: le taux d'apprenttisage(le changement à effectuer sur les poids
        # epochs: le nombre d'teration
        # choisir aleatoirement un ensemble de données et entrainer le reseau avec

    def pso(self, trainData, resultData, numParticles, maxEpochs, exitError, probDeath):
        epoch = 0;
        minX = -1;  # for each weight.assumes data has been normalized about 0
        maxX = 1;
        w = 0.729;  # inertia weight
        c1 = 1.49445;  # cognitive / local weight
        c2 = 1.49445;  # social /global weight
        error = sys.float_info.max;
        bestGlobalError = sys.float_info.max;
        self.weightSize()
        #totalweights = (self.layers[0] + 1) * self.layers[1] + (self.layers[1] + 1) * self.layers[2]
        totalweights=self.numWeight
        # def __init_(self,  position, error, velocity, bestPosition,  bestError):
        swarm = []
        bestGlobalPosition = np.zeros(shape=totalweights)
        # initialize each Particle in the swarm with random positions and velocities

        for i in range(numParticles):
            # initialiser la position aleatoirement
            randomPositions = np.zeros(shape=totalweights)
            randomVelocities = np.zeros(shape=totalweights)
            #for j in range(totalweights):
            randomPositions = (np.random.uniform(-1, 1,totalweights))
            randomVelocities = (np.random.uniform(-1, 1,totalweights))
                # initialiser les velocity  aleaoirement
                # print (i)
            swarm.append(Particle(randomPositions, error, randomVelocities, randomPositions, error))
            '''
            print ("position")
            print (swarm[ i ].position)
            print ("velocity")
            print (swarm[ i ].velocity)
            print ("error")
            print (swarm[ i ].error)'''
            # does current Particle have  global best position / solution?
            if swarm[i].error < bestGlobalError:
                bestGlobalError = swarm[i].error;
                bestGlobalPosition = copy.copy(swarm[i].position)

        print("numParticles",numParticles,totalweights)
        while epoch < maxEpochs:
            # print(epoch)
            if bestGlobalError < exitError: break
            for i in range(numParticles):  # process each particle
                # compute new velocity of curr particle
                # for k in range( len(swarm[i].velocity)):
                r1 = np.random.uniform(1, 16,totalweights)  # randomizations
                r2 = np.random.uniform(1, 16 , totalweights)
                swarm[i].velocity = ((w * swarm[i].velocity) +
                                           (c1 * r1 * (swarm[i].bestPosition - swarm[i].position)) +
                                           (c2 * r2 * (bestGlobalPosition- swarm[i].position)))
                swarm[ i ].velocity[swarm[ i ].velocity>maxX]=maxX
                swarm[ i ].velocity[ swarm[ i ].velocity < minX ] = minX

                # compute new position using new velocity
               # for k in range(0, len(swarm[i].position), h):
                swarm[i].position += swarm[i].velocity
                # choose a data frome train sample

                # compute error of new position
                swarm[i].error = self.MeanSquaredError(trainData, resultData, swarm[i].position)
                # is new position a new best for the particle?

                if swarm[i].error < swarm[i].bestError:
                    swarm[i].bestError = swarm[i].error
                    swarm[i].bestPosition = copy.copy(swarm[i].position)

                    # is new position a new best overall?
                if swarm[i].error < bestGlobalError:
                    bestGlobalError = swarm[i].error
                    bestGlobalPosition = copy.copy(swarm[i].position)
                    # if epoch%100 == 0:
                    # print("error")
                    # print(swarm[ i ].error)
                    # print("bestGlobal error")
                    # print(bestGlobalError)
            epoch += 1

        # while
        self.setWeight(bestGlobalPosition)

    def ComputeOutputs(self, xValues):
        hSums = np.zeros(shape=self.layers[1]);
        oSums = np.zeros(shape=self.numOutput);
        inputs = np.zeros(shape=(self.numInput))
        self.hOutput = np.zeros(shape=self.layers[len(self.layers)-2])
        inputs = xValues;
        # print("input value")
        # print(inputs)
        for j in range(self.layers[1]):
            for i in range(self.numInput):
                hSums[j] = hSums[j] + inputs[i] * self.ihWeight[i][j]
            hSums[j] = hSums[j] + self.hBiases[j];
       # for j in range(self.layers[2]):
        self.hOutput = tanh(hSums);

        # print (self.hOutput[ i ])
        #definir les input de /output
        hhInput=self.hOutput
        hhSum=np.zeros(shape=np.max(self.layers));
        #calculer les outputs des hiddenlayers
        # la couche i
        for i in range(len(self.hhWeight)):
            # le neud j
            for j in range(1,len(self.hhWeight[i])):
                #calculer la somme
                for k in range(len(self.hhWeight[i][j])):
                    hhSum[j]=hhSum[j]+self.hhWeight[i][j][k]*hhInput[k]
                hhSum[j] = hhSum[j] + self.hhBiases[i][j];
            hhInput= tanh(hhSum)

        self.hOutput=hhInput
        for j in range(self.numOutput):
            for i in range(self.layers[len(self.layers)-2]):
                oSums[j] += self.hOutput[i] * self.hoWeight[i][j];
                # print(oSums[ j ])
            oSums[ j ] = oSums[ j ] + self.oBiases[ j ];

            # print(oSums[ i ])
            # softOut = 1.0 - np.tanh(oSums) ** 2;
        softOut = (self.relu(oSums, 0.1))
        # print("soft out")
        # print(softOut)
        return softOut;

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        # recuperer le max
        max_x = np.max(x)
        scal = 0
        for i in range(len(x)):
            scal = scal + math.exp(x[i] - max_x)

        result = np.zeros(shape=len(x))
        for i in range(len(result)):
            result[i] = math.exp(x[i] - max_x) / scal

        return result

    def relu(self, data, epsilon):
        """Compute softmax values for each sets of scores in x."""
        # recuperer le max
        return np.maximum(epsilon * data, data)

    def MeanSquaredError(self, trainData, result_data, position):
        self.setWeight(position);
        sumSquaredError = 0;

        for i in range(len(trainData)):
            # following assumes data has all x-values first, followed by y-values!
            yValues = self.ComputeOutputs(
                trainData[i]);  # compute the outputs using centroids, widths, weights, bias values
            for j in range( len(yValues)):
                sumSquaredError = sumSquaredError + ((yValues[j] - result_data[i]) * (yValues[j] - result_data[i]));

        return sumSquaredError / len(trainData);


data4, target4 = sklearn.datasets.load_iris([True])
# print ("len \n\n\n\n")

class1 = []
result1 = []
class2 = []
result2 = []
class3 = []
result3 = []
train = []
resultTrain = []

# construire l'ensemble de tranning et de test
for i in range(len(target4)):
    if target4[i] == 0:
        class1.append(data4[i])
        result1.append(target4[i])
    if target4[i] == 1:
        class2.append(data4[i])
        result2.append(target4[i])
    if target4[i] == 2:
        class3.append(data4[i])
        result3.append(target4[i])

# choisir un echantillion aleatoire
class1 = np.array(class1)
class2 = np.array(class2)
class3 = np.array(class3)

for j in range(int(len(target4) / 6)):
    train.append(class1[j])
    train.append(class2[j])
    train.append(class3[j])
    resultTrain.append(result1[j])
    resultTrain.append (result2[j])
    resultTrain.append (result3[j])
    class1 = np.delete (class1, j, axis=0)
    class2 = np.delete (class2, j, axis=0)
    class3 = np.delete (class3, j, axis=0)
    result1 = np.delete(result1, j)
    result2 = np.delete(result2, j)
    result3 = np.delete(result3, j)

resultTest = np.concatenate([result1, result2, result3])
#resultTest = np.concatenate([result1, result2])
# creation du neural netwok

'''
X = np.array ([ [ 0, 1 ], [ 1, 0 ], [ 0, 0 ],
                [ 1, 1 ], [ 5, 4 ], [ 0, 5 ] ])
y = np.array ([ 1, 1, 0, 0, 0, 1 ])
'''
# valeur d'entrée
X = train
y = resultTrain

#try to find the best neural networ
netArch= [[4,4,1],[4,8,1],[4,16,1],[4,4,4 ,1],[4,4,8,1],
          [4,4,16,1], [4,8,4,1], [ 4,8,8,1], [ 4,8,16,1], [4,16,4,1 ],
          [ 4,16,8,1], [4,16,16,1], [4,4,4,4,1], [4,4,4,16,1],
          [ 4,4,8,4,1], [ 4,4,8,8,1], [ 4,4,8,16,1], [ 4,4,16,4,1], [ 4,4,16,8,1],
          [ 4,4,16,16,1], [4,8,4,4,1 ], [ 4,8,4,8,1], [ 4,8,4,16,1], [4,8,8,4,1 ],
          [ 4,8,8,8,1], [4,8,8,16,1 ], [ 4,8,16,4,1], [4,8,16,8,1 ], [4,8,16,16,1 ],
          [4,16,4,4,1], [ 4,16,4,8,1], [4,16,4,16,1 ], [4,16,8,4,1], [ 4,16,8,8,1],
          [ 4,16,8,16,1], [4,16,16,4,1 ], [4,16,16,8,1 ], [4,16,16,16,1]]


nn = NeuralNetwork([4,4,8,4,1], 'tanh')
# entrainer l'algo avec pso
nn.pso(X, y,21, 100, 0.02, 0.1)
nn.pso(X, y,21, 100, 0.02, 0.1) 



# nn.fit(X,y)
j = 0;
test = np.concatenate([class1, class2,class3])
result = []
for i in (test):
                    r = nn.ComputeOutputs(i)
                   # print ( r, resultTest[j])
                    result.append(r);
                    j = j + 1
result = np.array(result)
result = np.round(result)

print("accuracy", sklearn.metrics.accuracy_score(resultTest, result))