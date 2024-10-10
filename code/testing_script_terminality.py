import numpy as np
import torch
import torch.nn as nn
import training_functions as tf
from pickle import load
from sklearn.metrics import confusion_matrix
import sys
import ast

def OrderMatrix(weights):
    ''' Orders the columns of the weights matrix cyclically.
    Arg:
        weights: a list of lists of non-negative integers (given as the rows).
    Return:
        a list of lists of non-negative integers (now we have ordered this cyclically).
    '''
    # Get column presentation
    weights = [[weights[0][i], weights[1][i]] for i in range(len(weights[0]))]

    # Order according to top row first
    weights.sort(key = lambda x : x[0])

    # Separate into sections
    zerou = [i for i in weights if i[0] == 0]
    zerov = [i for i in weights if i[1] == 0]
    nonzero = [i for i in weights if i[0]*i[1] != 0]

    # Order each section (zerov is already ordered from the first .sort we have done)
    zerou.sort(key = lambda x: x[1])
    nonzero.sort(key = lambda x: x[1]/x[0])

    # Regroup the sections
    weights = zerov + nonzero + zerou

    # Return weights in row presentation
    return [[x[0] for x in weights], [x[1] for x in weights]]

# Total number of arguments
n = len(sys.argv)

# Hard-coded definition of the bound
bound = 7

# Set TRAIN_SIZE to be the argument supplied
if n < 2:
    raise RuntimeError('No TRAIN_SIZE supplied')
elif n == 2:
    TRAIN_SIZE = int(sys.argv[1])
else:
    raise RuntimeError('Too many TRAIN_SIZE supplied')

with open(f'../data/bound_{bound}_terminal.txt', 'r') as f:
    data = f.readlines()
    terminal = [ast.literal_eval(x) for x in data]

with open(f'../data/bound_{bound}_non_terminal.txt', 'r') as f:
    data = f.readlines()
    non_terminal = [ast.literal_eval(x) for x in data]

# Make sure that the column in the weight matrices are cyclically ordered in the same way
terminal = [OrderMatrix(x) for x in terminal]
non_terminal = [OrderMatrix(x) for x in non_terminal]

# Define configurable model
class Net(nn.Module):
    def __init__(self, layers = (200,200,200), slope = 0.1):
        super().__init__()

        # Sanity checks
        if len(layers) == 0:
            raise ValueError('Empty network')

        # Input layer
        self.inp = nn.Linear(20,layers[0])

        # Hidden layers
        self.hid = nn.ModuleList()
        for i in range(len(layers)-1):
            self.hid.append(nn.Linear(layers[i], layers[i+1]))
        
        # Outputlayer: 2 classes, so only one neuron
        self.out = nn.Linear(layers[-1],1)

        # Leaky ReLu activation function
        self.m = nn.LeakyReLU(slope)
    
    # We need to define how the data goes through the nn
    def forward(self,x):
        # Make x pass through every layer
        x = self.m(self.inp(x))

        for l in self.hid:
            x = self.m(l(x)) 

        x = self.out(x)
        return torch.sigmoid(x)

# Store the training and validation losses
accuracy_train = {}
accuracy_test = {}

# Define train set
X_train = terminal[:TRAIN_SIZE] + non_terminal[:TRAIN_SIZE]
y_train = [1]*TRAIN_SIZE + [0]*TRAIN_SIZE

# Define testing set
X_test = terminal[TRAIN_SIZE:] + non_terminal[TRAIN_SIZE:]
y_test = [1]*len(terminal[TRAIN_SIZE:]) + [0]*len(non_terminal[TRAIN_SIZE:])

# Shuffle
perm = np.random.permutation(len(X_train))
X_train = [X_train[i] for i in perm]
y_train = [y_train[i] for i in perm]

# Shuffle
perm = np.random.permutation(len(X_test))
X_test = [X_test[i] for i in perm]
y_test = [y_test[i] for i in perm]

# Turn into tensors and flatten
y_train = torch.Tensor(y_train)
X_train = torch.Tensor(X_train)
X_train = torch.flatten(X_train,start_dim=1)

y_test = torch.Tensor(y_test)
X_test = torch.Tensor(X_test)
X_test = torch.flatten(X_test,start_dim=1)

# Load scaler and transform testing and training
scaler = load(open(f'../trained_models/ml_terminality_scaler{TRAIN_SIZE}_dim8_bound{bound}.pkl', 'rb'))
X_test = torch.Tensor(scaler.transform(X_test))
X_train = torch.Tensor(scaler.transform(X_train))

# Get the training accuracy

# Final configuration for the model
config = {"layers": (512, 768, 512),
    "lr": 0.01,
    "batch_size": 128,
    "momentum": 0.99, 
    "slope": 0.01}

# Initialize the network 
net = Net(layers = config["layers"], slope = config["slope"])

# Accuracy train
# accuracy = tf.TestAccuracy(X_train, y_train, f'ml_terminality_nn{TRAIN_SIZE}_dim8_bound{bound}', net, device="cuda:0")[0]

accuracy = tf.TestAccuracy(X_train, y_train, f'ml_terminality_nn{TRAIN_SIZE}_dim8_bound{bound}', net, device="cpu")[0]

print('The accuracy on the training set is: ', accuracy)

# Get testing accuracy

# Initialize the network 
net = Net(layers = config["layers"], slope = config["slope"])

# Accuracy test
# accuracy, predictions = tf.TestAccuracy(X_test, y_test, f'ml_terminality_nn{TRAIN_SIZE}_dim8_bound{bound}', net, device="cuda:0")

accuracy, predictions = tf.TestAccuracy(X_test, y_test, f'ml_terminality_nn{TRAIN_SIZE}_dim8_bound{bound}', net, device="cpu")


print('The accuracy on the testing set is: ', accuracy)

# Normalise with respect to both the true values and the predicted values
mat_true = confusion_matrix(y_test, predictions.to('cpu'), normalize='true')
mat_pred = confusion_matrix(y_test, predictions.to('cpu'), normalize='pred')

# Plot and save the confusion matrices
tf.ConfusionMatricesPlotter(mat_true, f'confusion_matrix_terminality_{TRAIN_SIZE}_true_dim8_bound{bound}')
tf.ConfusionMatricesPlotter(mat_pred, f'confusion_matrix_terminality_{TRAIN_SIZE}_pred_dim8_bound{bound}')
