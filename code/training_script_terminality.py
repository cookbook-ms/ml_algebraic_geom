import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import training_functions as tf
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

# Hard-coded bound
bound = 7

# Set TRAIN_SIZE to be the argument supplied
if n < 2:
    raise RuntimeError('No TRAIN_SIZE supplied')
elif n == 2:
    TRAIN_SIZE = int(sys.argv[1])
else:
    raise RuntimeError('Too many TRAIN_SIZE supplied')
print('Importing data...')

# Import data
# with open(f'../data/bound_{bound}_terminal_augmented.txt', 'r') as f:
#     data = f.readlines()
#     terminal = [ast.literal_eval(x) for x in data]

# with open(f'../data/bound_{bound}_non_terminal_augmented.txt', 'r') as f:
#     data = f.readlines()
#     non_terminal = [ast.literal_eval(x) for x in data]

# Import data
with open(f'../data/bound_{bound}_terminal.txt', 'r') as f:
    data = f.readlines()
    terminal = [ast.literal_eval(x) for x in data]

with open(f'../data/bound_{bound}_non_terminal.txt', 'r') as f:
    data = f.readlines()
    non_terminal = [ast.literal_eval(x) for x in data]



print('Completed')

# Order the column of the matrices cyclically
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
print('Divide in train and test and scale...')

# Define train set
X_train = terminal[:TRAIN_SIZE] + non_terminal[:TRAIN_SIZE]
y_train = [1]*TRAIN_SIZE + [0]*TRAIN_SIZE

# Shuffle
perm = np.random.permutation(len(X_train))
X_train = [X_train[i] for i in perm]
y_train = [y_train[i] for i in perm]

# Turn into tensors and flatten
y_train = torch.Tensor(y_train)
X_train = torch.Tensor(X_train)
X_train = torch.flatten(X_train,start_dim=1)

# Final configuration for the model
config = {
        "layers": (512,768,512),
        "lr": 0.01,
        "batch_size": 128,
        "momentum": 0.99,
        "slope": 0.01,
    }

# Define the Scaler
scaler = StandardScaler()

# Scale the features (by fitting onto the training set)
scaler.fit(X_train)
X_train = torch.Tensor(scaler.transform(X_train))

# Save the trained scaler
from pickle import dump
dump(scaler, open(f'../trained_models/ml_terminality_scaler{TRAIN_SIZE}_dim8_bound{bound}.pkl', 'wb'))

print('Completed')

# Initialise the network
net = Net(layers = config["layers"], slope = config["slope"])

print('Start training loop...')

# Train the network for a maximum of 150 epochs, verbose=1 prints train and validation accuracy for each epoch
loss_train, loss_validation, net = tf.TrainFinal(config, X_train,y_train, net, f'ml_terminality_nn{TRAIN_SIZE}_dim8_bound{bound}', ep = 150, verbose = 1)

print('Generate learning curves')

# Generate learning curves
tf.LearningCurves(loss_train, loss_validation, f'learning_curve_terminality_{TRAIN_SIZE}_dim8_bound{bound}')

print('Save losses')

# Save the losses for reproducibility of figures
with open(f'losses/loss_train_{TRAIN_SIZE}_bound{bound}.txt', 'w') as f:
    for x in loss_train:
        f.write(str(x)+'\n')

with open(f'losses/loss_validation_{TRAIN_SIZE}_bound{bound}.txt', 'w') as f:
    for x in loss_validation:
        f.write(str(x)+'\n')