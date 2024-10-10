import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
from pickle import load
import torch
import torch.nn as nn
import math
import random
import training_functions as tf
from sklearn.metrics import confusion_matrix
import weights_functions as wf

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
    
def gcd_euclid(a, b):
    '''
    Calculates the A, B, g, such that Aa + Bb = g = gcd(a,b).

    Args:
        a: integer
        b: integer
    Return:
        Tuple (g,A,B) such that Aa + Bb = g = gcd(a,b).
    '''
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = gcd_euclid(b % a, a)
        return (g, x - (b // a) * y, y)


def ReidShepherdTai(u, d, v, origin=True):
    ''' It returns True or False depending on whether the Reid-Shepherds-Tai-like criterion fails or succeeds. The flag 'origin' tells us whether the polytope contains the origin or not.
    Args:
        u: list of non-negative integers.
        v: list of non-negative integers.
        d: positive integer.
        origin: True or False.
    Return:
        True or False.
    Exceptions:
        Return RuntimeError if the weights correspond to a fan whose rays are not linearly independent.  
    '''
    # Sum of the weights
    sum_u = int(abs(sum(u)))
    
    # Define dhat as d / the gcd of sum(v) and  d (if v exists)
    dhat = 1
    if d!=1:              
        sum_v = abs(sum(v))
        dhat = int(d/math.gcd(d,sum_v))

    for k in range(0, sum_u*dhat):
        for j in range(0, d):

            # Define linear combination of rational vectors
            lc = [k*u[i]/(sum_u*dhat) + j*v[i]/d for i in range(len(u))]
            
            # Take rational part 
            lc_red = [x - math.floor(x) for x in lc]

            # If the origin is in the simplex we have to check that we are not catching it by mistake
            if origin and d!=1:
                # Check if it is at height one (with tolerance)
                if abs(sum(lc_red)-1) < 0.001:

                    # Check that it does not have the same rational part as the origin
                    diff = [lc[i] - u[i]/sum_u for i in range(len(u))]
                    if sum([x-math.floor(x) for x in diff]) != 0:
                        return False
                    
            elif origin and d == 1:

                # Check if it is at height one (with tolerance)
                if k!=1 and abs(sum(lc_red)-1) < 0.001:
                    return False
            else:
                # Check if it is at height one (with tolerance)
                if abs(sum(lc_red)-1) < 0.001:
                    return False
    return True

def IsTerminal(w):
    ''' Returns True/False depending on whether the input weight matrix correspond to a terminal variety (where the 
    stability condition is taken as the sum of the weights). 
    Args:
        w: list of lists of non-negative integers.
    Return:
        True or False, depending on whether it is terminal or not.
    '''
    # Take the sum of the weight
    a = sum(w[0])
    b = sum(w[1])

    # Loop over the side of the matrix on the left hand side of the stability condition
    for i in range(len(w[0])):

        # Stop after we pass the stability condition
        if a*w[1][i] - b*w[0][i]<0:

            # Get A*x+B*y = g
            g, A, B = gcd_euclid(w[0][i], w[1][i])

            # Solve for g*e_i in terms of the other rays
            v = [A*w[0][j] + B*w[1][j] for j in range(len(w[0]))]

            # Find the origin in terms of all the rays except e_i
            u = [w[1][i]*w[0][j]/g - w[0][i]*w[1][j]/g for j in range(len(w[0]))]

            # Take out the ith element from u and v
            u.pop(i)
            v.pop(i)

            # Initialise the origin flag as false
            origin = False

            # Set origin false if the sign of the entries of u are all the same
            if len([x for x in u if x >=0]) == len(u):
                origin = True
            elif len([x for x in u if x <=0]) == len(u):
                origin = True
                u = [-x for x in u]

            # Check Reid-Shepherd-Tai for the two rational vectors
            if not ReidShepherdTai(u, g, v, origin = origin):
                return False
        else:
            break
    return True

# Fix bound on weights (7 or 10)
bound = 7

# Fix the train size
TRAINSIZE = 2500000

# Define low and high bounds
low_bound = 5
high_bound = 11
n_figures = high_bound - low_bound

# Define figure with four subfigures
plt.rcParams["figure.figsize"] = [n_figures*n_figures, n_figures]
plt.rcParams["figure.autolayout"] = True

fig, axs = plt.subplots(ncols=n_figures,  gridspec_kw=dict(width_ratios=[n_figures]*(n_figures-1)+[n_figures+1.5]))

# Loop over upper bounds
for ub in range(low_bound,high_bound):

    # Generate data with new bound on weights
    dim = 8

    # Generate 10 000 validated balanced examples with new upper bound on the weights
    weights = []
    terminality = []

    while len(weights) < 10000:

        # Initialise random terminality status
        status = bool(random.getrandbits(1))

        # Loop until we produce an example whose terminality matches the status
        while True:

            # Generate random weights
            wts = wf.GenerateWeights(dim+2, ub)

            # Order the weight matrix
            wts = wf.OrderMatrix(wts)

            # Validate the matrix
            if wf.Validate(wts) and wf.IsNormalForm(wts) and wf.IsWellFormed(wts):

                # Change to row presentation
                wts = [[x[0] for x in wts], [x[1] for x in wts]]

                # If the terminality agrees with the status we keep it
                if IsTerminal(wts) == status:
                    weights.append(wts)
                    terminality.append(status)
                    break

    # Turn into tensors and flatten
    y = torch.Tensor(terminality)
    X = torch.Tensor(weights)
    X = torch.flatten(X,start_dim=1)

    # Load scaler and transform X
    scaler = load(open(f'../trained_models/ml_terminality_scaler{TRAINSIZE}_dim8_bound{bound}.pkl', 'rb'))
    X = torch.Tensor(scaler.transform(X))

    # Final configuration for the model
    config = {
            "layers": (512,768,512),
            "lr": 0.01,
            "batch_size": 128,
            "momentum": 0.99,
            "slope": 0.01,
        }

    # Initialise the network
    net = Net(layers = config["layers"], slope = config["slope"])

    # Accuracy test
    accuracy, predictions = tf.TestAccuracy(X, y, f'ml_terminality_nn{TRAINSIZE}_dim8_bound{bound}', net, device="cuda:0")

    print(f'For bound {ub} the accuracy is {accuracy}')

    # Confusion matrix
    mat = confusion_matrix(y, predictions.to('cpu'))

    sns.set_theme()

    # Heatmap with 1 significant figure
    sns.heatmap(
        mat,
        cmap="rocket_r",
        annot=True,
        fmt='.5g',
        annot_kws={'fontsize':25},
        xticklabels=['NT', 'T'],
        yticklabels=['NT', 'T'],
        cbar=False, 
        ax=axs[ub-low_bound]
    )

    # Title
    axs[ub-low_bound].set_title(f'Upper Bound = {ub}', fontsize = 25)

    # Axis labels
    axs[ub-low_bound].set_xlabel("Predicted", fontsize=25)
    axs[ub-low_bound].set_ylabel("True", fontsize=25) 
    axs[ub-low_bound].tick_params(axis='both', which='major', labelsize=20)

fig.colorbar(axs[n_figures-1].collections[0])

fig.subplots_adjust(wspace=0.25)

# Save pictures
for dpi in [300,600,1200]:
    fig.savefig(f'../images_{dpi}/combined_limitations_cm_bound{bound}.png', dpi=dpi)