import numpy as np
from pickle import load
import torch
import torch.nn as nn
from sage.rings.integer import Integer # we import Integer from sage for the regression calculation
from ulid import ULID
import weights_functions as wf
import os

# Define job id (on the HPC this will be read from the environment)
job_id = 1

# # Read the job id from the environment
# try:
#     job_id = int(os.environ['PBS_ARRAY_INDEX'])
# except ValueError:
#     raise RuntimeError('malformed job id')
# except KeyError:
#     raise RuntimeError('cannot find job id')


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

# Terminal check using Neural Network (IsTerminal)
def IsTerminal(weights):
    ''' 
    Returns which weights-matrix in the list of 'weights' are terminal, along with their probabilities, using ML.

    Arg:
        weights: list of weight matrix, list of lists of non-negative integers (given as rows).

    Return:
        The subset of w of those weights that are terminal along with their probability output (both given as lists).
    
    '''
    weights = torch.Tensor(weights)

    # Flatten matrices
    weights_flat = torch.flatten(weights, start_dim = int(1))

    # Final configuration for the model
    config = {
                "layers": (512,768,512),
                "lr": 0.01,
                "batch_size": 128,
                "momentum": 0.99,
                "slope": 0.01,
            }
    
    # Import the scaler
    scaler = load(open(f'../trained_models_final/ml_terminality_scaler2500000_dim8_bound7.pkl', 'rb'))

    # Scale weights
    weights_scaled = torch.Tensor(scaler.transform(weights_flat))

    # Initialize the network 
    net = Net(layers = config["layers"], slope = config['slope'])
    
    net.load_state_dict(torch.load(f'../trained_models_final/ml_terminality_nn2500000_dim8_bound7.pt', map_location=torch.device('cpu')))

    # Forward step, calculate loss
    net_out = net(weights_scaled.view(-1,20).type(torch.float32))

    # Select the terminal indices (terminal is 1)
    indices = (net_out > 0.5).nonzero(as_tuple=True)[0]

    # Select terminal weights
    terminals = weights[indices]

    # Select output probabilities for the terminal weights
    probabilities = net_out[indices]

    # Cast weight matrices into integers
    terminals = terminals.to(torch.int32)

    return terminals.tolist(), probabilities.tolist()


# Initialise running weights
running_weights = []

# Fix entries upper bound
ub = 7

# Fix matrix dimension
dim = 10

# Generate some random weights that are in normal form, well-formed, rank2, and validated
while len(running_weights) < 1000:

    # Generate random weights
    wts = wf.GenerateWeights(dim, ub)

    # Order the weight matrix
    wts = wf.OrderMatrix(wts)

    # Validate the matrix
    if wf.Validate(wts) and wf.IsNormalForm(wts) and wf.IsWellFormed(wts):

        # If the matrix passes all checks, append to running_weights
        running_weights.append([[x[0] for x in wts], [x[1] for x in wts]])


# Checks terminality for running_weights
terminals, probabilities = IsTerminal(running_weights)

# Check that we have terminal examples
if len(terminals)!=0:

    # Store data
    data = []

    for i, w in enumerate(terminals):
        
        # Calculate the sum of the weights
        s = [sum(w[0]),sum(w[1])]        

        # Calculate alpha
        alpha = wf.Bisection(wf.Polynomial, w, -Integer(w[0][-1])/Integer(w[1][-1]), Integer(10), 0.0001)

        # Append data dictionary
        data.append({'Weights': str(terminals[i]),
                    'Probability': float(probabilities[i][0]), 
                    'Alpha': float(alpha), 
                    'Regression': str([wf.A(w, alpha), wf.B(w, alpha)]), 
                    'ULID': str(ULID()),
                    'K': str(s),
                    'FanoIndex':str(np.gcd(s[0],s[1]))
                    })

    # Store them in a txt file named by the job_id
    with open(f'../data_hpc/{job_id}.txt', 'w') as f:
        for x in data:
            f.write(str(x)+'\n')
