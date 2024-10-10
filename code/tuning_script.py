import ast
import itertools
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os
import ray
import logging
import sys

# Total number of arguments
n = len(sys.argv)

# Hard-coded definition of the bound
bound = 7

# Set data_directory to be the argument supplied
if n < 2:
    raise RuntimeError('No path supplied')
elif n == 2:
    path = sys.argv[1]
else:
    raise RuntimeError('Too many paths supplied')

def does_swell(x):
    """
    Returns True if x rises and falls and False otherwise.
    Args:
      x: a list of integers
    Returns:
      True or False
    """
    # h is half-way along the list
    h = int((len(x)-1)/2)
    for i in range(1, len(x)):
        if i <= h and x[i] < x[i-1]:
            return False
        if i > h and x[i] > x[i-1]:
            return False
    return True
def swell(a,b,cs):
    """
    Returns tuples of lengths in [a,b) with elements in cs that rise and then fall.
    """
    for l in range(a,b):
        for x in itertools.product(*[cs]*l):
            if does_swell(x):
                yield(x)


def load_data(data_dir="../data/"):
    """
    Prepares the training data.
    
    Returns:
        training_subset, validation_subset

    """
    # Import data from txt files
    with open(os.path.join(data_dir, f"terminal_tuning_bound{bound}.txt"), 'r') as f:
        terminal = [ast.literal_eval(x) for x in f]

    with open(os.path.join(data_dir, f"non_terminal_tuning_bound{bound}.txt"), 'r') as f:
        non_terminal = [ast.literal_eval(x) for x in f]

    # Define train set
    X_train = terminal + non_terminal
    y_train = [1]*len(terminal) + [0]*len(non_terminal)

    # Shuffle the data
    perm = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in perm]
    y_train = [y_train[i] for i in perm]

    # Turn into tensors and flatten
    y_train = torch.Tensor(y_train)
    X_train = torch.Tensor(X_train)
    X_train = torch.flatten(X_train,start_dim=1)

    # Fix a train-test split: 80%
    TRAIN_SIZE = int(0.8*len(y_train))

    # Split into training and validation data
    X_tr, X_val = X_train[:TRAIN_SIZE], X_train[TRAIN_SIZE:]
    y_tr, y_val = y_train[:TRAIN_SIZE], y_train[TRAIN_SIZE:]

    # Scale the features
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = torch.Tensor(scaler.transform(X_tr))
    X_val = torch.Tensor(scaler.transform(X_val))

    # Package training and validation sets for the data loaders
    train_subset = [(X_tr[i],y_tr[i]) for i in range(len(X_tr))]
    val_subset = [(X_val[i],y_val[i]) for i in range(len(X_val))]

    return train_subset, val_subset

def TrainTune(config, ep=10, data_dir = None):
    '''
    Trains the NN with parameters from config and reports the validation loss to Ray Tune.

    Args:
        config: dictionary containing the parameters (which have been tuned previously).
        ep: number of epochs (default 10).
        data_dir: directory of the data, default None.

    '''
    # Recover the training data
    if data_dir is None:
        raise ValueError("no training data supplied")
    (train_subset, val_subset) = load_data(data_dir)

    # Define the network using the configuration supplied
    net = Net(layers = config["layers"], slope = config["slope"])

    # Set device to cpu (or gpu if available) and move the network there
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    
    # Binary Cross Entropy loss function
    loss = nn.BCELoss()

    # Optimizer with configurable learning rate
    optimizer = optim.SGD(net.parameters(), lr = config["lr"], momentum = config["momentum"])

    # Loads the training set in configurable batch sizes
    trainloader = torch.utils.data.DataLoader(
        train_subset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        )

    # Loads the validation set in configurable batch sizes
    valloader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=config["batch_size"], 
        shuffle=True,    
        )

    # Loop over the dataset multiple times
    for epoch in range(ep):  

        # Switch on training mode
        net.train()

        for data in trainloader:
            # Get the input and the real_class from data
            input, real_class = data

            # Make sure they live on the same device as net
            input, real_class = input.to(device), real_class.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward step, calculate loss
            net_out = net(input)
            loss_train = loss(net_out.view(-1,1), real_class.view(-1,1))

            # Backprop
            loss_train.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        # Turn on evaluation mode
        net.eval()

        # Compute and report validation loss
        with torch.no_grad():
            # Keep track of the running loss
            running_loss = 0.0

            # And the number of batches
            n_batches = 0

            for data in valloader:
                # Get the input and the real_class from data
                input, real_class = data

                # Make sure they live on the same device as net
                input, real_class = input.to(device), real_class.to(device)

                # Forward step, calculate loss
                net_out = net(input)
                loss_val = loss(net_out.view(-1,1), real_class.view(-1,1))

                # Add to running loss
                running_loss += loss_val.cpu().numpy()

                # Add a step
                n_batches += 1
                torch.cuda.empty_cache()
            
            tune.report(loss=float(running_loss/n_batches))

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

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    """
    Runs the Ray Tune search
    """
    ray.init(logging_level=logging.ERROR)
    # Specify the allowed network architectures

    arch = list(swell(3,4,[256, 512, 768]))

    # Configure the search space
    config = {
        "layers": tune.choice(arch),
        "lr": tune.choice([0.1, 0.01, 0.001]),
        "batch_size": tune.choice([128, 256, 512]),
        "momentum": tune.choice([0.99, 0.9, 0.8, 0.7]), 
        'slope': tune.choice([0, 0.01, 0.05, 0.1])
    }

    # Prepare to tune
    scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=5,
            reduction_factor=2
            )
    wrap = tune.with_parameters(TrainTune, data_dir = path)

    result = tune.run(
        wrap,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=CLIReporter(),
        max_concurrent_trials=4, 
        verbose=1)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

if __name__ == "__main__":
    main(num_samples=100, max_num_epochs=20, gpus_per_trial=1)