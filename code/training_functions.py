import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

def EarlyStopping(loss_validation, min_loss_val, net, name, tolerance = 10, diff = 0.1, counter = 0):
    ''' 
    Checks whether the validation loss has been increasing for too may epochs. If it has it stops the training. Returns the net that gives the minimal validation loss.

    Args: 
        loss_validation: float.
        min_loss_val: float.
        net: neural network object.
        name: string, name for saving the file.
        tolerance: int (how many epochs we are allowed to increase for), default 10.
        diff: float (lower bound to determine whether the loss is increasing), default 0.1.
        counter: int (keeps track of how many times), default 0.

    Returns:
        A boolean determining whether to stop the training, the minimal validation loss achieved, and the counter.

    '''
    # Deal with the epoch 1 case
    if min_loss_val == 0:

        # Set the min loss as the validation loss
        min_loss_val = loss_validation
        
        # Save the network configuration
        torch.save(net.state_dict(), f"../trained_models/{name}.pt")
        return False, min_loss_val, counter
    else:
        # Case 1: we are much bigger than the minimum loss
        if loss_validation - min_loss_val > diff:

            # Increase the counter
            counter +=1

            # If we are over the tolerance we stop
            if counter > tolerance:
                return True, min_loss_val, counter
            
            # Otherwise we keep going
            else:
                return False, min_loss_val, counter

        # Case 2: we are less than the minimum loss
        elif loss_validation - min_loss_val < 0:

            # Zero the counter and update the min loss
            counter = 0
            min_loss_val = loss_validation

            # Save the network configuration since it has achieved a new minimum (it will overwrite the previous minimum)
            torch.save(net.state_dict(), f"../trained_models/{name}.pt")

            return False, min_loss_val, counter

        # Case 3: we are bigger than the minimum, but not enough to update counter
        else:
            return False, min_loss_val, counter

def TrainFinal(config, X_train, y_train, net, name, ep=10, verbose=1):
    ''' 
    Trains the neural network with parameters from config and returns the running training and testing loss and the trained network.

    Args:
        config: dictionary containing the parameters.
        X_train: training features.
        y_train: training labels.
        net: neural network (initialized outside the function).
        name: string, name for saving the pictures/files.
        ep: number of epochs, default 10.
        verbose: int, 1 = prints validation anf training loss after each epoch, 0: no printing, default 1.

    Returns:
        loss_train, loss_val: the training and validation losses calculated for each epoch, as lists.
        net: the final trained network.

    '''
    # Set device to cpu (or gpu if available)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    
    # Initizalise network on device
    net.to(device)

    # Binary Cross Entropy loss
    loss = nn.BCELoss()

    # SGD Optimizer with configurable learning rate and momentum
    optimizer = optim.SGD(net.parameters(), lr = config["lr"], momentum = config["momentum"])

    # Decaying learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Fix a train-test size: 80%
    VAL_SIZE = int(0.8*len(y_train))

    # Split into training and validation
    X_tr = X_train[:VAL_SIZE]
    y_tr = y_train[:VAL_SIZE]
    X_val = X_train[VAL_SIZE:]
    y_val = y_train[VAL_SIZE:]

    # Train set
    train_subset = [(X_tr[i],y_tr[i]) for i in range(len(X_tr))]

    # Validation set
    val_subset = [(X_val[i],y_val[i]) for i in range(len(X_val))]

    # Loads the training set in configurable batch sizes
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)

    # Loads the validation set in configurable batch sizes
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=int(config["batch_size"]), shuffle=True)

    # Storage for the running training loss for each epoch
    losst_tot = []

    # Storage for the running validation loss for each epoch
    lossv_tot = []

    # Initialize the minimum loss and counter (for early stopping)
    min_loss = 0
    counter = 0

    # Loop over the data for the number of epochs
    for epoch in range(ep):  

        # Switch on training mode
        net.train()

        # Keep track of the running loss
        running_loss = 0

        # And the number of batches
        n_batches = 0

        for data in trainloader:
            # Get the input and the real_class from data
            input, real_class = data

            # Make sure they live on the same device as net
            input, real_class = input.to(device), real_class.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward step, calculate loss
            net_out = net(input.view(-1, input.size()[1])).to(device)
            loss_train = loss(net_out.view(-1,1), real_class.view(-1,1))

            # Add to running loss
            running_loss += loss_train.item()

            # Add a step
            n_batches += 1

            # Backprop
            loss_train.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        # Calculate average loss and store
        losst = running_loss/n_batches
        losst_tot.append(losst)

        if verbose == 1:
            print(f'Epoch {epoch+1}. Training loss: {losst}')

        # Calculate validation loss

        # Keep track of the running loss
        running_loss = 0

        # And the number of batches
        n_batches = 0

        # Switch on testing mode
        net.eval()

        with torch.no_grad():
            for data in valloader:

                # Get the input and the real_class from data
                input, real_class = data

                # Make sure they live on the same device as net
                input, real_class = input.to(device), real_class.to(device)

                # Forward step, calculate loss
                net_out = net(input.view(-1, input.size()[1])).to(device)
                loss_val = loss(net_out.view(-1,1), real_class.view(-1,1))

                # Add to running loss
                running_loss += loss_val.item()

                # Add a step
                n_batches += 1

                torch.cuda.empty_cache()

        # Calculate average loss and store
        lossv = running_loss/n_batches
        lossv_tot.append(lossv)

        # Early stopping
        ES, min_loss, counter = EarlyStopping(lossv, min_loss, net, name, tolerance = 10, diff = 0.01, counter = counter)
        if ES:
            return losst_tot, lossv_tot, net

        # Take scheduler step
        scheduler.step(lossv)

        if verbose == 1:
            print(f'Epoch {epoch+1}. Validation loss: {lossv_tot[-1]}')

    return losst_tot, lossv_tot, net

def LearningCurves(loss_train, loss_val, name):
    ''' 
    Plots the learning curves for the training loss and the testing loss against the number of epochs.

    Args:
        loss_train: list of floats.
        loss_val: list of floats.
        name: string, name for the pictures/files.
    
    '''
    import seaborn as sns
    # Reset to default matplotlib style
    sns.reset_orig()
    sns.reset_defaults()

    # Get number of epochs
    epochs = range(len(loss_train))

    # Create figure
    fig, ax = plt.subplots()

    # Plot the training loss curve
    ax.plot(epochs, loss_train, label = 'Train', c='orange')

    # Plot the validation loss curve
    ax.plot(epochs, loss_val, label = 'Validation', c='green')

    # Labels and legend
    ax.set_xlabel('Epochs', fontsize=17)
    ax.set_ylabel('Loss', fontsize=17)
    ax.legend(fontsize = 15)

    # Ticks sizes
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save pictures
    for dpi in [300,600,1200]:
        fig.savefig(f'../images_{dpi}/{name}.png', dpi=dpi)

def TestAccuracy(X_test, y_test, name, net, device="cpu"):
    '''
    Tests the trained model on the testing set.

    Args:
        X_test: testing features.
        y_test: testing features.
        net: configured neural net.
        name: string, name for the pictures/files.
        device: where the calculation is taking place, default cpu.

    Return:
        Accuracy of model on testing set, the predicted labels and the real labels (both as tensors/arrays).
    
    '''
    # Import network
    net.load_state_dict(torch.load(f'../trained_models/{name}.pt'))

    # Put network on device
    net.to(device)

    # Initialise testing mode
    net.eval()

    # Test input and labels
    testset = [(X_test[i],y_test[i]) for i in range(len(X_test))]

    # Loads the testing
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024)

    # Store correct answers
    correct = 0
    total = len(testset)

    # Stores output
    output = torch.Tensor([]).to(device)

    with torch.no_grad():
        for data in testloader:

            # Get the input and the real_class from data
            input, real_classes = data

            # Make sure they are on the same device as net
            input = input.to(device)
            real_classes = real_classes.to(device).view(len(real_classes),1)

            # Forward step, calculate loss
            net_out = net(input.view(-1,input.size()[1]).type(torch.float32)).to(device)

            # Shift to [-0.5.0.5] from [0,1]
            net_out_shift = net_out - 0.5*torch.ones([len(net_out),1]).type(torch.float32).to(device)

            # Get predictions from heaviside 
            predicted = torch.heaviside(net_out_shift, torch.ones([len(net_out),1]).type(torch.float32).to(device))

            # Count correct
            correct += (predicted.to(device) == real_classes.to(device)).sum().item()

            # Store predicted values
            output = torch.cat([output,predicted], dim=-2)

            # Empty cache if using GPU
            if device == 'cuda:0':
                torch.cuda.empty_cache()

    return correct / total, output

def ConfusionMatricesPlotter(cm, name, ann_size = 25, fmt = ".1g"):
    ''' 
    Makes a heatmap of the confusion matrix cm and saves the resulting images as PNG files with given filename.

    Args:
        cm: confusion matrix
        name: string, name for the pictures/files.
        ann_size: integer, size of annotations, default 25.
        fmt: string, format for annotations, detaul '.1g'.
    '''
    import seaborn as sns
    sns.set_theme()

    fig, ax = plt.subplots()

    # Heatmap with 1 significant figure
    sns.heatmap(
        cm,
        cmap="rocket_r",
        annot=True,
        fmt=fmt,
        annot_kws={'fontsize':ann_size},
        xticklabels=['Non-terminal', 'Terminal'],
        yticklabels=['Non-terminal', 'Terminal']
    )

    # Axis labels
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15) 

    # Save pictures
    for dpi in [300,600,1200]:
        fig.savefig(f'../images_{dpi}/{name}.png', dpi=dpi, bbox_inches='tight')

