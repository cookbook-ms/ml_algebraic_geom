# Set bound
bound = 7

if bound==7: 
    # Accuracy outputs of testing_scipt_terminality.py (bound 7)
    accuracy_train = {500000: 0.880057, 1000000: 0.9281, 1500000: 0.946783, 2000000: 0.95465775, 2500000: 0.9551608} 
    accuracy_test = {500000: 0.8352698888888889, 1000000: 0.9116035, 1500000: 0.9383504285714286, 2000000: 0.9432193333333333, 2500000: 0.9509848}
elif bound==10:
    # Accuracy outputs of testing_scipt_terminality.py (bound 10)
    accuracy_train = {2500000: 0.9107992, 3000000: 0.9394496, 3500000: 0.937751, 4000000: 0.938071875, 4500000: 0.93800644, 5000000: 0.9387019} 
    accuracy_test = {2500000: 0.900165133, 3000000: 0.9282998571, 3500000: 0.93216561538, 4000000: 0.9318555, 4500000: 0.9346750909, 5000000: 0.9351023} 
else:
    raise ValueError('Bound supplied unknown')

# Plot learning curve for different train-test splits
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Train learning curve
ax.plot([2*i for i in accuracy_train.keys()], [accuracy_train[i] for i in accuracy_train.keys()] , label = 'Train', c='orange')

# Test learning curve
ax.plot([2*i for i in accuracy_test.keys()], [accuracy_test[i] for i in accuracy_test.keys()] , label='Test', c='blue')

ax.legend(fontsize = 15)
ax.set_xlabel('Training samples', fontsize = 17)
ax.set_ylabel('Accuracy', fontsize = 17)
ax.tick_params(axis='both', which='major', labelsize=12)

# Save pictures
for dpi in [300,600,1200]:
    fig.savefig(f'../images_{dpi}/learning_curve_train_size_bound{bound}.png', dpi=dpi,bbox_inches='tight')