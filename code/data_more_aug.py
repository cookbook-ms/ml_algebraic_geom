from sl2z_action import action_S, action_T, action_st2, action_st3
import numpy as np 
import ast

"here we augment the data by applying MORE S and T actions on the data"

# augment the data by applying the action of SL(2,Z) on the data
augmented_data = []
with open('../data/bound_7_terminal.txt', 'r') as f:
    # first read 100 lines of the data
    data = f.readlines()[:300000]
    # for each matrix in the data, apply the action of SL(2,Z) and append the result to the data
    for x in data:
        x = ast.literal_eval(x)
        augmented_data.append(x)
        # x_s = action_S(x)
        # augmented_data.append(x_s)
        # x_t = action_T(x)
        # augmented_data.append(x_t)
        x_s = action_st2(x)
        augmented_data.append(x_s)
        x_t = action_st3(x)
        augmented_data.append(x_t)
        
# save the augmented data
with open('../data/bound_7_terminal_more_augmented.txt', 'w') as f:
    for x in augmented_data:
        f.write(str(x)+'\n')
        
        
        
# augment the data by applying the action of SL(2,Z) on the data
augmented_data = []
with open('../data/bound_7_non_terminal.txt', 'r') as f:
    # first read 100 lines of the data
    data = f.readlines()[:300000]
    # for each matrix in the data, apply the action of SL(2,Z) and append the result to the data
    for x in data:
        x = ast.literal_eval(x)
        augmented_data.append(x)
        # x_s = action_S(x)
        # augmented_data.append(x_s)
        # x_t = action_T(x)
        # augmented_data.append(x_t)
        x_s = action_st2(x)
        augmented_data.append(x_s)
        x_t = action_st3(x)
        augmented_data.append(x_t)
        
        
# save the augmented data
with open('../data/bound_7_non_terminal_more_augmented.txt', 'w') as f:
    for x in augmented_data:
        f.write(str(x)+'\n')
        