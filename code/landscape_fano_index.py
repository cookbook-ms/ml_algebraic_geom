import matplotlib.pyplot as plt
import matplotlib as mpl
import ast

# Store regression data and fano indices
regression = []
fidx = []

# Read file from cleaned dataset of probable examples
with open('../data/terminal_dim8_probable.txt', 'r') as f:
   
    # Extract only the Fano indices, the line looks like 'FanoIndes: 1', and the regression data, the line looks like 'Regression': [..,..]
   for x in f:
       x = x.split(':')
       if 'Regression' in x[0]:
           regression.append(ast.literal_eval(x[1]))
       elif 'FanoIndex' in x[0]:
           fidx.append(int(x[1])) 

fig, ax = plt.subplots()

# Fix colour map
cm = plt.cm.get_cmap('viridis')

# Extract two regression coefficients
A = [x[0] for x in regression]
B = [x[1] for x in regression]

# Scatter using log norm for the colours
plt.scatter(A, B, s = .1, c = fidx, cmap = cm, alpha = .1, norm = mpl.colors.LogNorm())

ax.set_xlabel('A', fontsize = 17)
ax.set_ylabel('B', fontsize = 17)

# Colour bar
cbar = plt.colorbar()

# Label colour bar
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Fano Index', rotation=270, fontsize = 17)

# Reset alpha from colour bar
cbar.solids.set(alpha=1)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/fidx_hedgehog_dim8.png',dpi=dpi,bbox_inches='tight')