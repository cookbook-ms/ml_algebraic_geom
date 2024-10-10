import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast

# Store the regression coefficients
regression = []

# Read file from cleaned dataset of probable examples
with open('../data/terminal_dim8_probable.txt', 'r') as f:
   
    # Extract only the regression coefficients, the line will look like 'Regression: [..,..]'
   for x in f:
       x = x.split(':')
       if 'Regression' in x[0]:
           regression.append(ast.literal_eval(x[1]))

# Fix colour map
cm = plt.cm.get_cmap('inferno')

# Fix no values (because of the logarithm)
cm.set_bad((0,0,0))

# Define 2-dimensional histogram
heatmap, xedges, yedges = np.histogram2d([x[0] for x in regression], [x[1] for x in regression], bins=200)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Plot with log norm for the colours 
plt.imshow(heatmap.T, extent=extent, origin='lower', aspect = 'auto', cmap=cm, norm = mpl.colors.LogNorm())

# Add axes labels
plt.xlabel('A', fontsize = 17)
plt.ylabel('B', fontsize = 17)

# Colour bar
cbar = plt.colorbar()

# Label colour bar
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Frequency', rotation=270, fontsize = 17)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/heatmap_hedgehog_dim8.png',dpi=dpi,bbox_inches='tight')