from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast
from mycolorpy import colorlist as mcp
import numpy as np

# Store the regression coefficients and the Fano indices 
regression = []
fidx = []

# Read file from cleaned dataset of probable examples
with open('../data/terminal_dim8_probable.txt', 'r') as f:
   
   # Extract only the Fano indices, the line looks like 'FanoIndes: 1', and the regression data, the line looks like 'Regression: [..,..]'
   for x in f:
       x = x.split(':')
       if 'Regression' in x[0]:
           regression.append(ast.literal_eval(x[1]))
       elif 'FanoIndex' in x[0]:
           fidx.append(int(x[1])) 

fig, ax = plt.subplots(1, figsize=(8,8))

# Upped bound for Fano index
N=10

color1=mcp.gen_color(cmap="viridis",n=10)

# Loop over Fano indices
for i in range(1,N):
    points = np.array([regression[j]for j in range(len(regression)) if fidx[j]==i])

    # Get convex hull
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                    points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                    points[hull.vertices,1][0])
    # Plot shape
    plt.fill(x_hull, y_hull, alpha=0.2, c=color1[i], label = f'Fidx {i}')
    
# Labels
ax.set_xlabel('A', fontsize = 17)
ax.set_ylabel('B', fontsize = 17)

# Legend
leg = plt.legend()

# Adjust legend transparency
for lh in leg.legend_handles: 
    lh.set_alpha(1)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/overlap.png',dpi=dpi,bbox_inches='tight')