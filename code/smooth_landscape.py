import ast
import math
from sage.rings.integer import Integer
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import weights_functions as wf

# Get the weights of the smooth PR2 varieties
smooths = [[[ 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 7 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 6, 0, 1, 6 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 4, 0, 1, 5 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 2, 0, 1, 4 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 0, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 5, 5, 0, 1, 5 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 4, 3, 0, 1, 4 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 3, 1, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 2, 2, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 0, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 4, 4, 4, 0, 1, 4 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 3, 3, 2, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 2, 2, 0, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 2, 1, 1, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 0, 0, 0, 0, 1, 1 ]],
[[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 3, 3, 3, 3, 0, 1, 3 ]],
[[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 2, 2, 2, 1, 0, 1, 2 ]],
[[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 ]],
[[ 0, 0, 1, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 2, 2, 2, 2, 2, 0, 1, 2 ]],
[[ 0, 0, 1, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 ]],
[[ 0, 1, 1, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 1 ]],
[[ 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 6 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 5, 0, 1, 5 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 3, 0, 1, 4 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 4, 4, 0, 1, 4 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 3, 2, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 2, 0, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 3, 3, 3, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 2, 2, 1, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 ]],
[[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 2, 2, 2, 2, 0, 1, 2 ]],
[[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 ]],
[[ 0, 0, 1, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 1 ]],
[[ 0, 1, 1, 1, 1, 1, 1, 1, 5, 1 ],[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 1, 1, 1, 1, 1, 1, 1, 4, 1 ],[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 1, 1, 1, 1, 1, 1, 1, 3, 1 ],[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 1, 1, 1, 1, 1, 1, 1, 2, 1 ],[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 1, 1, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 4, 0, 1, 4 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 2, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 0, 0, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 0, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 3, 3, 0, 1, 3 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 2, 1, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 2, 2, 2, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 ]],
[[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 1 ]],
[[ 1, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],[ 3, 0, 1, 1, 1, 1, 1, 1, 3, 1 ]],
[[ 1, 0, 1, 1, 1, 1, 1, 1, 2, 1 ],[ 1, 1, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 1, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],[ 2, 0, 1, 1, 1, 1, 1, 1, 2, 1 ]],
[[ 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 ],[ 1, 1, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 1, 0, 1, 1, 1, 1, 1, 1, 1, 1 ],[ 1, 1, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 0, 1, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 0, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 2, 2, 0, 1, 2 ]],
[[ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 1, 1, 1, 0, 1, 1 ]],
[[ 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 ],[ 1, 1, 1, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 0, 0, 0, 0, 0, 1, 0 ]],
[[ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ],[ 1, 1, 1, 1, 0, 0, 0, 0, 1, 0 ]]]

# Store the regression data and the Fano indices for the smooth examples
A_smooths = []
B_smooths = []
fidx_smooths = []

for x in smooths:

    # Turn to column presentation and order the matrix
    x_cols = [[x[0][i],x[1][i]] for i in range(len(x[0]))]
    x_cols = wf.OrderMatrix(x_cols)

    # Turn back to row presentation
    x = [[y[0] for y in x_cols], [y[1] for y in x_cols]]

    # Compute growth coefficients
    alpha = wf.Bisection(wf.Polynomial, x, -Integer(x[0][-1])/Integer(x[1][-1]), Integer(10), 0.0001)
    A_smooths.append(wf.A(x, alpha))
    B_smooths.append(wf.B(x, alpha))

    # Compute Fano index
    a = sum(x[0])
    b = sum(x[1])
    fidx_smooths.append(math.gcd(a,b))

# Plot with logged colours
fig, ax = plt.subplots()

cm = plt.cm.get_cmap('viridis')
plt.scatter(A_smooths, B_smooths, c=fidx_smooths, cmap=cm, s=5, norm=mpl.colors.LogNorm(vmin = 1, vmax=54))

# Set axes labels
ax.set_xlabel('A', fontsize = 17)
ax.set_ylabel('B', fontsize = 17)

# Label colour bar
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Fano Index', rotation=270, fontsize = 17)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/smooth.png',dpi=dpi,bbox_inches='tight')

# Images over 100M dataset (gray)

# Store the regression coefficients
regression = []

# Read file from cleaned dataset of probable examples
with open('../data/terminal_dim8_probable.txt', 'r') as f:
   
    # Extract only the regression data, the line looks like 'Regression: [..,..]'
   for x in f:
       x = x.split(':')
       if 'Regression' in x[0]:
           regression.append(ast.literal_eval(x[1]))

plt.clf()

# Plot with logged colours
fig, ax = plt.subplots()

# Scatter plot 100M dataset
plt.scatter([x[0] for x in regression], [x[1] for x in regression], s=.1, alpha=.01, c='lightgray', label = '100M')

# Scatter plot smooth varieties
cm = plt.cm.get_cmap('viridis')
plt.scatter(A_smooths, B_smooths, c=fidx_smooths, cmap=cm, s=5, norm=mpl.colors.LogNorm(vmin = 1, vmax=54), label='ProdWPS')

line1 = Line2D([], [], color="white", marker='o', markerfacecolor="purple")
line2 = Line2D([], [], color="white", marker='o', markerfacecolor="lightgray")
plt.legend((line1, line2), ('Smooth RK2', '100M-dataset'), numpoints=1, loc=1)

# Set axes labels
ax.set_xlabel('A', fontsize = 17)
ax.set_ylabel('B', fontsize = 17)

# Label colour bar
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Fano Index', rotation=270, fontsize = 17)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/smooths_gray.png',dpi=dpi,bbox_inches='tight')