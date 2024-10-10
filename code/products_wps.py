import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations_with_replacement
import numpy as np
import copy
import ast
from matplotlib.lines import Line2D

def mu_and_nu(a,b):
    '''
    Returns mu and nu for a product of weighted projective spaces with weights a and b.

    Args:
        a, b: lists of positive integers (the weights for the weighted projective spaces)
    Return:
        Tuple (mu, nu)
    '''

    # Check that we are in dimension 8 and rank 2
    if len(a) + len(b) != 10:
        raise ValueError("length mismatch")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("both a and b must have length at least 2")
    
    # Sum of the weights
    aa = sum(a)
    bb = sum(b)

    # Calculate mu and nu
    mu = math.prod([x**x for x in b])**(1/bb)
    nu = math.prod([x**x for x in a])**(1/aa)

    return mu, nu

def A_and_B(a,b):
    '''
    Returns A, B, (the growth coefficients) and the Fano index for a product of weighted projective spaces with weights a and b.

    Args:
        a, b: lists of positive integers (the weights for the weighted projective spaces)
    Return:
        Tuple (A, B, Fano-index)
    '''

    # Calculate mu and nu for this product of weighted projective spaces
    mu, nu = mu_and_nu(a,b)

    # Store the probabilities
    p = []

    # Sum of the weights
    aa = sum(a)
    bb = sum(b)

    # Denominators for the probabilities
    den = mu*aa+nu*bb

    # Fano index
    ell = math.gcd(aa,bb)

    # Calculate the probabilities
    for i in range(10):
        if i<len(a):
            p.append(mu*a[i]/den)
        else:
            j = i-len(a)
            p.append(nu*b[j]/den)
    
    # Extra term in B
    y = []
    for i in range(10):
        if i<len(a):
            y.append(a[i]*bb*a[i]*bb/(ell*ell*p[i]))
        else:
            j = i-len(a)
            y.append(b[j]*aa*b[j]*aa/(ell*ell*p[i]))

    # Calculate A and B
    A = sum([-x*math.log(x) for x in p])
    B = -4*math.log(2*math.pi)-sum([0.5*math.log(x) for x in p]) - 0.5*math.log(sum(y))

    return A, B, ell

def IsTerminalWPS(w):
    '''
    Implementation of the terminality criterion for weighted projective spaces from 'Classifying terminal weighted projective space' by Alexander M. Kasprzyk (2013).

    Arg:
        w: list of positive integers (weights of the weighted projective space)
    Return:
        True (if terminal), False (if not terminal)
    '''
    
    # Sum of the weights
    h = sum(w)

    for k in range(2, h-1):
        # Rational vector
        rv = [k*x/h for x in w]

        # Check condition
        if round(sum(x-math.floor(x) for x in rv)) not in list(range(2,h)):
            return False
    return True

def IsWellFormedWPS(w):
    '''
    Checks if the weighted projective space given by the weights w is well-formed.

    Arg:
        w: list of positive integers (weights of the weighted projective space)
    Return:
        True (if well-formed), False (if not well-formed)
    '''
    
    if np.gcd.reduce(w) != 1:
        return False
    
    for x in w:
        w_copy = w.copy()
        w_copy.remove(x)
        if np.gcd.reduce(w_copy) != 1:
            return False
    return True


# Generate all wps of dimension 1,...,8
wps = {i:[] for i in range(2,9)}

for i in range(2,9):
    for x in combinations_with_replacement(range(1,8),i):
        x = list(x)
        # Check if wellformed and terminal
        if IsWellFormedWPS(x) and IsTerminalWPS(x):
            wps[i].append(x)

# Compute all possible products
prod_wps = []
for j in range(2,5):
    prod_wps = prod_wps +[[x,y] for x in wps[j] for y in wps[10-j]]

# Deal with symmetric case
fives = []
for x in wps[5]:
    for y in wps[5]:
        if [y,x] not in fives:
            fives.append([x,y])

# Add the ones coming from products of 4 dimensional weighted projective spaces
prod_wps = prod_wps + fives

# Store As, Bs, and Fano indies for all products of projective spaces
As = []
Bs = []
fidxs = []
for x in prod_wps:
    A, B, ell = A_and_B(x[0],x[1])
    As.append(A)
    Bs.append(B)
    fidxs.append(ell)

# Plot with logged colours
fig, ax = plt.subplots()

cm = plt.cm.get_cmap('viridis')
plt.scatter(As, Bs, c=fidxs, cmap=cm, s=0.1, norm=mpl.colors.LogNorm(vmin = 1, vmax=54))

# Set axes labels
ax.set_xlabel('A', fontsize = 17)
ax.set_ylabel('B', fontsize = 17)

# Label colour bar
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Fano Index', rotation=270, fontsize = 17)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/product_wps_new.png',dpi=dpi,bbox_inches='tight')


# Images over 100M dataset (gray)

# Store regression data
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

# Scatter 100M
plt.scatter([x[0] for x in regression], [x[1] for x in regression], s=.1, alpha=.01, c='lightgray', label = '100M')

# Scatter producs of wps
cm = plt.cm.get_cmap('viridis')
plt.scatter(As, Bs, c=fidxs, cmap=cm, s=0.1, norm=mpl.colors.LogNorm(vmin = 1, vmax=54), label='ProdWPS')

# Legend
line1 = Line2D([], [], color="white", marker='o', markerfacecolor="purple")
line2 = Line2D([], [], color="white", marker='o', markerfacecolor="lightgray")
plt.legend((line1, line2), ('Products-WPS', '100M-dataset'), numpoints=1, loc=1)

# Set axes labels
ax.set_xlabel('A', fontsize = 17)
ax.set_ylabel('B', fontsize = 17)

# Label colour bar
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Fano Index', rotation=270, fontsize = 17)

# Save the images
for dpi in [300,600,1200]:
    plt.savefig(f'../images_{dpi}/product_wps_gray.png',dpi=dpi,bbox_inches='tight')