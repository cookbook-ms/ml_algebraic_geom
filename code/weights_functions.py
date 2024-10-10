import random
import numpy as np
import math
from sage.rings.integer import Integer # we import Integer from sage for the regression calculation

# Functions to generate and validate the matrix (GenerateWeights, IsNormalForm, Validate, OrderMatrix, IsWellFormed, IsRankTwo)
def GenerateWeights(n,k):
    ''' 
    Generate a random 2xn matrix with entries bounded between 0 and k (inclusive).

    Args:
        n: integer
        k: integer

    Return: 
        A list of lists (the columns of the matrix)

    '''
    return [[random.randint(0,k) for i in range(2)] for j in range(n)]

def IsNormalForm(weights):
    ''' 
    Checks whether a weights matrix (cyclically ordered and given as columns) is in Normal Form.

    Arg:
        weights: a list of lists of non-negative integers (given as the columns).

    Return:
        True (if the weight matrix is in Normal Form), False (if it is not). 

    '''
    # Check that the first colum is of the form [a1,0]
    if weights[0][1] != 0 :
        return False
    
    # Check that the last column is of the form [aN, bN] for aN<bN
    elif weights[-1][0] >= weights[-1][1]:
        return False
    
    return True

def Validate(weights):
    ''' 
    Validates the weights matrix by checking whether it has a zero column and whether the sum of the weigths lies on a wall.

    Arg:
        weights: a list of lists of non-negative integers (given as the columns of the matrix).

    Return:
        True (if the weight matrix passes validation), False (if it does not pass validation).
    
    '''
    # Check whether the zero column is in the weight matrix
    if [0,0] in weights:
        return False
    
    else:
        # Compute the sum of the weights
        sweights = [sum(x[0] for x in weights), sum(x[1] for x in weights)]

        # Check whether the sum of the weights lies on a wall
        for x in weights:
            if x[0]*sweights[1] == x[1]*sweights[0]:
                return False
    
    return True

def OrderMatrix(weights):
    ''' 
    Orders the columns of the weights matrix cyclically.

    Arg:
        weights: a list of lists of non-negative integers (given as the columns).

    Return:
        a list of lists of non-negative integers (now we have ordered this cyclically).

    '''
    # Order according to top row first
    weights.sort(key = lambda x : x[0])

    # Separate into sections
    zerou = [i for i in weights if i[0] == 0]
    zerov = [i for i in weights if i[1] == 0]
    nonzero = [i for i in weights if i[0]*i[1] != 0]

    # Order each section (zerov is already ordered)
    zerou.sort(key = lambda x: x[1])
    nonzero.sort(key = lambda x: x[1]/x[0])

    # Regroup the sections
    return zerov + nonzero + zerou

def Determinant(v1,v2):
    ''' 
    Returns the determinant of the 2x2 matrix with columns v1 and v2.
    '''
    return v1[0]*v2[1]-v1[1]*v2[0]

def IsWellFormed(weights):
    ''' Checks whether the weight matrix is well-formed.
    Arg: 
        weights: list of lists of non-negative integers 
    Return
        True (if the weights are well-formed) False (otherwise)
    '''
    # Take the sum of the weights
    K = [sum(x[0] for x in weights), sum(x[1] for x in weights)]

    i = 0
    n = len(weights)

    # Define the left and right of the stability condition
    while i < n and Determinant(weights[i],K) > 0:
        i+=1

    # Check that it is truly rank 2
    if i == 1 or i == n-1:
        return False
    # Calculate all the determinants
    determinants = {i:[] for i in range(len(weights))}

    # Loop over the columns of the weight matrix
    for i, x in enumerate(weights):
        for j, y in enumerate(weights[i+1:]):
            
            d = Determinant(x,y)
            for k in range(len(weights)):
                if k!=i and k!=j+1+i:
                    determinants[k].append(d)

    # Check that the gcds are all one
    for k in range(len(weights)):
        if abs(np.gcd.reduce(determinants[k])) != 1:
            return False
        
    return True

# Alpha calculation functions (Bisection, BisectionInternal, Polynomial)
def Bisection(f, w, bound1, bound2, tol=0.01):
    ''' 
    Wrapper for _bisection, performs the sanity checks only once.

    Args:
        f: Polynomial function.
        w: weights, list of lists of non-negative integers (given as rows).
        bound1, bound2: extremes of the interval where we are performing bisection search (bound1<bound2).
        tol: tolerance (float), default value 0.01.

    Return:
        The approximated positive root (as a float).
    
    Raises:
        ValueError if bound1 > bound2.
        ValueError if tol is not positive.
    
    '''
    # Sanity checks
    if bound1 > bound2:
        raise ValueError('bound1 must be less than bound2')
    elif tol <= 0:
        return ValueError('the tolerance must be a positive number')
    
    return _bisection(f, w, bound1, bound2, tol)

def _bisection(f, w, bound1, bound2, tol):
    ''' 
    Computes the positive real solution of the polynomial f, with tolerance tol, using the Bisection method.

    Args:
        f: Polynomial function.
        w: weights, list of lists of non-negative integers (given as rows).
        bound1, bound2: extremes of the interval where we are performing bisection search (bound1<bound2).
        tol: tolerance (float).

    Return:
        The approximated positive root (as a float).
    
    '''  
    # Define low and high
    low = bound1
    high = bound2

    # If we do not have a change of sign in the interval, we shift along the positive real line
    if f(low, w)*f(high, w) > 0:
        return _bisection(f, w, bound2, 2*bound2-bound1, tol)
    
    # If there is a change of sign in the interval, we run the bisection 
    else:
        # Take the mid point of the interval
        mid = low + (high-low)/Integer(2)

        # If the value is less than the tolerance we return (or if we find the root)
        if abs(mid-low) < tol or f(mid, w) == 0:
            return float(mid)
        
        # Apply the bisection to the subinterval where we have the change of sign
        elif f(low, w)*f(mid, w) < 0:
            return _bisection(f, w, low, mid, tol)
        elif f(high, w)*f(mid, w) < 0:
            return _bisection(f, w, mid, high, tol)

def Polynomial(x,w):
    ''' 
    Returns the value of the polynomial at x, which is determined by the weight matrix w.

    Args: 
        x: float.
        w: weights, list of lists of non-negative integers (given as rows).

    Return: 
        A float (the value of the polynomial at x).

    '''
    # Take the sum of the weights
    a = Integer(sum(w[0]))
    b = Integer(sum(w[1]))

    # Initialise polynomials
    f = Integer(1)
    g = Integer(1)

    # Loop over the columns of the weight matrix
    for i in range(len(w[0])):
        if w[0][i]*b - w[1][i]*a > 0:
            f = f*(Integer(w[0][i]) + x*Integer(w[1][i]))**Integer(w[0][i]*b - w[1][i]*a)
        else:
            g = g*(Integer(w[0][i]) + x*Integer(w[1][i]))**Integer(-w[0][i]*b + w[1][i]*a)

    # Return the difference
    return f-g

# Regression calculation (A, B)
def A(wts, alpha):
    ''' 
    The linear term in the asymptotics for the quantum period of a Picard rank 2 toric variety.

    Args:
        wts: list os lists of non-negative integers (given as rows)
        alpha: float

    Return:
        Float

    '''
    # Take the sum of the weights
    a = sum(wts[0])
    b = sum(wts[1])
    
    # Calculate the probabilities
    p = [(wts[0][i]+wts[1][i]*alpha)/(a+alpha*b) for i in range(len(wts[0]))]

    return -sum(x*math.log(x) for x in p)

def B(wts, alpha):
    '''
    The constant term in the asymptotics for the quantum period of a Picard rank 2 toric variety.

    Args:
        wts: list os lists of non-negative integers (given as rows) 
        alpha: float

    Return:
        Float

    '''
    # Calculate the dimension
    D = len(wts[0])-2

    # Take the sum of the weights
    a = sum(wts[0])
    b = sum(wts[1])

    # Calculate the probabilities
    p = [(wts[0][i]+wts[1][i]*alpha)/(a+alpha*b) for i in range(len(wts[0]))]

    # Theta term 
    theta = [(wts[0][i]*b - a*wts[1][i])**2/p[i] for i in range(len(p))]

    return 0.5*(-D*math.log(2*math.pi) - sum(math.log(x) for x in p) + 2*math.log(math.gcd(a,b)) - math.log(sum(theta)))