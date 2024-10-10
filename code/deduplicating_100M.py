import ast

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

def gcd_euclid(a, b):
    '''
    Calculates the A, B, g, such that Aa + Bb = g = gcd(a,b).

    Args:
        a: integer
        b: integer
    Return:
        Tuple (g,A,B) such that Aa + Bb = g = gcd(a,b).
    '''
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = gcd_euclid(b % a, a)
        return (g, x - (b // a) * y, y)

def ComputePartner(w):

    # Turn to columns
    w = [[w[0][i],w[1][i]] for i in range(len(w[0]))]

    # If the outmost rays are on the axis then just swap the rows and reorder
    if w[-1][0] == 0:
        w = [[x[1],x[0]] for x in w]
        return OrderMatrix(w)
    
    # If you are not on the axis perform SL2(Z) transformations
    else:

        # Swap the rows and order
        w = OrderMatrix([[x[1],x[0]] for x in w])

        # Get the gcd of the first column
        G, A, B =  gcd_euclid(w[0][0], w[0][1])

        # Move the first column on the x-axis
        w = [[int(A*x[0]+B*x[1]), round(-w[0][1]*x[0]/G + w[0][0]*x[1]/G)] for x in w]

        # If we are not in the positive quadrant move there
        if w[-1][0]<0:
            n = 0
            while w[-1][0]+n*w[-1][1]<0:
                n+=1
            w = [[x[0]+n*x[1], x[1]] for x in w]
            return OrderMatrix(w)
        
        # If the last column is so that x>=y, move to x<y
        elif w[-1][0]>=w[-1][1]:
            n = 0
            while w[-1][0]+n*w[-1][1]>= w[-1][1]:
                n-=1
            w = [[x[0]+n*x[1], x[1]] for x in w]
            return OrderMatrix(w)
        
        # If neither of the above happens than just return
        else:
            return OrderMatrix(w)

import os

# List files output of hpc (saved in data folder)
files = os.listdir('../data_hpc/')

# Store the hashes
terminal = 0
i = 0
hashes = set()
print(files)

with open('../data/terminal_dim8_probable.txt', 'a') as s:
    while terminal<100000000:
        if i<len(files):
            print(i)
            if files[i]!='terminal_dim8_probable.txt':
                with open('../data_hpc/'+files[i],'r') as t:

                    # Each row is a dictionary {'Weights': string, 'Regression': string, 'FanoIndex':string, ...}
                    for row in t:
                        # Evaluate dictionary
                        d = eval(row)

                        # Extract weights
                        w = eval(d['Weights'])

                        # Compute weights partner
                        w_partner_cols = ComputePartner(w)
                        w_partner = [[x[0] for x in w_partner_cols], [x[1] for x in w_partner_cols]]

                        # Compute hashes
                        hw = hash(str(w))
                        hwp = hash(str(w_partner))

                        # Check for hash collision
                        if min(hw, hwp) not in hashes:
                            hashes.add(min(hw, hwp))
                            terminal+=1

                            # Write to deduplicated file
                            s.write('Weights: '+str(w)+'\n')
                            s.write('Regression: '+d['Regression']+'\n')
                            s.write('FanoIndex: '+d['FanoIndex']+'\n')
                            s.write('\n')
            i+=1
        else:
            break


