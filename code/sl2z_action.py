# # write a class that represents the action of SL(2,Z) on a matrix of positive integers of dimension 2x(n+2)
# import numpy 

# class SL2ZAction:
#     '''
#     This class represents the action of SL(2,Z) on a matrix of positive integers of dimension 2x(n+2).
#     '''
#     def __init__(self, matrix):
#         '''
#         Constructor for the SL2ZAction class.
#         Args:
#             matrix: a list of lists of non-negative integers of dimension 2x(n+2).
#         '''
#         # Sanity checks
#         if len(matrix) != 2:
#             raise ValueError('Matrix must have two rows')
#         if len(matrix[0]) != len(matrix[1]):
#             raise ValueError('Matrix must have the same number of columns')
#         if len(matrix[0]) < 2:
#             raise ValueError('Matrix must have at least two columns')
        
#         # Store the matrix
#         self.matrix = matrix
        
#         # Store the dimension
#         self.dim = len(matrix[0])-2
        
    
#     def action_S(self):
#         # Get the matrix
#         m = self.matrix

#         # make the entries of a list negative
#         return list([list(-numpy.array(m[1])),list(numpy.array(m[0]))])
    
#     def action_T(self):

#         # Get the matrix
#         m = self.matrix
        
#         # Get the action
#         return list([list(numpy.array(m[1]+m[0])),list(numpy.array(m[1]))])


# write a class that represents the action of SL(2,Z) on a matrix of positive integers of dimension 2x(n+2)
import numpy 
    
def action_S(m):
    # make the entries of a list negative
    return list([list(-numpy.array(m[1])),list(numpy.array(m[0]))])

def action_T(m):
    # Get the action
    return list([list(numpy.array(m[1])+numpy.array(m[0])),list(numpy.array(m[1]))])

def action_st3(m):
    return list([list(-numpy.array(m[1])),list(-numpy.array(m[0]))])

def action_st2(m):
    # t^-1 = (1 -1; 0 1)
    # s^-1 = (0 1; -1 0)
    # negative 
    return list([list(-numpy.array(m[1]+m[0])),list(numpy.array(m[0]))])
    