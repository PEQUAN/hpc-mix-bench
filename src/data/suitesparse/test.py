import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix

filename = "1138_bus.mtx"
A = mmread(filename)

A = csc_matrix(A)
A_dense = A.toarray()

cond_number = np.linalg.cond(A_dense)

print(f"Condition number of the matrix: {cond_number:e}")
