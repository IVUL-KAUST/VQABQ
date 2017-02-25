import numpy as np
from scipy.linalg import eigh

largest_eigenvalue = lambda A: eigh(A, eigvals_only=True, eigvals=(A.shape[1]-1, A.shape[1]-1))[0]

def PPA(A, b, l=1, x=None, iterations=100):
	#get the largest eigen value of AtA
	#TODO: replace with Lanczsos algorithm
	At = np.transpose(A)
	L = largest_eigenvalue(np.matmul(At, A))
	l2L = np.sqrt(2*l/L)

	if not x:
		x = np.random.rand(A.shape[1])

	for i in range(iterations):
		# x <- x - gradient/L
		x -= np.matmul(At, np.matmul(A,x)-b)/L

		# x <- argmin_{y} 0.5 ||y - x||_2^2 + l * ||y||_0
		x[np.abs(x)<=l2L] = 0

	return x