import numpy as np
from scipy.linalg import eigh

def computeObj(x,A,b):
	diff = np.matmul(A,x)-b
	fobj = 0.5*np.sum(diff**2)
	grad = np.matmul(np.transpose(A), diff)
	return fobj, grad

def PPA(x, A, b, l, verbose=False):
	hist = []
	AtA = np.matmul(np.transpose(A), A)

	#get the largest eigen value of AtA
	L = eigh(AtA, eigvals_only=True, eigvals=(A.shape[1]-1, A.shape[1]-1))[0]

	for i in range(100):
		fobj, grad = computeObj(x,A,b)
		if verbose:
			print('iter:'+str(i)+', fobj:'+str(fobj))
		x -= grad/L

		# solving the following OP:
		# x <- argmin_{y} 0.5 ||y - x||_2^2 + l * ||y||_0
		x[(x**2)/(2*l/L)<=1] = 0

		hist.append(fobj + l*np.count_nonzero(x))

	return x, hist

def plot(hist):
	import matplotlib.pyplot as plt
	plt.plot(hist)
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	plt.show()

def test():
	# min_{x} 0.5 || Ax-b||_2^2 + l * ||x||_0
	n = 100
	m = 100
	A = np.random.rand(m,n)
	b = np.random.rand(m,1)
	x = np.random.rand(n,1)
	l = 1
	x, hist = PPA(x,A,b,l, verbose=True)
	plot(hist)