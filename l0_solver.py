import numpy as np
from scipy.linalg import eigh

def prox_l0(a, l):
	# solving the following OP:
	# min_{x} 0.5 ||x - a||_2^2 + l * ||x||_0

	a[(a**2)/(2*l)<=1] = 0

	return a

def computeObj(x,A,b):
	diff = np.matmul(A,x)-b
	fobj = 0.5*np.sum(diff**2)
	grad = np.matmul(np.transpose(A), diff)
	return fobj, grad

def PPA(x, A, b, l, verbose=False):
	hist = []
	At = np.transpose(A)
	AtA = np.matmul(At, A)

	#get the largest eigen value of AtA
	L = eigh(AtA, eigvals_only=True, eigvals=(A.shape[1]-1, A.shape[1]-1))[0]

	for i in range(100):
		fobj, grad = computeObj(x,A,b)
		if verbose:
			print('iter:'+str(i)+', fobj:'+str(fobj))
		x = x - grad/L
		x = prox_l0(x, l/L)
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
	n = 1000
	m = 1000
	A = np.random.rand(m,n)
	b = np.random.rand(m,1)
	x = np.random.rand(n,1)
	l = 1
	x, hist = PPA(x,A,b,l, verbose=True)
	plot(hist)