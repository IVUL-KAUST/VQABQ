import numpy as np
from scipy.linalg import eigh, svd

def largest_eigen_AtA(A, At):
	#if one of the dimensions of A is small enough
	if np.min(A.shape)<5000:
		#consider an efficient method
		if A.shape[0]<A.shape[1]:
			AtA = np.matmul(A, At)
			s = A.shape[0]
		else:
			AtA = np.matmul(At, A)
			s = A.shape[1]
		return eigh(AtA, eigvals_only=True, eigvals=(s-1, s-1))[0]
	else: 
		#consider memory saving method
		_, s, _ = svd(A)
		L = np.max(s)
		return L*L

def PPA(A, b, x=None, l=1, iterations=100, history=False, verbose=False):
	if history:
		hist = []

	#get the largest eigenvalue of AtA
	At = np.transpose(A)
	L = largest_eigen_AtA(A, At)

	if x == None:
		if len(b.shape)>1:
			x = np.random.rand(A.shape[1], 1)
		else:
			x = np.random.rand(A.shape[1])

	l2L = np.sqrt(2*l/L)
	for i in range(iterations):
		diff = np.matmul(A,x)-b
		grad = np.matmul(At, diff)

		if verbose or history:
			fobj = 0.5*np.sum(diff**2)
		if verbose:
			print('iter:'+str(i)+', fobj:'+str(fobj))
		if history:
			hist.append(fobj + l*np.count_nonzero(x))

		# x <- x - gradient/L
		x = x - grad/L

		# x <- argmin_{y} 0.5 ||y - x||_2^2 + l * ||y||_0
		x[abs(x)<=l2L] = 0

	if history:
		return x, hist
	else:
		return x

def plot(hist):
	import matplotlib.pyplot as plt
	plt.plot(hist)
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	plt.show()

def save(A, file):
	m = A.shape[0]
	if len(A.shape)<2:
		n = 1
	else:
		n = A.shape[1]

	with open(file+'.txt', 'w') as f:
		for i in range(m):
			for j in range(n):
				f.write(str(A[i,j])+' ')
			f.write('\n')

def test():
	# min_{x} 0.5 || Ax-b||_2^2 + l * ||x||_0
	m = 1000
	n = 10000
	np.random.seed(0)
	A = np.random.rand(m,n)
	b = np.random.rand(m,1)
	x = np.random.rand(n,1)
	#save(A, 'A')
	#save(b, 'b')
	#save(x, 'x')
	l = 1
	x = PPA(A,b, verbose=True)