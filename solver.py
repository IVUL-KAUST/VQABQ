import numpy as np
from scipy.linalg import eigh, svd
from sklearn.linear_model import Lasso
from abc import ABCMeta, abstractmethod

class Solver(object):
	'''Solves a linear combination problem (Ax=b) with some regularization if needed'''
	__metaclass__ = ABCMeta
	def __init__(self):
		pass

	@abstractmethod
	def solve(self, A, b):
		''' Solves Ax=b with regularization if needed.
		Args:
			A: An MxN matrix of feature vectors. 
			b: An Mx1 matrix.
		Returns:
			The solution x.
		'''
		pass

class LeastLinearSquaresSolver(Solver):
	def __init__(self):
		pass

	def solve(self, A, b):
		#optimize ||Ax-b||_2 using linear least squares and return the solution x
		x, _, _, _ = np.linalg.lstsq(A, b)
		return x

class LassoSolver(Solver):
	def __init__(self, l=1):
		self.l = l

	def solve(self, A, b):
		#optimize (0.5/n_samples) || Ax-b||_2^2 + l * ||x||_1 using Lasso and return the solution x
		lasso = Lasso(alpha=self.l)
		return lasso.fit(A, b).coef_

class ProximalL0Solver(Solver):
	def __init__(self, l=1):
		self.l = l

	def largest_eigen_AtA(self, A, At):
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

	def PPA(self, A, b, x=None, l=1, iterations=100, history=False, verbose=False):
		if history:
			hist = []

		#get the largest eigenvalue of AtA
		At = np.transpose(A)
		L = self.largest_eigen_AtA(A, At)

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

	def solve(self, A, b):
		#optimize 0.5 || Ax-b||_2^2 + l * ||x||_0 using PPA and return the solution x
		return self.PPA(A, b, l=self.l)