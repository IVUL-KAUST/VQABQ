from l0_solver import PPA
from sklearn.linear_model import Lasso
from abc import ABCMeta, abstractmethod

class Solver(object):
	__metaclass__ = ABCMeta
	def __init__(self):
		pass

	@abstractmethod
	def solve(self, A, b):
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

	def solve(self, A, b):
		#optimize 0.5 || Ax-b||_2^2 + l * ||x||_0 using PPA and return the solution x
		return PPA(A, b, l=self.l)