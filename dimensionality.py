import numpy as np
from sklearn.manifold import MDS, Isomap
from abc import ABCMeta, abstractmethod

class Reducer(object):
	'''Dimensionality reduction tool.'''
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def reduced(self, A):
		'''Reduce the dimensionality of a given column feature vectors.
		Args:
			A: A matrix of column feature vectors.

		Returns:
			The reduced column feature vectors.
		'''
		pass

class MDS_Reducer(Reducer):
	'''The multidimensional scaling (MDS) reduction method'''
	def __init__(self, dimensionality=2500, seed=None):
		rnd_state = np.random.RandomState(seed=seed)
		self.mds = MDS(n_components=dimensionality, n_jobs=-1, random_state=rnd_state, dissimilarity="precomputed")

	def reduced(self, A):
		embd = self.mds.fit(A).embedding_
		return np.transpose(embd)

class ISO_Reducer(Reducer):
	'''Iso map reduction method'''
	def __init__(self, dimensionality=2500):
		self.iso = Isomap(n_neighbors=5, n_components=dimensionality, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=-1)

	def reduced(self, A):
		embd = self.iso.fit(A).embedding_
		return np.transpose(embd)