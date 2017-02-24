import numpy as np
from sklearn.manifold import MDS
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