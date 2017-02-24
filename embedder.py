import numpy as np
from abc import ABCMeta, abstractmethod

class Embedder(object):
	'''Embeds questions into some vector space.

	Attributes:
		dataset: The list of questions.
		embedded_dataset: The embedding of the dataset (should only be accessed after a call to embed()).
	'''
	__metaclass__ = ABCMeta

	def __init__(self, dataset):
		'''Initializes the embedder'''
		self.dataset = dataset
		self.embedded_dataset = None

	@abstractmethod
	def embed(self, question):
		'''Embed a question into some vector space.

		The overriding methods must update self.embedded_dataset if needed
		
		Args:
			questions: A general string question.

		Returns:
			The embedded question as a column vector.
		'''
		raise NotImplementedError

class SimilarityEmbedder(Embedder):
	'''Embeds questions using some similarity measure.

	Generate a symmetric similarity matrix out of the questions 
	and reduce the dimensionality of the matrix using MDS(multidimensional scaling).
	'''
	def __init__(self, dataset, similarity_measure=None, reducer=None):
		'''Initializes the dataset, similarity measure, dimensionality, and the seed
		
		Args:
			dataset: The name of a json file that contains a list of simple questions (e.g. yes/no question).
			similarity_measure: A function that takes two questions and return a score in [0,1].
			reducer: A reducer object. If None is given, no dimensionality reduction is used.
		'''
		Embedder.__init__(self, dataset=dataset)
		if similarity_measure != None:
			self.__compare = similarity_measure
		else:
			self.__compare = SimilarityEmbedder.__jaccard
		self._sim_mat = self.__generate_similarity_matrix(self.dataset)
		self.reducer = reducer

	def embed(self, question):
		#compute similarity scores as a column vector
		scores = np.array([self.__compare(question, q) for q in self.dataset])
		scores = np.reshape(scores, (len(scores), 1))
		
		if self.reducer:
			#append to similarity matrix
			sim_mat = np.concatenate((self._sim_mat, scores), axis=1)
			#append 1 to the end and make it into row vector
			scores = np.transpose(np.concatenate((scores, [[1]]), axis=0))
			#append to similarity matrix
			sim_mat = np.concatenate((sim_mat, scores), axis=0)

			#reduce dimensionality using MDS
			questions_embedding = self.reducer.reduced(sim_mat)

			#update self._embedded
			self.embedded_dataset = questions_embedding[:,:-1]

			#return embedded question
			return questions_embedding[:,-1]
		else:
			self.embedded_dataset = self._sim_mat
			return scores

	@staticmethod
	def __jaccard(str1, str2):
		'''Compares two questions using Jaccard method (intersection over union).

		Returns:
			Similarity score between zero and one.
		'''
		a = set(str1.strip('?').split(' '))
		b = set(str2.strip('?').split(' '))
		intersection = float(len(a&b))
		union = float(len(a|b))
		jaccard = intersection/union
		return jaccard

	def __generate_similarity_matrix(self, questions):
		'''Generates the similarity matrix of the questions using the similarity measure.
	
		Given a list of N questions and a similarity measure (compares two questions)
		it will generate NxN similarity symmetric matrix.

		Returns:
			The similarity matrix.
		'''
		#return [compare_to_all(q) for q in dataset]
		#OR...
		N = len(questions)
		sim_mat = np.zeros((N, N))
		for i in range(N):
			for j in range(i, N):
				sim_mat[i][j] = self.__compare(questions[i], questions[j])
		return sim_mat + np.transpose(sim_mat) - np.identity(N)*np.diagonal(sim_mat)