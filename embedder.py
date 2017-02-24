import pickle
import numpy as np
from os.path import isfile
import skpt.skipthoughts as sk
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
	_sim_mat_file = './models/sim_mat.pickle'
	def __init__(self, dataset, similarity_measure=None, reducer=None, load=True, save=False):
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
		if load and isfile(self._sim_mat_file):
			sim_mat = pickle.load(open(self._sim_mat_file, 'rb'))
		else:
			print('generating similarity matrix for the dataset...')
			sim_mat = self.__generate_similarity_matrix(self.dataset)
			if save:
				print('saving generated similarity matrix...')
				pickle.dump(sim_mat, open(self._sim_mat_file, 'wb'))
		self._sim_mat = sim_mat
		self.reducer = reducer

	def embed(self, question):
		#compute similarity scores as a column vector
		scores = np.array([self.__compare(question, q) for q in self.dataset])
		
		if self.reducer:
			scores = np.reshape(scores, (len(scores), 1))
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
			return np.reshape(scores, (len(scores),))

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

class SkipThoughtEmbedder(Embedder):
	_embedded_dataset_file = './models/embedded_dataset.pickle'
	def __init__(self, dataset, load=True, save=False):
		'''Initializes SkipThoughtEmbedder

		Args:
			dataset: The name of a json file that contains a list of simple questions (e.g. yes/no question).
			load: Load the processed dataset if True
			save: Save the processed dataset if True
		'''
		Embedder.__init__(self, dataset=dataset)
		self._model = sk.load_model()
		if load and isfile(self._embedded_dataset_file):
			print('loading processed dataset...')
			embedded_dataset = pickle.load(open(self._embedded_dataset_file, "rb"))
		else:
			print('preprocessing dataset...')
			embedded_dataset = sk.encode(self._model, dataset, verbose=False)
			embedded_dataset = np.transpose(embedded_dataset)
			if save:
				print('saving processed dataset...')
				pickle.dump(embedded_dataset, open(self._embedded_dataset_file, "wb"))
		self.embedded_dataset = embedded_dataset
			

	def embed(self, question):
		encd = sk.encode(self._model, [question], verbose=False)
		return np.reshape(encd, (encd.shape[1],))