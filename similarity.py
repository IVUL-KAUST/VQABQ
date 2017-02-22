import json
import numpy as np
#from sklearn.manifold import MDS

class QuestionDecomposer:
	'''Composes any given question into simpler questions.

	It uses a dataset and a similarity measure.

	Attributes:
		dataset: The name of a json file that contains a list of simple questions (e.g. yes/no question).
		similarity_measure: A function that takes to input questions and returns a similarity score in [0,1].
		seed: Seed to the pseudo-random number generator.
	'''
	def __init__(self, dataset='questions.json', similarity_measure=None, seed=None):
		'''Initializes the dataset and similarity_measure.'''
		self.dataset = QuestionDecomposer.__load_questions(dataset)
		#TODO: To be removed
		self.dataset = self.dataset[:100] 
		self.N = len(self.dataset)
		if similarity_measure != None:
			self.__compare = similarity_measure
		else:
			self.__compare = QuestionDecomposer.__jaccard
		self.sim_mat = self.__generate_similarity_matrix()
		#rnd_state = np.random.RandomState(seed=seed)
		#mds = MDS(n_components=512, n_jobs=-1, random_state=rnd_state, dissimilarity="precomputed")
		#questions_embedding = mds.fit(sim_mat).embedding_
		#questions_embedding = np.transpose(questions_embedding) #dxN


	@staticmethod
	def __load_questions(input_file):
		'''Loads the list of questions from json file.
	
		Args:
			input_file: The path to the json file.

		Returns:
			A list of loaded questions.
		'''
		with open(input_file, 'r') as f:
			return json.load(f)

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

	def __generate_similarity_matrix(self):
		'''Generates the similarity matrix of the dataset using the similarity measure.
	
		Given a list of N questions and a similarity measure (compares two questions)
		it will generate NxN similarity symmetric matrix.

		Returns:
			The similarity matrix.
		'''
		#return [compare_to_all(q, dataset, measure) for q in dataset]
		#OR...
		sim_mat = np.zeros((self.N, self.N))
		for i in range(self.N):
			for j in range(i, self.N):
				sim_mat[i][j] = self.__compare(self.dataset[i], self.dataset[j])
		return sim_mat + np.transpose(sim_mat) - np.identity(self.N)*np.diagonal(sim_mat)

	def __compare_to_all(self, question):
		'''Compares a question against all questions in the dataset and returns a list of the similarity scores.
		
		Given a question and a list of questions we measure the similarity of the question
		against all the questions in the list using a similarity measure and return the values in a list
		The returned list is a numpy array of shape (1, len(dataset)).

		Args:
			question: The question to compare.

		Return:
			The list of similarity scores.
		'''
		return [self.__compare(question, q) for q in self.dataset]

	def decompose(self, question, number=10):
		'''Decomposes a question into basic questions.

		Given a question and a number n it outputs n basic questions from the dataset
		that are, when linearly combined, give the highest similarity score
		(according to the provided similarity measure) to the input question.

		Args:
			question: A general string question.
			number: The number of basic questions to generate.

		Returns:
			A list of basic questions.
		'''
		sim_list = self.__compare_to_all(question)
		sim_list = np.reshape(sim_list, (1, len(sim_list)))
		#question_embedding = mds.fit_transform(sim_list)
		question_embedding = np.transpose(sim_list)
		#Optimize using linear least square and Ax = b return the solution x
		x, _, _, _ = np.linalg.lstsq(self.sim_mat, question_embedding)
		nth_largest_element = np.sort(x, axis=0)[-number]
		mask = [1 if a>=nth_largest_element else 0 for a in x]
		indecies = np.nonzero(mask)[0]
		top_questions = [(x[i], self.dataset[i]) for i in indecies]
		top_questions = sorted(top_questions, key=lambda tup: tup[0], reverse=True)
		top_questions = [tup[1] for tup in top_questions]
		return top_questions[:number]