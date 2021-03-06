import numpy as np
from solver import LinearLeastSquaresSolver

class QuestionDecomposer:
	'''Composes any given question into simpler questions.

	It uses an Embedder to encode questions.

	Attributes:
		embedder: The embedder.
		solver: A solver to linearly combine questions.
	'''
	def __init__(self, embedder, solver=None):
		'''Initializes QuestionDecomposer.'''
		self.embedder = embedder
		if solver == None:
			self.solver = LinearLeastSquaresSolver()
		else:
			self.solver = solver

	def __get_basic_questions(self, x, eps, sort):
		decomposition = [(self.embedder.dataset[i], float(x[i])) for i in range(len(x))]
		
		if sort:
			decomposition = sorted(decomposition, key=lambda x:x[1], reverse=True)
		if eps != None:
			decomposition = [d for d in decomposition if d[1]>eps]

		return decomposition

	def decompose(self, question, eps=None, sort=True):
		'''Decomposes a question into basic questions.

		Given a question it outputs the decomposed basic questions 
		from the dataset that are, when linearly combined, give the highest similarity score
		(according to the provided similarity measure) to the input question.
		This method minimizes 0.5||Ax-b||_2^2+\lambda||x||_0

		Args:
			question: A general string question.

		Returns:
			A list of (basic questions, score) tuples.
		'''

		#get the embedding of the question and the dataset
		b = self.embedder.embed(question)
		A = self.embedder.embedded_dataset

		x = self.solver.solve(A, b)

		return self.__get_basic_questions(x, eps, sort)

	def decompose_all(self, questions, eps=None, sort=True):
		'''Decomposes a list of questions into basic questions.

		Given a question it outputs the decomposed basic questions 
		from the dataset that are, when linearly combined, give the highest similarity score
		(according to the provided similarity measure) to the input question.
		This method minimizes 0.5||Ax-b||_2^2+\lambda||x||_0

		Args:
			question: A list of general string questions.

		Returns:
			A list of list of (basic questions, score) tuples.
		'''

		#get the embedding of the question and the dataset
		B = [self.embedder.embed(question) for question in questions]
		b = np.stack(B, axis=1)
		A = self.embedder.embedded_dataset

		x = self.solver.solve_all(A, b)
		if len(x.shape)>1:
			decompositions = [0]*len(questions)
			for i in range(len(questions)):
				decompositions[i] = self.__get_basic_questions(x[:,i])

			return decompositions
		else:
			return [self.__get_basic_questions(x, eps, sort)]