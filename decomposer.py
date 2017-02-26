import numpy as np
from solver import LeastLinearSquaresSolver

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
			self.solver = LeastLinearSquaresSolver()
		else:
			self.solver = solver

	def decompose(self, question):
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

		decomposition = [(self.embedder.dataset[i], x[i]) for i in range(len(x))]
		decomposition = sorted(decomposition, key=lambda x:x[1], reverse=True)

		return decomposition