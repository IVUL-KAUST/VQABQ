import numpy as np

class QuestionDecomposer:
	'''Composes any given question into simpler questions.

	It uses an Embedder to encode questions.

	Attributes:
		embedder: The embedder.
	'''
	def __init__(self, embedder):
		'''Initializes QuestionDecomposer.'''
		self.embedder = embedder

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

		#optimize using linear least square and Ax = b return the solution x
		x, _, _, _ = np.linalg.lstsq(A, b)

		decomposition = [(self.embedder.dataset[i], x[i]) for i in range(len(x))]
		decomposition = sorted(decomposition, key=lambda x:x[1], reverse=True)

		return decomposition