import lutorpy as lua
from abc import ABCMeta, abstractmethod

class VQA(object):
	'''A general Visual Question Answering solver'''
	__metaclass__ = ABCMeta
	def __init__(self):
		pass
	@abstractmethod
	def answer(self, image_path, question):
		'''Generate an answer to the VQA problem.

		Args:
			image_path: The path to the input image.
			question: A general string question.
		Returns:
			The answer to the question based on the image.
		'''
		pass

class CoAttenVQA(VQA):
	'''HieCoAttenVQA solver using VGG-19 and VQA dataset
	as described here: https://github.com/jiasenlu/HieCoAttenVQA
	'''
	def __init__(self):
		require('load')
		require('predict')
		self.__protos = load()

	def answer(self, image_path, question):
		splt = question.split(' ')
		if len(splt)>26:
			question = ' '.join(splt[:26])
		return predict(image_path, question, self.__protos)