import os
import time
from vqa import CoAttenVQA

class VQAEvaluator(object):
	'''Evaluates a VQA dataset of questions and generate the answers'''
	def __init__(self, vqa=None, concatenate=None):
		'''Initializes the VQAEvaluator

		Args:
			vqa: A VQA object.
			concatenate: A function that take a question and a list of basic questions
					with a score value and append them into one question
					basic = [{'question':'DUMMY'
							  'score':0.123
					}]
		'''
		if vqa == None:
			self.__vqa = CoAttenVQA()
		else:
			self.__vqa = vqa
		if concatenate == None:
			self.__concat = VQAEvaluator.__concatenate
		else:
			self.__concat = concatenate

	@staticmethod
	def __concatenate(question, basic):
		'''Concatenate the main question with the basic questions

		Args:
			question: The main question.
			basic: The list of basic questions with their similarity score.
		Returns:
			The concatenated question.
		'''
		basic = [b['question'] for b in basic]
		return question+' '+' '.join(basic)

	def evaluate(self, dataset, image_folder='.'):
		'''Compute the answers of the dataset and return them in a list
		
		Args:
			dataset: The dataset should have the following structure
				dataset = [{'question_id':123,
							'question':'DUMMY', 
							'image_path':'DUMMY', 
							'basic':[
								{'question':'DUMMY','score':0.123}
							]
				}]
			image_folder: The folder where all the images are.
		Returns:
			The list of answers as
				result = [{'question_id':123, 'answer':'DUMMY'}]
		'''
		skipped = 0
		result = [0]*len(dataset)
		start_time = time.time()
		for i in range(len(dataset)):
			if i%100 == 0:
				if i != 0:
					elapsed_time = time.time() - start_time
					print('finished '+str(i+1-skipped)+' questions')
					print('current speed: '+str(100.0/elapsed_time))
					start_time = time.time()
			d = dataset[i]
			image_path = os.path.join(image_folder, d['image_path'])
			quesiton = self.__concat(d['question'], d['basic'])
			try:
				answer = self.__vqa.answer(image_path, quesiton)
				result[i] = {'question_id':d['question_id'], 'answer':answer}
			except:	
				skipped += 1
				print('skipped question at index #'+str(i))
		print('finished '+str(i+1-skipped)+' questions and skipped '+str(skipped)+' questions')
		return result