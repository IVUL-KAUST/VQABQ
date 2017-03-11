import json
import numpy as np
from embedder import SkipThoughtEmbedder
from nltk.tokenize import word_tokenize
from solver import LeastLinearSquaresSolver


#---------------------------------------------------------------------------
#	Utility Functions
#---------------------------------------------------------------------------

#load json file
#return the loaded file
def load(file):
	with open(file, 'r') as f:
		data = json.load(f)
	return data

#save given data to json file
def save(file, data):
	with open(file, 'w') as f:
		json.dump(data, f)

#clean a given sentence
#return the cleaned sentence
def clean(sentence):
	sentence = word_tokenize(str(sentence).lower())
	return ' '.join(sentence)

#---------------------------------------------------------------------------
#	Handy Functions
#---------------------------------------------------------------------------

#embed questions into feature vectors
#return the feature vectors
def embed(questions, load=None, save=None, embedder=None):
	if type(questions) is list:
		questions = [clean(q) for q in questions]
	else:
		questions = [clean(questions)]
	if embedder or load:
		if load:
			embedder = SkipThoughtEmbedder(questions, load=load)
		mat = np.zeros(4800, len(questions))
		for i in range(len(questions)):
			mat[:,i] = embedder.embed(questions[i])
		return mat
	else:
		embdr = SkipThoughtEmbedder(questions, load=load, save=save)
	return embdr.embedded_dataset, embdr

#decompose an embedded question into basic questions given questions and their embeddings
#return the list of dictionaries of top 20 decomposed questions with their score
def decompose(question, questions, embeddings):
	slvr = LeastLinearSquaresSolver()
	x = slvr.solve(embeddings, question)
	decomposition = [(questions[i], float(x[i])) for i in range(len(x)) if float(x[i])!=0]
	decomposition = sorted(decomposition, key=lambda x:x[1], reverse=True)
	return decomposition