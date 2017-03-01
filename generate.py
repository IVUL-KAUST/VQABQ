import os
import json
import numpy as np
from datetime import datetime
from solver import LassoSolver
from embedder import SkipThoughtEmbedder
from decomposer import QuestionDecomposer

#top_N = 10
lmda = 1e-5
num_threads = 1000
dataset_file = './data/vqa_train_val_questions.json'
questions_file ='./data/vqa_test_questions.json'
output_folder = './data/basic_vqa_questions/'
embedded_dataset = './models/skipthoughts_vqa_train_val_dataset.npy'

def load_questions(input_file):
	with open(input_file, 'r') as f:
		return json.load(f)

def save(data, n):
	print('saving to file...')
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	with open(output_folder+str(n)+'.json', 'w') as f:
		json.dump(data, f)
	print('done')

solver = None
dataset = None
embedder = None
questions = None
decomposer =None
chunk_size = None

def run(n):
	global dataset, questions, chunk_size, embedder, solver, decomposer
	ques = questions[n*chunk_size:(n+1)*chunk_size]

	print('decomposing '+str(len(ques))+' questions...')
	basics = decomposer.decompose_all(ques)
	data = [0]*len(ques)
	for i in range(len(ques)):
		data[i] = {
			'question':ques[i],
			'basic':[{'question':q,'score':s} for q, s in basics[i]]#[:top_N]
		}

	return data

dataset = load_questions(dataset_file)
questions = load_questions(questions_file)
chunk_size = int(np.ceil(float(len(questions))/num_threads))
embedder = SkipThoughtEmbedder(dataset, load=embedded_dataset)
solver = LassoSolver(l=lmda)
decomposer = QuestionDecomposer(embedder, solver=solver)

for i in range(500, 1000):
	t = datetime.now()
	print('Running job #'+str(i)+' ['+str(t)+']')
	data = run(i)
	save(data, i)
	print('Total time '+str(datetime.now()-t))