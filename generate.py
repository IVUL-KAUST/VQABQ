import json
import numpy as np
from threading import Thread
from solver import LassoSolver
from embedder import SkipThoughtEmbedder
from decomposer import QuestionDecomposer

top_N = 10
lmda = 1e-5
num_threads = 31
dataset_file = './data/vqa_train_val_questions.json'
questions_file ='./data/vqa_test_questions.json'
output_file = './data/basic_vqa_questions.json'
embedded_dataset = './models/skipthoughts_vqa_train_val_dataset.npy'

output = [0]*num_threads

def load_questions(input_file='./data/questions.json'):
	with open(input_file, 'r') as f:
		return json.load(f)

def decompose(t, decomposer, questions):
	print('thread #'+str(t)+' has started')
	basics = decomposer.decompose_all(questions)
	data = [0]*len(questions)
	for i in range(len(questions)):
		data[i] = {
			#'id':t*len(questions)+i,
			'question':questions[i],
			'basic':[{'question':q,'score':s} for q, s in basics[i]][:top_N]
		}
	output[t] = data
	print('thread #'+str(t)+' has finished')

def main():
	dataset = load_questions(dataset_file)
	questions = load_questions(questions_file)
	embedder = SkipThoughtEmbedder(dataset, save=embedded_dataset)
	solver = LassoSolver(l=lmda)
	questions_decomposer = QuestionDecomposer(embedder, solver=solver)

	chunk_size = int(np.ceil(float(len(questions))/num_threads))

	threads = []
	for i in range(num_threads):
		t = Thread(target=decompose, args=(i, questions_decomposer, questions[chunk_size*i:chunk_size*(i+1)]))
		t.start()
		threads.append(t)

	for t in threads:
		t.join()

	print('combining data...')
	merged = []
	for i in range(num_threads):
		merged.extend(output[i])

	print('saving to file...')
	with open(output_file, 'w') as f:
		json.dump(merged, f)
	print('done')

if __name__ == '__main__':
	main()