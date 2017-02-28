import os
import json
import numpy as np
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

def main(n):
	dataset = load_questions(dataset_file)
	questions = load_questions(questions_file)
	chunk_size = int(np.ceil(float(len(questions))/num_threads))
	questions = questions[n*chunk_size:(n+1)*chunk_size]
	embedder = SkipThoughtEmbedder(dataset, load=embedded_dataset)
	solver = LassoSolver(l=lmda)
	decomposer = QuestionDecomposer(embedder, solver=solver)

	print('decomposing '+str(len(questions))+' questions...')
	basics = decomposer.decompose_all(questions)
	data = [0]*len(questions)
	for i in range(len(questions)):
		data[i] = {
			'question':questions[i],
			'basic':[{'question':q,'score':s} for q, s in basics[i]]#[:top_N]
		}

	print('saving to file...')
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	with open(output_folder+str(n)+'.json', 'w') as f:
		json.dump(data, f)
	print('done')

if __name__ == '__main__':
	main(int(os.environ['SLURM_ARRAY_TASK_ID']))
