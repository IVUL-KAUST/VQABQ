import os
import sys
import json
import numpy as np
from datetime import datetime
from solver import LassoSolver
from embedder import SkipThoughtEmbedder
from decomposer import QuestionDecomposer

top_N = 20
lmda = 1e-5
dataset_file = './data/vqa_train_val_questions.json'
questions_file ='./data/vqa_test_questions.json'
output_folder = './data/basic_vqa_questions/'
embedded_dataset = './models/skipthoughts_vqa_train_val_dataset.npy'

job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
core_id = int(sys.argv[1])

def load_questions(input_file):
	with open(input_file, 'r') as f:
		return json.load(f)

incomplete = load_questions('incomplete.json')[job_id*8 + core_id]
question = load_questions(questions_file)[incomplete[0]*1421+incomplete[1]]
file = output_folder+str(incomplete[0])+'/'+str(incomplete[1])+'.json'
if os.path.isfile(file):
	exit()

dataset = load_questions(dataset_file)
embedder = SkipThoughtEmbedder(dataset, load=embedded_dataset)
solver = LassoSolver(l=lmda)
decomposer = QuestionDecomposer(embedder, solver=solver)

basic = decomposer.decompose(question)
data = {
	'question':question,
	'basic':[{'question':q,'score':s} for q, s in basic[:top_N]]
}

with open(file, 'w') as f:
	json.dump(data, f)

def get_missing():
	import os

	folder = './data/basic_vqa_questions/'

	data = []
	for i in range(100):
		files = os.listdir(folder+str(i))
		for j in range(1421):
			if str(j)+'.json' not in files:
				data.append([i, j])

	import json

	with open('incomplete.json', 'w') as f:
		json.dump(data, f)