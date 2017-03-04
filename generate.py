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

num_tasks = 16
num_threads = 100
thread_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = int(sys.argv[1])

def load_questions(input_file):
	with open(input_file, 'r') as f:
		return json.load(f)

questions = load_questions(questions_file)
chunk_size = int(np.ceil(float(len(questions))/num_threads))
questions = questions[(thread_id*chunk_size):((thread_id+1)*chunk_size)]
if len(questions) == 0:
	exit()

if not os.path.exists(output_folder):
	os.makedirs(output_folder)
path = output_folder+str(thread_id)+'/'
if not os.path.exists(path):
	os.makedirs(path)

dataset = load_questions(dataset_file)
embedder = SkipThoughtEmbedder(dataset, load=embedded_dataset)
solver = LassoSolver(l=lmda)
decomposer = QuestionDecomposer(embedder, solver=solver)

print('decomposing '+str(len(questions))+' questions...')
for i in range(len(questions)):
	if i%num_tasks != task_id:
		continue

	file = path+str(i)+'.json'
	if os.path.isfile(file):
		continue

	print('question #'+str(thread_id*chunk_size+i)+' ['+str(datetime.now())+']')	
	basic = decomposer.decompose(questions[i])
	data = {
		'question':questions[i],
		'basic':[{'question':q,'score':s} for q, s in basic[:top_N]]
	}

	with open(file, 'w') as f:
		json.dump(data, f)
