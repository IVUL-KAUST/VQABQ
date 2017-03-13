import os
import sys
import numpy as np
from solver import LinearLeastSquaresSolver

job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
core_id = int(sys.argv[1])

file_id = job_id*15+core_id

A_file = './models/skipthoughts_vqa_mc_dataset.npy'
B_folder = './models/B/'
output_folder = './models/X/'

loaded = False
def load():
	global loaded
	A = np.load(A_file)
	B = np.load(B_folder+str(file_id)+'.npy')
	loaded = True

slvr = LinearLeastSquaresSolver()
for i in range(B.shape[1]):
	file = output_folder+str(file_id)+'_'+str(i)+'.npy'
	if not os.path.isfile(file):
		print('Solving question #'+str(i))
		if not loaded:
			load()
		x = slvr.solve(A, B[:,i])
		np.save(file, x)
print('Done')