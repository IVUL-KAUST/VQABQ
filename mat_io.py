import numpy as np
from scipy.io import savemat, loadmat

A_file = './models/skipthoughts_vqa_train_val_dataset.npy'
B_file = './models/skipthoughts_vqa_test_dataset.npy'
AB_file = './models/AB'
X_file = './models/X'

def generate():
	A = np.load(A_file)
	B = np.load(B_file)
	dic = {
		'A':A,
		'B':B,
	}
	savemat(AB_file, mdict=dic)

def load():
	X = loadmat(X_file)['X']
	return X