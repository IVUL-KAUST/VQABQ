import json
import numpy as np
#from sklearn.manifold import MDS

def nth_largest(n, data):
	'''
	returns the nth largest element of the list data
	'''
	return np.sort(data, axis=0)[-n]

def lls(A, b):
	'''
	Optimize using linear least square and Ax = b return the solution
	'''
	x, _, _, _ = np.linalg.lstsq(A, b)
	return x

def compare(str1, str2):
	'''
	will return a number between zero and one inclusive to compare to 
	questions given as strings
	'''
	a = set(str1.strip('?').split(' '))
	b = set(str2.strip('?').split(' '))
	intersection = float(len(a&b))
	union = float(len(a|b))
	jaccard																																																																																																																																																																																																																																																																																																										 = intersection/union
	return jaccard

def load_questions(input_file='questions.json'):
	'''
	will return a list of yes-no questions as strings
	'''
	with open(input_file, 'r') as f:
		return json.load(f)

def compare_to_all(question, dataset, measure):
	'''
	given a question and a list of questions we measure the similarity of the question
	against all the questions in the list using a similarity measure and return the values in a list
	The returned list is a numpy array of shape (1, len(dataset))
	'''
	return [measure(question, q) for q in dataset]

def generate_similarity_matrix(dataset, measure):
	'''
	given a list of N questions and a similarity measure (compares two questions)
	it will generate NxN similarity symmetric matrix
	'''
	#return [compare_to_all(q, dataset, measure) for q in dataset]
	#OR...
	N = len(dataset)
	sim_mat = np.zeros((N, N))
	for i in range(N):
		for j in range(i, N):
			sim_mat[i][j] = measure(dataset[i], dataset[j])
	return sim_mat + np.transpose(sim_mat) - np.identity(N)*np.diagonal(sim_mat)

def init():
	'''
	loads the dataset of yes-no questions and generate the questions_embedding and saves the MDS
	'''
	global dataset, mds, questions_embedding
	dataset = load_questions()[:1000]
	sim_mat = generate_similarity_matrix(dataset, compare)
	rnd_state = np.random.RandomState(seed=0)
	#mds = MDS(n_components=512, n_jobs=-1, random_state=rnd_state, dissimilarity="precomputed")
	#questions_embedding = mds.fit(sim_mat).embedding_
	#questions_embedding = np.transpose(questions_embedding) #dxN
	questions_embedding = sim_mat

def get_top_similar(question, n=10):
	'''
	given a question return the top n similar questions sorted from most similar to least similar
	'''
	sim_list = compare_to_all(question, dataset, compare)
	sim_list = np.reshape(sim_list, (1, len(sim_list)))
	#question_embedding = mds.fit_transform(sim_list)
	question_embedding = np.transpose(sim_list)
	x = lls(questions_embedding, question_embedding)
	nth_largest_element = nth_largest(n, x)
	mask = [1 if a>=nth_largest_element else 0 for a in x]
	indecies = np.nonzero(mask)[0]
	top_questions = [(x[i], dataset[i]) for i in indecies]
	top_questions = sorted(top_questions, key=lambda tup: tup[0], reverse=True)
	top_questions = [tup[1] for tup in top_questions]
	return top_questions[:n]

#--------------------------------------------------------------------

def test():
	new_question = 'is the adult wearing white?'
	print('Main question: '+new_question)
	sim = get_top_similar(new_question)
	print('Top 10 similar questions:')
	for q in sim:
		print(q)

init()
test()