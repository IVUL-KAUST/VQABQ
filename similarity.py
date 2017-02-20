import heapq
import numpy as np
from sklearn.manifold import MDS

def nth_largest(n, data):
    return heapq.nlargest(n, data)

def compare(str1, str2):
	'''
	will return a number between zero and one inclusive to compare to 
	questions given as strings
	'''
	#TODO
	raise NotImplementedError

def load_questions():
	'''
	will return a list of yes-no questions as strings
	'''
	#TODO
	raise NotImplementedError

def generate_similarity_matrix(data, measure):
	'''
	given a list of N questions and a similarity measure (compares two questions)
	it will generate NxN similarity symmetric matrix
	'''
	#TODO
	raise NotImplementedError

def compare_to_all(question, dataset, measure):
	'''
	given a question and a list of questions we measure the similarity of the question
	against all the questions in the list using a similarity measure and return the values in a list
	'''
	#TODO
	raise NotImplementedError

def init():
	'''
	loads the dataset of yes-no questions and generate the questions_embedding and saves the MDS
	'''
	global dataset, mds, questions_embedding
	dataset = load_questions()
	sim_mat = generate_similarity_matrix(dataset, compare)
	rnd_state = np.random.RandomState(seed=0)
	mds = MDS(n_components=512, n_jobs=-1, random_state=rnd_state, dissimilarity='precomputed')
	questions_embedding = mds.fit_transform(sim_mat)

def get_top_similar(question, n=10):
	'''
	given a question return the top n similar questions
	'''
	sim_list = compare_to_all(question, dataset, compare)
	question_embedding = mds.fit(sim_list)
	x = np.linalg.lstsq(questions_embedding, question_embedding)
	nth_largest = nth_largest(n, x)
	mask = [1 if x>=nth_largest else 0 for x in a]
	indecies = np.nonzero(mask)[0]
	top_questions = [dataset[i] for i in indecies]
	return top_questions

#--------------------------------------------------------------------

def test():
	init()
	new_question = 'What is your name?'
	print('Main question: '+new_question)
	sim = get_top_similar(new_question)
	print('Top 10 similar questions:')
	for q in sim:
		print(q)

test()



