import json
from embedder import SimilarityEmbedder, SkipThoughtEmbedder
from decomposer import QuestionDecomposer
from dimensionality import MDS_Reducer, ISO_Reducer

def load_questions(input_file='./data/questions.json'):
	with open(input_file, 'r') as f:
		return json.load(f)

def test(quesd, question, number):
	print('---------------------------------')
	print('Main question: '+question)
	dques = quesd.decompose(question)
	dques = dques[:number]
	print('Top '+str(number)+' similar questions:')
	for q in dques:
		print(q[0]+'\t\t'+str(q[1]))

if __name__ == '__main__':
	dataset = load_questions()

	#reduc = MDS_Reducer(dimensionality=512, seed=0)
	#reduc = ISO_Reducer(dimensionality=512)
	
	#embdr = SimilarityEmbedder(dataset, reducer = None)
	embdr = SkipThoughtEmbedder(dataset, load=True)
	quesd = QuestionDecomposer(embdr)

	test(quesd, 'is the adult wearing white?', 10)
	test(quesd, 'is it on my right?', 10)
	test(quesd, 'is this new?', 10)