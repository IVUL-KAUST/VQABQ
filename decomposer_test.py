import json
from embedder import SimilarityEmbedder
from decomposer import QuestionDecomposer

def load_questions(input_file='questions.json'):
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
	dataset = load_questions()[:1000]

	embdr = SimilarityEmbedder(dataset, dimensionality=512, seed=0)
	quesd = QuestionDecomposer(embdr)

	test(quesd, 'is the adult wearing white?', 3)
	test(quesd, 'is it on my right?', 2)
	test(quesd, 'is this new?', 4)