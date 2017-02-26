import json
from embedder import SkipThoughtEmbedder
from decomposer import QuestionDecomposer

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
	dataset = load_questions(input_file='./data/vqa_questions.json')
	embdr = SkipThoughtEmbedder(dataset, load='./models/skipthoughts_vqa_dataset.npy')
	quesd = QuestionDecomposer(embdr)

	test(quesd, 'What color is the ball?', 10)
	test(quesd, 'What is her name?', 10)
	test(quesd, 'Why he doesn\'t look happy?', 10)
	test(quesd, 'How is the weather?', 10)
	test(quesd, 'When will you come back?', 10)
	test(quesd, 'Is the man in blue eating a bunch of bananas?', 10)
	test(quesd, 'What is the guy furthest on the left wearing?', 10)
