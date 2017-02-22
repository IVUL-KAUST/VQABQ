import similarity

def test(qd, question, number):
	print('---------------------------------')
	print('Main question: '+question)
	sim = qd.decompose(question, number=number)
	print('Top '+str(number)+' similar questions:')
	for q in sim:
		print(q)

qd = similarity.QuestionDecomposer()
test(qd, 'is the adult wearing white?', 3)
test(qd, 'is it on my right?', 2)
test(qd, 'is this new?', 4)