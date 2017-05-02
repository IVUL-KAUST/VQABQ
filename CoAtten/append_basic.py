import json
from nltk.tokenize import word_tokenize

basic_file = './basic.json'
test_file = './vqa_raw_test.json'
output_file = './vqa_raw_test_appended.json'

def load(file):
	with open(file,'r') as f:
		return json.load(f)

def save(data, file):
	with open(file,'w') as f:
		json.dump(data, f)

def preprocess(sentence):
	return ' '.join(word_tokenize(str(sentence).lower()))

def concatenate(question, basic):
	s1, s2, s3 = (0.43, 0.82, 0.53)

	first_basic = preprocess(basic[0]['question'])
	if first_basic==preprocess(question):
		return question, 0
	else:
		basic = basic[0:3]
	basic = [b for b in basic if b['score']>0]
	number = 0
	append = ''
	if len(basic)>0:
		if basic[0]['score']>s1:
			append = ' '+basic[0]['question']
			number = 1
			if len(basic)>1:
				if basic[1]['score']/basic[0]['score']>s2:
					append += ' '+basic[1]['question']
					number = 2
					if len(basic)>2:
						if basic[2]['score']/basic[1]['score']>s3:
							append += ' '+basic[2]['question']
							number = 3
	return question+append, number

basic = load(basic_file)
test = load(test_file)

ignored = []
concatenated = [0]*4
for i in range(len(test)):
	if type(basic[i]) is not dict:
		basic[i] = {}
		basic[i]['question'] = test[i]['question']
		basic[i]['basic'] = []
	if test[i]['question']!=basic[i]['question']:
		ignored.append(i)
	else:
		new_question, number = concatenate(basic[i]['question'], basic[i]['basic'])
		test[i]['question'] = new_question
		concatenated[number]+=1

print('The following questions were ignored:')
print(ignored)
print('The following questions were appended:')
print(concatenated)

save(test, output_file)