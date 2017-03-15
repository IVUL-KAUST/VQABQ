import os
import sys
import json
import time
from nltk.tokenize import word_tokenize
import lutorpy as lua
require('combined_answering')

number_of_experiments = 15
#expr 01: ?%
#expr 02: ?%
#expr 03: ?%
#expr 04: ?%
#expr 05: ?%
#expr 06: ?%
#expr 07: ?%
#expr 08: ?%
#expr 09: ?%
#expr 10: ?%
#expr 11: ?%
#expr 12: ?%
#expr 13: ?%
#expr 14: ?%
#expr 15: ?%

images_folder = '/home/modar/test2015/'
devtest = '/home/modar/VQA/data/OpenEnded_mscoco_test-dev2015_basic_questions.json'
output_file = '/home/modar/VQA/data/devtest/dev_test2015_answers_8_{}.json'

def _clean(sentence):
	return ' '.join(word_tokenize(str(sentence).lower()))

def _prog(strn, at, total):
	"""Prints progress."""
	p = round(float(100*at)/total, 2)
	if at>=total:
		sys.stdout.write(strn+' ['+str(p)+'%]...\n')
	else:
		sys.stdout.write(strn+' ['+str(p)+'%]...\r')
	sys.stdout.flush()

expr = 0
def concat7(data):
	if (expr-1)%3 == 0:
		#[53547, 7317, 0, 0]
		s1, s2, s3 = 0.4, 1, 1
	elif (expr-1)%3 == 1:
		#[47792, 9555, 3517, 0]
		s1, s2, s3 = 0.3, 0.5, 1
	elif (expr-1)%3 == 2:
		#[43302, 10751, 2568, 4243]
		s1, s2, s3 = 0.25, 0.5, 0.5

	questions = [data['question']]
	basic = [b for b in data['basic'][:4] if b['score']>0]
	if basic[0]['question']==data['question']:
		basic = basic[1:4]
	else:
		basic = basic[0:3]

	if len(basic)>0:
		if basic[0]['score']>s1:
			questions.append(basic[0]['question'])
			if len(basic)>1:
				if basic[1]['score']/basic[0]['score']>s2:
					questions.append(basic[1]['question'])
					if len(basic)>2:
						if basic[2]['score']/basic[1]['score']>s3:
							questions.append(basic[2]['question'])
	return questions

def evaluate():
	with open(devtest, 'r') as f:
		dataset = json.load(f)
	#dataset = dataset[20:30]

	count = [0]*4
	for d in dataset:
		count[len(concat7(d))-1] += 1
	print('The number of concatenated questions: '+str(count))

	skipped = 0
	result = [0]*len(dataset)
	elapsed_time = 1
	start_time = time.time()
	for i in range(len(dataset)):
		if i%100 == 0:
			if i != 0:
				elapsed_time = time.time() - start_time
				_prog('current speed: '+str(round(100/elapsed_time,2)), i, len(dataset))
				start_time = time.time()
		d = dataset[i]
		image_path = os.path.join(images_folder, d['image_path'])
		question = _clean(d['question'])
		basic = d['basic']
		for b in basic:
			b['question'] = _clean(b['question'])
		try:
			questions = [repr(str(q)) for q in concat7(d)]
			questions = '{'+(', '.join(questions))+'}'
			questions = lua.eval(questions)
			answer = get_answer(image_path, questions, expr)
			result[i] = {'question_id':d['question_id'], 'answer':answer}
		except:
			skipped += 1
			result[i] = {'question_id':d['question_id'], 'answer':'NOTGIVENTON'}#just dummy answer
			print('skipped question at index #'+str(i))
		#except Exception, err:
		#	import traceback
		#	traceback.print_exc()
		#	exit()	
	_prog('current speed: '+str(round(100/elapsed_time,2)), i, len(dataset))
	print('finished '+str(i+1-skipped)+' questions and skipped '+str(skipped)+' questions')

	with open(output_file.format(expr), 'w') as f:
		json.dump(result, f)

if __name__ == '__main__':
	for i in range(1,number_of_experiments+1):
		if not os.path.isfile(output_file.format(i)):
			expr = i
			evaluate()
			lua.eval('collectgarbage()')