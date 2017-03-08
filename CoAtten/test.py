import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from evaluate import VQAEvaluator

images_folder = '/home/modar/test2015/'
devtest = '/home/modar/VQA/data/OpenEnded_mscoco_test-dev2015_basic_questions.json'
output_file = '/home/modar/VQA/data/dev_test2015_answers.json'


def concat0(question, basic):
	#No concatenation
	return question

def concat1(question, basic):
	#Appending the questions that is not
	#exactly the main question and with score > 0
	#then take the first 26 words in the concatenation
	basic = [b for b in basic if b['score']>0]
	basic = [b['question'] for b in basic]
	if basic[0]==question:
		basic = basic[1:]
	strn = question+' '+' '.join(basic)
	strn = strn.split(' ')
	strn = strn[:26]
	return ' '.join(strn)

def concat2(question, basic):
	#Take the questions that is not
	#exactly the main question and with score > 0
	#then take the first 26 unique words in the concatenation
	basic = [b for b in basic if b['score']>0]
	basic = [b['question'] for b in basic]
	if basic[0]==question:
		basic = basic[1:]
	strn = question+' '+' '.join(basic)
	strn = set(strn.split(' '))
	cont = strn.pop()
	for i in range(min(25, len(strn))):
		cont = cont+' '+strn.pop()
	return cont

#pick your concatenation method
method = concat2

vqa = VQAEvaluator(concatenate=method)

with open(devtest, 'r') as f:
	dataset = json.load(f)
#dataset = dataset[10:100]

data = vqa.evaluate(dataset, images_folder)
with open(output_file, 'w') as f:
	json.dump(data, f)

def show(i, view=True):
	question = dataset[i]['question']
	basic = dataset[i]['basic']
	concatenated = method(question, basic)
	answer = data[i]['answer']
	print('Question: '+question)
	print('Concatenated: '+method)
	print('Answer: '+answer)
	if view:
		image_file = images_folder+dataset[i]['image_path']
		image = mpimg.imread(image_file)
		plt.imshow(image)
		plt.show()

#now try
#show(0)