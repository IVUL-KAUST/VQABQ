import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from evaluate import VQAEvaluator

images_folder = '/home/modar/test2015/'
devtest = '/home/modar/VQA/data/OpenEnded_mscoco_test-dev2015_basic_questions.json'
output_file = '/home/modar/VQA/data/dev_test2015_answers.json'


def concatenate(question, basic):
		basic = [b for b in basic if b['score']>0]
		basic = [b['question'] for b in basic]
		if basic[0]==question:
			basic = basic[1:]
		strn = question+' '+' '.join(basic)
		strn = strn.split(' ')
		strn = strn[:26]
		return ' '.join(strn)

vqa = VQAEvaluator(concatenate=concatenate)

with open(devtest, 'r') as f:
	dataset = json.load(f)
#dataset = dataset[10:100]

data = vqa.evaluate(dataset, images_folder)
with open(output_file, 'w') as f:
	json.dump(data, f)

def show(i, view=True):
	question = dataset[i]['question']
	basic = dataset[i]['basic']
	concatenated = concatenate(question, basic)
	answer = data[i]['answer']
	print('Question: '+question)
	print('Concatenated: '+concatenated)
	print('Answer: '+answer)
	if view:
		image_file = images_folder+dataset[i]['image_path']
		image = mpimg.imread(image_file)
		plt.imshow(image)
		plt.show()

#now try
#show(0)