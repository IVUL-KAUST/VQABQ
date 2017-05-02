import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from evaluate import VQAEvaluator

images_folder = '/home/modar/test2015/'
devtest = '/home/modar/VQA/data/OpenEnded_mscoco_test2015_basic_questions.json'
output_file = '/home/modar/VQA/data/test/vqa_OpenEnded_mscoco_test2015_0_VGG_results.json'

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
	return question+' '+' '.join(basic)

def concat2(question, basic):
	#Take the questions that is not
	#exactly the main question and with score > 0
	#then take the first 26 unique words in the concatenation
	basic = [b for b in basic if b['score']>0]
	basic = [b['question'] for b in basic]
	if basic[0]==question:
		basic = basic[1:]
	strn = question+' '+' '.join(basic)
	strn = list(set(strn.split(' ')))
	return ' '.join(strn)

def concat3(question, basic):
	#Concatenate the top basic questions without the main question
	basic = [b for b in basic if b['score']>0]
	basic = [b['question'] for b in basic]
	return ' '.join(basic)

def concat4(question, basic):
	#leave the main question alone 
	#and concatenate the union of the words in the basic questions
	basic = [b for b in basic if b['score']>0]
	basic = [b['question'] for b in basic]
	strn = ' '.join(basic)
	strn = list(set(strn.split(' ')))
	return ' '.join(strn)

def concat5(question, basic):
	#the main question with the top question only
	basic = [b for b in basic if b['score']>0]
	basic = [b['question'] for b in basic]
	if basic[0]==question:
		if len(basic)>=2:
			basic = basic[1]
		else:
			basic = ''
	else:
		basic = basic[0]
	return question+' '+basic

def concat6(question, basic):
	#the main question with the top two questions
	basic = [b for b in basic if b['score']>0]
	basic = [b['question'] for b in basic]
	if basic[0]==question:
		basic = basic[1:3]
	else:
		basic = basic[0:2]
	return question+' '+' '.join(basic)

def concat7(question, basic):
	#the main questions with the top 3 questions using thresholding
	
	#for the 1st top question:
	##the average is 0.43092766002311983
	##the standard deviation is 0.30615083368834917
	##we are trying 0,{avg(+,-)(0,std)},1 : [0.43, 0.74, 0.12, 0, 1]
	##the accuracy for each s1 value with s2=s3=1 : [59.46, 59.42, 58.89, 58.85, 59.42]
	##experiments numbers [7_1, 7_2, 7_3, 5, 0]
	s1 = 0.43
	#for the 2nd top question:
	##the average is 0.11948131109699911
	##the standard deviation is 0.078193463710913444
	##the average of 1st/2nd is 0.4891305580687027
	##the standard deviation is 0.33401283456913849
	##we are trying avg(+,-)(std,0),1 = [0.1195, 0.1977, 0.4891, 0.8231, 0.1551, 0.5, 1]
	##the accuracy for each s2 value with s1=0.43 and s3=1 : [59.32, 59.37, 59.45, 59.46, 59.34, 59.45, 59.46]
	##experiments numbers [7_4, 7_5, 7_6, 7_7, 7_8, 7_9, 7_1]
	s2 = 0.82
	#for the 3rd top question:
	##the average is 0.082631150319170704
	##the standard deviation is 0.051838998571211305
	##the average of 2nd/3rd is 0.7316048417811265
	##the standard deviation is 0.20423493247506838
	##we are trying avg(+,-)(0,std), 1 = [0.7316, 0.9358, 0.5274, 1]
	##the accuracy for each s1 value with s1=0.43 and s2=0.82 : [59.46, 59.46, 59.46, 59.46]
	##experiments numbers [7_10, 7_11, 7_12, 7_7]
	s3 = 0.53

	basic = [b for b in basic if b['score']>0]
	if basic[0]['question']==question:
		basic = basic[1:4]
	else:
		basic = basic[0:3]
	append = ''
	if len(basic)>0:
		if basic[0]['score']>s1:
			append = ' '+basic[0]['question']
			if len(basic)>1:
				if basic[1]['score']/basic[0]['score']>s2:
					append += ' '+basic[1]['question']
					if len(basic)>2:
						if basic[2]['score']/basic[1]['score']>s3:
							append += ' '+basic[2]['question']
	return question+append


#pick your concatenation method
method = concat0

with open(devtest, 'r') as f:
	dataset = json.load(f)
#dataset = dataset[10:100]

vqa = VQAEvaluator(concatenate=method)

data = vqa.evaluate(dataset, images_folder)
with open(output_file, 'w') as f:
	json.dump(data, f)

def show(i, view=True):
	question = dataset[i]['question']
	basic = dataset[i]['basic']
	concatenated = method(question, basic)
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