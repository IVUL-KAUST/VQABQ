import os
import json

basic_questions = '../data/basic_vqa_questions/basic.json'
image_folder = '/home/modar/test2015/'
dataset_file = '/home/modar/test.json'
output_file = '../data/test.json'

def get_files(folder):
	files = []
	for d in os.listdir(folder):
		path = os.path.join(folder, d)
		if os.path.isfile(path):
			files.append(path)
		else:
			files.extend(get_files(path))
	return files

images = os.listdir(image_folder)
def get_path(image_id):
	global images
	image_id = str(image_id)
	image_id = '0'*(12-len(image_id))+image_id
	#for img in images:
	#	if image_id in img:
	#		return img
	#return None
	return 'COCO_test2015_'+image_id+'.jpg'

print('preparing the basic questions dataset...')
with open(basic_questions, 'r') as f:
	basics = json.load(f)
def get_basic(question):
	global basics
	for b in basics:
		if question == b['question']:
			return b['basic']
	return None

def combine():
	global dataset_file
	with open(dataset_file, 'r') as f:
		dataset = json.load(f)
	dataset = dataset['questions']
	ignored = []
	print('combining questions...')
	for i in range(len(dataset)):
		#print('combining question #'+str(i))
		image_id = dataset[i].pop('image_id')
		path = get_path(image_id)
		if path == None:
			print('#'+str(i)+' path failed!')
			ignored.append(i)
			continue
		basic = get_basic(dataset[i]['question'])
		if basic == None:
			print('#'+str(i)+' basic failed!')
			ignored.append(i)
			continue
		dataset[i]['image_path'] = path
		dataset[i]['basic'] = basic
	with open('ignored.json','w') as f:
		json.dump(ignored, f)
	return dataset

data = combine()
with open(output_file, 'w') as f:
	json.dump(data, f)