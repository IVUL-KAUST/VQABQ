import os
import json
from nltk.tokenize import word_tokenize

#Download the files from http://www.visualqa.org/download.html

path_to_vqa = './data/vqa/'
output_file = './data/vqa_questions.json'

def tokenize(sentence):
	return word_tokenize(str(sentence).lower())

def get_files_paths(directory):
	files = []
	for f in os.listdir(directory):
		file = os.path.join(directory, f)
		if os.path.isfile(file):
			files.append(file)
		else:
			subfiles = get_files_paths(file)
			files.extend(subfiles)
	return files

def get_questions(file):
	print('Extracting \"'+file+'\"...')
	with open(file, 'r') as f:
		data = json.load(f)['questions']
	data = set([' '.join(tokenize(d['question'])) for d in data])
	print('Extracted questions: '+str(len(data)))
	return data

if __name__ == '__main__':
	#get json files for OpenEnded questions
	files = get_files_paths(path_to_vqa)
	files = [f for f in files if 'OpenEnded' in f]

	dataset = set()
	for f in files:
		dataset |= get_questions(f)
	dataset = list(dataset)
	print('Total number of questions: '+str(len(dataset)))
	print('Writing to file...')
	with open(output_file, 'w') as outfile:
		json.dump(dataset, outfile)
	print('Done')