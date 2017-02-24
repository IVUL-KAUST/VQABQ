import gzip
import json

path_to_guesswhat = './data/'
output_file = './questions.json'

def get_questions(jsonl_gz):
	print('Extracting \"'+jsonl_gz+'\"...')
	with gzip.open(path_to_guesswhat+jsonl_gz) as f:
		data = []
		for line in f:
			line = line.decode("utf-8")
			game = json.loads(line.strip('\n'))
			questions = [qas['question'] for qas in game['qas']]
			data.extend(questions)
		return set(data)
	return None

dataset = set()
dataset |= get_questions('guesswhat.train.jsonl.gz')
dataset |= get_questions('guesswhat.valid.jsonl.gz')
dataset |= get_questions('guesswhat.test.jsonl.gz')
dataset = list(dataset)

print('Writing to file...')
with open(output_file, 'w') as outfile:
	json.dump(dataset, outfile)
print('Done')