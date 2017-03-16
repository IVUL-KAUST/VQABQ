import os
import sys
import json
import numpy as np
from nltk.tokenize import word_tokenize
import skpt.skipthoughts as sk
from solver import LassoSolver

#---------------------------------------------------------------------------
#	Utility Functions
#---------------------------------------------------------------------------

def _prog(strn, at, total):
	"""Prints progress."""
	p = round(float(100*at)/total, 2)
	if at>=total:
		sys.stdout.write(strn+' ['+str(p)+'%]...\n')
	else:
		sys.stdout.write(strn+' ['+str(p)+'%]...\r')
	sys.stdout.flush()

def _load(file):
	"""Loads a JSON file.

	Args:
		file (str): Path to JSON file.

	Returns:
		obj: The loaded object.
	"""
	with open(file, 'r') as f:
		data = json.load(f)
	return data

def _save(data, file):
	"""Saves a list or dict in a JSON file.

	Args:
		data (list:dict): A python list or dict.
		file (str): Path to output JSON file. 
	"""
	with open(file, 'w') as f:
		json.dump(data, f)

def _clean(sentence):
	"""Cleans a given sentence by separating the tokens by spaces.

	Args:
		sentence (str): A general str sentence.

	Returns:
		str: The cleaned sentence.
	"""
	sentence = word_tokenize(str(sentence).lower())
	return ' '.join(sentence)

def _overwrite_dict(dict_dist, dict_src):
	"""Overwrite `dict_dist` with `dict_src`."""
	for k in dict_dist.keys():
		dict_dist.pop(k)
	for k in dict_src.keys():
		dict_dist[k] = dict_src[k]

def _get_files_paths(directory, extension=None):
	"""Traverse a folder and its subdirectories.

	Args:
		directory (str): A path to a directory.
		extension (str, optional): The extension of the target files. Defaults to None.

	Returns:
		list: The list of all files with the specified extension.
			if `extension` is None, it returns the list of all files.
	"""
	files = []
	for f in os.listdir(directory):
		file = os.path.join(directory, f)
		if os.path.isfile(file):
			if extension:
				if file[file.rfind('.')+1:]!=extension:
					continue
			files.append(file)
		else:
			subfiles = _get_files_paths(file)
			files.extend(subfiles)
	return files

def _apply_on_each(dataset, keys, funcs, output_file=None, force=False, verbose=True):
	"""Apply a function on each element in the dataset.

	Args:
		dataset (dict): A dataset dictionary with keys {'data':{'question'}}.
		keys (list): A list of keys under the key `data` in the dataset.
		funcs (list): A list of functions to call on each element for each key in `keys`.
		output_file (str, optional): Path to output JSON file. Defaults to None.
		force (bool, optional): Forces writing to output file. Defaults to False.
		verbose (bool, optional): Forces printing progress. Defaults to True.
	"""
	if not force and output_file and os.path.isfile(output_file):
		return
	for i in range(len(dataset['data']['question'])):
		if verbose and i%500==0: _prog('processing questions', i+1, len(dataset['data']['question']))
		for j in range(len(keys)):
			dataset['data'][keys[j]][i] = funcs[j](dataset['data'][keys[j]][i])
	if verbose: _prog('processing questions', i+1, len(dataset['data']['question']))
	if output_file:
		_save(dataset, output_file)

def _remove_file(dataset, indx):
	"""Removes all the questions of a specific file from the dataset.

	Args:
		dataset (dict): The dataset.
		indx (int): The index of the file.
	"""
	s = sum([l for l in dataset['lengths'][:indx]])
	count = dataset['lengths'][indx]
	keys = dataset['data'].keys()
	for k in keys:
		dataset['data'][k] = dataset['data'][k][:s]+dataset['data'][k][s+count:]
	dataset['files'] = dataset['files'][:indx]+dataset['files'][indx+1:]
	dataset['lengths'] = dataset['lengths'][:indx]+dataset['lengths'][indx+1:]

def _occurrences_indices(lst, strn):
	"""Find the indecies of all the occurrences of `strn` in .

	Args:
		lst (list): A list of of str.
		strn (str): A keyword.

	Returns:
		list: Indices of the occurrences.
	"""
	rslt = []
	for i in range(len(lst)):
		if strn.lower() in lst[i].lower():
			rslt.append(i)
	return rslt

#---------------------------------------------------------------------------
#	Handy Functions
#---------------------------------------------------------------------------

def extract_vqa(folder, output_file=None, force=False, verbose=True):
	"""Extract all the questions in VQA dataset into a JSON file.

	Args:
		folder (str): Path to VQA dataset directory.
		output_file (str, optional): Path to output JSON file. Defaults to None.
		force (bool, optional): Forces writing to the output file. Defaults to False.
		verbose (bool, optional): Forces printing progress. Defaults to True.

	Returns:
		dict: The loaded dataset with keys {'files', 'lengths', 'data':{'question',...}}.
	"""
	if verbose: print('[Extract VQA dataset]')
	if not force and output_file and os.path.isfile(output_file):
		return _load(output_file)

	if os.path.isfile(folder):
		files = [folder]
	else:
		files = _get_files_paths(folder, extension='json')

	if verbose: print('processing '+str(len(files))+' files...')
	extract = {
		'files':[0]*len(files),
		'lengths':[0]*len(files),
		'data':{}, #parallel arrays
	}
	accum = 0
	for i in range(len(files)):
		if verbose: print('processing `'+files[i]+'`...')
		#add file name to extract
		extract['files'][i] = files[i][files[i].rfind('/')+1:]
		#add length of data to extract
		data = _load(files[i])['questions']
		extract['lengths'][i] = len(data)
		#add the keys of the data
		if i == 0:
			keys = data[0].keys()
			for k in keys:
				extract['data'][k] = [0]*len(data)
		else:
			for k in keys:
				extract['data'][k].extend([0]*len(data))
		#fill in the data
		for j in range(accum, accum+len(data)):
			if verbose and j%500==0: _prog('extracted questions', j+1-accum, len(data))
			for k in keys:
				extract['data'][k][j] = data[j-accum][k]
		accum += len(data)
		if verbose: _prog('extracted questions', j+1-accum, len(data))
	if output_file:
		if verbose: print('saving to `'+output_file+'`...')
		_save(extract, output_file)
		if verbose: print('done')
	return extract

def preprocess(dataset, output_file=None, force=False, verbose=True):
	"""Preprocess all the questions in the dataset.

	This will add `_copy_of` key as {'data':{'_copy_of'}} to `dataset`.
	Args:
		dataset (dict): A dataset dictionary with keys {'data':{'question'}}.
		output_file (str, optional): Path to output JSON file. Defaults to None.
		force (bool, optional): Forces writing to file. Defaults to False.
		verbose (bool, optional): Forces printing progress. Defaults to True.
	"""
	if verbose: print('[Preprocess dataset]')
	if not force and output_file and os.path.isfile(output_file):
		_overwrite_dict(dataset, _load(output_file))
		return

	if force or dataset['data'].get('_copy_of', None) == None:
		#clean all questions
		_apply_on_each(dataset=dataset, keys=['question'], funcs=[_clean], output_file=output_file, force=force, verbose=verbose)
		#mark all repeated questions as copy of the first occurrence
		N = sum([l for l in dataset['lengths']])
		if verbose: print('dealing with duplicate questions...')
		##get a list of list of indecies of the questions that are equal
		questions = [(i,dataset['data']['question'][i].__hash__()) for i in range(N)]
		questions = sorted(questions, key=lambda x:x[1])
		indecies = []
		i = 0
		while i < N:
			if verbose: _prog('finding duplicate questions', i+1, N)
			indecies.append([i])
			for j in range(i+1,N):
				if questions[i][1]!=questions[j][1]:
					i = j-1
					break
				indecies[-1].append(j)
			i = i+1
		if verbose: _prog('finding duplicate questions', i+1, N)
		##mark them as copies
		dataset['data']['_copy_of'] = [0]*N
		for ind in indecies:
			ind = sorted(ind)
			for i in ind:
				dataset['data']['_copy_of'][i] = ind[0]

	if output_file:
		if verbose: print('saving to `'+output_file+'`...')
		_save(dataset, output_file)
		if verbose: print('done')

def embed(dataset, output_file=None, force=False, verbose=True):
	"""Embed all the questions in the dataset into skip thoughts vector space.

	This will add `embedded` key as {'data':{'_embedded'}} to `dataset`.
	Args:
		dataset (dict): A dataset dictionary with keys {'data':{'question'}}.
		output_file (str, optional): Path to output JSON file. Defaults to None.
		force (bool, optional): Forces embedding the questions. Defaults to False.
		verbose (bool, optional): Forces printing progress. Defaults to True.
	"""
	if verbose: print('[Embed VQA dataset]')
	if not force and output_file and os.path.isfile(output_file):
		_overwrite_dict(dataset, _load(output_file))
		return

	if force or dataset['data'].get('_embedded', None) == None:
		model = sk.load_model()
		N = sum([l for l in dataset['lengths']])
		dataset['data']['_embedded'] = [None]*N
		for i in range(N):
			if verbose: _prog('embedding questions', i+1, N)
			if dataset['data']['_copy_of'][i] != i:
				dataset['data']['_embedded'][i] = dataset['data']['_embedded'][dataset['data']['_copy_of'][i]]
			else:
				encd = sk.encode(model, [dataset['data']['question'][i]], verbose=False)
				dataset['data']['_embedded'][i] = np.reshape(encd, (encd.shape[1],))
		if verbose: _prog('embedding questions', i+1, N)

	if output_file:
		if verbose: print('saving to `'+output_file+'`...')
		_save(dataset, output_file)
		if verbose: print('done')

def vqa_subset(vqa_file, output_file=None, force=False, abstract=False, real=False, 
train=False, validation=False, test=False, dev=False, open_ended=False, multiple_choice=False):
	"""Load subset of VQA dataset from VQA JSON file.

	Args:
		vqa_file (str): Path to VQA JSON file.
		output_file (str, optional): Path to output JSON file. Defaults to None.
		force (bool, optional): Forces saving the subset. Defaults to False.
		abstract (bool, optional): Include abstract questions. Defaults to False.
		real (bool, optional): Include real questions. Defaults to False.
		train (bool, optional): Include train questions. Defaults to False.
		validation (bool, optional): Include validation questions. Defaults to False.
		test (bool, optional): Include test questions. Defaults to False.
		dev (bool, optional): Include test-dev questions. Defaults to False.
		open_ended (bool, optional): Include OpenEnded questions. Defaults to False.
		multiple_choice (bool, optional): Include MultipleChoice questions. Defaults to False.

	Returns:
		dict: The loaded dataset.
	"""
	if verbose: print('[Extract subset of VQA dataset]')
	if not force and output_file and os.path.isfile(output_file):
		return _load(output_file)

	dataset = _load(vqa_file)
	files = dataset['files']

	def process(cond, strn):
		if not cond:
			occ = _occurrences_indices(files, strn)
			for o in occ:
				_remove_file(dataset, o)

	process(abstract, '_abstract_')
	process(train, 'train20')
	process(validation, 'val20')
	process(test, 'test20')
	process(dev, 'test-dev20')
	process(open_ended, 'OpenEnded')
	process(multiple_choice, 'MultipleChoice')

	if not real:
		occ = _occurrences_indices(files, '_abstract_')
		for i in range(len(files)):
			if i not in occ:
				_remove_file(dataset, i)

	if output_file:
		_save(dataset, output_file)

	return dataset

def get_embedding(dataset, chunks=1, output=None, force=False):
	"""Merge the embeddings of the entire dataset.

	Args:
		dataset (dict): The dataset.
		chunks (int, optional): Total number of chunks. Defaults to 1.
		output (str, optional): Path to NPY file or to a directory if `chunks`>1. Defaults to None.
		force (bool, optional): Forces embedding the questions. Defaults to False.

	Returns:
		numpy.ndarray: The embeddings as a matrix.
	"""
	if verbose: print('[Extract embedding]')
	if chunks > 1:
		if not os.path.exists(output):
			if verbose: print('creating folder `'+output+'`...')
			os.makedirs(output)
		else:
			files = os.listdir(output)
			if not force and len(files)>=chunks:
				try:
					result = [0]*chunks
					for i in range(chunks):
						result[i] = np.load(os.path.join(output, str(i)+'.npy'))
					return result
				except:
					pass
	else:
		if not force and output and os.path.isfile(output):
			return np.load(output)

	m = dataset['data']['_embedding'][0].shape[0]
	N = sum([l for l in dataset['lengths']])
	indecies = [0]*N
	n = 0
	for i in range(N):
		if dataset['data']['_copy_of'][i]==i:
			indecies[n] = i
			n += 1
	indecies = indecies[:n]
	items_per_chunk = np.ceil(float(n)/chunks)

	result = [0]*chunks
	for c in range(chunks):
		mat = np.zeros(m, items_per_chunk)
		for i in range(c*items_per_chunk, min((c+1)*items_per_chunk, N)):
			if dataset['data']['_copy_of'][i]==i:
				mat[:,i-c*items_per_chunk] = dataset['data']['_embedding'][i]
		result[c] = mat

	if len(result)==1:
		if output:
			np.save(output, result[0])
		return result[0]
	else:
		if output:
			for i in range(len(result)):
				np.save(os.path.join(output, str(i)+'.npy'), result[i])
		return result

'''
#decompose an embedded question into basic questions given questions and their embeddings
#return the list of dictionaries of top n decomposed questions with their score
def decompose(question, questions, embeddings, top_n=20, l=1e-5):
	slvr = LassoSolver(l=l)
	x = slvr.solve(embeddings, question)
	decomposition = [(questions[i], float(x[i])) for i in range(len(x)) if float(x[i])!=0]
	decomposition = sorted(decomposition, key=lambda x:x[1], reverse=True)
	return decomposition[:top_n]
'''