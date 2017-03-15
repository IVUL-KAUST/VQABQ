from utilities import *

#---------------------------------------------------------------------------
#	Global Variables
#---------------------------------------------------------------------------

#the folder where all 
vqa_questions_path = './data/questions/vqa_v1/'
vqa_questions_file = './data/questions/vqa_v1.json'
vqa_processed_questions_file = './data/questions/vqa_v1_processed.json'
vqa_embedded_questions_file = './data/questions/vqa_v1_embedded.json'
A_file = './models/A.npy'
A_data_file = './data/A.json'
B_data_file = './data/B.json'

#---------------------------------------------------------------------------
#	Utility Functions
#---------------------------------------------------------------------------

#extract VQA dataset to parallel arrays structure
dataset = extract_vqa(folder=vqa_questions_path, output_file=vqa_questions_file, force=False, verbose=True)

#preprocess the dataset and save it into the file
preprocess(dataset, output_file=vqa_processed_questions_file, force=False, verbose=True)

#embed the dataset of questions using skip thoughts embedding
embed(dataset=dataset, output_file=vqa_embedded_questions_file, force=False, verbose=True)

#get the set of real train+val questions
A_data = vqa_subset(vqa_embedded_questions_file, output_file=A_data_file, force=False, real=True, train=True, validation=True, open_ended=True)
#use them as the entire set of questions
A = get_embedding(A_data, output_file=A_file, force=False)

#get the set of real dev-test questions
B_data = vqa_subset(vqa_embedded_questions_file, output_file=A_data_file, force=False,
												abstract=False, real=True,
												train=False, validation=False, test=False, dev=True, 
												open_ended=True, multiple_choice=False)