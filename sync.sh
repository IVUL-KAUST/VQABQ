while true
do
	echo syncing to jiahonghuang...
	rsync -au alfadlmm@login.cbrc.kaust.edu.sa:~/VQA/data/basic_vqa_questions/ ./data/basic_vqa_questions
	echo syncing to cbrc...
	rsync -au ./data/basic_vqa_questions/ alfadlmm@login.cbrc.kaust.edu.sa:~/VQA/data/basic_vqa_questions/
	echo everything is synced.
	sleep 1m
done

