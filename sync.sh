#!/bin/bash
((i = 1))
((first = $(find ./data/basic_vqa_questions/ -type f | wc -l)))
((old = $first))
while true
do
	echo --------------------------
	echo syncing to jiahonghuang...
	rsync -au alfadlmm@login.cbrc.kaust.edu.sa:~/VQA/data/basic_vqa_questions/ ./data/basic_vqa_questions
	echo syncing to cbrc...
	rsync -au ./data/basic_vqa_questions/ alfadlmm@login.cbrc.kaust.edu.sa:~/VQA/data/basic_vqa_questions/
	((num = $(find ./data/basic_vqa_questions/ -type f | wc -l)))
	echo number of synced files: $((num - old))
	echo total number of files: $num
	echo average speed per minute over $i minutes: $(((num - first)/i))
	((old = num))
	((i++))
	sleep 1m
done