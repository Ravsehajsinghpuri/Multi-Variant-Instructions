import json
import os
import spacy
from statistics import mean, variance
import nltk
from rouge_score import rouge_scorer
import pylev 
import math

tasks_metadata_file = open("mvi_dataset/metadata.json")
tasks_metadata = json.load(tasks_metadata_file)
tasks_metadata_file.close()

def get_average_num_variants(tasks_metadata):
	task_ids = list(tasks_metadata.keys())
	total_num_variants = 0
	for task_id in task_ids:
		num_variants = tasks_metadata[task_id]["num_variants"]
		total_num_variants += num_variants
	avg_num_variants = total_num_variants/len(task_ids)
	return avg_num_variants

def get_average_num_instances(tasks_metadata):
	task_ids = list(tasks_metadata.keys())
	total_num_instances = 0
	for task_id in task_ids:
		task_instances = []
		original_task = tasks_metadata[task_id]["original_task"]
		variants = tasks_metadata[task_id]["task_variants"]
		combined_tasks = [original_task] + variants
		for task in combined_tasks:
			task_file = open("final_dataset/{}".format(task))
			task_dict = json.load(task_file)
			instances = task_instances.extend([json.dumps(a, sort_keys=True) for a in task_dict["Instances"]])
		num_instances = len(list(set(task_instances)))
		total_num_instances += num_instances
	avg_num_instances = total_num_instances/len(task_ids)
	return avg_num_instances


def get_average_num_pos_examples(tasks_metadata):
	task_ids = list(tasks_metadata.keys())
	total_num_pos_examples = 0
	for task_id in task_ids:
		task = tasks_metadata[task_id]["original_task"]
		task_file = open("final_dataset/{}".format(task))
		task_dict = json.load(task_file)
		pos_examples = task_dict["Positive Examples"]
		total_num_pos_examples += len(pos_examples)
	avg_num_pos_examples = total_num_pos_examples/len(task_ids)
	return avg_num_pos_examples

def get_average_num_neg_examples(tasks_metadata):
	task_ids = list(tasks_metadata.keys())
	total_num_neg_examples = 0
	for task_id in task_ids:
		task = tasks_metadata[task_id]["original_task"]
		task_file = open("final_dataset/{}".format(task))
		task_dict = json.load(task_file)
		neg_examples = task_dict["Negative Examples"]
		total_num_neg_examples += len(neg_examples)
	avg_num_neg_examples = total_num_neg_examples/len(task_ids)
	return avg_num_neg_examples

def get_average_len_definition(tasks_metadata):
	task_ids = list(tasks_metadata.keys())
	def_len_dict = {}
	for task_id in task_ids:
		def_len_dict[task_id] = {}
		original_task = tasks_metadata[task_id]["original_task"]
		original_task_file = open("final_dataset/{}".format(original_task))
		original_task_dict = json.load(original_task_file)
		original_task_definition = original_task_dict["Definition"]
		if isinstance(original_task_definition,list):
			original_task_definition = original_task_definition[0]
		original_task_definition = "Definition: {}".format(original_task_definition)
		original_definition_length = len(nltk.word_tokenize(original_task_definition))
		def_len_dict[task_id]["original"] = original_definition_length
		variants = tasks_metadata[task_id]["task_variants"]
		def_len_dict[task_id]["variants"] = []
		for task in variants:
			task_file = open("final_dataset/{}".format(task))
			task_dict = json.load(task_file)
			task_definition = task_dict["Definition"]
			task_definition = "Definition: {}".format(task_definition)
			definition_length = len(nltk.word_tokenize(task_definition))
			def_len_dict[task_id]["variants"].append(definition_length-2)
	return def_len_dict


def get_semantic_similarity_metrics(tasks_metadata):
	"""
	returns a dict of task_ids as keys and 
	each key having mean and variance of its semantic similarity scores
	"""
	nlp = spacy.load("en_core_web_lg")
	task_ids = list(tasks_metadata.keys())
	semantic_similarity = {}
	for task_id in task_ids:
		print(task_id)
		semantic_similarity[task_id] = {}
		similarity_scores = []
		task_definitions_list = []
		original_task = tasks_metadata[task_id]["original_task"]
		variants = tasks_metadata[task_id]["task_variants"]
		combined_tasks = [original_task] + variants
		for task in combined_tasks:
			task_file = open("final_dataset/{}".format(task))
			task_dict = json.load(task_file)
			task_definitions_list.append(task_dict["Definition"])
		for i in range(len(task_definitions_list)):
			for j in range(i+1, len(task_definitions_list)):
				doc1 = nlp(task_definitions_list[i])
				doc2 = nlp(task_definitions_list[j])
				similarity_scores.append(doc1.similarity(doc2))

		semantic_similarity[task_id]["mean"] = mean(similarity_scores)
		semantic_similarity[task_id]["stddev"] = math.sqrt(variance(similarity_scores)) if len(similarity_scores) > 1 else 0

	return semantic_similarity

def get_word_similarity_metrics(tasks_metadata):
	"""
	returns a dict of task_ids as keys and 
	each key having mean and variance of its word similarity scores
	"""

	task_ids = list(tasks_metadata.keys())
	scorer = rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
	word_similarity = {}
	for task_id in task_ids:
		print(task_id)
		word_similarity[task_id] = {}
		word_similarity_scores = []
		task_definitions_list = []
		original_task = tasks_metadata[task_id]["original_task"]
		variants = tasks_metadata[task_id]["task_variants"]
		combined_tasks = [original_task] + variants
		for task in combined_tasks:
			task_file = open("final_dataset/{}".format(task))
			task_dict = json.load(task_file)
			task_definitions_list.append(task_dict["Definition"])
		for i in range(len(task_definitions_list)):
			for j in range(i+1, len(task_definitions_list)):
				edit_distance = pylev.levenshtein(task_definitions_list[i].split(" "), task_definitions_list[j].split(" "))
				normalized_edit_distance = edit_distance/max(len(task_definitions_list[i].split(" ")),len(task_definitions_list[j].split(" ")))
				word_similarity_scores.append(normalized_edit_distance)
		word_similarity[task_id]["mean"] = mean(word_similarity_scores)
		word_similarity[task_id]["stddev"] = math.sqrt(variance(word_similarity_scores)) if len(word_similarity_scores) > 1 else 0

	return word_similarity

avg_num_variants = get_average_num_variants(tasks_metadata)
avg_num_instances = get_average_num_instances(tasks_metadata)
avg_num_pos_examples = get_average_num_pos_examples(tasks_metadata)
avg_num_neg_examples = get_average_num_neg_examples(tasks_metadata)

print(avg_num_variants)
print(avg_num_instances)
print(avg_num_pos_examples)
print(avg_num_neg_examples)

print("Calculating definition lengths")
len_def_dict = get_average_len_definition(tasks_metadata)
with open('definition_lengths.json', 'w') as f:
	json.dump(len_def_dict, f, indent=4, ensure_ascii=False)

print("Calculating semantic similarity scores")
semantic_similarity = get_semantic_similarity_metrics(tasks_metadata)

with open('semantic_similarity_scores.json', 'w') as f:
	json.dump(semantic_similarity, f, indent=4, ensure_ascii=False)

print("Calculating word similarity scores")

word_similarity = get_word_similarity_metrics(tasks_metadata)
with open('word_similarity_scores_edit_distance.json', 'w') as f:
	json.dump(word_similarity, f, indent=4, ensure_ascii=False)



