from cmath import tan
import json
import os
from re import I
import pandas as pd
import argparse
import random
import numpy as np
import nltk

parser = argparse.ArgumentParser(description='Generating required model readable dataset')
parser.add_argument('experiment_id', help='What is the number of the experiment')
parser.add_argument('dataset_type', help='Which type of dataset do you need? Variants or Original?')
parser.add_argument('percentage_instances_to_sample', help='What percentage of samples do you need in this dataset')
parser.add_argument('percentage_tasks_to_sample', help='What percentage of tasks do you need in this dataset')
parser.add_argument('path_to_data_dir', help='path to data directory')
args = parser.parse_args()
experiment_id = int(args.experiment_id)
percentage_instances_to_sample = int(args.percentage_instances_to_sample)
dataset_type = args.dataset_type
percentage_tasks_to_sample = int(args.percentage_tasks_to_sample)
path_to_data_dir = args.path_to_data_dir

tasks_metadata_file = open(path_to_data_dir + "/final_dataset/metadata.json")
tasks_metadata = json.load(tasks_metadata_file)
tasks_metadata_file.close()

split_metadata_file = open(path_to_data_dir + "/experiment_{}/split_metadata.json".format(experiment_id))
split_metadata = json.load(split_metadata_file)


if not os.path.isdir(path_to_data_dir + "/experiment_{}".format(experiment_id)):
	os.mkdir(path_to_data_dir + "/experiment_{}".format(experiment_id))
os.mkdir(path_to_data_dir + "/experiment_{}/model_dataset_{}_{}%_instances_{}%_tasks".format(experiment_id,dataset_type,percentage_instances_to_sample,percentage_tasks_to_sample))
train_header, test_header = True, True
train_task_ids = split_metadata["train"]
train_task_ids = random.Random(experiment_id).sample(train_task_ids,int(len(train_task_ids)*(percentage_tasks_to_sample/100)))
test_task_ids = split_metadata["test"]
combined_task_ids = train_task_ids + test_task_ids

for task_id in combined_task_ids:
	task_data = tasks_metadata[task_id]
	task_df = pd.DataFrame()
	input_list = []
	output_list = []
	original_task = task_data["original_task"]
	if dataset_type == 'variants':
		task_variants = task_data["task_variants"]
		combined_tasks = task_variants
		combined_tasks.append(original_task)
	else:
		combined_tasks = [original_task]
	original_task_file = open(path_to_data_dir + "/final_dataset/" + original_task)
	original_task_dict = json.load(original_task_file)
	original_task_file.close()
	
	instances = original_task_dict["Instances"]
	if task_id in split_metadata["train"]:
		#randomly sampling some of the instances defined by the parameter percentage_instances_to_sample
		instances = random.Random(experiment_id).sample(instances,int(len(instances)*(percentage_instances_to_sample/100)))
	
	for task in combined_tasks:
		task_file = open(path_to_data_dir + "/final_dataset/" + task)
		task_dict = json.load(task_file)
		task_file.close()
		prefix_string = ""
		task_definition = task_dict["Definition"]
		prefix_string += "Definition: {}\n".format(task_definition)

		negative_examples = task_dict["Negative Examples"]
		prefix_string += "Negative Examples: "
		for example in negative_examples[:2]:
			prefix_string += "Input: {} ".format(example["input"])
			prefix_string += "Output: {} ".format(example["output"])
			prefix_string += "Explanation: {} ".format(example["explanation"])
		prefix_string += "\n"
		positive_examples = task_dict["Positive Examples"]
		prefix_string += "Positive Examples: "
		for example in positive_examples[:2]:
			prefix_string += "Input: {} ".format(example["input"])
			prefix_string += "Output: {} ".format(example["output"])
			prefix_string += "Explanation: {} ".format(example["explanation"])
		prefix_string += "\n"

		for instance in instances:
			if len(instance["output"]):
				instruction_input = "{}Input: {}\nOutput:".format(prefix_string,instance["input"])
				input_list.append(instruction_input)
				if task_id in split_metadata["train"]:
					output_list.append(random.Random(experiment_id).choice(instance["output"]))
				else:
					output_list.append(instance["output"])

	task_df["Input"] = input_list
	task_df["Output"] = output_list
	if task_id in split_metadata["train"]:
		task_df.to_csv(path_to_data_dir + "/experiment_{}/model_dataset_{}_{}%_instances_{}%_tasks/train.csv".format(experiment_id, dataset_type,percentage_instances_to_sample,percentage_tasks_to_sample),mode="a",header=train_header,index=False,encoding='utf-8')
		train_header = False
		split = "train"
	if task_id in split_metadata["test"]:
		task_df.to_csv(path_to_data_dir + "/experiment_{}/model_dataset_{}_{}%_instances_{}%_tasks/test.csv".format(experiment_id,dataset_type, percentage_instances_to_sample,percentage_tasks_to_sample),mode="a",header=test_header,index=False,encoding='utf-8')
		test_header = False
		split = "test"
