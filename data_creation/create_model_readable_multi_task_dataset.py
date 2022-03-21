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
parser.add_argument('path_to_data_dir', help='path to data directory')
args = parser.parse_args()
experiment_id = int(args.experiment_id)
percentage_instances_to_sample = int(args.percentage_instances_to_sample)
dataset_type = args.dataset_type
path_to_data_dir = args.path_to_data_dir

tasks_metadata_file = open(path_to_data_dir + "/final_dataset/metadata.json")
tasks_metadata = json.load(tasks_metadata_file)
tasks_metadata_file.close()

random.seed(experiment_id)

def create_train_test_dev_splits(task, percentage_instances_to_sample):
	instances = task["Instances"]
	total_num_instances = len(instances)
	num_train = int(0.70*total_num_instances)
	num_dev = int(0.10*total_num_instances)
	num_test = total_num_instances - (num_train + num_dev)
	random.Random(experiment_id).shuffle(instances)
	train = instances[:num_train]
	test = instances[num_train:num_train+num_test]
	dev = instances[num_train+num_test:]
	assert len(train)+len(test) + len(dev) == total_num_instances
	split_metadata = {}
	split_metadata["train"] = random.Random(experiment_id).sample(train,int(len(train)*(percentage_instances_to_sample/100)))
	split_metadata["test"] = test
	split_metadata["dev"] = dev
	return split_metadata


if not os.path.isdir(path_to_data_dir + "/experiment_{}".format(experiment_id)):
	os.mkdir(path_to_data_dir + "/experiment_{}".format(experiment_id))
os.mkdir(path_to_data_dir + "/experiment_{}/model_dataset_{}_{}%_instances".format(experiment_id,dataset_type,percentage_instances_to_sample))
"""
list of task ids to use in this experiment, you can add a single task id for single-task seting
 or multiple task ids for multi-task setting
"""
task_ids = ['task058']
train_header, dev_header, test_header = True, True, True

for task_id in task_ids:
	task_data = tasks_metadata[task_id]
	original_task = task_data["original_task"]
	original_task_file = open(path_to_data_dir + "/final_dataset/" + original_task)
	original_task_dict = json.load(original_task_file)
	original_task_file.close()
	if dataset_type == 'variants':
		task_variants = task_data["task_variants"]
		combined_tasks = task_variants
		combined_tasks.append(original_task)
	else:
		combined_tasks = [original_task]
	
	split_metadata = create_train_test_dev_splits(original_task_dict, percentage_instances_to_sample)
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
		
		splits = list(split_metadata.keys())
		for split in splits:
			task_df = pd.DataFrame()
			input_list = []
			output_list = []
			instances = split_metadata[split]
			for instance in instances:
				if len(instance["output"]):
					instruction_input = "{}Input: {}\nOutput:".format(prefix_string,instance["input"])
					input_list.append(instruction_input)
					if instance in split_metadata["train"]:
						output_list.append(random.choice(instance["output"]))
					else:
						output_list.append(instance["output"])
			task_df["Input"] = input_list
			task_df["Output"] = output_list
			if split == "train":
				task_df.to_csv(path_to_data_dir + "/experiment_{}/model_dataset_{}_{}%_instances/train.csv".format(experiment_id, dataset_type,percentage_instances_to_sample),mode="a",header=train_header,index=False,encoding='utf-8')
				train_header = False
			if split == "test":
				task_df.to_csv(path_to_data_dir + "/experiment_{}/model_dataset_{}_{}%_instances/test.csv".format(experiment_id,dataset_type, percentage_instances_to_sample),mode="a",header=test_header,index=False,encoding='utf-8')
				test_header = False
			if split == "dev":
				task_df.to_csv(path_to_data_dir + "/experiment_{}/model_dataset_{}_{}%_instances/dev.csv".format(experiment_id,dataset_type, percentage_instances_to_sample),mode="a",header=dev_header,index=False,encoding='utf-8')
				dev_header = False