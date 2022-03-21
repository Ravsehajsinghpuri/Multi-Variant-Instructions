#!/bin/bash

# Load required modules for job's environment
module load anaconda/py3
source activate nlpmiv

#specify an experiment id to maintain consistency of different experiments and communication between data and model output
experiment_id=29 #REPLACE

#Change this path to your desired output data directory
output_data_dir=/scratch/rpuri8/nlpmiv_dataset #REPLACE

<<com
Following code will generate the required datasets for comparing single-instruction learning and our
approach(multi-variant instruction learning) across different # of tasks in CROSS TASK setting.
In this setting, we experiment on task-level and sample % # of tasks to perform different experiments 
by keeping % # of instances fixed. Training is done on a set of sampled tasks and evaluation on 8 selected tasks
described in the paper, achieving cross-task generalization.
In this setting, % # of tasks are varied by keeping % # of instances fixed. You can change and fix the % # of instances and
then vary the tasks.

We perform our experiments on 1%, 5%, 10%, 50% and 100% tasks and thus generate dataset for these numbers
for both single-intruction setting named as "original" and multi-variant instruction setting named as "variants"
com
instances=#TO_ADD

python3 create_model_readable_cross_task_dataset.py ${experiment_id} "variants" ${instances} 100 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "original" ${instances} 100 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "variants" ${instances} 50 ${output_data_dir} 
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "original" ${instances} 50 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "variants" ${instances} 10 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "original" ${instances} 10 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "variants" ${instances} 5 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "original" ${instances} 5 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "variants" ${instances} 1 ${output_data_dir}
python3 create_model_readable_cross_task_dataset.py ${experiment_id} "original" ${instances} 1 ${output_data_dir}