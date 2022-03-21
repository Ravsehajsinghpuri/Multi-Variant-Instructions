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
approach(multi-variant instruction learning) across different # of instances in MULTI TASK setting.
In this setting, we perform our experiments on 8 different tasks mentioned in the paper. Training and evaluation
on same set of tasks achieving instance-generalization.
In this setting, instances are varied.

We perform our experiments on 1%, 5%, 10%, 50% and 100% instances and thus generate dataset for these numbers
for both single-intruction setting named as "original" and multi-variant instruction setting named as "variants"
com

python3 create_model_readable_multi_task_dataset.py ${experiment_id} "variants" 100 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "original" 100 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "variants" 50 ${output_data_dir} 
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "original" 50 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "variants" 10 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "original" 10 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "variants" 5 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "original" 5 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "variants" 1 ${output_data_dir}
python3 create_model_readable_multi_task_dataset.py ${experiment_id} "original" 1 ${output_data_dir}