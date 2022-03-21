#!/bin/bash

# Load required modules for job's environment
module load anaconda/py3
source activate nlpmiv

#Specify the required parameters for this script
experiment_id=29 #REPLACE
instances=1% #REPLACE
dataset_type=original #REPLACE
path_to_data_dir=/scratch/rpuri8/nlpmiv_dataset #REPLACE
path_to_output_dir=/scratch/rpuri8/nlpmiv_output #REPLACE

# training the model
train_opts=(--model_name_or_path "facebook/bart-base" 
        --do_train true --do_eval false
        --train_file ${path_to_data_dir}/experiment_${experiment_id}/model_dataset_${dataset_type}_${instances}_instances/train.csv
        --output_dir ${path_to_output_dir}/experiment_${experiment_id}/nlpmiv_${dataset_type}_output_${instances}_instances/experiment
        --per_device_train_batch_size=4
        --per_device_eval_batch_size=4
	--gradient_accumulation_steps=32
        --predict_with_generate
	--save_strategy=steps)

python3 run_model.py ${train_opts[@]}

# evaluate the selected model and generate predictions on the test set in single-instruction setting
eval_opts=( --model_name_or_path ${path_to_output_dir}/experiment_${experiment_id}/nlpmiv_${dataset_type}_output_${instances}_instances/experiment
            --do_train false --do_eval false --do_predict true
            --train_file ${path_to_data_dir}/experiment_${experiment_id}/model_dataset_${dataset_type}_${instances}_instances/train.csv
            --test_file ${path_to_data_dir}/experiment_${experiment_id}/model_dataset_original_100%_instances/test.csv
            --output_dir ${path_to_output_dir}/experiment_${experiment_id}/nlpmiv_${dataset_type}_output_${instances}_instances/experiment/eval
            --per_device_train_batch_size=4
            --per_device_eval_batch_size=4
            --predict_with_generate)
python3 run_model.py ${eval_opts[@]}

# evaluate the selected model and generate predictions on the test set in multi-variants instruction setting
eval_opts=( --model_name_or_path ${path_to_output_dir}/experiment_${experiment_id}/nlpmiv_${dataset_type}_output_${instances}_instances/experiment
            --do_train false --do_eval false --do_predict true
            --train_file ${path_to_data_dir}/experiment_${experiment_id}/model_dataset_${dataset_type}_${instances}_instances/train.csv
            --test_file ${path_to_data_dir}/experiment_${experiment_id}/model_dataset_variants_100%_instances/test.csv
            --output_dir ${path_to_output_dir}/experiment_${experiment_id}/nlpmiv_${dataset_type}_output_${instances}_instances/experiment/eval_on_variants
            --per_device_train_batch_size=4
            --per_device_eval_batch_size=4
            --predict_with_generate)
python3 run_model.py ${eval_opts[@]}
