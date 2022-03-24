# Multi-Variant-Instructions

This repository is an implementation of the paper [How Many Data Samples is an Additional Instruction Worth?](https://arxiv.org/pdf/2203.09161.pdf)
We release our dataset of Multi-Variant Instructions and the code associated with all the experiments we have performed to analyse our dataset and our approach(Multi-Variant Instruction Learning) on this dataset. The dataset is available [here.](https://drive.google.com/drive/folders/1jDmfU7nuTXOLEXXsr27zFmLBM-qp8gQT?usp=sharing)

Broadly, we have performed our experiments in three settings: single-task, multi-task and cross-task. This repository is structured in a similar way to accomodate these three settings.
## Dataset Analysis
Download the dataset and run the script to generate all the statistics for our dataset. The script is available at `dataset_analysis/generate_data_statistics.py`
Provide the appropriate path to the dataset in the script
## Single-Task Experiments
### Data Creation

To generate the model readable data for the single-task, you need to run the bash script available at `data_creation/submit_create_dataset_multi_task.sh`. We follow an experiment setting which asscoiates every experiment with an id for consistency between data and model. Due to this, the script requires you to update a few parameters: 
`experiment_id` and `output_data_dir`. This bash script uses the script `data_creation/create_model_readable_multi_task_dataset.py`. Change the taskid for which you need to perform this experiment. You need to change this in the variable named `task_ids` in `data_creation/create_model_readable_multi_task_dataset.py`. We use the same script for multi-task and single-task experiments as the two settings are quite similar in terms of data creation step.
### Model Training and Evaluation

Once the model readable data is created, you can run the bash script `run_model/submit_run_model_multi_task.sh`. For this, you need to update a few parameters: `experiment_id`(experiment_id you used for data creation for this experiment), `instances`(% instances), `dataset_type`(*original* for single-instruction and *variants* for multi-variant instruction learning), `path_to_data_dir`(path to model readable dataset created in the previous step), `path_to_output_dir`(path where you want to save the model, evaluation results and generations). This script will perform the following operations:

 1. Train the model in the desired setting (single-instruction or multi-variant instruction learning)
 2. Evaluate the model in single-instruction setting
 3. Evaluate the model in multi-variant instruction setting

## Multi-Task Experiments

This setting is quite similar to the single-task setting, so similar instructions can be followed to reproduce the results for this setting.

### Data Creation

As mentioned before, single-task and multi-task data creation steps are similar. Follow the same steps instead add multiple tasks(task_ids) in the script `data_creation/create_model_readable_multi_task_dataset.py` against the variable `task_ids` for this setting. To reproduce our results, you can use the task_ids mentioned in the paper(appendix B Table 4) which we used to run our experiments in this setting.

### Model Training and Evaluation

Follow the same training and evaluation instructions mentioned in the section **Single-Task Experiments**

## Cross-Task Experiments
### Data Creation

To generate the model readable data for the cross-task, you need to run the bash script available at `data_creation/submit_create_dataset_cross_task.sh`. We follow an experiment setting which asscoiates every experiment with an id for consistency between data and model. Due to this, the script requires you to update a few parameters: 
`experiment_id`, `output_data_dir` and `instances`(specify % instances as we only vary tasks in this experiment keeping instances fixed). This bash script uses the script `data_creation/create_model_readable_cross_task_dataset.py`. For this experiment setting, we have used a held-out split for tasks and provide that at `data_creation/split_metadata.json`. You need to first create a folder named `experiment_<your experiment_id_>` at the location you mentioned in the parameter `output_data_dir` and add this file (split_metadata.json) to this folder prior to running the data creation script.
We use 274 tasks in training and 8 tasks for evaluation in this setting and accordingly define the split_metadata.json.

### Model Training and Evaluation

Once the model readable data is created, you can run the bash script `run_model/submit_run_model_cross_task.sh`. For this, you need to update a few parameters: `experiment_id`(experiment_id you used for data creation for this experiment), `instances`(% instances), `tasks`(% tasks)`dataset_type`(*original* for single-instruction and *variants* for multi-variant instruction learning), `path_to_data_dir`(path to model readable dataset created in the previous step), `path_to_output_dir`(path where you want to save the model, evaluation results and generations). This script will perform the following operations:

 1. Train the model in the desired setting (single-instruction or multi-variant instruction learning)
 2. Evaluate the model in single-instruction setting
 3. Evaluate the model in multi-variant instruction setting



