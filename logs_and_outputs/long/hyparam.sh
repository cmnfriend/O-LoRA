#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
 
# bash scripts/long.sh> logs_and_outputs/long/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/1-yelp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0 \
   --lamda_2 0

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/1-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round2 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/MNLI \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/3-MNLI \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 2 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/3-MNLI/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/CB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/4-CB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round4 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/4-CB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/COPA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/5-COPA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round5 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0.1 

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/5-COPA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/QQP \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/6-QQP \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round6 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0.1 

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/6-QQP/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/RTE \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/7-RTE \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round7 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0.3 

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/7-RTE/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/IMDB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/8-IMDB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round8 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/8-IMDB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/SST-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/9-SST-2 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round9 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0.1

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/9-SST-2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/10-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round10 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 5 \
   --lamda_2 0

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/10-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/11-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round11 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 5 \
   --lamda_2 0

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/11-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/12-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round12 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 5 \
   --lamda_2 0.1

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/12-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/MultiRC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/13-MultiRC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round13 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 5 \
   --lamda_2 0

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/13-MultiRC/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/BoolQA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/14-BoolQA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round14 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 5 \
   --lamda_2 0.1

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/long/outputs/14-BoolQA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/long_configs/WiC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/long/outputs/15-WiC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name long_round15 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 5 \
   --lamda_2 0.3