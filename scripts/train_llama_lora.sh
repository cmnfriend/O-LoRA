#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=1
port=$(shuf -i25000-30000 -n1)

# T5-large + LoRA
# model_name_or_path: 如果是第一次运行: /root/MODELS/t5-large (模型路径); 否则: ./logs_and_output/EXPERIMENT_NAME (如: order8)/output/ID (如: 1)/adapter
# output_dir e.g. ./logs_and_output/EXPERIMENT_NAME (如: order8)/output/ID (如: 1)
# cl_task_configs: 在新的任务上训练时, 修改configs/cl_task_configs/train_task.json, dev和test的配置保持不变
# lora_r=8, lora_alpha=32, lora_dropout=0.1 (hard coded)
# 需要修改的参数: model_name_or_path (输入模型路径), lamda_1/2 (正则化系数, 在参数的最后), output_dir (lora参数保存路径)
# 运行这个指令: nohup bash ./scripts/train_t5_lora.sh > ./logs_and_output/EXPERIMENT_NAME/logs/ID (如: 1).txt 2>&1 &

# in tmux
# bash scripts/train_llama_lora.sh 2>&1 | tee -a ./logs_and_output/llama/output/1

model_name_or_path=decapoda-research/llama-7b-hf

order=1
for step in 1 2 3 4
do
   output_dir=logs_and_output/order-$order/$step

   python configs/llama_task_configs/generate.py $order $step

   export CUDA_VISIBLE_DEVICES=`gpu_monitor.py -n 6 'echo $CUDA_VISIBLE_DEVICES'`
   deepspeed --master_port $port \
      src/run_uie_lora.py \
      --do_train \
      --do_predict \
      --predict_with_generate \
      --model_name_or_path $model_name_or_path \
      --data_dir /mnt/data/user/xia_han/dataset/CL_Benchmark \
      --task_config_dir configs/llama_task_configs \
      --instruction_file configs/instruction_config_cl.json \
      --instruction_strategy single \
      --output_dir $output_dir \
      --input_record_file llama.record \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 8 \
      --learning_rate 1e-03 \
      --num_train_epochs 1 \
      --deepspeed configs/ds_configs/stage2.config \
      --run_name llama-experiment-olora \
      --max_source_length 512 \
      --max_target_length 50 \
      --generation_max_length 50 \
      --max_num_instances_per_task 10000 \
      --max_num_instances_per_eval_task 200 \
      --add_task_name True \
      --add_dataset_name True \
      --num_examples 0 \
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
      
      if [ $? -ne 0 ]
      then 
         break
      fi
      model_name_or_path=$output_dir
done