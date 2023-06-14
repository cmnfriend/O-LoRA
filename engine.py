"""
For continual learning params search.

Xiao Wang  20230613
"""

from detector import Detector
from gpu_scheduler import GPUScheduler
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TASK_CONFIGS_DIR = '/workspace/AutoCL/configs/auto_CL_configs'

# task training and testing configs
TASK_CONFIGS = {
    "BoolQA_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "BoolQA_40_1")},
    "COPA_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "COPA_40_1")},
    "MultiRC_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "MultiRC_40_1")},
    "QQP_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "QQP_40_1")},

}

# params list for each task
TASK_PARAMS = {
    "BoolQA_40_1": [
        {"lamda_1": 0.5, "lamda_2": 0, "lr": 1e-3, "num_train_epochs": 1},
        {"lamda_1": 0.5, "lamda_2": 0.3, "lr": 1e-3, "num_train_epochs": 1},
        {"lamda_1": 0.1, "lamda_2": 0, "lr": 1e-3, "num_train_epochs": 1}
    ],
    "COPA_40_1":  [
        {"lamda_1": 0.5, "lamda_2": 0, "lr": 1e-3, "num_train_epochs": 1},
        {"lamda_1": 0.5, "lamda_2": 0.3, "lr": 1e-3, "num_train_epochs": 1}
    ],
    "MultiRC_40_1": [
        {"lamda_1": 0.5, "lamda_2": 0, "lr": 1e-3, "num_train_epochs": 1},
        {"lamda_1": 0.5, "lamda_2": 0.3, "lr": 1e-3, "num_train_epochs": 1}
    ],
    "QQP_40_1": [
        {"lamda_1": 0.5, "lamda_2": 0, "lr": 1e-3, "num_train_epochs": 1},
        {"lamda_1": 0.5, "lamda_2": 0.3, "lr": 1e-3, "num_train_epochs": 1}
    ]
}

# add average accuracy list to save history results
CMD_TEMPLATE = "nohup /opt/conda/bin/python src/run_uie_lora.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path {} \
    --data_dir {} \
    --task_config_dir {} \
    --instruction_file {} \
    --instruction_strategy {} \
    --output_dir {} \
    --per_device_train_batch_size {} \
    --per_device_eval_batch_size {} \
    --gradient_accumulation_steps 1 \
    --learning_rate {} \
    --num_train_epochs {} \
    --run_name t5-large-experiment-olora \
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
    --lamda_1 {} \
    --lamda_2 {} \
    --lora_dim {} \
    --seed 42 \
    --average_accuracy_list {} \
    --sequential_results_file {} > {} 2>&1 &"


# IO settings
DATA_DIR = "/workspace/CL_Benchmark"
INIT_MODEL = "/workspace/MODELS/t5-small"
RESULT_FILE = "/workspace/AutoCL/autoCL_result.txt"
MODEL_OUTPUT_DIR = "/workspace/AutoCL_MODELS/"
INSTRUCTION_FILE = "/workspace/AutoCL/configs/instruction_config_cl.json"
LOG_DIR = "/workspace/AutoCL/logs/"
INSTRUCTION_STRATEGY = "single"


# general params settings
LoRA_DIM = 8
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8

# gpu settings
MAX_GPU_COUNT = 6
GPU_THRESHOLD = 0.8
TOP_K = 2


class Engine:
    """
    1. continual detect result file;
    2. generate params list for next task;
    3. query gpu status and schedule tasks;
    4. run tasks and save model;
    5. back to step 1 until all tasks are finished.

    """
    def __init__(self, task_list) -> None:
        self.check_task_list(task_list)
        self.task_list = task_list
        self.detector = Detector(RESULT_FILE)
        self.gpu_scheduler = GPUScheduler(MAX_GPU_COUNT, GPU_THRESHOLD)
        self.cmd_pool = []

    def check_task_list(self, task_list):
        for task in task_list:
            assert task in TASK_PARAMS.keys(), "task {} not in TASK_PARAMS".format(task)

    def run(self):
        training_params = self.generate_training_params(self.task_list[0])
        self.detector.clear()
        self.detector.write_beginner(self.task_list[0])
        self.insert_cmd_pool(training_params)
        self.gpu_scheduler.schedule(self.cmd_pool)
        # init model num is 1
        init_model_num = 1
        params_num = init_model_num * len(TASK_PARAMS[self.task_list[0]])
    
        for task in self.task_list[1:]:
            # reset for next task
            self.cmd_pool = []
            # get top k results from params_num models
            top_k_results = self.detector.detect(params_num, top_k=TOP_K)
            # generate training params and then insert to cmd pool
            training_params = self.generate_training_params(task, top_k_results)
            self.detector.write_beginner(task)
            self.insert_cmd_pool(training_params)
            # schedule 
            self.gpu_scheduler.schedule(self.cmd_pool)
            params_num = TOP_K * len(TASK_PARAMS[task])

    # generate params list for next task
    def generate_training_params(self, task_name, top_k_results=None):
        # init models with their history average metrics
        if top_k_results:
            init_infos = []
            for result in top_k_results:
                init_infos.append([result['model_save_path'], result['average_metrics']])
        else:
            init_infos = [[INIT_MODEL, []]]

        # construct out dir name
        out_dir = MODEL_OUTPUT_DIR + task_name + "_"

        original_params = {
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": TEST_BATCH_SIZE,
            "lora_dim": LoRA_DIM,
            "data_dir": DATA_DIR,
            "instruction_file": INSTRUCTION_FILE,
            "instruction_strategy": INSTRUCTION_STRATEGY,
            "sequential_results_file": RESULT_FILE
        }

        task_params_list = TASK_PARAMS[task_name]
        training_params = []

        # return len(init_models) * len(task_params_list) params
        base_id = 0
        for init_model, average_metrics in init_infos:
            for id, task_params in enumerate(task_params_list):
                task_params_ = task_params.copy()
                task_params_.update(original_params)
                task_params_["model_name_or_path"] = init_model
                task_params_["task_config_dir"] = TASK_CONFIGS[task_name]["task_config_dir"]
                task_params_["output_dir"] = out_dir + str(base_id * len(task_params_list) + id)
                task_params_["average_accuracy_list"] = average_metrics.copy()
                task_params_["log_file"] = os.path.join(LOG_DIR, task_name + "_" + str(base_id * len(task_params_list) + id) + ".log")
                training_params.append(task_params_)
            base_id += 1

        return training_params

    # generate training cmds
    def insert_cmd_pool(self, training_params):
        for params in training_params:
            cmd = CMD_TEMPLATE.format(
                params["model_name_or_path"],
                params["data_dir"],
                params["task_config_dir"],
                params["instruction_file"],
                params["instruction_strategy"],
                params["output_dir"],
                params["per_device_train_batch_size"],
                params["per_device_eval_batch_size"],
                params["lr"],
                params["num_train_epochs"],
                params["lamda_1"],
                params["lamda_2"],
                params["lora_dim"],
                json.dumps(params["average_accuracy_list"]).replace(' ', ''),
                params["sequential_results_file"],
                params["log_file"]
            )
            self.cmd_pool.append(cmd)
        logger.info("insert {} cmds into cmd pool".format(len(self.cmd_pool)))


if __name__ == "__main__":
    task_list = ["COPA_40_1", "QQP_40_1", "BoolQA_40_1"]
    engine = Engine(task_list)
    engine.run()
