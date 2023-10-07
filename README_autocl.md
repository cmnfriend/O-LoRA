# AutoCL

持续学习自动化调参功能。对于T个时序来的任务，对每个step的任务搜索最优参数，并且根据结果，决定下个step的任务运行，最终得到效果最好的一组。

## Tutorial

假如有T个任务，每个任务8个超参。本脚本会对每组超参创建一个command，并且提交到单个GPU上。当前任务结束之后，选择topk个最好的ckpt，作为下个任务的初始模型，所以下个任务会运行 8*k 次，然后再在里面找 top k个，周而复始，直到所有任务运行结束。

最后的结果文件都会写到**autoCL_result.txt**里。每次运行都会将该次运行log写到logs文件夹下，分别存储。不同 step 任务的运行是串行的，一个任务的不同超参是不同GPU并行运行的，GPU是否可用、如何调度，全部代码自动实现。

运行入口为 engine.py , 运行需要设置以下参数。

1. 输入输出相关参数

```
# IO settings
DATA_DIR = "/workspace/CL_Benchmark"
INIT_MODEL = "/workspace/MODELS/t5-small"
RESULT_FILE = "/workspace/O-LoRA/autoCL_result.txt"		# 所有结果文件将保存在这里
MODEL_OUTPUT_DIR = "/workspace/AutoCL_MODELS/"				# 模型保存文件夹
INSTRUCTION_FILE = "/workspace/O-LoRA/configs/instruction_config_cl.json"
LOG_DIR = "/workspace/AutoCL/logs/"		# 每组超参都会写个log文件到这里，可以在这里检查任务运行状态
INSTRUCTION_STRATEGY = "single"
TASK_CONFIGS_DIR = '/workspace/AutoCL/configs/auto_CL_configs'   # 每个task的configs文件夹的夫文件夹

# task training and testing configs
TASK_CONFIGS = {
    "BoolQA_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "BoolQA_40_1")},
    "COPA_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "COPA_40_1")},
    "MultiRC_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "MultiRC_40_1")},
    "QQP_40_1": {"task_config_dir": os.path.join(TASK_CONFIGS_DIR, "QQP_40_1")},

}
```

注意，需要**提前准备**好训练任务的 configs，**每个任务一个config文件夹**，放在TASK_CONFIGS_DIR下，我的是auto_CL_configs。比如：

```
auto_CL_configs/
|-- BoolQA_40_1
|   |-- dev_tasks.json
|   |-- test_tasks.json
|   `-- train_tasks.json
|-- COPA_40_1
|   |-- dev_tasks.json
|   |-- test_tasks.json
|   `-- train_tasks.json
|-- MultiRC_40_1
|   |-- dev_tasks.json
|   |-- test_tasks.json
|   `-- train_tasks.json
`-- QQP_40_1
    |-- dev_tasks.json
    |-- test_tasks.json
    `-- train_tasks.json
```

2.通用训练相关参数

```
# general params settings
LoRA_DIM = 8
TRAIN_BATCH_SIZE = 8		# 每个任务单卡运行，请根据GPU和模型情况自己设置合适 batch size
TEST_BATCH_SIZE = 8

# gpu settings
MAX_GPU_COUNT = 6				# 使用前几张卡
GPU_THRESHOLD = 0.8			# 当 gpu 剩余显存超过这个比例时占用
TOP_K = 2								# 保留当前所有参数组中最好的 k 组，作为下个 step 的起始模型
```

3.任务训练调节超参

```
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
```

代码简要说明

`|-- engine.py   
|-- detector.py	# 输出文件监测
|-- gpu_scheduler.py	# GPU监测与任务调度`

此外，src/run_uie_lora.py 里的输出需要满足解析脚本，此处逻辑 T5 版本已修改。



## TODO

1. support **lora_dim** for new task

   

## 适配T5之外模型

当前代码支持T5版本，如果需要适配其他模型，请注意检查以下事项。

1. 任务训练调度接口
   支持cmd传参（cuda、model input、model output、result file、r、epoch、lr、data dir、train bs、test bs、lamda_1、lamda_2）
2. 任务输出格式化接口
   a. model output path  （RESULT_FILE）
   b. predict result      注意格式 (eval result, training params, model output)
3. 输出结果监控程序 
   a. step起始标志
   b. 调参任务是否全部完成
4. GPU状态监控、任务提交程序
   a. 检测GPU显存占用，输出可用GPU
   b. 根据cmd pool，提交单卡任务





# O-LoRA

TODO

