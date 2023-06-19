#!/opt/conda/bin/python
import os
import sys
import time
import pynvml
import argparse
import subprocess
from datetime import datetime
# nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

pynvml.nvmlInit()

def get_gpus_info():
    ans = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        ans.append((name, info))
    return ans

parser = argparse.ArgumentParser()
parser.add_argument('cmd', type = str, help='Command to be run')
parser.add_argument('-l', type=int, default=30, dest='loop_interval', help='Loop interval (sec)')
parser.add_argument('-p', type=int, default=5, dest='percentage', help='Percentage threshhold of GPU usage')
parser.add_argument('--name', type=str, default=None, help='GPU name')
parser.add_argument('-n', type=int, default=1, dest='num_gpu', help='Number of GPU required')

args = parser.parse_args()

# if not os.path.isfile(args.script):
#     raise ValueError('Invalid path of script to be run!')
# if os.path.splitext(args.script)[-1] != '.sh':
#     raise ValueError('Only bash(.sh) scripts are supported!')
# if args.num_gpu > pynvml.nvmlDeviceGetCount():
#     raise ValueError('GPU required more than %d!'%pynvml.nvmlDeviceGetCount())

def check():
    infos = get_gpus_info()
    ans_idx = []
    for idx, (name, info) in enumerate(infos):
        rate = info.used / info.total * 100
        if rate <= args.percentage:
            if args.name is None or args.name in name:
                ans_idx.append((idx, rate))
    if len(ans_idx) >= args.num_gpu:
        return [i[0] for i in sorted(ans_idx, key=lambda x: x[1])[:args.num_gpu]]
    else:
        return None

while True:
    gpu_idx = check()
    if gpu_idx is not None:
        # run cmd
        set_cuda_cmd = 'export CUDA_VISIBLE_DEVICES=%s'%(','.join([str(i) for i in gpu_idx]))
        cmd = set_cuda_cmd + ' && ' + args.cmd
        print(f'[{datetime.now()}]', cmd, file=sys.stderr)
        os.system(cmd)   # CUDA_VISIBLE_DEVICES环境变量不会影响pynvml（大概）
        break
    time.sleep(args.loop_interval)
