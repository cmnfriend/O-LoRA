"""
多线程调参，生成所有参数的组合，加入任务池，多线程后台执行
"""
import os
from pynvml import *
import multiprocessing
from multiprocessing import Process
import time


nvmlInit()
# lock = multiprocessing.Lock()

# lock = threading.Lock()


# from pynvml.smi import nvidia_smi
# nvsmi = nvidia_smi.getInstance()
# nvsmi.DeviceQuery('memory.free, memory.total')
# # return by gpu ID 
# {
#     'gpu': [
#         {'fb_memory_usage': {'total': 24268.3125, 'free': 24265.5625, 'unit': 'MiB'}},
#         {'fb_memory_usage': {'total': 24268.3125, 'free': 20354.5625, 'unit': 'MiB'}},
#         {'fb_memory_usage': {'total': 24268.3125, 'free': 20352.5625, 'unit': 'MiB'}},
#         {'fb_memory_usage': {'total': 24220.3125, 'free': 14466.3125, 'unit': 'MiB'}},
#         {'fb_memory_usage': {'total': 24268.3125, 'free': 14188.5625, 'unit': 'MiB'}}
#     ]
# }


class GPUScheduler:
    def __init__(self, gpu_count=None, gpu_threshold=0.8):
        self.gpu_count = gpu_count if gpu_count else nvmlDeviceGetCount()
        self.gpu_threshold = gpu_threshold
    
    # API for detect GPU status and schedule tasks
    @staticmethod
    def sniffing_gpu(command, gpu_id, task_id, gpu_threshold=0.8):
        nvmlInit()
        while(True):
            handle = nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            total = meminfo.total / (1024**2)
            used = meminfo.used / (1024**2)
            free = meminfo.free / (1024**2)
            print("gpu_id: {}, total: {}, used: {}, free: {}, Memory-Usage: {:.2f}".format(
                gpu_id, total, used, free, used / total
                ))

            if used < total * (1 - gpu_threshold):
                command = "CUDA_VISIBLE_DEVICES={}".format(gpu_id) + ' ' + command
                time.sleep(3)

                print('Start! gpu_id is {}, command id is {}'.format(gpu_id, task_id))
                print('----Start submit! command is :{}. ----'.format(command))

                os.system(command)
                time.sleep(30)
                print('----End submit! gpu_id is {}, command id is {}. ------'.format(gpu_id, task_id))
                break

    def schedule(self, command_pool):
        print('command pool len is :{}'.format(len(command_pool)))

        sniffings_pool = []
        for id, command in enumerate(command_pool):
            gpu_id = id % self.gpu_count
            sniffing = Process(name='gpu_{}'.format(gpu_id), target=self.sniffing_gpu, args=(command, gpu_id, id, self.gpu_threshold))
            sniffings_pool.append(sniffing)

        for sniffing in sniffings_pool:
            sniffing.start()
        
        # join for return after all sniffing threads finished
        for sniffing in sniffings_pool:
            sniffing.join()
    
