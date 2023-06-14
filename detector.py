"""
1. Detect whether all params of one step task are finished.
2. Parse task test results and generate params list for next task.

step beginner identify:
-------------{task name} beginer---------------

format of test result:
{   
    "training_params": {},
    "average_metrics": [0.5,0.1],
    "test_result": [
        ['task_1', 50], 
        ['task_2', 43]
    ],
    "model_save_path": "path/to/model"
}    
    
"""

import os
import json
import time
import regex as re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, result_file) -> None:
        self.result_file = result_file

    def clear(self):
        with open(self.result_file, "w") as f:
            f.write("")

    def write_beginner(self, task_name):
        # update history metrics and out to data_args.sequential_results_file
        out_dir, results_file_name = os.path.split(self.result_file)
        # create an identify file for output dir as a Lock
        identify_file_path = os.path.join(out_dir, "identify_for_cl")
        while os.path.exists(identify_file_path):
            time.sleep(1)
        logger.info("Create lock file {}, write beginner identify to {}".format(identify_file_path, self.result_file))
        with open(identify_file_path, 'w') as file:
            pass
        with open(self.result_file, "a") as f:
            f.write("-------------{} beginner---------------\n".format(task_name))

        os.remove(identify_file_path)
        logger.info("Clear lock file {}, finish beginner writing!".format(identify_file_path))

    # results are writing with model training API.
    # detect whether all params of one step task are finished. Break until all finished and return top 3 highest score params.
    def detect(self, params_num, top_k=3):
        beginner_pattern = re.compile(r"-------------\w+ beginner---------------")

        while True:
            lines = None
            with open(self.result_file, "r") as f:
                lines = f.readlines()
            reverse_split_index = 1

            for line in lines[::-1]:
                beginner = beginner_pattern.findall(line)
                if beginner:
                    break
                reverse_split_index += 1

            # get params results, check, and return rank results
            parms_results = lines[-reverse_split_index:][1:]
            if self.check(params_num, parms_results):
                # rank params results
                rank_results = self.rank(parms_results)
                break
            else:
                time.sleep(3)
        
        return rank_results[:top_k]
    
    def check(self, params_num, parmas_results):
        if len(parmas_results) == params_num:
            return True
        else:
            logger.info("params num {} is not equal to params results num {}.".format(params_num, len(parmas_results)))
            return False

    def rank(self, params_results):
        # parse params results
        params_results = [json.loads(result) for result in params_results]
        # rank key is the average metrics of param list
        rank_key = []
        for params in params_results:
            params_list_metrics = [result[-1] for result in params["test_result"]]
            rank_key.append(sum(params_list_metrics)/len(params_list_metrics))
        
        # rank params results by rank_key
        rank_results = [result for _, result in sorted(zip(
            rank_key, params_results), key=lambda x: x[0], reverse=True)]
        
        return rank_results

