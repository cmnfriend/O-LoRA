import sys
import os
import json

DB = 'dbpedia'
AM = 'amazon'
YA = 'yahoo'
AG = 'agnews'

DATASET_TYPE = {
    AM: 'SC',
    AG: 'TC',
    YA: 'TC',
    DB: 'TC'
}
DATASET_TEST_NAME = {
    AM: 'amazon_100_1',
    AG: 'agnews_80_1',
    YA: 'yahoo_200_1',
    DB: 'dbpedia_280_1'
}
DATASET_TRAIN_NAME = {
    AM: 'amazon_5000',
    AG: 'agnews_4000',
    YA: 'yahoo_10000',
    DB: 'dbpedia_14000'
}

ORDER = {
    1: [DB, AM, YA, AG],
    2: [DB, AM, AG, YA],
    3: [YA, AM, AG, DB]
}

def generate_json(task_list, is_train):
    # 给定数据集本名列表，返回单个json-like object
    if isinstance(task_list, str):
        task_list = [task_list]
    ans = dict()
    for task in task_list:
        dataset_type = DATASET_TYPE[task]
        if is_train:
            dataset_name = DATASET_TRAIN_NAME[task]
        else:
            dataset_name = DATASET_TEST_NAME[task]
        if dataset_type not in ans:
            ans[dataset_type] = []
        ans[dataset_type].append({
            "sampling strategy": "full",
            "dataset name": dataset_name
        })
    return ans

def generate(order, step):
    task_list = ORDER[order]
    train_config = generate_json(task_list[step-1], True)
    dev_config = generate_json(task_list[:step], False)
    test_config = generate_json(task_list[:step], False)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    for config, fname in zip([train_config, dev_config, test_config], ['train_tasks.json','dev_tasks.json','test_tasks.json']):
        fpath = os.path.join(dir_path, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

if __name__ == '__main__':
    order = int(sys.argv[1])
    step = int(sys.argv[2])
    assert order in {1,2,3}
    assert step in {1,2,3,4}
    print(f' * order: {order}')
    print(f' * step: {step}')
    generate(order, step)