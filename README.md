# O-LoRA

- This repo releases our implementation for the O-LoRA model.
- It is built based on the pretrained T5-large model, and finetuned on our data.


## Setup

You can install the required libraries by running 

```
pip install -r requirements.txt
```

You are also required to download the t5-large model from huggingface, and put it to the folder named ```initial_model```.


## Training and Evaluation

You can reproduce our experiments of order 1 & 2 & 3 by simply running

order1:

```
bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &
```

order2:

```
bash scripts/order_2.sh> logs_and_outputs/order_2/logs/train_and_infer.log 2>&1 &
```

order3:

```
bash scripts/order_3.sh> logs_and_outputs/order_3/logs/train_and_infer.log 2>&1 &
```

The model you have trained will be saved in ```logs_and_outputs/order_1(2 or 3)/outputs```.

The result of each task will be saved in ```logs_and_outputs/order_1(2 or 3)/outputs/TASK_NAME/predict_results.json```.

You can also check the logs during training and infering in  ```logs_and_outputs/order_1(2 or 3)/logs/train_and_infer.log```

