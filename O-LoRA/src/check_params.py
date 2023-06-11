from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import torch

lora_model_name_or_path = "/root/InstructUIE/logs_and_output/CL_benchmark_15/output/3/adapter"
config = PeftConfig.from_pretrained(lora_model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model1 = PeftModel.from_pretrained(model, lora_model_name_or_path)
# model = AutoModelForSeq2SeqLM.from_pretrained("/root/MODELS/t5-large")

# fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
# fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
# optional: lora_A/B is trainable but should not move too far from lorapre_A/B (constrained in "training_step"[uie_trainer_lora.py])
for name, param in model.named_parameters():
    if name.find("loranew_") != -1:
        param.requires_grad = True
    elif name.find("lora_") != -1:
        param.requires_grad = False
    # this module should always be frozen because we change the vocabulary
    elif name.find("shared") != -1:
        param.requires_grad = False

for name, param in model.named_parameters():
    print(name)
