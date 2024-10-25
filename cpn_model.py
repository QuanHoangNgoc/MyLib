import os
import sys
import torch
from huggingface_hub import login, HfApi, HfFolder
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig


class ModelCPN:
    def __init__(self) -> None:
        pass

    def push_model_to_hub(model, repo_id, hf_token):
        # Create the API client
        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            private=False,
            repo_type="model",
            exist_ok=True,
            token=hf_token,
        )
        model.push_to_hub(repo_id)

    pattern_lora = {
        "r": 16,
        "lora_alpha": 32,  # Added lora_alpha
        "lora_dropout": 0.1,  # Added lora_dropout
        "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj"],
        "bias": "lora_only",
        "task_type": "CAUSAL_LM",
        "fan_in_fan_out": True,  # Added fan_in_fan_out initzation
    }

    def get_lora_from_model(_model, pdic=pattern_lora):
        lora_config = LoraConfig(**pdic)
        model = get_peft_model(_model, lora_config)
        model.print_trainable_parameters()
        return model
