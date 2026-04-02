DPO = "dpo"
GRPO = "grpo"
INSTRUCT = "instruct"
import re
from huggingface_hub import HfApi
from transformers import AutoConfig
import glob
from safetensors.torch import load_file
from pathlib import Path
import torch
import os
import json
import torch

MODEL_CONFIG = {
    "TinyLlama/TinyLlama_v1.1": {"model_size": 1_100_000_000},
    "TitanML/tiny-mixtral": {"model_size": 246_000_000},
    "dltjdgh0928/test_instruction": {"model_size": 7_200_000_000},
    "microsoft/Phi-3-mini-128k-instruct": {"model_size": 3_800_000_000},
    "microsoft/Phi-3-mini-4k-instruct": {"model_size": 3_800_000_000},
    "microsoft/Phi-3.5-mini-instruct": {"model_size": 3_800_000_000},
    "microsoft/phi-1_5": {"model_size": 1_400_000_000},
    "microsoft/phi-2": {"model_size": 2_780_000_000},
    "numind/NuExtract-v1.5": {"model_size": 3_800_000_000},
    "unsloth/Mistral-Nemo-Base-2407": {"model_size": 12_000_000_000},
    "unsloth/Phi-3-medium-4k-instruct": {"model_size": 13_000_000_000},
    "unsloth/Phi-3-mini-4k-instruct": {"model_size": 3_800_000_000},
    "unsloth/Phi-3.5-mini-4k-instruct": {"model_size": 3_800_000_000},
    "unsloth/tinyllama": {"model_size": 1_100_000_000},
    "unsloth/tinyllama-chat": {"model_size": 1_100_000_000},
    "unsloth/zephyr-sft": {"model_size": 7_200_000_000},
}

hf_api = HfApi()


def get_model_architecture(model_path: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_path)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except:
        return "Unknown"


def get_use_liger(architecture: str) -> str:
    if architecture.lower() in [
        "qwen2forcausallm",
        "llamaforcausallm",
        "gemma2forcausallm",
        "mixtralforcausallm",
        "mistralforcausallm",
        "qwen3forcausallm",
        "phi3forcausallm",
        "gemmaforcausallm",
    ]:
        return "True"
    else:
        return "False"


def count_params_from_safetensors(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not shards:
        return None

    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        tensors = load_file(shard_path)
        total_params += sum(v.numel() for v in tensors.values())

    return total_params


def count_params_from_bin(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.bin"))
    if not shards:
        return None

    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        try:
            state_dict = torch.load(shard_path, map_location="cpu")
            total_params += sum(v.numel() for v in state_dict.values())
        except Exception as e:
            print(f"cannot load {shard_path}: {e}")
            continue

    return total_params


def get_model_size_from_local_path(model_path: str) -> int:
    size = count_params_from_safetensors(model_path)
    if size is not None and size > 1000:
        print(f"Model size from safetensors: {size}")
        return size
    size = count_params_from_bin(model_path)
    if size is not None and size > 1000:
        print(f"Model size from bin: {size}")
        return size
    return None


def get_gpu_count():
    return torch.cuda.device_count()


def get_model_num_params(model_id: str, model_path: str) -> int:
    # 1. Try to extract size directly from the model_id string (e.g. "7B", "1.3b", "0.5M")
    size_match = re.search(r"(\d+(?:\.\d+)?)([mMbB])\b", model_id)
    if size_match:
        amount = float(size_match.group(1))
        suffix = size_match.group(2).lower()
        multiplier = 1_000_000 if suffix == "m" else 1_000_000_000
        model_size = int(amount * multiplier)
        print(f"Model size from model_id regex: {model_size}")
        return model_size

    # 2. Fall back to the hard-coded config table
    if model_id in MODEL_CONFIG:
        return MODEL_CONFIG[model_id]["model_size"]

    # 3. Count parameters directly from local safetensors / bin files
    try:
        size = get_model_size_from_local_path(model_path)
        if size is not None:
            return size
        raise Exception(f"Cannot get model size from {model_path}")
    except Exception as e:
        print(f"Error getting model size from local path: {e}")
        return None


def disable_flash_attention(architecture: str, model: str) -> str:
    if model == "microsoft/phi-2":  
        return "True"
    if "falcon-rw" in model.lower():  # ex, tiiuae/falcon-rw-1b
        return "True"
    # if model == "databricks/dolly-v2-3b":
    #    return "True"
    if architecture.strip().lower() in ["gptneoforcausallm", "bloomforcausallm", "gptossforcausallm"]:
        return "True"
    else:
        return "False"


def get_use_vllm(architecture: str, model: str) -> str:
    if model in [
        "Eurdem/Defne_llama3_2x8B",
        "heegyu/WizardVicuna-open-llama-3b-v2",
        "openlm-research/open_llama_3b",
        "TitanML/tiny-mixtral",
        "dunzhang/stella_en_1.5B_v5",
        "oopsung/llama2-7b-n-ox-test-v1",
        "microsoft/phi-2",
        "databricks/dolly-v2-3b",
    ]:
        return False
    if "falcon-rw" in model.lower():
        return False

    if architecture in ["gptneoforcausallm", "bloomforcausallm"]:
        return False
    else:
        return True


def get_gradient_checkpointing(model: str) -> str:
    if "falcon-rw" in model.lower():
        return "False"
    return "True"


def get_data_size(data_path: str) -> int:
    with open(data_path, "r") as f:
        data = json.load(f)
    return len(data)
