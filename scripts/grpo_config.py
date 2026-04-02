from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_use_vllm,
    get_gradient_checkpointing,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import get_grpo_lr, get_grpo_python_lr


def _bs_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 16
    p = param_nums
    if p < 2_000_000_000:       # < 2 B  — small model, vllm overhead is minor
        return 40
    if p < 6_000_000_000:       # 2 B – 6 B
        return 24
    if p < 9_000_000_000:       # 6 B – 9 B
        return 16
    if p < 12_000_000_000:      # 9 B – 12 B
        return 8
    if p < 15_000_000_000:      # 12 B – 15 B  — high vllm memory pressure
        return 4
    if p < 40_000_000_000:      # 15 B – 40 B  — vllm off, 4-bit quant
        return 8
    return 4                    # 40 B +  — 4-bit, large model


def _lr_from_param_nums(param_nums) -> float:
    if param_nums is None:
        return 8e-6
    p = param_nums
    if p < 1_000_000_000:       # < 1 B
        return 1e-5
    if p < 4_000_000_000:       # 1 B – 4 B
        return 8e-6
    if p < 9_000_000_000:       # 4 B – 9 B
        return 6e-6
    if p < 15_000_000_000:      # 9 B – 15 B
        return 5e-6
    if p < 40_000_000_000:      # 15 B – 40 B
        return 4e-6
    return 3e-6                 # 40 B +


def _gpu_count_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 4
    p = param_nums
    if p < 2_000_000_000:       # < 2 B
        return 1
    if p < 6_000_000_000:       # 2 B – 6 B
        return 2
    if p < 20_000_000_000:      # 6 B – 20 B
        return 4
    return 8                    # 20 B +


def _use_lora_from_param_nums(param_nums) -> bool:
    if param_nums is None:
        return True
    return param_nums >= 2_000_000_000


def _vllm_mem_from_param_nums(param_nums) -> float:
    if param_nums is None:
        return 0.3
    p = param_nums
    if p < 6_000_000_000:       # < 6 B
        return 0.3
    if p < 9_000_000_000:       # 6 B – 9 B  — model+LoRA+optimizer leaves ~30 % free
        return 0.3
    if p < 12_000_000_000:      # 9 B – 12 B
        return 0.4
    if p < 15_000_000_000:      # 12 B – 15 B
        return 0.5
    return 0.4                  # 15 B + (vllm likely disabled anyway)


def _use_4bit_from_param_nums(param_nums) -> bool:
    if param_nums is None:
        return False
    return param_nums >= 20_000_000_000


def _slow_reward_bs_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 8
    p = param_nums
    if p < 1_000_000_000:       # < 1 B
        return 8
    if p < 2_000_000_000:       # 1 B – 2 B
        return 10
    if p < 12_000_000_000:      # 2 B – 12 B
        return 16
    if p < 20_000_000_000:      # 12 B – 20 B
        return 2
    if p < 40_000_000_000:      # 20 B – 40 B (4-bit)
        return 16
    return 2                    # 40 B +


def if_contain_slow_reward_function(dataset_type: dict) -> bool:
    reward_functions = dataset_type["reward_functions"]
    for reward_func in reward_functions:
        func_def = reward_func["reward_func"]
        keywords = [
            "import langcheck",
            "from langcheck",
            "import detoxify",
            "from detoxify",
            "import textstat",
            "from textstat",
        ]
        if any(keyword in func_def for keyword in keywords):
            return True
    return False


def contain_python_execution(dataset_type: dict) -> bool:
    reward_functions = dataset_type["reward_functions"]
    for reward_func in reward_functions:
        func_def = reward_func["reward_func"]
        keywords = ["sat_reward_function", "ded_reward_function", "abd_reward_function"]
        if any(keyword in func_def for keyword in keywords):
            return True
    return False


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "vllm_gpu_memory_utilization",
        "num_generations",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    run_type = config["distributed"]
    if run_type == "ds":
        start_cmd = "deepspeed"

    template = (
        start_cmd
        + """ train_grpo.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size {eval_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --num_generations {num_generations} --vllm_mode colocate --vllm_gpu_memory_utilization {vllm_gpu_memory_utilization} \
    --disable_fa {disable_fa}"""
    )

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"
        )

    if config.get("use_vllm", True):
        template += " --use_vllm True"
    else:
        template += " --use_vllm False"

    if run_type == "ds":
        template += """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("tensor_parallel", False):
        template += f" --vllm_tensor_parallel_size {gpu_nums}"

    if config.get("use_4bit", False):
        template += (
            " --load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4"
        )

    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)

    run_config = {
        "epoch_num": 2,
        "batch_size": _bs_from_param_nums(param_nums),
        "learning_rate": _lr_from_param_nums(param_nums),
        "min_lr_rate": 0.25,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": _use_lora_from_param_nums(param_nums),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": _gpu_count_from_param_nums(param_nums),
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": "ddp",
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 4,
        "vllm_gpu_memory_utilization": _vllm_mem_from_param_nums(param_nums),
        "num_generations": 2,
        "use_vllm": get_use_vllm(model_architecture, model_name),
        "tensor_parallel": False,
        "use_4bit": _use_4bit_from_param_nums(param_nums),
    }

    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500

    if if_contain_slow_reward_function(train_info["dataset_type"]):
        train_request["save_before_remaining_time"] = 12
        run_config["batch_size"] = _slow_reward_bs_from_param_nums(param_nums)

    # 🔁 Recompute gradient accumulation after batch size is finalised
    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]
    if total_batch_size < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / total_batch_size))

    run_config["eval_batch_size"] = 4
    if run_config["batch_size"] <= 4:
        run_config["eval_batch_size"] = 2

    # 🚀 DYNAMIC LR + BATCH SIZE
    if train_info["find_lk_lr"]:
        dataset_path = train_info.get("dataset", "")
        dataset_type_dict = train_info.get("dataset_type", {})
        has_python_execution = contain_python_execution(train_info["dataset_type"])

        if not has_python_execution:
            lr_result = get_grpo_lr(model_name, model_path, param_nums, dataset_path, dataset_type_dict)
        else:
            lr_result = get_grpo_python_lr(model_name, model_path, param_nums, dataset_path, dataset_type_dict)

        if lr_result is not None:
            lr = lr_result.get("lr")
            bs = lr_result.get("batch_size")

            if lr is not None:
                print(f"Using lr from dynamic finder: {lr}", flush=True)
                run_config["learning_rate"] = lr
            else:
                fallback_lr = _lr_from_param_nums(param_nums)
                print(f"LR finder returned no lr, using param-based fallback: {fallback_lr}", flush=True)
                run_config["learning_rate"] = fallback_lr

            if bs is not None:
                print(f"Using batch size from dynamic finder: {bs}", flush=True)
                run_config["batch_size"] = bs
        else:
            print(f"LR finder failed, using param-based fallback: {run_config['learning_rate']}", flush=True)

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    return {"train_request": train_request, "run_cmd": run_cmd}
