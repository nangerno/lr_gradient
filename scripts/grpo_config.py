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

from lrs_lookup import apply_tokenized_lr_finder_to_run_config, effective_lr_finder_seq_len


def _bs_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 16
    p = param_nums
    if p < 2_000_000_000:
        return 40
    if p < 6_000_000_000:
        return 24
    if p < 9_000_000_000:
        return 16
    if p < 12_000_000_000:
        return 8
    if p < 15_000_000_000:
        return 4
    if p < 40_000_000_000:
        return 8
    return 4              


def _lr_from_param_nums(param_nums) -> float:
    if param_nums is None:
        return 8e-6
    p = param_nums
    if p < 1_000_000_000: 
        return 1e-5
    if p < 4_000_000_000: 
        return 8e-6
    if p < 9_000_000_000: 
        return 6e-6
    if p < 15_000_000_000:
        return 5e-6
    if p < 40_000_000_000:
        return 4e-6
    return 3e-6           


def _use_lora_from_param_nums(param_nums) -> bool:
    if param_nums is None:
        return True
    return param_nums >= 2_000_000_000


def _vllm_mem_from_param_nums(param_nums) -> float:
    if param_nums is None:
        return 0.2
    p = param_nums
    if p < 12_000_000_000:
        return 0.2
    if p < 20_000_000_000:
        return 0.25
    return 0.3     


def _use_4bit_from_param_nums(param_nums) -> bool:
    if param_nums is None:
        return False
    return param_nums >= 20_000_000_000


def _slow_reward_bs_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 8
    p = param_nums
    if p < 1_000_000_000: 
        return 8
    if p < 2_000_000_000: 
        return 10
    if p < 12_000_000_000:
        return 16
    if p < 20_000_000_000:
        return 2
    if p < 40_000_000_000:
        return 16
    return 2              


def if_contain_slow_reward_function(dataset_type: dict) -> bool:
    reward_functions = dataset_type.get("reward_functions") or []
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
        "max_completion_length",
        "disable_fa",
        "distributed",
        "gradient_checkpointing",
        "gradient_accumulation_steps",
        "eval_batch_size",
        "output_dir",
        "request_path",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

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
    --max_completion_length {max_completion_length} --mask_truncated_completions True \
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


def get_training_json(train_info: dict, *, run_lr_finder: bool = True) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    
    gpu_nums = max(1, get_gpu_count())
    
    run_config = {
        "epoch_num": 2,
        "batch_size": _bs_from_param_nums(param_nums),
        "learning_rate": _lr_from_param_nums(param_nums),
        "min_lr_rate": 0.25,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": _use_lora_from_param_nums(param_nums),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": gpu_nums,
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": "ddp",
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 4,
        "vllm_gpu_memory_utilization": _vllm_mem_from_param_nums(param_nums),
        "num_generations": 4,
        "max_completion_length": 256,
        "use_vllm": False if gpu_nums <= 1 else get_use_vllm(model_architecture, model_name),
        "tensor_parallel": False,
        "use_4bit": _use_4bit_from_param_nums(param_nums),
        # Probe: mini-train on 2% of tokenized JSON; batch = safety (synthetic × headroom, OOM-safe).
        "lr_finder_lr_probe_points": int(
            train_info.get(
                "lr_finder_lr_probe_points",
                train_info.get("lr_finder_steps", 28),
            )
        ),
        "lr_finder_mini_train_batches": int(
            train_info.get("lr_finder_mini_train_batches", 3)
        ),
        "lr_finder_seq_len": effective_lr_finder_seq_len(train_info),
        "lr_finder_stratify_length": bool(train_info.get("lr_finder_stratify_length", True)),
        "lr_finder_sample_seed": int(train_info.get("lr_finder_sample_seed", 42)),
        "lr_finder_batch_headroom": float(train_info.get("lr_finder_batch_headroom", 0.8)),
    }

    _slow_reward_funcs = if_contain_slow_reward_function(train_info["dataset_type"])
    if run_lr_finder and _slow_reward_funcs:
        print(
            "[LR Finder] Slow GRPO rewards detected: LR/batch AutoML uses **SFT CE proxy** "
            "on tokenized JSON only (no reward imports, no rollouts). "
            "Full training still uses your configured reward functions.",
            flush=True,
        )

    if run_lr_finder:
        apply_tokenized_lr_finder_to_run_config(
            run_config,
            train_info,
            model_name,
            model_path,
            param_nums,
            fallback_learning_rate=_lr_from_param_nums(param_nums),
            grpo_slow_reward_proxy_probe=_slow_reward_funcs,
        )

    train_request = deepcopy(train_info)
    train_request["find_lk_lr"] = True
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 250

    if _slow_reward_funcs:
        train_request["save_before_remaining_time"] = 12
        run_config["batch_size"] = _slow_reward_bs_from_param_nums(param_nums)
        run_config["max_completion_length"] = 128
        print(
            "Slow reward functions (e.g. textstat/langcheck/detoxify): "
            "max_completion_length=128 to shorten generation and reward cost.",
            flush=True,
        )

    num_gen = run_config["num_generations"]
    bs = run_config["batch_size"]
    if num_gen > 1 and bs % num_gen != 0:
        snapped = max(num_gen, (bs // num_gen) * num_gen)
        print(
            f"Snapping batch size {bs} → {snapped} to be divisible by "
            f"num_generations={num_gen}",
            flush=True,
        )
        run_config["batch_size"] = snapped

    # After final batch_size (slow rewards, num_generations snap).
    effective_gpu_nums = max(1, run_config["gpu_nums"])
    total_batch_size = run_config["batch_size"] * effective_gpu_nums
    if total_batch_size < 64:
        run_config["gradient_accumulation_steps"] = max(1, 64 // total_batch_size)
    else:
        run_config["gradient_accumulation_steps"] = 1

    run_config["eval_batch_size"] = 4
    if run_config["batch_size"] <= 4:
        run_config["eval_batch_size"] = 2

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    return {"train_request": train_request, "run_cmd": run_cmd}
