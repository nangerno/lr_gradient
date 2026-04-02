from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_gradient_checkpointing,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import get_dpo_lr


def _bs_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 8
    p = param_nums
    if p < 1_000_000_000:       
        return 16
    if p < 3_000_000_000:       
        return 8
    if p < 7_000_000_000:       
        return 6
    if p < 13_000_000_000:      
        return 4
    if p < 30_000_000_000:      
        return 2
    return 1                   


def _lr_from_param_nums(param_nums) -> float:
    if param_nums is None:
        return 1e-5
    p = param_nums
    if p < 1_000_000_000:       
        return 1.5e-5
    if p < 3_000_000_000:       
        return 1e-5
    if p < 7_000_000_000:       
        return 7e-6
    if p < 13_000_000_000:      
        return 5e-6
    if p < 30_000_000_000:      
        return 4e-6
    if p < 70_000_000_000:      
        return 3e-6
    return 2e-6                


def _gpu_count_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 4
    p = param_nums
    if p < 1_330_000_000:       
        return 1
    if p < 4_000_000_000:       
        return 2
    if p < 13_330_000_000:      
        return 4
    return 8                    


def _distributed_from_param_nums(param_nums) -> str:
    if param_nums is None or param_nums < 9_000_000_000:
        return "ddp"
    return "ds"


def _use_lora_from_param_nums(param_nums) -> bool:
    if param_nums is None:
        return True
    return param_nums >= 2_000_000_000


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config.get("distributed", "ddp")
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_dpo.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
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
    --use_liger {use_liger} --disable_fa {disable_fa}"""
    )

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"
        )

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("use_attn_implementation", ""):
        template += f""" --use_attn_implementation {config["use_attn_implementation"]}"""

    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)

    run_config = {
        "epoch_num": 3,
        "batch_size": _bs_from_param_nums(param_nums),
        "learning_rate": _lr_from_param_nums(param_nums),
        "min_lr_rate": 0.1,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": _use_lora_from_param_nums(param_nums),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": _gpu_count_from_param_nums(param_nums),
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": _distributed_from_param_nums(param_nums),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 1,
        "use_attn_implementation": (
            "kernels-community/vllm-flash-attn3"
            if train_info.get("is_openai", False)
            else ""
        ),
    }

    # 🔁 Recompute gradient accumulation after batch size is set
    data_per_step = run_config["batch_size"] * run_config["gpu_nums"]
    if data_per_step < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / data_per_step))

    # 🚀 DYNAMIC LR + BATCH SIZE
    if train_info["find_lk_lr"]:
        dataset_path = train_info.get("dataset", "")
        dataset_type_dict = train_info.get("dataset_type", {})

        lr_result = get_dpo_lr(model_name, model_path, param_nums, dataset_path, dataset_type_dict)

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
    if run_config["disable_fa"] == "False":
        run_cmd = run_cmd + " --padding_free True"

    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["checking_step"] = 60

    return {"train_request": train_request, "run_cmd": run_cmd}
