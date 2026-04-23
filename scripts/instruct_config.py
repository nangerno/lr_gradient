from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import (
    apply_tokenized_lr_finder_to_run_config,
    effective_lr_finder_probe_seq_len,
    effective_lr_finder_seq_len,
)


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
        "distributed",
        "gradient_checkpointing",
        "gradient_accumulation_steps",
        "output_dir",
        "request_path",
        "warmup_ratio",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    start_cmd = "python"
    run_type = config["distributed"]

    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_instruct.py \
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
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_ratio {warmup_ratio} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} \
    --packing {packing} \
    --disable_fa {disable_fa}"""
    )

    if run_type == "ds":
        template += """ --deepspeed ds_config/zero3.json"""

    if config["use_lora"]:
        template += """ --use_lora True"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("use_attn_implementation", ""):
        template += f""" --use_attn_implementation {config["use_attn_implementation"]}"""

    return template


def _bs_from_param_nums(param_nums) -> int:
    if param_nums is None:
        return 16
    p = param_nums
    if p < 1_000_000_000:       
        return 32
    if p < 3_000_000_000:       
        return 16
    if p < 7_000_000_000:       
        return 8
    if p < 13_000_000_000:      
        return 4
    if p < 30_000_000_000:      
        return 2
    return 1                    


def _lr_from_param_nums(param_nums) -> float:
    if param_nums is None:
        return 2e-5
    p = param_nums
    if p < 1_000_000_000:      
        return 5e-5
    if p < 3_000_000_000:      
        return 3e-5
    if p < 7_000_000_000:      
        return 2e-5
    if p < 13_000_000_000:      
        return 1e-5
    if p < 30_000_000_000:      
        return 8e-6
    if p < 70_000_000_000:      
        return 5e-6
    return 3e-6                


def get_training_json(train_info: dict, *, run_lr_finder: bool = True) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]

    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    _grad_ckpt = True

    run_config = {
        "epoch_num": 3,
        "batch_size": _bs_from_param_nums(param_nums),
        "learning_rate": _lr_from_param_nums(param_nums),
        "warmup_ratio": float(train_info.get("warmup_ratio", 0.03)),
        "min_lr_rate": 0.05,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": False,
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "packing": True,
        "gpu_nums": get_gpu_count(),
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": "ddp",
        "gradient_checkpointing": _grad_ckpt,
        "gradient_accumulation_steps": 4,
        "use_attn_implementation": (
            "kernels-community/vllm-flash-attn3"
            if train_info.get("is_openai", False)
            else ""
        ),
        # LR finder: SFT CE on ``train_tokenized_*.json``; batch from synthetic × headroom (+ caps).
        "lr_finder_lr_probe_points": int(
            train_info.get(
                "lr_finder_lr_probe_points",
                train_info.get("lr_finder_steps", 28),
            )
        ),
        "lr_finder_mini_train_batches": int(
            train_info.get("lr_finder_mini_train_batches", 20)
        ),
        "lr_finder_samples_per_lr": int(train_info.get("lr_finder_samples_per_lr", 80)),
        "lr_finder_lr_pick_mode": str(
            train_info.get("lr_finder_lr_pick_mode", "descending_segment")
        ),
        "lr_finder_pick_trim_low_lr_frac": float(
            train_info.get("lr_finder_pick_trim_low_lr_frac", 0.2)
        ),
        "lr_finder_pick_explosion_rel_rolling_min": float(
            train_info.get("lr_finder_pick_explosion_rel_rolling_min", 2.5)
        ),
        "lr_finder_pick_explosion_step_ratio": float(
            train_info.get("lr_finder_pick_explosion_step_ratio", 1.5)
        ),
        "lr_finder_pick_segment_pick_frac": float(
            train_info.get("lr_finder_pick_segment_pick_frac", 0.4)
        ),
        "lr_finder_seq_len": effective_lr_finder_seq_len(train_info),
        "lr_finder_probe_seq_len": effective_lr_finder_probe_seq_len(train_info),
        "lr_finder_b_train_cap": int(train_info.get("lr_finder_b_train_cap", 0)),
        "lr_finder_sample_seed": int(train_info.get("lr_finder_sample_seed", 42)),
        "lr_finder_batch_headroom": float(train_info.get("lr_finder_batch_headroom", 0.8)),
        "lr_finder_tight_after_job_kill": bool(
            train_info.get("lr_finder_tight_after_job_kill", False)
        ),
        "lr_finder_quadratic_interp_steps": int(
            train_info.get("lr_finder_quadratic_interp_steps", 10)
        ),
        # Peak slack still used when quadratic fit is unavailable (fallback rule).
        "lr_finder_peak_rel_slack": float(train_info.get("lr_finder_peak_rel_slack", 0.28)),
        "lr_finder_min_lr": float(train_info.get("lr_finder_min_lr", 1e-6)),
        "lr_finder_max_lr": float(train_info.get("lr_finder_max_lr", 9e-3)),
        "lr_finder_lora_r": int(train_info.get("lora_r", 128)),
        "lr_finder_lora_alpha": int(train_info.get("lora_alpha", 512)),
        "lr_finder_lora_dropout": float(train_info.get("lora_dropout", 0.05)),
        "lr_finder_gradient_checkpointing": _grad_ckpt,
        "lr_finder_task": "instruct",
        "lr_finder_linear_scale_lr_to_effective_batch": bool(
            train_info.get("lr_finder_linear_scale_lr_to_effective_batch", False)
        ),
    }

    # Keep scheduling math safe even when GPU auto-detection fails.
    effective_gpu_nums = max(1, run_config["gpu_nums"])
    data_per_step = run_config["batch_size"] * effective_gpu_nums

    if data_per_step >= 64:
        run_config["gradient_accumulation_steps"] = 1
    else:
        run_config["gradient_accumulation_steps"] = max(1, 64 // data_per_step)

    if run_lr_finder:
        apply_tokenized_lr_finder_to_run_config(
            run_config,
            train_info,
            model_name,
            model_path,
            param_nums,
            fallback_learning_rate=_lr_from_param_nums(param_nums),
        )

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    train_request = deepcopy(train_info)
    train_request["find_lk_lr"] = True
    train_request["save_before_remaining_time"] = 3
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 250
    train_request["checking_step"] = 50

    return {"train_request": train_request, "run_cmd": run_cmd}
