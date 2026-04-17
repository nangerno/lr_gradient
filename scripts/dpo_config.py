from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_gradient_checkpointing,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import get_dpo_lr, is_dataset_available_for_lr_finder

_DPO_LR_FINDER_SCALE = 0.32
_DPO_LR_CAP_MULT = 1.75
_DPO_LR_FLOOR_MULT = 0.30


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
        "gradient_checkpointing",
        "gradient_accumulation_steps",
        "output_dir",
        "request_path",
        "warmup_steps",
        "beta",
        "max_grad_norm",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

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
    --warmup_steps {warmup_steps} \
    --max_grad_norm {max_grad_norm} \
    --beta {beta} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} \
    --disable_fa {disable_fa}"""
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
        "gpu_nums": get_gpu_count(),
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
        "lr_finder_steps": int(train_info.get("lr_finder_steps", 28)),
        "lr_finder_points": int(train_info.get("lr_finder_points", 30)),
        "lr_finder_seq_len": int(train_info.get("lr_finder_seq_len", 512)),
        "lr_finder_smith_micro_batches": int(train_info.get("lr_finder_smith_micro_batches", 1)),
        "lr_finder_smith_early_stop": bool(train_info.get("lr_finder_smith_early_stop", True)),
        "lr_finder_smith_divergence_mult": float(
            train_info.get("lr_finder_smith_divergence_mult", 10.0)
        ),
        "lr_finder_smith_min_points": int(train_info.get("lr_finder_smith_min_points", 5)),
        "lr_finder_sample_frac": float(train_info.get("lr_finder_sample_frac", 0.02)),
        "lr_finder_sample_min": int(train_info.get("lr_finder_sample_min", 200)),
        "lr_finder_sample_max": int(train_info.get("lr_finder_sample_max", 3000)),
        "lr_finder_stratify_length": bool(train_info.get("lr_finder_stratify_length", True)),
        "lr_finder_sample_seed": int(train_info.get("lr_finder_sample_seed", 42)),
        "lr_finder_batch_headroom": float(train_info.get("lr_finder_batch_headroom", 0.8)),
        "lr_finder_smith_curve_mode": train_info.get(
            "lr_finder_smith_curve_mode", "max_decreasing"
        ),
        "warmup_steps": int(train_info.get("dpo_warmup_steps", 180)),
        "beta": float(train_info.get("dpo_beta", 0.08)),
        "max_grad_norm": float(train_info.get("dpo_max_grad_norm", 0.35)),
    }

    dataset_path = train_info.get("dataset", "")
    dataset_type_dict = train_info.get("dataset_type", {})

    if not is_dataset_available_for_lr_finder(dataset_path):
        print(
            "[LR Finder] Skipping: no dataset path; using param-based learning rate.",
            flush=True,
        )
    else:
        lr_result = get_dpo_lr(
            model_name,
            model_path,
            param_nums,
            dataset_path,
            dataset_type_dict,
            seq_len=run_config["lr_finder_seq_len"],
            steps=run_config["lr_finder_steps"],
            lr_points=run_config["lr_finder_points"],
            optimizer_name=run_config["optimizer"],
            smith_micro_batches=run_config["lr_finder_smith_micro_batches"],
            smith_early_stop_divergence=run_config["lr_finder_smith_early_stop"],
            smith_divergence_vs_min=run_config["lr_finder_smith_divergence_mult"],
            smith_min_points_before_divergence=run_config["lr_finder_smith_min_points"],
            lr_sample_frac=run_config["lr_finder_sample_frac"],
            lr_sample_min=run_config["lr_finder_sample_min"],
            lr_sample_max=run_config["lr_finder_sample_max"],
            lr_sample_stratify=run_config["lr_finder_stratify_length"],
            lr_sample_seed=run_config["lr_finder_sample_seed"],
            batch_headroom=run_config["lr_finder_batch_headroom"],
            smith_curve_mode=run_config["lr_finder_smith_curve_mode"],
        )

        if lr_result is not None:
            lr = lr_result.get("lr")
            bs = lr_result.get("batch_size")

            if lr is not None:
                fallback_lr = _lr_from_param_nums(param_nums)
                scaled = lr * _DPO_LR_FINDER_SCALE
                cap = fallback_lr * _DPO_LR_CAP_MULT
                floor = fallback_lr * _DPO_LR_FLOOR_MULT
                applied = max(floor, min(scaled, cap))
                run_config["learning_rate"] = applied
                print(
                    f"Using lr from dynamic finder: raw={lr}, "
                    f"DPO-scaled×{_DPO_LR_FINDER_SCALE}={scaled}, "
                    f"band=[{floor},{cap}], applied={applied} "
                    f"(fallback heuristic={fallback_lr})",
                    flush=True,
                )
            else:
                fallback_lr = _lr_from_param_nums(param_nums)
                print(f"LR finder returned no lr, using param-based fallback: {fallback_lr}", flush=True)
                run_config["learning_rate"] = fallback_lr

            if bs is not None:
                print(f"Using batch size from dynamic finder: {bs}", flush=True)
                run_config["batch_size"] = bs
        else:
            print(
                "[LR Finder] Probe failed or errored; using param-based learning rate: "
                f"{run_config['learning_rate']}",
                flush=True,
            )

    # Keep scheduling math safe even when GPU auto-detection fails.
    effective_gpu_nums = max(1, run_config["gpu_nums"])
    data_per_step = run_config["batch_size"] * effective_gpu_nums
    if data_per_step < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / data_per_step))
    
    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])
    if run_config["disable_fa"] is False:
        run_cmd = run_cmd + " --padding_free True"

    train_request = deepcopy(train_info)
    train_request["find_lk_lr"] = True
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 250
    train_request["checking_step"] = 50

    return {"train_request": train_request, "run_cmd": run_cmd}
