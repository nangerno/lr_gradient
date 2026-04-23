#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import json
import os
import shutil
import copy
import subprocess
import sys
import re
import time
from datetime import datetime, timezone, timedelta

from state_manager import get_state, set_state
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
import pathlib
from transformers import AutoConfig
import lr_utils

def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None) -> int:
    """Run ``cmd`` streaming to stdout and ``log_file_path``; return the process exit code."""
    return_code = -1
    with open(log_file_path, "w") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")

    return return_code


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    # Use (\s+|$) so the pattern matches even when the arg is at end of string.
    match = re.search(f"(?P<p>--{arg_name}(\\s+)([^\\s]+))(\\s+|$)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        print(f"[WARN] replace_args_in_cmd: --{arg_name} not found in command", flush=True)
        return cmd  # return original command unchanged rather than None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(f"(?P<p>--{arg_name}(\\s+)(?P<value>[^\\s]+))(\\s+|$)", cmd)
    if match:
        return match.group("value")
    else:
        return None


def _train_bundle_after_tokenize(
    task_id: str,
    ds_folder: str,
    bundle: dict,
    get_training_json_fn,
    task_type: str,
) -> dict:
    """
    Run **after** the tokenize subprocess succeeds. Rebuild config via
    ``get_training_json(..., run_lr_finder=True)`` when the task-specific dataset
    for LR search exists (instruct/chat: ``train_tokenized_*.json``; DPO:
    ``dpo_train_*.json``; GRPO: ``grpo_train_*.json``).
    """
    if task_type in (TaskType.INSTRUCTTEXTTASK.value, TaskType.CHATTASK.value):
        probe_path = os.path.join(ds_folder, f"train_tokenized_{task_id}.json")
    elif task_type == TaskType.DPOTASK.value:
        probe_path = os.path.join(ds_folder, f"dpo_train_{task_id}.json")
    elif task_type == TaskType.GRPOTASK.value:
        probe_path = os.path.join(ds_folder, f"grpo_train_{task_id}.json")
    else:
        probe_path = os.path.join(ds_folder, f"train_tokenized_{task_id}.json")

    if not os.path.isfile(probe_path):
        print(
            "[LR Finder] Skipping probe: "
            f"{probe_path} not found after tokenization (keeping param-based LR/batch).",
            flush=True,
        )
        return bundle
    req = copy.deepcopy(bundle["train_request"])
    tokenized_path = os.path.join(ds_folder, f"train_tokenized_{task_id}.json")
    if os.path.isfile(tokenized_path):
        req["tokenized_train_path"] = tokenized_path
    dt = req.get("dataset_type")
    if isinstance(dt, dict):
        dt = dict(dt)
        dt["tokenized_train_path"] = tokenized_path
        req["dataset_type"] = dt
    print(
        f"[LR Finder] Data ready; running LR/batch probe (gate file: {probe_path})",
        flush=True,
    )
    return get_training_json_fn(req, run_lr_finder=True)


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
    return False


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"

# Maximum number of task-level fallback attempts after all inner retries fail.
MAX_TASK_FALLBACKS = 3


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


def _next_task_fallback(
    train_cmd: str, task_type: str, fallback_num: int
) -> tuple:
    strategies = []

    # Strategy 0: halve batch size
    bs_str = extract_value_from_cmd(train_cmd, "per_device_train_batch_size")
    if bs_str and int(bs_str) > 1:
        strategies.append("reduce_batch")

    # Strategy 1: disable vLLM (GRPO only, and only when it is currently on)
    if task_type == TaskType.GRPOTASK.value:
        vllm_val = extract_value_from_cmd(train_cmd, "use_vllm")
        if vllm_val and vllm_val.lower() != "false":
            strategies.append("disable_vllm")

    # Strategy 2: enable 4-bit quant (only when not already active)
    if "--load_in_4bit" not in train_cmd:
        strategies.append("use_4bit")

    if fallback_num >= len(strategies):
        return None, ""

    strategy = strategies[fallback_num]

    if strategy == "reduce_batch":
        bs = int(bs_str)
        new_bs = max(1, bs // 2)
        # For GRPO, keep new_bs divisible by num_generations.
        if task_type == TaskType.GRPOTASK.value:
            num_gen_str = extract_value_from_cmd(train_cmd, "num_generations")
            num_gen = int(num_gen_str) if num_gen_str else 2
            if num_gen > 1 and new_bs % num_gen != 0:
                new_bs = max(num_gen, (new_bs // num_gen) * num_gen)
        cmd = replace_args_in_cmd(train_cmd, "per_device_train_batch_size", str(new_bs))
        return cmd, f"halved batch size {bs} → {new_bs}"

    if strategy == "disable_vllm":
        cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")
        return cmd, "disabled vLLM colocate mode"

    if strategy == "use_4bit":
        cmd = (
            train_cmd
            + " --load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4"
        )
        return cmd, "enabled 4-bit NF4 quantisation"

    return None, ""


def extract_output_dir(train_cmd: str) -> str:
    match = re.search(r"--output_dir\s+(.*?)\s+", train_cmd)
    if match:
        return match.group(1)
    else:
        return None


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
) -> tuple:
    for i in range(retries):
        print(
            f"************* Training attempt {i+1}/{retries} for task {task_id}*************",
            flush=True,
        )
        if i > 0:
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    if current_batch_size is None:
                        print("CUDA OOM — cannot extract batch size, skipping reduction", flush=True)
                        continue
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        # For GRPO, keep new_batch_size divisible by num_generations.
                        if task_type == TaskType.GRPOTASK.value:
                            num_gen_str = extract_value_from_cmd(train_cmd, "num_generations")
                            num_gen = int(num_gen_str) if num_gen_str else 2
                            if num_gen > 1 and new_batch_size % num_gen != 0:
                                new_batch_size = max(num_gen, (new_batch_size // num_gen) * num_gen)
                        print(
                            f"CUDA OOM — reducing batch size {current_batch_size} → {new_batch_size}",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                    else:
                        print(
                            "CUDA OOM — batch size already 1, cannot reduce further",
                            flush=True,
                        )
                        if task_type == TaskType.GRPOTASK.value:
                            print("Disabling vLLM colocate mode", flush=True)
                            train_cmd = replace_args_in_cmd(
                                train_cmd, "use_vllm", "False"
                            )
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print("vLLM OOM — disabling vLLM colocate mode", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")
                else:
                    print(
                        f"Unknown error on attempt {i}/{retries}, retrying with same config",
                        flush=True,
                    )

        # Clear the log so the next get_error_type call reads fresh output.
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        training_env_vars = {
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
        }

        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        if output_dir and os.path.exists(os.path.join(output_dir, "success.txt")):
            return True, train_cmd
        time.sleep(5)

    return False, train_cmd


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def delete_poor_checkpoints(train_runs: list[dict]):
    lowest_loss = min([run["current_loss"] for run in train_runs])
    for run in train_runs:
        if run["current_loss"] > lowest_loss:
            if os.path.exists(run["output_dir"]):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run["output_dir"])


def get_log_scale(task_type: str):
    log_scale_map = {
        TaskType.INSTRUCTTEXTTASK.value: 0.15,
        TaskType.DPOTASK.value: 0.15,
        TaskType.GRPOTASK.value: 0.15,
        TaskType.CHATTASK.value: 0.15,
    }
    return log_scale_map[task_type]


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=5)
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )

    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.0
    )

    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))

    is_openai = False
    if is_openai_model(original_model_name):
        print("Upgrading python packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0",
            os.path.join(ds_folder, f"upgrade_transformers.log"),
        )
        # upgrade deepspeed
        run_cmd_with_log(
            "pip uninstall -y deepspeed && pip install deepspeed==0.17.4",
            os.path.join(ds_folder, f"upgrade_deepspeed.log"),
        )
        # install kernel
        run_cmd_with_log(
            "pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log")
        )
        is_openai = True

    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "checking_mode": "first_time",
    }

    if (
        args.task_type == TaskType.INSTRUCTTEXTTASK.value
        or args.task_type == TaskType.CHATTASK.value
    ):
        get_training_json_fn = get_instruct_training_json
        tokenize_cmd = (
            f"/workspace/axo_py/bin/python tokenize_instruct.py {request_path}"
        )
    elif args.task_type == TaskType.DPOTASK.value:
        get_training_json_fn = get_dpo_training_json
        tokenize_cmd = f"python tokenize_dpo.py {request_path}"
    elif args.task_type == TaskType.GRPOTASK.value:
        get_training_json_fn = get_grpo_training_json
        tokenize_cmd = f"python tokenize_grpo.py {request_path}"
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    # Training config pipeline (LR/batch probe after tokenize when the task dataset exists):
    # 1) Tokenize → task-specific JSON under ``datasets/`` (incl. ``train_tokenized_*`` for SFT).
    # 2) Safe batch: synthetic max batch × headroom, then optimizer fit (inside lr_finder.find_lr).
    # 3) LR: mini-train grid at batch ≤ safe batch — loss matches ``lr_finder_task``
    #    (SFT CE / DPO sigmoid / GRPO clipped surrogate).
    # 4) Apply finder batch_size + learning_rate to ``run_cmd``.
    #
    # DPO/GRPO: ``train_dpo.py`` / ``train_grpo.py`` may still run an extra tokenization pass
    # for TRL; LR search uses ``dpo_train_*`` / ``grpo_train_*`` directly.
    #
    # Phase 1: param-based LR/batch only — tokenized JSON does not exist yet.
    train_info = get_training_json_fn(train_info, run_lr_finder=False)

    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    print(
        "[Pipeline] Step 1/2: Tokenizing dataset (must finish before LR finder). "
        f"Command: {tokenize_cmd}",
        flush=True,
    )
    _tok_rc = run_cmd_with_log(
        tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
    )
    if _tok_rc != 0:
        raise RuntimeError(
            f"Tokenization failed with exit code {_tok_rc}; not running LR finder. "
            f"See log: {os.path.join(ds_folder, f'tokenize_{args.task_id}.log')}"
        )

    # Phase 2: LR finder only after tokenize — uses ``train_tokenized_{task_id}.json`` when present.
    print(
        "[Pipeline] Step 2/2: Rebuilding training config + LR finder on tokenized output …",
        flush=True,
    )
    train_info = _train_bundle_after_tokenize(
        args.task_id, ds_folder, train_info, get_training_json_fn, args.task_type
    )
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    train_cmd = train_info["run_cmd"]
    original_train_cmd = train_cmd
    train_success = False
    state = {"mode": "initial"}
    set_state(state)
    count = 0
    while True:
        state = get_state()
        train_cmd = original_train_cmd  # will replace based on the state later
        c_train_info = copy.deepcopy(train_info)
        final_output_dir = None
        if args.task_type == TaskType.GRPOTASK.value:
            state["mode"] = "finish" # do not run this for GRPO task
            c_train_info["train_request"]["checking_mode"] = "none"
        else:
            if state["mode"] == "initial":
                c_train_info["train_request"]["checking_mode"] = "first_time"
                
            elif state["mode"] == "continue":
                c_train_info["train_request"]["checking_mode"] = "second_time"
                n_runs = state["next_runs"]
                if "lrs" not in state: # first time of continue
                    current_lr = float(state["train"]["lr"])
                    state["lrs"] = lr_utils.extend_learning_rates(current_lr, n_runs, log_range=get_log_scale(args.task_type))
                    assert len(state["lrs"]) == n_runs, f"Number of learning rates {state['lrs']} should be equal to number of runs {n_runs}"
                    state["runs"] = []
                
                set_state(state)
                state["runs"].append(state["train"].copy())
                delete_poor_checkpoints(state["runs"])
                if len(state["runs"]) < n_runs:
                    index = len(state["runs"])
                    current_lr = state["lrs"][index]
                    train_cmd = replace_args_in_cmd(train_cmd, "learning_rate", str(state["lrs"][index]))
                else: # the final run
                    # first find from runs the best loss
                    c_train_info["train_request"]["checking_mode"] = "none"
                    index = np.argmin([run["current_loss"] for run in state["runs"]])
                    print(f"BL;{index};{state['runs'][index]['current_loss']}; {state['lrs'][index]}", flush=True)
                    train_cmd = state["runs"][index]["train_cmd"]  #replace_args_in_cmd(train_cmd, "learning_rate", str(state["lrs"][index]))
                    final_output_dir = state["runs"][index]["output_dir"]
                    state["mode"] = "finish"
            else: # the state = finish; no need to run more
                assert state["mode"] == "finish"
                break
        
        set_state(state)
        if train_cmd:
            run_output_dir = output_dir + f"_{count}" if not final_output_dir else final_output_dir
            train_cmd = replace_args_in_cmd(train_cmd, "output_dir", run_output_dir)
            
            current_request_path = os.path.join(ds_folder, f"training_request_{args.task_id}_{count}.json")
            with open(current_request_path, "w") as f:
                json.dump(c_train_info, f, indent=4, ensure_ascii=False)
            
            train_cmd = replace_args_in_cmd(train_cmd, "request_path", current_request_path)
            
            state["train"] = {
                "train_cmd": train_cmd,
                "log_path": os.path.join(ds_folder, f"train_{args.task_id}.log"),
                "lr": extract_value_from_cmd(train_cmd, "learning_rate"),
                "output_dir": run_output_dir
            }
            state["train"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            set_state(state)
            
            log_path = state["train"]["log_path"]
            success, train_cmd = run_training(
                train_cmd,
                log_path,
                args.task_id,
                args.retries,
                args.task_type,
                args.expected_repo_name,
            )
            time.sleep(5)
            
            if not success:
                print(
                    f"All inner retries failed for task {args.task_id} "
                    f"(count={count}). Attempting task-level fallbacks …",
                    flush=True,
                )
                for fb_num in range(MAX_TASK_FALLBACKS):
                    fb_cmd, fb_desc = _next_task_fallback(
                        train_cmd, args.task_type, fb_num
                    )
                    if fb_cmd is None:
                        print(
                            "No more task-level fallback strategies available.",
                            flush=True,
                        )
                        break
                    print(
                        f"Task fallback {fb_num + 1}/{MAX_TASK_FALLBACKS}: {fb_desc}",
                        flush=True,
                    )
                    # Update output_dir so each fallback run goes to its own dir.
                    fb_output_dir = output_dir + f"_{count}_fb{fb_num}"
                    fb_cmd = replace_args_in_cmd(fb_cmd, "output_dir", fb_output_dir)
                    success, train_cmd = run_training(
                        fb_cmd,
                        log_path,
                        args.task_id,
                        args.retries,
                        args.task_type,
                        args.expected_repo_name,
                    )
                    if success:
                        print(
                            f"Task fallback {fb_num + 1} succeeded.", flush=True
                        )
                        break
                    time.sleep(5)

            if not success:
                print(
                    f"Training failed for task {args.task_id} at count={count} "
                    f"after all retries and fallbacks.",
                    flush=True,
                )
                break
        
        count += 1

    if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        print(f"Training failed for task {args.task_id}", flush=True)
    else:
        print(f"Training successfully done for task {args.task_id}", flush=True)
        train_success = True

    if not train_success:
        print(f"Training failed for task {args.task_id}", flush=True)
        # add noise to the model
        add_noise_cmd = f"python add_random_noise.py {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
