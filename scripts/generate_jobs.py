#!/usr/bin/env python3
"""
Grid Search + Ablation Job Generator (PBS, tiny-vit/resnet example)

這支程式會自動產生：
1) 個別 PBS shell 腳本：jobs/<CODE>.sh
2) 批次提交腳本：submit_all.sh (用 qsub 送出所有 jobs)
3) 對應表：mapping.csv（代號 ↔ 參數 ↔ 指令 ↔ 日誌）

使用方式：
$ python3 generate_jobs.py
$ bash submit_all.sh     # 送出所有工作（或自行挑選 qsub jobs/XXXX.sh）

如需調整：直接改動下方 SEARCH_SPACE、ABLATIONS、PBS_TEMPLATE、ENV 變數即可。
"""

import os
import csv
import itertools
from pathlib import Path
from datetime import datetime

# =============== 可自訂參數區 ==================
# 你的環境設定（PBS 標頭與環境啟動）
PBS_TEMPLATE = """#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=32GB
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06
#PBS -N {job_name}
""".strip()

ENV = {
    # 若不需要可留空字串
    "module_lines": [
        "module load cuda/12.6.2",
        # "module load python3/3.10.4",
    ],
    # 建議改成你的 venv/conda 啟動路徑
    "venv_activate": "/scratch/rp06/sl5952/BPM/.venv/bin/activate",
    # PBS 的 -l wd 會把工作目錄設為 qsub 當下目錄
    # 若你的 train.py 在提交目錄的上上層，就保留 "../.."；否則改成合適的相對路徑或空字串
    "workdir": "../..",
}

# 搜尋空間（可自由擴增/刪減）。
# 注意：這裡的數值以字串為主，可避免 1e-3 格式化差異。
SEARCH_SPACE = {
    "dataset": ["cotton80"],
    "model": [
        "tiny_vit_21m_384.dist_in22k_ft_in1k",
        # 如需加其他模型可在此加入，例如："resnet50"
    ],
    "img_size": ["384"],
    "epochs": ["100"],
    "batch_size": ["32"],
    "lr": ["1e-3"],
    "weight_decay": ["1e-4"],
    "proto_mode": ["mean"],  # 可改為 ["mean", "momentum"]
    "momentum_tau": ["0.99"], # 僅 proto_mode==momentum 時會帶入
    "ema": ["0.99"],
}

# Ablation 設定（名稱, alpha_inv, beta_uni, gamma_sd）
ABLATIONS = [
    ("ce",          "0.0", "0.0", "0.0"),   # cross-entropy only baseline
    ("alpha_only",  "0.5", "0.0", "0.0"),
    ("beta_only",   "0.0", "0.2", "0.0"),
    ("gamma_only",  "0.0", "0.0", "1.0"),
    ("full",        "0.5", "0.2", "1.0"),
]

# 代號前綴與位數（J0001, J0002, ...）
CODE_PREFIX = "J"
CODE_PAD = 4

# 產物輸出位置
JOBS_DIR = Path("jobs")
RESULTS_DIR = Path("results")
MAPPING_CSV = Path("mapping.csv")
SUBMIT_ALL = Path("submit_all.sh")

# 其他：是否在腳本裡頭先 mkdir -p results
ENSURE_RESULTS_DIR_IN_JOB = True

# =============== 產生工具函式 ==================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_command(params: dict) -> str:
    """根據參數組出 train.py 命令。"""
    parts = [
        "python3 train.py",
        f"--dataset {params['dataset']}",
        f"--model {params['model']}",
        f"--epochs {params['epochs']}",
        f"--img-size {params['img_size']}",
        f"--batch-size {params['batch_size']}",
        f"--lr {params['lr']}",
        f"--weight-decay {params['weight_decay']}",
        f"--ema {params['ema']}",
        f"--alpha-inv {params['alpha_inv']}",
        f"--beta-uni {params['beta_uni']}",
        f"--gamma-sd {params['gamma_sd']}",
        f"--proto-mode {params['proto_mode']}",
    ]
    if params.get("proto_mode") == "momentum":
        parts.append(f"--momentum-tau {params['momentum_tau']}")
    return " ".join(parts)


def write_job_script(code: str, params: dict) -> Path:
    """寫出單一 PBS 工作腳本，回傳檔案路徑。"""
    job_path = JOBS_DIR / f"{code}.sh"
    log_path = RESULTS_DIR / f"{params['dataset']}_{code}.log"
    gpu_info = RESULTS_DIR / f"gpu-info-{code}.txt"

    cmd = build_command(params) + f" >> {log_path}"

    pbs_head = PBS_TEMPLATE.format(job_name=code)

    lines = ["#!/bin/bash", pbs_head, ""]
    lines += ENV["module_lines"]
    lines.append(f"nvidia-smi >> {gpu_info}")
    if ENV.get("venv_activate"):
        lines.append(f"source {ENV['venv_activate']}")
    if ENV.get("workdir"):
        lines.append(f"cd {ENV['workdir']}")
    if ENSURE_RESULTS_DIR_IN_JOB:
        lines.append("mkdir -p results")
    lines.append(cmd)
    content = "\n".join(lines) + "\n"

    with open(job_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(job_path, 0o755)
    return job_path


def write_submit_all(codes):
    lines = ["#!/bin/bash", "set -e"]
    for c in codes:
        lines.append(f"qsub jobs/{c}.sh")
        lines.append("sleep 1")  # 避免一次送太快
    with open(SUBMIT_ALL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(SUBMIT_ALL, 0o755)


def write_mapping(rows: list):
    headers = [
        "code", "label", "dataset", "model", "img_size", "epochs", "batch_size",
        "lr", "weight_decay", "ema", "alpha_inv", "beta_uni", "gamma_sd",
        "proto_mode", "momentum_tau", "command", "job_script", "log_file",
    ]
    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =============== 主程式 ==================

def main():
    ensure_dir(JOBS_DIR)
    ensure_dir(RESULTS_DIR)

    # 展開基礎格子（不含 ablation 先）
    keys = [
        "dataset", "model", "img_size", "epochs", "batch_size",
        "lr", "weight_decay", "proto_mode", "momentum_tau", "ema",
    ]
    grid_values = [SEARCH_SPACE[k] for k in keys]

    codes = []
    mapping_rows = []
    counter = 1

    for combo in itertools.product(*grid_values):
        base = dict(zip(keys, combo))
        # momentum_tau 僅在 proto_mode==momentum 時有效，否則留著也無妨
        for label, a, b, g in ABLATIONS:
            params = base.copy()
            params.update({
                "alpha_inv": a,
                "beta_uni": b,
                "gamma_sd": g,
            })

            code = f"{CODE_PREFIX}{counter:0{CODE_PAD}d}"
            job_script_path = write_job_script(code, params)
            log_file = RESULTS_DIR / f"{params['dataset']}_{code}.log"
            cmd = build_command(params)

            mapping_rows.append({
                "code": code,
                "label": label,
                "dataset": params["dataset"],
                "model": params["model"],
                "img_size": params["img_size"],
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "ema": params["ema"],
                "alpha_inv": params["alpha_inv"],
                "beta_uni": params["beta_uni"],
                "gamma_sd": params["gamma_sd"],
                "proto_mode": params["proto_mode"],
                "momentum_tau": params["momentum_tau"],
                "command": cmd,
                "job_script": str(job_script_path),
                "log_file": str(log_file),
            })

            codes.append(code)
            counter += 1

    write_submit_all(codes)
    write_mapping(mapping_rows)

    print(f"Generated {len(codes)} jobs.")
    print(f" - jobs/: {len(codes)} scripts")
    print(f" - {SUBMIT_ALL}")
    print(f" - {MAPPING_CSV}")


if __name__ == "__main__":
    main()
