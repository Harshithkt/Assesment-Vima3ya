"""Run the GPU work on Modal, pinned to the same hardware as the assignment (one NVIDIA T4 per job).

    modal run modal_app.py::prefetch                 # download the base model into a cache volume (once)
    modal run modal_app.py::experiments              # Part B + D: baselines and 5 experiments, in parallel, 1 T4 each
    modal run modal_app.py::experiments --names ref,aug
    modal run modal_app.py::notebook [--gpu A10G]    # execute notebook.ipynb top to bottom on a T4 (final model,
                                                     #   test predictions; Parts B/D shown from saved results)
    modal volume get nimbus-triage out ./modal_out   # fetch executed notebook, adapter, predictions, results

The notebook stays runnable on Colab unchanged: it detects Colab / Modal / local and uses the same
pinned library versions in each.
"""
import os
import shutil
import subprocess
import sys

import modal

APP_DIR = os.path.dirname(os.path.abspath(__file__))
app = modal.App("nimbus-triage")
vol = modal.Volume.from_name("nimbus-triage", create_if_missing=True)
hf_cache = modal.Volume.from_name("nimbus-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "transformers==5.22.2", "peft==0.21.2", "accelerate==1.15.0", "bitsandbytes",
                 "scikit-learn", "pandas", "tabulate", "matplotlib", "nbclient", "nbconvert", "ipykernel")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(APP_DIR, "/src", ignore=["runs", "results", "adapter", "modal_out", ".git", "__pycache__",
                                            "*.ipynb_checkpoints", "data/smoke_*", "executed*.ipynb"])
)
BASE = "Qwen/Qwen2.5-3B-Instruct"
JOBS = ["baseline_fixed_prompt", "baseline_rules_in_prompt", "ref", "raw", "qlora", "ep3", "aug"]


def _workspace(path):
    """Fresh private copy of code + data, then build the cleaned / augmented / raw-noleak files exactly as the notebook does."""
    shutil.rmtree(path, ignore_errors=True)
    shutil.copytree("/src", path)
    os.chdir(path)
    sys.path.insert(0, path)
    subprocess.run([sys.executable, "audit.py"], check=True)
    subprocess.run([sys.executable, "augment.py", "--n", "700", "--seed", "0"], check=True)
    import csv, json
    leak = {r["id"] for r in csv.DictReader(open("cleaning_log.csv")) if r["reason"].startswith("same ticket as a dev")}
    with open("data/train_raw_noleak.jsonl", "w") as f:
        for line in open("data/train.jsonl"):
            if json.loads(line)["id"] not in leak:
                f.write(line)


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=3600)
def prefetch_model():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE)
    hf_cache.commit()


@app.function(image=image, gpu="T4", volumes={"/vol": vol, "/hf": hf_cache}, timeout=4 * 3600)
def run_job(name: str):
    import json
    _workspace(f"/tmp/work_{name}")
    os.makedirs("results", exist_ok=True)
    import experiments as X
    from train_lib import load_jsonl
    dev = load_jsonl("data/dev.jsonl")
    res = X.run_baseline(name, dev) if name in X.BASELINES else X.run_experiment(X.EXPERIMENTS[name], dev)
    os.makedirs("/vol/results", exist_ok=True)
    for f in os.listdir("results"):
        shutil.copy(os.path.join("results", f), "/vol/results/")
    if os.path.exists(f"runs/{name}"):
        shutil.copytree(f"runs/{name}", f"/vol/runs/{name}", dirs_exist_ok=True)
    vol.commit()
    return {k: res.get(k) for k in ["name", "exact_match", "mean_field_acc", "json_valid", "train_minutes", "peak_gpu_mem_gb"]}


@app.function(image=image, gpu="T4", volumes={"/vol": vol, "/hf": hf_cache}, timeout=8 * 3600)
def run_notebook(run_baselines: bool = False, run_experiments: bool = False):
    import time
    import nbformat
    from nbclient import NotebookClient
    work = "/tmp/nbwork"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree("/src", work)
    if os.path.exists("/vol/results"):                       # saved Part B / D results
        shutil.copytree("/vol/results", os.path.join(work, "results"), dirs_exist_ok=True)
    os.environ.update({"WORKDIR": work, "RUN_BASELINES": "1" if run_baselines else "0",
                       "RUN_EXPERIMENTS": "1" if run_experiments else "0"})
    nb = nbformat.read(os.path.join(work, "notebook.ipynb"), as_version=4)
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=4 * 3600, kernel_name="python3", resources={"metadata": {"path": work}}).execute()
    finally:
        out = "/vol/out"
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        nbformat.write(nb, os.path.join(out, "notebook.ipynb"))
        for f in ["predictions_test.jsonl", "cleaning_log.csv", "audit_summary.json"]:
            if os.path.exists(os.path.join(work, f)):
                shutil.copy(os.path.join(work, f), out)
        for d in ["adapter", "results"]:
            if os.path.exists(os.path.join(work, d)):
                shutil.copytree(os.path.join(work, d), os.path.join(out, d))
        vol.commit()
    return f"notebook executed in {(time.time() - t0) / 60:.1f} min"


@app.local_entrypoint()
def prefetch():
    prefetch_model.remote()


@app.local_entrypoint()
def experiments(names: str = ",".join(JOBS)):
    for r in run_job.map(names.split(","), return_exceptions=True):
        print(r)


@app.local_entrypoint()
def notebook(run_baselines: bool = False, run_experiments: bool = False, gpu: str = "T4"):
    print(run_notebook.with_options(gpu=gpu).remote(run_baselines, run_experiments))
