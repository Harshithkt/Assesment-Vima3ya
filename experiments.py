"""Baselines (Part B), controlled experiments (Part D) and the final run, shared by notebook.ipynb
and modal_app.py so both execute exactly the same code."""
import gc
import json
import os
from dataclasses import replace

import torch

from train_lib import (FIXED_SYSTEM, KEYS, Cfg, generate, lenient_parse, load_for_inference, load_jsonl,
                       score_preds, train, write_preds)

BASE = "Qwen/Qwen2.5-3B-Instruct"
SCHEMA_SYSTEM = (FIXED_SYSTEM + "\n\nFollow this labelling specification exactly. Output only the JSON object.\n\n"
                 + open("SCHEMA.md", encoding="utf-8").read())
EVAL_BS = 8                 # baselines / experiments: left-padded batches of 8 (speed); final run uses 1

REF = Cfg(name="ref", base_model=BASE, train_files=["data/train_clean.jsonl"])
EXPERIMENTS = {
    "ref": REF,
    "raw": replace(REF, name="raw", train_files=["data/train_raw_noleak.jsonl"]),
    "qlora": replace(REF, name="qlora", load_4bit=True),
    "ep3": replace(REF, name="ep3", epochs=3),
    "aug": replace(REF, name="aug", train_files=["data/train_clean.jsonl", "data/train_aug.jsonl"]),
}
FINAL = replace(REF, name="final", train_files=["data/train_clean.jsonl", "data/train_aug.jsonl"], epochs=2)
BASELINES = {"baseline_fixed_prompt": None, "baseline_rules_in_prompt": SCHEMA_SYSTEM}
META_KEYS = ["train_minutes", "peak_gpu_mem_gb", "trainable_params", "train_sequences", "optimizer_steps", "cfg", "log"]


def free(model=None):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lenient_score(rows, preds):
    fixed = []
    for p in preds:
        o = lenient_parse(p["output"])
        fixed.append({"id": p["id"], "output": json.dumps(o) if o else ""})
    return score_preds(rows, fixed)["all"]


def summarize(name, rows, preds, extra=None):
    s = score_preds(rows, preds)
    res = {"name": name, "n": len(rows), "json_valid": s["all"]["json_valid"], "exact_match": s["all"]["exact_match"],
           "mean_field_acc": s["all"]["mean_field_acc"], "fields": s["all"]["fields"], "groups": s}
    res.update(extra or {})
    return res


def run_baseline(name, rows, base=BASE, bs=EVAL_BS):
    tok, model = load_for_inference(base)
    preds = generate(tok, model, rows, system=BASELINES[name], batch_size=bs)
    free(model)
    write_preds(preds, f"results/{name}_dev_preds.jsonl")
    res = summarize(name, rows, preds, {"lenient": lenient_score(rows, preds)})
    json.dump(res, open(f"results/{name}.json", "w"), indent=1)
    return res


def run_experiment(cfg, rows, bs=EVAL_BS):
    meta = train(cfg)
    tok, model = load_for_inference(cfg.base_model, os.path.join(cfg.out_dir, "adapter"), cfg.load_4bit)
    preds = generate(tok, model, rows, batch_size=bs)
    free(model)
    write_preds(preds, f"results/{cfg.name}_dev_preds.jsonl")
    res = summarize(cfg.name, rows, preds, {k: meta[k] for k in META_KEYS})
    json.dump(res, open(f"results/{cfg.name}.json", "w"), indent=1)
    return res


def exact_ids(name, rows):
    preds = {p["id"]: p["output"] for p in load_jsonl(f"results/{name}_dev_preds.jsonl")}
    ok = set()
    for g in rows:
        gold = json.loads(g["messages"][-1]["content"])
        try:
            p = json.loads(preds.get(g["id"], ""))
        except Exception:
            continue
        if isinstance(p, dict) and set(p) == set(KEYS) and all(type(p[k]) is type(gold[k]) and p[k] == gold[k] for k in KEYS):
            ok.add(g["id"])
    return ok


def mcnemar(a_ids, b_ids):
    """Exact two-sided McNemar test on the discordant rows."""
    from math import comb
    b, c = len(a_ids - b_ids), len(b_ids - a_ids)
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)
