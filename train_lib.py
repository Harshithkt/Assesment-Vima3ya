"""Training / inference library used by notebook.ipynb.

Design choices (see report.md for the evidence behind each):
  * Prompt = the model's own chat template applied to [system, user] with add_generation_prompt=True,
    exactly what the graders do at inference. Training sequences are built from the *same* token ids
    (asserted), followed by the answer tokens and <|im_end|>.
  * Loss only on answer tokens (+ <|im_end|> so the model learns to stop). The logits are computed
    only at answer positions: avoids a [batch, seq, 152k] fp32 tensor for the long quoted-email rows.
  * Plain PyTorch loop: fp16 autocast + GradScaler (T4 has no bf16), length-grouped micro-batches,
    gradient accumulation normalised by the number of answer tokens in the whole optimizer step.
"""
import gc
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

FIXED_SYSTEM = "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."
KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Cfg:
    name: str = "ref"
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    train_files: list = field(default_factory=lambda: ["data/train_clean.jsonl"])
    load_4bit: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: list(ALL_LINEAR))
    lr: float = 2e-4
    epochs: int = 2
    micro_bs: int = 4
    grad_accum: int = 4
    warmup_frac: float = 0.05
    weight_decay: float = 0.0
    max_len: int = 2400
    seed: int = 42
    out_root: str = "runs"

    @property
    def out_dir(self):
        return os.path.join(self.out_root, self.name)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- tokenisation
def prompt_text(tok, system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   add_generation_prompt=True, tokenize=False)


def encode_example(tok, msgs, max_len):
    """input_ids = prompt ids (identical to inference) + answer ids + <|im_end|>; labels mask the prompt."""
    prompt = prompt_text(tok, msgs[0]["content"], msgs[1]["content"])
    full = tok.apply_chat_template(msgs[:3], tokenize=False)
    assert full.startswith(prompt), "chat template: prompt is not a prefix of the full conversation"
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids
    assert f_ids[: len(p_ids)] == p_ids, "tokenisation of the prompt differs inside the full sequence"
    ans = f_ids[len(p_ids):]
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    last = len(ans) - 1 - ans[::-1].index(end_id)
    ans = ans[: last + 1]                         # drop the template's trailing "\n" after <|im_end|>
    ids = p_ids + ans
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + ans}


def build_dataset(tok, rows, max_len):
    data, dropped = [], 0
    for r in rows:
        ex = encode_example(tok, r["messages"], max_len)
        if ex is None:
            dropped += 1
        else:
            data.append(ex)
    return data, dropped


def length_grouped_batches(lengths, bs, seed, mega=16):
    """Shuffle, then sort inside mega-batches of bs*mega so each micro-batch has similar lengths."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    for i in range(0, len(idx), bs * mega):
        chunk = sorted(idx[i: i + bs * mega], key=lambda j: -lengths[j])
        batches += [chunk[k: k + bs] for k in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def collate(examples, pad_id, device):
    L = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        lab[i, :n] = torch.tensor(e["labels"])
        att[i, :n] = 1
    return ids.to(device), lab.to(device), att.to(device)


# ----------------------------------------------------------------------------- model
def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(name, load_4bit, device_map=None):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kw = {"dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    if torch.cuda.is_available():
        kw["device_map"] = device_map or {"": 0}
    try:
        return AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", **kw)
    except TypeError:                       # older transformers: torch_dtype instead of dtype
        kw["torch_dtype"] = kw.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", **kw)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def answer_loss_sum(peft_model, ids, labels, att):
    """Sum of token CE over answer positions; logits only computed where needed."""
    base = peft_model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    tgt = labels[:, 1:]
    m = tgt != -100
    logits = base.lm_head(h[:, :-1][m]).float()
    return F.cross_entropy(logits, tgt[m], reduction="sum"), int(m.sum())


# ----------------------------------------------------------------------------- training
def train(cfg: Cfg, log_every=10):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_cosine_schedule_with_warmup

    os.makedirs(cfg.out_dir, exist_ok=True)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    seed_all(cfg.seed)
    tok = load_tokenizer(cfg.base_model)
    rows = [r for f in cfg.train_files for r in load_jsonl(f)]
    data, dropped = build_dataset(tok, rows, cfg.max_len)
    print(f"[{cfg.name}] {len(data)} training sequences ({dropped} over max_len={cfg.max_len} dropped)")

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_base(cfg.base_model, cfg.load_4bit)
    gc_kwargs = {"use_reentrant": False}
    if cfg.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=gc_kwargs)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()
    model.config.use_cache = False
    seed_all(cfg.seed)                                   # LoRA init is seeded
    model = get_peft_model(model, LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                                             target_modules=list(cfg.target_modules), bias="none",
                                             task_type="CAUSAL_LM"))
    n_train = count_trainable(model)
    print(f"[{cfg.name}] trainable params: {n_train / 1e6:.2f}M")
    assert n_train <= 50_000_000, "assignment limit: at most 50M trainable parameters"

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    lengths = [len(e["input_ids"]) for e in data]
    steps_per_epoch = math.ceil(math.ceil(len(data) / cfg.micro_bs) / cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_frac * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    pad_id = tok.pad_token_id

    log, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(cfg.epochs):
        batches = length_grouped_batches(lengths, cfg.micro_bs, cfg.seed + epoch)
        for g in range(0, len(batches), cfg.grad_accum):
            group = batches[g: g + cfg.grad_accum]
            n_tok = sum(sum(1 for x in data[j]["labels"][1:] if x != -100) for b in group for j in b)
            loss_acc = 0.0
            for b in group:
                ids, lab, att = collate([data[j] for j in b], pad_id, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    s, _ = answer_loss_sum(model, ids, lab, att)
                loss = s / n_tok
                scaler.scale(loss).backward()
                loss_acc += loss.item()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == total:
                rec = {"step": step, "epoch": round(step / steps_per_epoch, 2), "loss": round(loss_acc, 4),
                       "grad_norm": round(gnorm, 3), "lr": sched.get_last_lr()[0],
                       "elapsed_min": round((time.time() - t0) / 60, 2)}
                log.append(rec)
                print(rec)
    minutes = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
    model.save_pretrained(os.path.join(cfg.out_dir, "adapter"))
    meta = {"cfg": asdict(cfg), "trainable_params": n_train, "train_sequences": len(data),
            "dropped_over_max_len": dropped, "optimizer_steps": total, "train_minutes": round(minutes, 2),
            "peak_gpu_mem_gb": round(peak, 2), "log": log}
    json.dump(meta, open(os.path.join(cfg.out_dir, "train_meta.json"), "w"), indent=1)
    print(f"[{cfg.name}] done in {minutes:.1f} min, peak GPU memory {peak:.2f} GB")
    del model, opt
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return meta


# ----------------------------------------------------------------------------- inference
def load_for_inference(base_name, adapter_dir=None, load_4bit=False):
    """Exactly how a grader would load the submission: base model + PEFT adapter, no merging."""
    from peft import PeftModel
    tok = load_tokenizer(base_name)
    model = load_base(base_name, load_4bit)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, rows, system=None, batch_size=1, max_new_tokens=200, progress=50, **gen_kw):
    """Greedy decoding with the model's generation_config otherwise untouched.
    batch_size=1 (no padding) is what the final predictions use; >1 uses left padding for speed."""
    tok.padding_side = "left"
    prompts = [prompt_text(tok, system or r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    outs, t0 = [], time.time()
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i: i + batch_size], return_tensors="pt", padding=batch_size > 1,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, **gen_kw)
        for j in range(gen.shape[0]):
            outs.append(tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        if progress and (i // batch_size) % progress == 0:
            print(f"  {i + len(gen)}/{len(prompts)}  {(time.time() - t0) / 60:.1f} min")
    return [{"id": r["id"], "output": o} for r, o in zip(rows, outs)]


def write_preds(preds, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- scoring / analysis
def score_preds(gold_rows, preds):
    from score import score
    return score(gold_rows, preds)


def lenient_parse(text):
    """Analysis only (never used for predictions): first {...} block in the text."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


def failures(gold_rows, preds):
    pm = {p["id"]: p["output"] for p in preds}
    out = []
    for g in gold_rows:
        gold = json.loads(g["messages"][-1]["content"])
        raw = pm.get(g["id"])
        try:
            pred = json.loads(raw)
        except Exception:
            pred = None
        if not isinstance(pred, dict):
            out.append({"id": g["id"], "wrong": ["INVALID_JSON"], "gold": gold, "pred": raw,
                        "ticket": g["messages"][1]["content"]})
            continue
        wrong = [k for k in KEYS if not (k in pred and type(pred[k]) is type(gold[k]) and pred[k] == gold[k])]
        if wrong or set(pred) != set(KEYS):
            out.append({"id": g["id"], "wrong": wrong or ["EXTRA_KEYS"], "gold": gold, "pred": pred,
                        "ticket": g["messages"][1]["content"]})
    return out
