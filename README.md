# NimbusPay ticket triage: LoRA fine-tune of a ≤4B model

| | |
|---|---|
| **Base model** | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (3.09B parameters) |
| **Loaded in 4-bit?** | **No.** The base is loaded in fp16 for training and inference (16-bit LoRA). |
| **Adapter** | `adapter/` (PEFT LoRA, r=16, alpha=32, all 7 linear projections, 29.9M trainable params) |
| **Best result (dev, `score.py`)** | **92.5% exact match, 98.6% mean field accuracy, 100% valid JSON** (final adapter, greedy, batch size 1) |

## Running the submitted adapter (how `predictions_test.jsonl` was made)

```python
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(model, "adapter").eval()

msgs = row["messages"][:2]                                   # exactly the system + user messages
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=200, do_sample=False)    # greedy, batch size 1
text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

- Qwen2.5 has a native `system` role, so the fixed system prompt is passed as-is.
- No merging, no post-processing beyond `.strip()`.
- The model's shipped `generation_config` is left untouched (it includes `repetition_penalty=1.05`). The notebook also measures how many test outputs change with `repetition_penalty=1.0`.
- Library versions: `transformers==5.22.2`, `peft==0.21.2`, `accelerate==1.15.0`, `torch==2.15.0`. Baselines and experiments ran on NVIDIA T4s (Modal). The final notebook run, which produced `adapter/` and the predictions, ran on a Modal A10G. The adapter is a standard fp16-base LoRA and runs on a T4 (see report §7).

## Files

| file | what |
|---|---|
| `notebook.ipynb` | the full pipeline with outputs: audit → baselines → pipeline checks → experiments → final model → error analysis |
| `report.md` | audit findings, experiment table, error analysis, next steps, AI-tool usage |
| `cleaning_log.csv` | every dropped / fixed training row (`id,action,reason`) |
| `predictions_test.jsonl` | raw model outputs for the 400 test tickets |
| `adapter/` | final LoRA adapter (PEFT format) |
| `triage_rules.py` | rule checker written from SCHEMA.md (used for auditing and to label augmented rows, never at inference) |
| `audit.py` | cleaning: writes `cleaning_log.csv` and `data/train_clean.jsonl` |
| `augment.py` | rule-grounded augmentation: writes `data/train_aug.jsonl` |
| `train_lib.py` | tokenisation, answer-only loss, PyTorch training loop, inference |
| `experiments.py` | baseline / experiment / final configurations, shared by the notebook and `modal_app.py` |
| `modal_app.py` | runs the same code on Modal, pinned to one T4 per job |
| `results/` | saved metrics and dev predictions for every baseline and experiment |

## Reproducing

The data and the candidate-pack files are not in this repo. Put `data/{train,dev,test_inputs}.jsonl`, `SCHEMA.md`
and `score.py` next to the code.

- **Colab (free T4):** copy the folder to `MyDrive/nimbuspay-triage/`, open `notebook.ipynb`, and run all. This retrains the final adapter and regenerates the test predictions in about 1 hour. Parts B and D are shown from `results/` unless `RUN_BASELINES=1` / `RUN_EXPERIMENTS=1` are set.
- **Modal (what I used for the GPU runs, T4 only):** `modal run modal_app.py::experiments` (Parts B and D, in parallel), then `modal run modal_app.py::notebook`.
