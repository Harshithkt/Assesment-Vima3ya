# NimbusPay triage: fine-tuning report

**Model:** Qwen2.5-3B-Instruct + LoRA (r=16, all 7 linear projections, 29.9M trainable), fp16 base (not 4-bit).
**Final result on dev:** **92.5% exact match, 98.6% mean field accuracy, 100% valid JSON** (greedy, batch size 1).

## 1. Data audit (Part A)
**Method.** I wrote a rule checker (`triage_rules.py`) from SCHEMA.md. It re-derives every field except `category`:
- ids, and amounts with units / `k` / `lakh`, skipping balance, limit and fee sentences
- absolute, year-less and relative dates, ignoring quoted `>` lines
- channel, language, legal threats and repeat contact
- priority and needs_human from the SCHEMA formulas

**On clean dev it reproduces 200/200 labels on all 7 fields**, so I treated each disagreement on train as a
candidate error and read every one. For category I used cross-validated confident learning. For duplicates and
leakage I used normalised text matching.

| problem (how found) | rows | action |
|---|---|---|
| **Dev leakage**: train ticket = dev ticket (30 exact + 15 differing only in whitespace / trailing "Thx"); 0 with test | 45 | drop (it would mean training on dev and inflate the score) |
| **Duplicates** within train, same normalisation; no group had conflicting labels | 91 | drop extra copies |
| **Junk tickets** ("test", "asdf asdf", empty) with full fabricated labels (amounts, ids, dates) | 10 | drop (they teach hallucination) |
| **Truncated label JSON** (`json.loads` fails) | 25 | fix: rebuilt from the surviving `category` + rules; kept only if every surviving field agreed (25/25) |
| **Old-spec labels**: key `prio` = low/medium/high; `txn_date` as DD/MM/YYYY | 70 | fix: low/medium/high map 44/15/11 → exactly P3/P2/P1 under v2, so rename; dates → ISO |
| **Priority noise**: random flips (₹495 refund = P1; fraud = P3; RBI threat = P3). `needs_human` in those rows matches the *correct* priority | 60 | fix: recompute |
| **Amount units**: rupees instead of paise (₹674 → `674`); `k`/`lakh` misapplied (34.3k → `34300`) | 25 | fix from the checker |
| **Un-normalised ids** (`np 81633 47892`, `NP-…`) | 20 | fix → `NP`+10 digits |
| **Category noise**: 10-fold TF-IDF+LR, out-of-fold accuracy 97.7%; I read all 30 disagreements | 0 | none: the classifier was wrong in nearly every case; a few are ambiguous ("my card's frozen") |

Result: 1,445 → **1,299 rows** (146 dropped, 200 fixed), all logged in `cleaning_log.csv`. Dates, channel and
language had no errors.

**Train–test shift (from the unlabelled test inputs).** These phrasings appear in test and never in train:
- `N P 5221 235 957` ids (3.0% of test)
- "last month on the 18th" / "the 28th of this month" (5.0%)
- "The sum involved is…" / "Poore Rs.X/- ka mamla" (9.3%)
- order numbers (6.5%), and fee distractors (3.2% vs 0.8% in train)

All are covered by SCHEMA rules, which motivated experiment D.4.

## 2. Baselines (Part B), dev
| | JSON valid | exact | mean field acc | priority | needs_human |
|---|---|---|---|---|---|
| fixed system prompt | 0.5% | 0.0% | 0.0% | 0% | 0% |
| + SCHEMA.md in the prompt | 100% | 26.0% | 82.0% | 49.5% | 75.0% |

- **Fixed prompt.** The model invents its own schema inside markdown fences (`"category": "ID Acceptance"`).
- **Rules in the prompt.** Format and extraction are mostly solved (txn_id 91.5%), but the rules are not: priority 49.5%, exact match 7% on fraud and 0% on account_access. The fine-tune has to learn the rules from data.

## 3. Training pipeline (Part C)
- **Stack.** HF `transformers` + `peft` with a small explicit PyTorch loop, so masking, gradient-accumulation normalisation and fp16 handling are visible and don't depend on library versions. Inference uses the plain `generate` path the graders use.
- **Chat template.** The training prompt is `apply_chat_template([system, user], add_generation_prompt=True)`, the graders' exact string. I assert the prompt ids are a prefix of the full tokenised conversation. Qwen2.5 has a native system role.
- **Loss.** Only on the answer JSON + `<|im_end|>` (so the model learns to stop). Logits are computed only at answer positions; summed token CE is divided by the answer tokens in the whole optimizer step.
- **Sequence length from evidence.** Lengths are bimodal: p95 ≈ 225 tokens, but quoted e-mail threads reach 2,214 (train) / 2,089 (test), and answers are ≤ 84 tokens. `max_len = 2400` keeps every row; truncating would cut off the answer. Length-grouped batches and answer-only logits keep the long rows cheap.
- **Settings.**
  - r=16 / alpha=32: 29.9M parameters. r=32 would be 59.9M, over the limit.
  - lr 2e-4 cosine with 5% warm-up, effective batch 16, 2 epochs, dropout 0.05.
  - fp16 autocast + GradScaler (the T4 has no bf16); gradient checkpointing for the 2.3k-token rows.

## 4. Controlled experiments (Part D), each vs `ref`, one NVIDIA T4 per run
| run | what changed | dev exact | dev mean field acc | peak GPU mem | train min | McNemar p |
|---|---|---|---|---|---|---|
| `ref` | cleaned data, 16-bit LoRA, r16, lr 2e-4, 2 ep | 90.5% | 98.19% | 9.95 GB | 15.6 | – |
| D.1 `raw` | shipped train.jsonl (minus the 45 dev-leak rows) | 87.0% | 97.63% | 9.95 GB | 16.3 | 0.17 |
| D.2 `qlora` | base in 4-bit NF4 | 89.5% | 98.13% | **8.28 GB** | 20.1 | 0.80 |
| D.3 `ep3` | 3 epochs | 92.0% | 98.25% | 9.95 GB | 23.0 | 0.55 |
| D.4 `aug` | + 700 rule-grounded augmented rows (my idea) | 92.0% | **98.50%** | 10.16 GB | 24.5 | 0.51 |
| **final** | `aug` config, retrained in the notebook | **92.5%** | **98.56%** | 10.16 GB | (A10G, §7) | – |

The p-value is an exact McNemar test on per-row exact match. **None is significant at n=200**: one row is 0.5 pt,
and `raw`'s −3.5 pt is 13 rows lost vs 6 gained.
- **D.1 raw.** Worse on exactly the noised fields: txn_date 97.0 vs 99.0, priority 90.5 vs 92.5, account_access exact 89 vs 100.
- **D.2 QLoRA.** −1.7 GB of memory, but 30% slower (dequantisation) and no better. On a 15 GB T4, a 3B model doesn't need 4-bit.
- **D.3 epochs.** +1.5 pt (7 rows gained, 4 lost) for 50% more time; train loss is already ~1e-3 after epoch 2.
- **D.4 augmentation.** Best field accuracy (amount and channel 100%, needs_human 95.5 → 97.0), and it is the only change aimed at the test-only phrasings, which dev doesn't contain. I chose it for the final model rather than an untested `aug`+3-epoch combination.

The final retrain differs from the `aug` run by one dev row. Two things changed: it ran on a different GPU, and its
dev pass used batch size 1 instead of padded batches of 8.

## 5. Error analysis (Part E), final model on dev
15/200 rows are not exact. **Weakest categories:** fraud 80.0% exact (category and priority 87%), refund 87.3%
(priority 91%, needs_human 93%), payment_failed 90.6%. The other four categories are 100%.
**Weakest fields:** priority 95.0%, needs_human 97.0%, category 98.5%, txn_date 98.5%, txn_id 99.5%; amount,
channel and language are 100%. The amount is right in *every* failing row, but 10/15 failures are priority errors,
and all 6 needs_human errors follow from a wrong P1 decision.

| id | wrong | failure | my explanation |
|---|---|---|---|
| dv-00011 | prio, nh | ₹84,137 refund → P2 | missed ≥₹50k |
| dv-00141 | prio, nh | ₹61,597 → P2 | missed ≥₹50k |
| dv-00037 | prio, nh | `284669 rupees` → P2 | magnitude of a 6-digit number without commas |
| dv-00034 | prio, nh | ₹36,737 → P1 | over-escalation below ₹50k |
| dv-00056 | prio, nh | ₹44,206.75 → P1 | same, near the threshold |
| dv-00098 | prio, nh | 41.5k → P1 | same; `k` form |
| dv-00127 | prio | ₹4,836 → P2 | rounded up across ₹5k |
| dv-00180 | prio | ₹3,130 refund → P2, with a ₹2,41,670 balance in the ticket | distractor leaked into priority (amount itself extracted right) |
| dv-00046 | cat, prio | "A payment on my statement is wrong" → refund (gold fraud) | ambiguous: no explicit "not me" |
| dv-00157 | cat, prio | "Kisi ne OTP manag ke mere piase nikal liye, yeh maine nahi kiya" → account_access | typos plus "OTP" outweighed "maine nahi kiya" |
| dv-00177 | cat | "red flag on my top up" → fraud | "red flag" read as a fraud cue |
| dv-00006 | date | "day before yesterday" → recv−1 | off by one on relative-date arithmetic |
| dv-00120 | date | "parso hi hua yeh" → recv | Hinglish "parso" not resolved |
| dv-00087 | date | "It was 4 days ago" at the end of a long ticket → null | date dropped after several distractor sentences |
| dv-00124 | txn_id | id is the only transaction field → null | collapsed to the "no transaction details" pattern |

**Why priority is weak.** The schema's key order puts `priority` *before* `amount_paise`, so the model must
compare the amount with ₹5k/₹50k before writing it, with no scratch space. The errors sit right at the thresholds.

**Harder test slice (diagnostic only; no labels, predictions unchanged).** On test rows with the new phrasings, the
model's amount and txn_id agree with the rule checker on 100% of rows, so that augmentation transferred. But on
rows with a **convenience fee**, priority agrees only 6/13. Each time the claim amount is right but priority
escalates as if the fee counted (₹931 claim + ₹4,658 fee → P2).

My augmentation caused this blind spot: it only inserted small fees (≤ ₹2,033), so no training row had a
distractor that crossed a threshold. dv-00180 shows the same thing with a balance.

## 6. With one more week
1. **Emit `amount_paise` before `priority`.** The scorer compares parsed JSON, so key order is free. The model then decides priority after writing the amount. This targets 10/15 dev failures, for one run.
2. **Contrastive distractors** (fee, balance or limit on the other side of ₹5k/₹50k from the claim) to fix the fee blind spot.
3. **Boundary drills in every amount format** (`284669 rupees`, `49.9k`, `0.5 lakh`) and relative dates across month ends ("parso", "day before yesterday").
4. **3 seeds per config.** Current gaps are 1–3 rows, so I can't separate `ep3` from `aug`; then try `aug` + 3 epochs.
5. **Ambiguous category cases** ("wrong payment", "red flag", typo-heavy Hinglish fraud): agree the labels with the labelling owner.
6. **A hard dev set** from train tickets rewritten with templates the augmenter never uses, to measure the test-only phrasings without looking at test.

## 7. Compute
- Baselines and the 5 experiments ran on **NVIDIA T4s** via Modal (`modal_app.py`), using the notebook's own code and pinned versions.
- The final top-to-bottom notebook execution ran on a Modal **A10G** to save time (55.9 min, including 12.6 min of training).
- The adapter is a standard fp16-base LoRA and runs on a T4 (peak training memory 10.2 GB). Greedy outputs can differ in rare tokens across GPU types; the in-notebook reproducibility check (10/10 identical) ran on the A10G.
- 98/100 test outputs are unchanged if `repetition_penalty` is switched from Qwen's default 1.05 to 1.0.

## 8. AI tools
I used Claude (Claude Code) as a pair programmer for:
- the rule-checker regexes and the audit and augmentation scripts
- the training loop, notebook and Modal launcher
- drafting this report

I chose what to check and fix, reviewed the flagged rows, and checked the numbers against the notebook outputs.
No model or API produced labels or predictions. Labels come from `train.jsonl` + SCHEMA rules; predictions are raw
outputs of the fine-tuned model.
