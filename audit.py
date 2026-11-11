"""Audit + clean data/train.jsonl.

    python audit.py            # writes cleaning_log.csv, data/train_clean.jsonl, audit_summary.json

Every decision here uses only train.jsonl and SCHEMA.md. dev.jsonl / test_inputs.jsonl are read for
ONE purpose: to detect train rows that duplicate a dev/test ticket (leakage), which are dropped.
The rule checker in triage_rules.py was validated on dev (it reproduces 200/200 dev labels for every
field except category) - dev labels are never copied into train.
"""
import collections
import csv
import json
import re
import sys

from triage_rules import (KEYS, TXN_CATS, derive_needs_human, derive_priority, load, parse_ticket,
                          signals)

TRAIN, DEV, TEST = "data/train.jsonl", "data/dev.jsonl", "data/test_inputs.jsonl"


def norm_body(user_text):
    """Ticket body with whitespace collapsed and a trailing 'Thx' removed (the two kinds of noise
    seen between otherwise identical tickets)."""
    body = user_text.split("\n\n", 1)[1]
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"\s*(Thx|thx)$", "", body)
    return body.lower()


def norm_full(user_text):
    return user_text.split("\n\n", 1)[0] + "||" + norm_body(user_text)


def salvage_truncated(raw):
    """Pull the complete `"key": value` pairs out of a truncated JSON string."""
    out = {}
    for m in re.finditer(r'"(\w+)":\s*("([^"]*)"|null|true|false|-?\d+)(?=,)', raw):
        k, v = m.group(1), m.group(2)
        out[k] = json.loads(v)
    return out


def reference_label(category, sig):
    txn = category in TXN_CATS
    amount = sig["amount"] if txn else None
    priority = derive_priority(category, amount, sig["legal"])
    return {
        "category": category,
        "priority": priority,
        "amount_paise": amount,
        "txn_date": sig["date"] if txn else None,
        "txn_id": sig["txn_id"] if txn else None,
        "channel": (sig["channels"][0] if len(sig["channels"]) == 1 else None) if txn else None,
        "language": sig["language"],
        "needs_human": derive_needs_human(priority, sig["repeat"]),
    }


def main():
    train, dev, test = load(TRAIN), load(DEV), load(TEST)
    dev_bodies = {norm_body(r["messages"][1]["content"]) for r in dev}
    test_bodies = {norm_body(r["messages"][1]["content"]) for r in test}

    log = []                       # (id, action, reason)
    stats = collections.Counter()
    seen = {}
    clean = []

    for r in train:
        rid, user, raw = r["id"], r["messages"][1]["content"], r["messages"][2]["content"]
        _, _, text = parse_ticket(user)
        reasons = []

        # ---- 1. junk tickets: no customer text, labels cannot be grounded ----------------
        if len(text.strip()) < 25 and re.fullmatch(r"(?i)[\s\d]*((test|asdf)[\s\d]*)*", text.strip()):
            log.append((rid, "drop", f"junk ticket body {text.strip()!r} with a fabricated non-empty label"))
            stats["drop_junk"] += 1
            continue

        # ---- 2. leakage into dev / test --------------------------------------------------
        nb = norm_body(user)
        if nb in dev_bodies:
            log.append((rid, "drop", "same ticket as a dev row (after whitespace/'Thx' normalisation): dev leakage"))
            stats["drop_dev_leak"] += 1
            continue
        if nb in test_bodies:
            log.append((rid, "drop", "same ticket as a test row: test leakage"))
            stats["drop_test_leak"] += 1
            continue

        # ---- 3. duplicates inside train --------------------------------------------------
        key = norm_full(user)
        if key in seen:
            log.append((rid, "drop", f"duplicate of {seen[key]} (identical ticket up to whitespace/'Thx'; same label)"))
            stats["drop_duplicate"] += 1
            continue
        seen[key] = rid

        sig = signals(user)

        # ---- 4. parse the label ----------------------------------------------------------
        try:
            label = json.loads(raw)
        except json.JSONDecodeError:
            part = salvage_truncated(raw)
            if "category" not in part:
                log.append((rid, "drop", "truncated label JSON without a category; cannot be repaired"))
                stats["drop_truncated"] += 1
                continue
            ref = reference_label(part["category"], sig)
            clash = [k for k in part if k in ref and k != "priority" and part[k] != ref[k]]
            if clash:
                log.append((rid, "drop", f"truncated label JSON; surviving fields {clash} disagree with the spec"))
                stats["drop_truncated"] += 1
                continue
            label = ref
            reasons.append(f"label JSON truncated after {len(raw)} chars; rebuilt from category + SCHEMA rules "
                           f"({len(part)} surviving fields all agree)")
            stats["fix_truncated"] += 1

        if "prio" in label:
            old = label.pop("prio")
            label["priority"] = old           # placeholder, recomputed below
            reasons.append(f"old-spec key 'prio'={old!r} renamed to 'priority' and recomputed with v2 rules")
            stats["fix_prio_key"] += 1

        d = label.get("txn_date")
        if isinstance(d, str) and re.fullmatch(r"\d{2}/\d{2}/\d{4}", d):
            iso = "-".join(reversed(d.split("/")))
            reasons.append(f"txn_date {d!r} in old DD/MM/YYYY format -> {iso!r}")
            label["txn_date"] = iso
            stats["fix_date_format"] += 1

        # ---- 5. field-level repairs -------------------------------------------------------
        cat = label["category"]
        txn = cat in TXN_CATS

        if txn and label["txn_id"] is not None and label["txn_id"] != sig["txn_id"]:
            if sig["txn_id"] and re.sub(r"\D", "", label["txn_id"]) == sig["txn_id"][2:]:
                reasons.append(f"txn_id {label['txn_id']!r} not normalised -> {sig['txn_id']!r}")
                label["txn_id"] = sig["txn_id"]
                stats["fix_txn_id_format"] += 1

        if txn and label["amount_paise"] != sig["amount"] and sig["amount"] is not None and label["amount_paise"] is not None:
            a, b = label["amount_paise"], sig["amount"]
            if a == b // 100:
                why = "label is in rupees, not paise"
            elif a * 100 == b or a * 10 == b or a * 1000 == b:
                why = "k/lakh multiplier applied wrongly"
            else:
                why = "does not match the stated claim amount"
            reasons.append(f"amount_paise {a} -> {b} ({why})")
            label["amount_paise"] = b
            stats["fix_amount"] += 1

        if not txn:
            for k in ("amount_paise", "txn_date", "txn_id", "channel"):
                if label[k] is not None:
                    reasons.append(f"{k} must be null for category {cat}")
                    label[k] = None
                    stats["fix_txn_null"] += 1

        pr = derive_priority(cat, label["amount_paise"], sig["legal"])
        if label["priority"] != pr:
            if not any("prio" in x for x in reasons):
                reasons.append(f"priority {label['priority']} -> {pr} (category={cat}, amount={label['amount_paise']}, "
                               f"legal_threat={sig['legal']})")
                stats["fix_priority"] += 1
            label["priority"] = pr

        nh = derive_needs_human(pr, sig["repeat"])
        if label["needs_human"] != nh:
            reasons.append(f"needs_human {label['needs_human']} -> {nh} (priority={pr}, repeat_contact={sig['repeat']})")
            label["needs_human"] = nh
            stats["fix_needs_human"] += 1

        # ---- 6. anything else the checker disagrees with is reported, not auto-fixed -----------
        ref = reference_label(cat, sig)
        for k in ("txn_date", "channel", "language", "txn_id", "amount_paise"):
            if label[k] != ref[k]:
                stats[f"unresolved_{k}"] += 1
                print(f"UNRESOLVED {rid} {k}: label={label[k]!r} checker={ref[k]!r}", file=sys.stderr)

        label = {k: label[k] for k in KEYS}
        if reasons:
            log.append((rid, "fix", "; ".join(reasons)))
            stats["rows_fixed"] += 1
        msgs = [dict(m) for m in r["messages"]]
        msgs[2]["content"] = json.dumps(label, ensure_ascii=False)
        clean.append({"id": rid, "messages": msgs})

    with open("cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action", "reason"])
        w.writerows(log)
    with open("data/train_clean.jsonl", "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["rows_in"] = len(train)
    stats["rows_out"] = len(clean)
    stats["rows_dropped"] = sum(1 for x in log if x[1] == "drop")
    json.dump(dict(sorted(stats.items())), open("audit_summary.json", "w"), indent=1)
    for k, v in sorted(stats.items()):
        print(f"{k:24} {v}")


if __name__ == "__main__":
    main()
