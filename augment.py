"""Rule-grounded augmentation of the cleaned training set.

    python augment.py [--n 700] [--seed 0]     # writes data/train_aug.jsonl (augmented rows only)

Source data: data/train_clean.jsonl only. Each augmented row is a copy of a cleaned train ticket with
1-3 surface edits whose effect on the label is fully determined by SCHEMA.md:

  ids      re-format the NimbusPay id with spaces / hyphens / symbols   (txn_id unchanged, normalised)
  date     add or replace the date with a relative / year-less phrase    (txn_date recomputed)
  amount   re-phrase the claim amount (Rs.X/-, X k, X lakh, "sum involved") (amount unchanged)
  thresh   move the amount just below / above Rs 5,000 or Rs 50,000      (amount, priority, needs_human)
  distract add a fee / balance / limit / order no. / bank ref sentence   (label unchanged)
  quote    add quoted earlier support e-mails with dates (email only)    (label unchanged)
  channel  state the payment mode in a new phrasing                       (channel set)
  legal    add a legal threat                                             (priority P1, needs_human)
  repeat   add a "contacted before" statement                             (needs_human)

Every generated row is re-checked with the independent reference checker in triage_rules.py and
discarded if the two disagree.
"""
import argparse
import calendar
import json
import random
import re
from datetime import timedelta

from audit import reference_label
from triage_rules import KEYS, TXN_CATS, TXN_RX, derive_needs_human, derive_priority, load, parse_ticket, signals

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"]


def ordinal(d):
    return f"{d}{'th' if 11 <= d % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


# ------------------------------------------------------------------ text helpers
def split_ticket(user):
    head, body = user.split("\n\n", 1)
    return head, body


def main_line_index(lines):
    """Index of the line holding the customer's message (longest non-quoted, non-header line)."""
    best, idx = -1, None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith(">") or s.startswith("Subject:") or s.startswith("Regards"):
            continue
        if len(s) > best:
            best, idx = len(s), i
    return idx


def insert_sentence(user, sentence, rng):
    head, body = split_ticket(user)
    lines = body.split("\n")
    i = main_line_index(lines)
    parts = re.split(r"(?<=[.!?])\s+", lines[i].strip())
    pos = rng.randint(1, max(1, len(parts) - 1)) if len(parts) > 1 else len(parts)
    parts.insert(pos, sentence)
    lines[i] = " ".join(parts)
    return head + "\n\n" + "\n".join(lines)


def replace_sentence(user, pred, new, rng):
    """Replace the first sentence of the main text for which pred(sentence) is true."""
    head, body = split_ticket(user)
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip().startswith(">"):
            continue
        parts = re.split(r"(?<=[.!?])\s+", ln)
        for j, p in enumerate(parts):
            if pred(p):
                parts[j] = new
                lines[i] = " ".join(parts)
                return head + "\n\n" + "\n".join(lines), True
    return user, False


def fmt_inr(n):
    """Indian digit grouping: 150000 -> 1,50,000"""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return head + "," + tail


def amount_text(paise, rng, hinglish):
    rupees = paise // 100
    whole = paise % 100 == 0
    opts = [f"Rs.{rupees}/-", f"₹ {fmt_inr(rupees)}", f"INR {fmt_inr(rupees)}", f"Rs {fmt_inr(rupees)}"]
    if not whole:
        opts = [f"Rs {fmt_inr(rupees)}.{paise % 100:02d}", f"₹{fmt_inr(rupees)}.{paise % 100:02d}"]
    if whole and rupees >= 1000 and rupees % 100 == 0:
        opts.append(f"{rupees / 1000:g}k")
    if whole and rupees >= 100000 and rupees % 10000 == 0:
        opts.append(f"{rupees / 100000:g} lakh")
    a = rng.choice(opts)
    if hinglish:
        return rng.choice([f"Poore {a} ka mamla hai.", f"{a} ka nuksaan hua hai.", f"Kul {a} kate hain."])
    return rng.choice([f"The sum involved is {a}.", f"I am out {a} because of this.",
                       f"{a} has been debited.", f"We are talking about {a} here."])


# ------------------------------------------------------------------ operators
def op_ids(user, label, rng):
    if not label["txn_id"]:
        return None
    d = label["txn_id"][2:]
    variant = rng.choice([
        f"NP {d[:4]} {d[4:7]} {d[7:]}", f"N P {d[:4]} {d[4:7]} {d[7:]}", f"Np-{d[:5]}-{d[5:]}",
        f"np.{d}", f"NP#{d}", f"NP/{d[:5]}/{d[5:]}", f"N.P. {d}", f"np-{d[:3]}-{d[3:6]}-{d[6:]}",
    ])
    m = TXN_RX.search(user.split("\n\n", 1)[1])
    if not m:
        return None
    hi = label["language"] == "hi-en"
    phrase = (f"Receipt pe reference {variant} likha hai." if hi else
              rng.choice([f"Reference number on the receipt reads {variant}.", f"The reference shows {variant}.",
                          f"Payment ref: {variant}."]))
    new, ok = replace_sentence(user, lambda s: TXN_RX.search(s) is not None, phrase, rng)
    return (new, label) if ok else None


def date_phrase(recv, rng, hinglish):
    kind = rng.choice(["last_month", "this_month", "ago", "named", "named", "rel"])
    if kind == "this_month" and recv.day > 1:
        d = rng.randint(1, recv.day)
        t = recv.replace(day=d)
        s = f"Yeh is mahine ki {d} tareekh ko hua." if hinglish else f"This happened on the {ordinal(d)} of this month."
    elif kind == "last_month":
        last = recv.replace(day=1) - timedelta(days=1)
        d = rng.randint(1, last.day)
        t = last.replace(day=d)
        s = f"Pichle mahine ki {d} tareekh ko hua." if hinglish else f"It was last month on the {ordinal(d)}."
    elif kind == "ago":
        n = rng.randint(2, 12)
        t = recv - timedelta(days=n)
        s = f"{n} din pehle ki baat hai." if hinglish else rng.choice([f"It happened {n} days ago.", f"That was {n} days back."])
    elif kind == "rel":
        n, s = rng.choice([(0, "aaj hua." if hinglish else "It happened earlier today."),
                           (1, "kal hua yeh." if hinglish else "It happened yesterday."),
                           (2, "parso hua yeh." if hinglish else "It was day before yesterday.")])
        t = recv - timedelta(days=n)
    else:   # named date without year: most recent such date on or before Received
        back = rng.randint(0, 120)
        t = recv - timedelta(days=back)
        nm = rng.choice([f"{t.day} {MON[t.month - 1]}", f"{MON[t.month - 1]} {t.day}", f"{ordinal(t.day)} {MONTH[t.month - 1]}"])
        s = f"{nm} ko hua." if hinglish else rng.choice([f"It took place on {nm}.", f"Date: {nm}."])
    return s, t.isoformat()


def op_date(user, label, rng):
    if label["category"] not in TXN_CATS:
        return None
    recv = parse_ticket(user)[0]
    s, iso = date_phrase(recv, rng, label["language"] == "hi-en")
    if label["txn_date"] is None:
        new = insert_sentence(user, s, rng)
    else:
        sig_dates = signals(user)["dates"]
        if len(sig_dates) != 1:
            return None
        from triage_rules import ref_date
        new, ok = replace_sentence(user, lambda x: ref_date(x, recv)[0] is not None, s, rng)
        if not ok:
            return None
    lab = dict(label, txn_date=iso)
    return new, lab


def op_amount(user, label, rng, new_amount=None):
    if label["category"] not in TXN_CATS or label["amount_paise"] is None:
        return None
    from triage_rules import DISTRACTOR_RX, amounts_in
    amt = label["amount_paise"] if new_amount is None else new_amount
    phrase = amount_text(amt, rng, label["language"] == "hi-en")
    new, ok = replace_sentence(user, lambda s: not DISTRACTOR_RX.search(s) and label["amount_paise"] in amounts_in(s),
                               phrase, rng)
    if not ok:
        return None
    lab = dict(label, amount_paise=amt)
    return new, lab


def op_thresh(user, label, rng):
    if label["category"] not in ("payment_failed", "refund", "fraud", "offers") or label["amount_paise"] is None:
        return None
    rupees = rng.choice([4999, 5000, 5001, 4950, 5250, 49999, 50000, 50001, 49500, 52000, 100000, 150000, 4999.5])
    return op_amount(user, label, rng, new_amount=int(round(rupees * 100)))


def op_distract(user, label, rng):
    hi = label["language"] == "hi-en"
    x = rng.choice([75, 120, 2033.25, 15, 49.5, 300])
    big = rng.choice([1200, 25000, 60000, 150000, 4999])
    s = rng.choice([
        (f"Iske upar Rs {x} convenience fee bhi kata." if hi else f"A convenience fee of Rs {x} was also shown."),
        (f"Order number {rng.randint(10000, 99999)} hai." if hi else f"Order number {rng.randint(10000, 99999)} if that helps."),
        (f"Account mein abhi Rs {fmt_inr(big)} pade hain." if hi else f"My account balance is Rs {fmt_inr(big)} right now."),
        (f"Meri daily limit {fmt_inr(big)} rupaye hai." if hi else f"FYI my daily limit is Rs {fmt_inr(big)}."),
        (f"Bank ref {rng.randint(10**11, 10**12 - 1)} hai." if hi else f"Bank ref no {rng.randint(10**11, 10**12 - 1)}."),
        (f"Ticket number #{rng.randint(10000, 99999)} tha." if hi else f"Ticket number #{rng.randint(10000, 99999)}."),
    ])
    return insert_sentence(user, s, rng), label


def op_quote(user, label, rng):
    if "Source: email" not in user:
        return None
    recv = parse_ticket(user)[0]
    blocks = []
    for _ in range(rng.randint(1, 4)):
        d = recv - timedelta(days=rng.randint(1, 60))
        blocks.append(f"> On {d.day:02d} {MON[d.month - 1]} {d.year}, NimbusPay Support wrote:\n"
                      f"> Thank you for contacting NimbusPay. Your request has been registered with ticket "
                      f"#{rng.randint(10000, 99999)}. Our team will get back to you within 48 working hours.")
    return user.rstrip() + "\n\n" + "\n\n".join(blocks), label


def op_channel(user, label, rng):
    if label["category"] not in TXN_CATS or label["channel"] is not None:
        return None
    ch = rng.choice(["upi", "card", "netbanking", "wallet"])
    hi = label["language"] == "hi-en"
    s = {
        "upi": ["UPI se payment kiya tha." if hi else "The UPI app was used for it.", "Paid through UPI."],
        "card": ["Credit card wala transaction tha." if hi else "I paid with my debit card.", "Paid using my Visa card."],
        "netbanking": ["Net banking se kiya tha." if hi else "It went through internet banking.", "Paid via netbanking."],
        "wallet": ["NimbusPay wallet se gaya tha." if hi else "The money went from my NimbusPay wallet.", "Paid from the wallet."],
    }[ch]
    sent = s[0] if hi else rng.choice(s)
    return insert_sentence(user, sent, rng), dict(label, channel=ch)


def op_legal(user, label, rng):
    hi = label["language"] == "hi-en"
    s = rng.choice(["RBI mein complaint karunga.", "Consumer court jaunga.", "Lawyer se legal notice bhejunga."] if hi else
                   ["I will file a complaint with the banking ombudsman.", "I am going to take legal action.",
                    "Expect a legal notice from my lawyer.", "I will report this to the RBI.",
                    "I will take this to consumer court."])
    lab = dict(label, priority="P1", needs_human=True)
    return insert_sentence(user, s, rng), lab


def op_repeat(user, label, rng):
    hi = label["language"] == "hi-en"
    s = rng.choice(["Baar baar complaint kar raha hoon.", "Pehle bhi complaint kar chuka hoon.", "Teesri baar likh raha hoon."] if hi else
                   ["This is the third time I am contacting you.", "I have already contacted support about this.",
                    "This is my second complaint on the same issue.", "I keep raising this again and again."])
    return insert_sentence(user, s, rng), dict(label, needs_human=True)


OPS = {"ids": op_ids, "date": op_date, "amount": op_amount, "thresh": op_thresh, "distract": op_distract,
       "quote": op_quote, "channel": op_channel, "legal": op_legal, "repeat": op_repeat}
WEIGHTS = {"ids": 3, "date": 3, "amount": 2, "thresh": 2, "distract": 3, "quote": 1, "channel": 1, "legal": 1, "repeat": 1}


def finalize(label):
    lab = dict(label)
    if lab["priority"] != "P1" or lab["category"] != "fraud":
        pass
    return {k: lab[k] for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default="data/train_clean.jsonl")
    ap.add_argument("--out", default="data/train_aug.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    src = load(a.src)
    out, tries, rejected, used = [], 0, 0, {}
    names, w = list(WEIGHTS), [WEIGHTS[k] for k in WEIGHTS]
    while len(out) < a.n and tries < a.n * 20:
        tries += 1
        r = rng.choice(src)
        user, label = r["messages"][1]["content"], json.loads(r["messages"][2]["content"])
        applied = []
        for op in rng.sample(names, k=3, counts=w)[: rng.randint(1, 3)]:
            if op in applied:
                continue
            res = OPS[op](user, label, rng)
            if res is None:
                continue
            user, label = res
            applied.append(op)
        if not applied:
            continue
        # recompute the derived fields from the (possibly edited) ticket, then cross-check everything
        sig = signals(user)
        label["priority"] = derive_priority(label["category"], label["amount_paise"], sig["legal"])
        label["needs_human"] = derive_needs_human(label["priority"], sig["repeat"])
        ref = reference_label(label["category"], sig)
        if any(ref[k] != label[k] for k in KEYS):
            rejected += 1
            continue
        for op in applied:
            used[op] = used.get(op, 0) + 1
        msgs = [dict(m) for m in r["messages"]]
        msgs[1]["content"] = user
        msgs[2]["content"] = json.dumps({k: label[k] for k in KEYS}, ensure_ascii=False)
        out.append({"id": f"{r['id']}-aug{len(out):04d}", "source_id": r["id"], "ops": applied, "messages": msgs})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows to {a.out}; rejected by cross-check: {rejected}; op counts: {used}")


if __name__ == "__main__":
    main()
