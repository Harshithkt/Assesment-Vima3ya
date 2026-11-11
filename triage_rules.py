"""Rule-based reference labeller built ONLY from SCHEMA.md (+ phrasings seen in train.jsonl).

Used for two things, never for producing predictions:
  1. Auditing train.jsonl: every field where the stored label disagrees with what the
     spec implies gets flagged, then inspected by hand.
  2. Repairing labels that are *derived* by the spec (priority, needs_human, nulling of
     transaction fields, txn_id normalisation) and generating augmented training rows.

The model at inference time never sees any of this code.
"""
import json
import re
from datetime import date, timedelta

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
CATEGORIES = ["payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"]
TXN_CATS = {"payment_failed", "refund", "fraud", "offers"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MON_RX = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"

LEGAL_RX = re.compile(r"\bRBI\b|ombudsman|consumer (court|forum)|lawyer|legal (notice|action)|\bcourt\b|\bvakil\b", re.I)
REPEAT_RX = re.compile(
    r"third time|again and again|contacted (support|you)|contacting you|baar baar|already (written|contacted|complained|raised)|"
    r"\d+ times about|pehle bhi|kitni baar|teesri baar|second (complaint|time)|"
    r"keep raising|still no (reply|response)|no one (has )?(replied|responded)", re.I)
DISTRACTOR_RX = re.compile(r"balance|pade hain|limit|\bfees?\b|charge|convenience", re.I)


# --------------------------------------------------------------------------- parsing
def parse_ticket(user_text):
    head, _, body = user_text.partition("\n\n")
    m = re.search(r"Received:\s*(\d{4})-(\d{2})-(\d{2})", head)
    recv = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    src = re.search(r"Source:\s*(\S+)", head).group(1)
    lines = []
    for ln in body.split("\n"):
        s = ln.strip()
        if s.startswith(">"):          # quoted earlier emails: ignored by spec
            continue
        if s.startswith("Subject:"):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    # drop email signature ("Regards,\nName")
    text = re.sub(r"\n(Regards|Thanks|Thank you|Best),?\s*\n[^\n]*\s*$", "", text.rstrip(), flags=re.I)
    return recv, src, text


def sentences(text):
    text = re.sub(r"\b(Rs|No|no)\.", r"\1 ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+|(?<=[\dk])\s+(?=(?!(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[A-Z][a-z]|FYI)", text) if s.strip()]


# --------------------------------------------------------------------------- txn_id
TXN_RX = re.compile(r"(?<![A-Za-z])N\s*P[\s\-:#.]*((?:\d[\s\-]?){9}\d)(?![\d])", re.I)


def find_txn_ids(text):
    return [re.sub(r"\D", "", m.group(1)) for m in TXN_RX.finditer(text)]


def ref_txn_id(text):
    ids = find_txn_ids(text)
    return ("NP" + ids[0]) if ids else None


# --------------------------------------------------------------------------- amount
def _num(s):
    return float(s.replace(",", ""))


def _strip_non_amounts(s):
    s = TXN_RX.sub(" ", s)
    s = re.sub(r"#\s*\d+", " ", s)                                     # ticket numbers
    s = re.sub(r"(bank )?ref(erence)?( no| number)?:?\s*\d{6,}", " ", s, flags=re.I)
    s = re.sub(r"order (number|no|id)\.?:?\s*\d+", " ", s, flags=re.I)
    s = re.sub(r"\d+\.\d+\.\d+", " ", s)                               # app versions
    s = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", s)            # dates
    s = re.sub(r"\b\d{1,2}(st|nd|rd|th)?\s+(of\s+)?" + MON_RX + r"(\s+\d{4})?", " ", s, flags=re.I)
    s = re.sub(MON_RX + r"\s+\d{1,2}(st|nd|rd|th)?\b(,?\s*\d{4})?", " ", s, flags=re.I)
    s = re.sub(r"\b\d{1,2}(st|nd|rd|th)\b", " ", s)
    s = re.sub(r"\b\d+\s*(days?|din|times|baar|working hours|hours|minutes|ghante)\b", " ", s, flags=re.I)
    s = re.sub(r"(Realme|Redmi Note|Samsung [A-Z]|iPhone|Vivo [A-Z]|Pixel|OnePlus|Oppo [A-Z])\s*\d+\w*", " ", s, flags=re.I)
    return s


AMT_RX = re.compile(
    r"(?:(?:Rs\.?|INR|₹)\s*)?(\d[\d,]*(?:\.\d+)?)\s*(k\b|lakhs?\b|lac\b)?", re.I)


def amounts_in(sentence):
    s = _strip_non_amounts(sentence)
    out = []
    for m in AMT_RX.finditer(s):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        if raw.replace(",", "").replace(".", "") == "":
            continue
        v = _num(raw)
        if unit == "k":
            v *= 1000
        elif unit.startswith("lakh") or unit == "lac":
            v *= 100000
        out.append(round(v * 100))
    return out


def ref_amount(text):
    cands = []
    for s in sentences(text):
        if DISTRACTOR_RX.search(s):
            continue
        cands += amounts_in(s)
    cands = [c for c in cands if c > 0]
    return cands[0] if len(cands) >= 1 else None, cands


# --------------------------------------------------------------------------- date
def _most_recent(day, month, recv, year=None):
    if year:
        return date(year, month, day)
    d = date(recv.year, month, day)
    return d if d <= recv else date(recv.year - 1, month, day)


def ref_date(text, recv):
    t = text
    found = []
    for m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", t):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        found.append(date(y, mo, d))
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MON_RX + r"(?:,?\s+(\d{4}))?", t, re.I):
        found.append(_most_recent(int(m.group(1)), MONTHS[m.group(2).lower()[:3]], recv,
                                  int(m.group(3)) if m.group(3) else None))
    for m in re.finditer(r"\b" + MON_RX + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(\d{4}))?", t, re.I):
        found.append(_most_recent(int(m.group(2)), MONTHS[m.group(1).lower()[:3]], recv,
                                  int(m.group(3)) if m.group(3) else None))
    m = re.search(r"last month on the (\d{1,2})(?:st|nd|rd|th)", t, re.I)
    if m:
        first = recv.replace(day=1) - timedelta(days=1)
        found.append(first.replace(day=int(m.group(1))))
    m = re.search(r"the (\d{1,2})(?:st|nd|rd|th) of this month", t, re.I)
    if m:
        found.append(recv.replace(day=int(m.group(1))))
    m = re.search(r"\b(\d+)\s*(days? ago|din pehle)", t, re.I)
    if m:
        found.append(recv - timedelta(days=int(m.group(1))))
    if re.search(r"day before yesterday|\bparso\b", t, re.I):
        found.append(recv - timedelta(days=2))
    elif re.search(r"\byesterday\b|\bkal\b", t, re.I):
        found.append(recv - timedelta(days=1))
    if re.search(r"\btoday\b|\baaj\b", t, re.I):
        found.append(recv)
    return (found[0].isoformat() if found else None), found


# --------------------------------------------------------------------------- channel
def ref_channel(text):
    t = text.lower()
    hits = []
    if re.search(r"\bupi\b", t):
        hits.append("upi")
    if re.search(r"credit card|debit card|card payment|\bvisa\b|mastercard|\bcard\b", re.sub(r"scratch card", "", t)):
        hits.append("card")
    if re.search(r"net ?banking|internet banking", t):
        hits.append("netbanking")
    if re.search(r"wallet", t):
        hits.append("wallet")
    return hits


# --------------------------------------------------------------------------- language
HI_WORDS = set("""hai hain nahi nahin kar karo kiya kiye ho gaya gaye gayi mera mere meri mujhe paise
paisa kat kaat wapas abhi tak tha thi raha rahi rahe ki ke ko se mein par bhi ka kya kyun kaise
yeh woh hua hui jaldi karo bhool aaya aayi mila mile chahiye diya liye baat pehle din kal parso
aaj baar likha wala lekin phir maine kisi koi rakam poore mamla nikal gaye bata batao sakta sakte
kaise karun kitna ktina lagega atka atki hua kab kyon nahi tak bhai ji""".split())


def ref_language(text):
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in HI_WORDS for w in words)
    return "hi-en" if hits >= 2 else "en", hits


# --------------------------------------------------------------------------- derived fields
def derive_priority(category, amount_paise, legal):
    if category == "fraud" or legal or (amount_paise is not None and amount_paise >= 50_000_00):
        return "P1"
    if category in ("payment_failed", "refund") and amount_paise is not None and amount_paise >= 5_000_00:
        return "P2"
    if category == "account_access":
        return "P2"
    return "P3"


def derive_needs_human(priority, repeat):
    return priority == "P1" or bool(repeat)


def signals(user_text):
    recv, src, text = parse_ticket(user_text)
    amt, amts = ref_amount(text)
    dt, dts = ref_date(text, recv)
    lang, hi_hits = ref_language(text)
    return {
        "recv": recv, "src": src, "text": text,
        "legal": bool(LEGAL_RX.search(text)), "repeat": bool(REPEAT_RX.search(text)),
        "txn_id": ref_txn_id(text), "txn_ids": find_txn_ids(text),
        "amount": amt, "amounts": amts, "date": dt, "dates": dts,
        "channels": ref_channel(text), "language": lang, "hi_hits": hi_hits,
    }


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
