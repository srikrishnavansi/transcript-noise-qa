"""Seeded ASR-style noise for call transcripts.

A transcript is a list of turns: {"speaker": "agent" | "customer", "text": str}.
corrupt(turns, wer, seed, key) returns (noisy_turns, ops). The word edits add up to
round(wer * N) under word-level Levenshtein (N = reference words), so the measured
WER of the output matches the requested level. Speaker-label swaps (diarization
errors) are applied on top and do not count towards WER.
"""
import hashlib
import random
import re
import string

STRIP = string.punctuation + "€“”‘’"

# Plausible ASR confusions (sound-alikes and language-model guesses), keyed by normalized word.
CONFUSIONS = {
    "recorded": ["recoded", "reported", "recorder"], "recording": ["according", "recoding"],
    "quality": ["quantity", "equality"], "training": ["trading", "raining"],
    "purposes": ["porpoises", "purpose"], "call": ["car", "cool"], "calling": ["coding", "falling"],
    "refund": ["refined", "we fund"], "refunded": ["refined", "refunding"],
    "replacement": ["placement", "replacements"], "reference": ["preference", "reverence"],
    "working": ["walking", "worrying"], "days": ["daze", "this"], "day": ["they", "date"],
    "sorry": ["so we", "story"], "understand": ["understood", "under stand"],
    "frustrating": ["frustrated", "fascinating"], "apologise": ["apologies", "a policy"],
    "verify": ["very fine", "ferry"], "security": ["secure it", "securely"],
    "postcode": ["post code", "postboard"], "email": ["female", "e-mail"],
    "address": ["a dress", "addresses"], "account": ["a count", "count"],
    "order": ["older", "border"], "number": ["lumber", "member"],
    "delivery": ["the livery", "deliver"], "delivered": ["deliver", "the lever"],
    "dispatched": ["despatch", "dispatch"], "damaged": ["damage", "manage"],
    "faulty": ["fault", "forty"], "warranty": ["warrant", "war and tea"],
    "repair": ["prepare", "re pair"], "return": ["retain", "we turn"], "returns": ["returned", "we turns"],
    "charged": ["changed", "charge"], "twice": ["price", "twist"], "help": ["health", "hell"],
    "anything": ["nothing", "everything"], "else": ["elf", "less"], "thank": ["tank", "thanks"],
    "thanks": ["tanks", "thank"], "can": ["can't", "could"], "can't": ["can", "cant"],
    "won't": ["want", "will"], "will": ["well", "we'll"], "i'll": ["all", "aisle"],
    "i've": ["i", "five"], "not": ["now", "knot"], "no": ["know", "now"], "yes": ["yeah", "jess"],
    "to": ["two", "too"], "for": ["four", "far"], "there": ["their", "they're"],
    "their": ["there", "they're"], "your": ["you're", "you"], "you're": ["your", "you"],
    "we": ["wee", "with"], "that": ["the", "it"], "the": ["a", "this"], "a": ["the", "uh"],
    "and": ["an", "in"], "in": ["and", "on"], "on": ["in", "an"], "of": ["off", "have"],
    "is": ["his", "as"], "it's": ["its", "is"], "have": ["of", "half"], "right": ["write", "white"],
    "know": ["no", "now"], "here": ["hear", "her"], "see": ["sea", "she"], "week": ["weak", "wake"],
    "through": ["threw", "true"], "today": ["to say", "to day"], "morning": ["mourning", "warning"],
    "afternoon": ["after new", "after noon"], "name": ["names", "nay"], "my": ["by", "mine"],
    "me": ["be", "my"], "please": ["police", "pleas"], "sure": ["shore", "show"],
    "really": ["rarely", "rally"], "arrange": ["a range", "arranged"], "parcel": ["partial", "parcels"],
    "label": ["level", "table"], "confirm": ["conform", "con firm"], "fee": ["free", "feet"],
    "minus": ["mine is", "minor"], "duplicate": ["duplicated", "do placate"], "bank": ["back", "thank"],
    "issue": ["tissue", "is you"], "problem": ["probably", "program"], "wait": ["weight", "white"],
    "tracking": ["trucking", "tracing"], "received": ["deceived", "receive"], "broken": ["spoken", "brokers"],
    "cracked": ["tracked", "crack"], "lamp": ["lump", "lamb"], "kettle": ["cattle", "kettles"],
    "processed": ["process", "professed"], "photo": ["photos", "for to"], "photos": ["photo", "for those"],
    "upset": ["up set", "offset"], "annoying": ["enjoying", "annoyed"], "terrible": ["terrific", "horrible"],
    "hello": ["yellow", "hollow"], "hi": ["hey", "high"], "good": ["could", "would"], "great": ["grade", "grey"],
    "free": ["three", "fee"], "full": ["fall", "fool"], "eligible": ["illegible", "legible"],
    "policy": ["policies", "polish"], "only": ["any", "lonely"], "within": ["with in", "with them"],
    "after": ["half to", "laughter"], "before": ["for", "be for"], "detail": ["details", "the tail"],
    "details": ["detail", "the tails"], "check": ["chuck", "cheque"], "pass": ["path", "past"],
    "sent": ["cent", "scent"], "send": ["sand", "spend"], "glad": ["glass", "clad"],
}

CONFUSIONS.update({
    "i": ["a", "eye"], "you": ["your", "ya"], "it": ["at", "a"], "with": ["where", "which"],
    "that's": ["that", "this"], "i'm": ["i", "um"], "so": ["sew", "said"], "at": ["it", "that"],
    "how": ["now", "who"], "you'll": ["you", "yule"], "okay": ["ok", "hokay"], "or": ["of", "are"],
    "get": ["got", "good"], "like": ["light", "liked"], "this": ["these", "the"], "one": ["won", "want"],
    "now": ["no", "know"], "fine": ["find", "five"], "set": ["said", "sit"], "had": ["have", "hat"],
    "two": ["to", "too"], "want": ["won't", "what"], "our": ["are", "hour"], "when": ["went", "then"],
    "all": ["or", "old"], "doesn't": ["does", "dozen"], "just": ["joost", "jest"], "oh": ["old", "so"],
    "take": ["make", "tick"], "be": ["me", "bee"], "has": ["is", "as"], "got": ["god", "get"],
    "back": ["black", "bag"], "out": ["at", "how"], "by": ["buy", "bye"], "could": ["good", "can"],
    "what's": ["what", "once"], "been": ["bin", "being"], "need": ["knead", "needed"], "you'd": ["you", "you would"],
    "same": ["sane", "say"], "are": ["our", "or"], "was": ["is", "what's"], "i'd": ["i", "id"],
    "don't": ["do", "dont"], "we've": ["we", "weave"], "it'll": ["it", "until"], "they'll": ["they", "there"],
})

FILLERS = ["uh", "um", "the", "a", "and", "so", "yeah", "i", "it", "you know", "thank you"]

NAMES = {
    "tidewell": ["tidal", "tide well", "titwell"], "sanne": ["sunny", "sana"], "joost": ["just", "yost"],
    "lotte": ["lot", "lottie"], "bram": ["brown", "brum"], "femke": ["fem key", "femme"],
    "ruud": ["rude", "root"], "siobhan": ["shivon", "show bun"], "niamh": ["neve", "neeve"],
    "priya": ["pria", "prior"], "ahmed": ["ahmad", "a mad"], "mark": ["marc", "mike"],
    "eva": ["ava", "ever"], "tom": ["tim", "tam"], "visser": ["fisher", "visor"],
    "de": ["the", "dee"], "vries": ["freeze", "fries"], "jansen": ["johnson", "jensen"],
    "bakker": ["backer", "baker"], "smit": ["smith", "smitt"], "mulder": ["moulder", "mold"],
    "dekker": ["decker", "deck"], "meijer": ["meyer", "mayor"], "willems": ["williams", "willem"],
    "kowalski": ["kovalski", "kowalsky"], "okafor": ["oak for", "okafur"], "haddad": ["had dad", "hadad"],
    "rossi": ["rosie", "rossy"], "nguyen": ["win", "new yen"], "dubois": ["do boy", "dubwa"],
}

PHONETIC = [("ph", "f"), ("v", "f"), ("w", "v"), ("ij", "ay"), ("oo", "o"), ("uu", "oo"),
            ("j", "y"), ("sch", "sk"), ("ck", "k"), ("ie", "ee"), ("ee", "ea"), ("th", "t"), ("c", "k")]

DIGIT_CONF = {"0": "8", "1": "7", "2": "3", "3": "2", "4": "1", "5": "9", "6": "8", "7": "1", "8": "6", "9": "5"}
TEEN = {str(13 + i): str(30 + 10 * i) for i in range(7)}  # 13<->30 ... 19<->90
TEEN.update({v: k for k, v in TEEN.items()})


def norm(tok):
    return tok.lower().strip(STRIP)


def words(text):
    return [w for w in (norm(t) for t in text.split()) if w]


def wer(ref_text, hyp_text):
    """Word error rate via Levenshtein distance on normalized words."""
    r, h = words(ref_text), words(hyp_text)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / max(len(r), 1)


def _rng(*parts):
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _split(tok):
    """'(Tidewell,' -> ('(', 'Tidewell', ',')"""
    m = re.match(r"^([^\w€]*)(.*?)([^\w]*)$", tok)
    return m.group(1), m.group(2), m.group(3)


def _phonetic(core, rng):
    low = core.lower()
    hits = [(a, b) for a, b in PHONETIC if a in low]
    if hits:
        a, b = rng.choice(hits)
        return low.replace(a, b, 1)
    return low + "s"


def _mangle_number(core, rng):
    if core in TEEN:
        return TEEN[core]
    idx = [i for i, ch in enumerate(core) if ch.isdigit()]
    i = rng.choice(idx)
    if len(idx) >= 4 and rng.random() < 0.3:
        return core[:i] + core[i + 1:]  # dropped digit
    return core[:i] + DIGIT_CONF[core[i]] + core[i + 1:]


def _substitute(tok, entity, rng):
    pre, core, post = _split(tok)
    low = core.lower()
    if entity and any(ch.isdigit() for ch in core):
        new = _mangle_number(core, rng)
    elif entity and "@" in core:
        local, _, domain = core.partition("@")
        new = _phonetic(local, rng) + "@" + domain
    elif entity or low in NAMES:
        new = rng.choice(NAMES[low]) if low in NAMES else _phonetic(core, rng)
    elif low in CONFUSIONS:
        new = rng.choice(CONFUSIONS[low])
    elif "'" in low:
        new = low.split("'")[0]  # dropped contraction: "you've" -> "you"
    elif low.endswith("ing"):
        new = low[:-1]
    elif low.endswith("ed"):
        new = low[:-1] if low.endswith("ded") else low[:-2]
    elif low.endswith("s") and len(low) > 3:
        new = low[:-1]
    else:
        new = low + "s"
    if norm(new) == norm(tok):
        new = low + "s"
    return pre + new + post


def _is_entity(tokens, i):
    tok = tokens[i][1]
    _, core, _ = _split(tok)
    if any(ch.isdigit() for ch in core) or "@" in core or core.lower() in NAMES:
        return True
    starts_sentence = i == 0 or tokens[i - 1][0] != tokens[i][0] or tokens[i - 1][1][-1:] in ".?!"
    return core[:1].isupper() and not starts_sentence and core not in ("I", "I'm", "I'll", "I've", "I'd")


def corrupt(turns, wer_level, seed, key=""):
    """Return (noisy_turns, ops). ops: list of {"turn", "op", "from", "to", "cost"}."""
    rng = _rng(key, wer_level, seed)
    tokens = [(ti, tok) for ti, t in enumerate(turns) for tok in t["text"].split()]
    n_ref = sum(len(words(t["text"])) for t in turns)
    budget = round(wer_level * n_ref)
    entity = [_is_entity(tokens, i) for i in range(len(tokens))]
    # ponytail: names and numbers are 3x more likely to be hit, a fixed prior not fitted to a real ASR system.
    order = sorted(range(len(tokens)), key=lambda i: rng.random() ** (1 / (3 if entity[i] else 1)), reverse=True)

    edits, taken, spent = {}, set(), 0
    for i in order:
        if spent >= budget:
            break
        if not norm(tokens[i][1]) or {i - 1, i, i + 1} & taken:
            continue  # spacing keeps the Levenshtein cost equal to the planned cost
        r = rng.random()
        if entity[i] and r < 0.8:
            op = "entity"
        else:
            op = "sub" if r < 0.6 else "del" if r < 0.85 else "ins"
        if op in ("sub", "entity"):
            new = _substitute(tokens[i][1], entity[i], rng)
            new_words = new.split()
            cost = 1 + max(len([w for w in new_words if norm(w)]) - 1, 0)
        elif op == "del":
            new, cost = "", 1
        else:
            filler = rng.choice(FILLERS)
            new, cost = filler + " " + tokens[i][1], len(filler.split())
        if spent + cost > budget:
            continue
        edits[i] = (op, new, cost)
        taken.add(i)
        spent += cost

    out = [[] for _ in turns]
    ops = []
    for i, (ti, tok) in enumerate(tokens):
        if i in edits:
            op, new, cost = edits[i]
            ops.append({"turn": ti, "op": op, "from": tok, "to": new, "cost": cost})
            if new:
                out[ti].append(new)
        else:
            out[ti].append(tok)

    noisy = [{"speaker": t["speaker"], "text": " ".join(out[ti])} for ti, t in enumerate(turns)]
    # Diarization errors: about half the WER rate, as a share of turns.
    for ti in rng.sample(range(len(turns)), round(0.5 * wer_level * len(turns))):
        noisy[ti]["speaker"] = "customer" if noisy[ti]["speaker"] == "agent" else "agent"
        ops.append({"turn": ti, "op": "speaker", "from": turns[ti]["speaker"], "to": noisy[ti]["speaker"], "cost": 0})
    return noisy, ops


def render(turns):
    return "\n".join(f"{t['speaker'].capitalize()}: {t['text']}" for t in turns)
