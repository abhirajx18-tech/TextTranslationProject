import re, os, json
import requests
import torch
import gradio as gr
from transformers import MarianMTModel, MarianTokenizer
from wordfreq import zipf_frequency

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGS = {"English": "en", "Hindi": "hi", "French": "fr"}

MODEL_PATHS = {
    ("en", "hi"): "models/en-hi-final",              # your fine-tuned model
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
}

_cache = {}  # models are loaded lazily, then reused

def get_model(src, tgt):
    key = (src, tgt)
    if key not in _cache:
        path = MODEL_PATHS[key]
        tok = MarianTokenizer.from_pretrained(path)
        mdl = MarianMTModel.from_pretrained(path).to(DEVICE).eval()
        _cache[key] = (tok, mdl)
    return _cache[key]

def _translate_lines(lines, src, tgt):
    tok, mdl = get_model(src, tgt)
    batch = tok(lines, return_tensors="pt", padding=True,
                truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**batch, num_beams=4, max_length=256)
    return tok.batch_decode(out, skip_special_tokens=True)

# ================= name protection =================
# Placeholder names the model already translates well, with their Hindi forms
PLACEHOLDERS = [("Rahul", "राहुल"), ("Priya", "प्रिया"), ("Amit", "अमित"),
                ("Neha", "नेहा"), ("Rohan", "रोहन")]

STRONG_INTRO = re.compile(r"\b(?:my name is|name's|named|call me)\b[,:]?\s+([A-Za-z][A-Za-z'\-]*)", re.I)
WEAK_INTRO = re.compile(r"\b(?:i am|i'm|this is|meet)\b[,:]?\s+([A-Za-z][A-Za-z'\-]*)", re.I)
GREET = re.compile(r"\b(?:hi|hello|hey|dear|thanks|thank you)\b[,:]?\s+([A-Za-z][A-Za-z'\-]*)", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
BRACKET = re.compile(r"\[([^\]]+)\]")

NAMES_FILE = "names.json"          # manual corrections (always win)
AUTO_FILE = "auto_names.json"      # cache of automatic results

def zipf(w):
    return zipf_frequency(w.lower(), "en")

def load_names():
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, encoding="utf-8") as f:
            return {k.lower(): v for k, v in json.load(f).items()}
    return {}

def _load_auto():
    if os.path.exists(AUTO_FILE):
        with open(AUTO_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def is_known(w):
    k = w.lower()
    return k in load_names() or k in _load_auto()

def _local_translit(word):
    """IndicXlit helper service (xlit_server.py). Returns None if it isn't running."""
    try:
        r = requests.get("http://127.0.0.1:5005/", params={"w": word}, timeout=2)
        cand = r.json().get("hi", "")
        if re.fullmatch(r"[\u0900-\u097F]+", cand):
            return cand
    except Exception:
        pass
    return None

def _online_translit(word):
    """Latin -> Devanagari via Google Input Tools (unofficial endpoint)."""
    try:
        r = requests.get(
            "https://inputtools.google.com/request",
            params={"text": word, "itc": "hi-t-i0-und", "num": 1,
                    "cp": 0, "cs": 1, "ie": "utf-8", "oe": "utf-8", "app": "demopage"},
            timeout=3)
        data = r.json()
        if data[0] == "SUCCESS":
            cand = data[1][0][1][0]
            if re.fullmatch(r"[\u0900-\u097F]+", cand):
                return cand
    except Exception:
        pass
    return None

def to_devanagari(name):
    key = name.lower()
    manual = load_names()
    if key in manual:                       # 1. manual override
        return manual[key]
    auto = _load_auto()
    if key in auto:                         # 2. cache
        return auto[key]
    result = _local_translit(name) or _online_translit(name)   # 3. IndicXlit, then online
    if result:
        auto[key] = result
        with open(AUTO_FILE, "w", encoding="utf-8") as f:
            json.dump(auto, f, ensure_ascii=False, indent=2)
        return result
    return name                             # 4. safe fallback: keep Latin spelling

def find_names(s):
    spans = []   # (start, end, name_text)
    for m in BRACKET.finditer(s):                       # explicit: [Abhiraj]
        spans.append((m.start(), m.end(), m.group(1)))
    for m in STRONG_INTRO.finditer(s):                  # "my name is X": always a name
        spans.append((m.start(1), m.end(1), m.group(1)))
    for pat in (WEAK_INTRO, GREET):                     # "I am X", "Hi X"
        for m in pat.finditer(s):
            w = m.group(1)
            if is_known(w) or zipf(w) < 3.2:
                spans.append((m.start(1), m.end(1), w))
    for m in WORD.finditer(s):                          # capitalised mid-sentence
        w = m.group(0)
        before = s[:m.start()].rstrip()
        if (m.start() > 0 and w[0].isupper() and w != "I" and "'" not in w
                and before and before[-1] not in ".!?\"“"
                and (is_known(w) or zipf(w) < 3.5)):
            spans.append((m.start(), m.end(), w))
    spans.sort()
    merged = []                                         # drop overlaps
    for sp in spans:
        if merged and sp[0] < merged[-1][1]:
            continue
        merged.append(sp)
    return merged

def mask(sentence, tgt):
    spans = find_names(sentence)[:len(PLACEHOLDERS)]
    if not spans:
        return sentence, []
    out, last, mapping = [], 0, []
    for i, (a, b, txt) in enumerate(spans):
        ph_en, ph_hi = PLACEHOLDERS[i]
        out.append(sentence[last:a]); out.append(ph_en); last = b
        if tgt == "hi":
            mapping.append((ph_hi, to_devanagari(txt)))
        else:
            mapping.append((ph_en, txt))
    out.append(sentence[last:])
    return "".join(out), mapping

def unmask(text, mapping):
    for expected, replacement in mapping:
        if expected not in text:
            return None          # placeholder was lost: caller falls back
        text = text.replace(expected, replacement)
    return text

def clean_english(s):
    """Fix common typing habits before translation."""
    s = s.strip()
    if not s:
        return s
    s = re.sub(r"\bi\b", "I", s)                                     # "i" -> "I"
    s = re.sub(r"\bi'm\b", "I'm", s, flags=re.I)
    s = re.sub(r"^(hi|hello|hey)\s+(?!,)", lambda m: m.group(1) + ", ", s, flags=re.I)
    s = s[0].upper() + s[1:]                                         # capital first letter
    if s[-1] not in ".!?":                                           # end punctuation
        s += "."
    return s

def translate_en_out(lines, tgt):
    """English -> hi/fr with input cleanup and name protection."""
    lines = [clean_english(l) for l in lines]
    masked = [mask(l, tgt) for l in lines]
    outs = _translate_lines([m[0] for m in masked], "en", tgt)
    final = []
    for line, o, (_, mp) in zip(lines, outs, masked):
        if not mp:
            final.append(o); continue
        r = unmask(o, mp)
        if r is None:
            plain = BRACKET.sub(r"\1", line)
            r = _translate_lines([plain], "en", tgt)[0]
        final.append(r)
    return final
# ===================================================

def translate_lines(lines, src, tgt):
    if src == "en" and tgt in ("hi", "fr"):
        return translate_en_out(lines, tgt)
    if (src, tgt) in MODEL_PATHS:
        return _translate_lines(lines, src, tgt)
    mid = _translate_lines(lines, src, "en")            # hi<->fr via English
    if tgt in ("hi", "fr"):
        return translate_en_out(mid, tgt)
    return _translate_lines(mid, "en", tgt)

def translate(text, src_name, tgt_name):
    if not text or not text.strip():
        return "Please enter some text."
    src, tgt = LANGS[src_name], LANGS[tgt_name]
    if src == tgt:
        return "Source and target languages are the same. Please choose different languages."
    try:
        lines = text.split("\n")
        idx = [i for i, l in enumerate(lines) if l.strip()]
        outs = translate_lines([lines[i].strip() for i in idx], src, tgt)
        result = list(lines)
        for i, o in zip(idx, outs):
            result[i] = o
        note = "  (via English)" if (src, tgt) not in MODEL_PATHS else ""
        return "\n".join(result) + (f"\n\n[{src_name} → {tgt_name}{note}]" if note else "")
    except Exception as e:
        return f"Translation failed: {e}"

with gr.Blocks(title="Text Translator") as demo:
    gr.Markdown("# Text Translator\nEnglish, Hindi and French. Pick the languages, type your text, and press Translate.\n\n"
                "Tip: put a name in [square brackets] to make sure it is kept as a name.")
    with gr.Row():
        src = gr.Dropdown(list(LANGS), value="English", label="Source language")
        swap = gr.Button("⇄ Swap", scale=0)
        tgt = gr.Dropdown(list(LANGS), value="Hindi", label="Target language")
    inp = gr.Textbox(lines=6, label="Text to translate", placeholder="Type or paste text here...")
    btn = gr.Button("Translate", variant="primary")
    out = gr.Textbox(lines=6, label="Translation", interactive=False)

    btn.click(translate, [inp, src, tgt], out)
    inp.submit(translate, [inp, src, tgt], out)
    swap.click(lambda s, t: (t, s), [src, tgt], [src, tgt])

if __name__ == "__main__":
    demo.launch()
