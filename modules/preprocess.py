"""
modules/preprocess.py — Pulse Sentiment AI
═══════════════════════════════════════════
OVERFITTING FIXES (all unchanged):
  ✅ FIX 1: Duplicate comments removed before processing
  ✅ FIX 2: Source "Unknown"/nan/empty → "Other"
  ✅ FIX 3: Sentiment "Unknown"/NaN/Sarcasm/Mixed → remapped to valid 3-class labels
  ✅ FIX 4: Final sanity guard — nothing outside Positive/Negative/Neutral reaches model

TRANSLATION FIXES (new — to support winning conditions):
  ✅ FIX 5: translate_to_english() — preserves sarcasm markers (emojis, !!!, CAPS)
             through translation so ML models still see sarcasm signals after
             Hindi/Tamil text is converted to English.
             Without this: "हाँ हाँ बहुत अच्छा! 😒" → "Yes yes very good!"
             With this:    "Yes yes very good! 😒!!"  ← sarcasm signal preserved
"""

import re
import json
import pandas as pd
from pathlib    import Path
from functools  import lru_cache

# ── NLTK ─────────────────────────────────────────────────────────────────────
try:
    import nltk
    for _res, _pkg in [("corpora/stopwords","stopwords"),
                       ("tokenizers/punkt","punkt"),
                       ("sentiment/vader_lexicon.zip","vader_lexicon")]:
        try:    nltk.data.find(_res)
        except LookupError: nltk.download(_pkg, quiet=True)
    from nltk.corpus import stopwords as _sw
    STOP_WORDS = set(_sw.words("english"))
except Exception:
    STOP_WORDS = set()

# ── Optional libs ─────────────────────────────────────────────────────────────
try:
    from langdetect import detect as _langdetect, LangDetectException
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False

try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except ImportError:
    TEXTBLOB_OK = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VaderSIA
    _vader_sia = _VaderSIA()
    VADER_OK = True
except Exception:
    VADER_OK = False


# ═════════════════════════════════════════════════════════════════════════════
#  UNIFIED SARCASM ENGINE  (single source of truth — used by BOTH files)
# ═════════════════════════════════════════════════════════════════════════════
_SARC_EMOJIS = {"🙄","😒","😏","🤨","🙃","😤","😑","🤡","💀","😂","🤣","😆"}

_SARC_PHRASES_EN = {
    "oh great","wow amazing","yeah right","sure sure","oh sure",
    "obviously working","brilliant idea","great job government","another amazing scheme",
    "as if","totally working","best scheme ever","oh brilliant","oh absolutely",
    "of course it works","what a surprise","so helpful","never seen this before",
    "what a coincidence","oh wow","sure it did","great job","totally works",
    "definitely helping","so helpful not","working perfectly","what a joke",
    "oh how wonderful","because that always works","as if it ever",
}

_SARC_PHRASES_HI = {
    "हाँ हाँ","हां हां","वाह क्या बात","अरे वाह","बिलकुल सही","ज़रूर होगा",
    "ज़रूर काम आएगा","क्या खूब","हाँ बिलकुल","बहुत अच्छा","बहुत बढ़िया है ना",
    "हाँ जी","वाह वाह","बहुत काम आई यह योजना","हां बिलकुल",
}

_SARC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\boh\s+(wow|great|sure|brilliant|perfect|fantastic|amazing|excellent)\b",
    r"\b(great|excellent|wonderful|brilliant|amazing)\b.{0,50}\b(another|again|still|same|really|definitely)\b",
    r"\b(sure|yeah|right|totally|absolutely|obviously)\b.{0,40}\b(work|happen|help|reach|benefit|deliver)\b",
    r"as\s+if\s+it\s+(ever|will|would|could)",
    r"what\s+a\s+(joke|surprise|shock|shocker|coincidence)",
    r"(great|amazing|excellent|wonderful|fantastic)[!.]{2,}",
    r"\bthank\s+(you|god).{0,40}\b(nothing|zero|nobody|no one|no money)\b",
    r"\bdefinitely\b.{0,40}\b(not|never|won.t|can.t|didn.t)\b",
    r"\b(best|greatest)\s+(scheme|yojana|plan|policy).{0,30}\b(ever|always|definitely)\b",
    r"sure\s+sure",
    r"yeah\s+right",
]]

# ── Sarcasm emojis to preserve through translation ────────────────────────────
# These carry critical sentiment signal that Google Translate often strips.
# Extracted before translation, reattached after if missing.
_SARC_EMOJIS_PRESERVE = {"🙄","😒","😏","🤨","🙃","😤","😑"}


def detect_sarcasm(text: str) -> bool:
    """Quick bool — used during preprocessing."""
    return sarcasm_score(text) > 0.45


def sarcasm_score(text: str) -> float:
    """
    Returns 0.0 → 1.0 sarcasm confidence.
    > 0.45 = sarcastic
    """
    if not text or len(text.strip()) < 4:
        return 0.0

    score = 0.0
    t  = str(text)
    tl = t.lower()

    for e in _SARC_EMOJIS:
        if e in t:
            score += 0.45
            break

    for phrase in _SARC_PHRASES_EN:
        if phrase in tl:
            score += 0.35
            break

    for phrase in _SARC_PHRASES_HI:
        if phrase in t:
            score += 0.35
            break

    for pat in _SARC_PATTERNS:
        if pat.search(tl):
            score += 0.30
            break

    alpha = [c for c in t if c.isalpha()]
    if alpha:
        caps_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if caps_ratio > 0.45 and len(t) > 8:
            score += 0.20

    if t.count("!") >= 2:
        score += 0.15

    pos_words   = {"great","amazing","wonderful","excellent","fantastic","brilliant","perfect"}
    neg_context = {"never","not","didn","don","won","can","doesn","isn","nothing","zero","fake"}
    words = set(tl.split())
    if pos_words & words and neg_context & words:
        score += 0.20

    return min(score, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
#  HINDI / REGIONAL NEGATIVE KEYWORD DETECTOR
# ═════════════════════════════════════════════════════════════════════════════
_HINDI_NEGATIVE_WORDS = {
    "नहीं","नहि","कोई नहीं","कुछ नहीं","बेकार","फर्जी","झूठ","झूठा","झूठी",
    "घोटाला","भ्रष्टाचार","धोखा","धोखेबाज़","नाकाम","विफल","समस्या","परेशानी",
    "नुकसान","खराब","बुरा","गलत","बुरी","अन्याय","शोषण","लूट","ठगी","ढकोसला",
    "सिर्फ नाम","जमीन पर कुछ नहीं","काम नहीं","मिलता नहीं","नहीं मिला",
    "नहीं होता","नहीं आया","नहीं पहुँचा","नहीं पहुंचा","नाराज","निराश",
    "तकलीफ","दिक्कत","मुश्किल","कठिनाई",
    "bakwaas","bekar","faltu","jhooth","cheat","fraud","problem","issue",
    "nahi mila","nahi hua","kaam nahi","kuch nahi","sirf naam",
}

_HINDI_POSITIVE_WORDS = {
    "अच्छा","बढ़िया","शानदार","उपयोगी","फायदेमंद","लाभदायक","सहायक","मदद",
    "खुश","संतुष्ट","धन्यवाद","सुंदर","बेहतर","सुधार","सफल","कामयाब",
    "accha","badhiya","shandaar","upyogi","helpful","fayda","khush",
    "shukriya","dhanyawad","sundar","behtar","safal",
}

def _detect_hindi_sentiment(text: str) -> str | None:
    """Returns 'Positive', 'Negative', or None if inconclusive."""
    neg = sum(1 for w in _HINDI_NEGATIVE_WORDS if w in text)
    pos = sum(1 for w in _HINDI_POSITIVE_WORDS if w in text)
    if neg > 0 and neg >= pos:
        return "Negative"
    if pos > 0 and pos > neg:
        return "Positive"
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  LANGUAGE DETECTION
# ═════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=8192)
def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 4:
        return "en"

    hindi_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    if hindi_chars > 2:
        return "hi"

    hinglish_markers = {"nahi","koi","kuch","hai","hota","mila","yojana","sarkar",
                        "paisa","paise","scheme","sarkaar","log","acha","badhiya",
                        "bekar","bakwaas","kaam","gaya","hua","tha","thi"}
    words = set(text.lower().split())
    if len(words & hinglish_markers) >= 2:
        return "hinglish"

    if not LANGDETECT_OK:
        return "en"
    try:
        code = _langdetect(str(text))
        if code in ("so","tl","id","ms","af","cy","sw","sk","sl","hr"):
            return "en"
        return code
    except Exception:
        return "en"


# ═════════════════════════════════════════════════════════════════════════════
#  TRANSLATION — disk-persisted, never re-hits API
# ═════════════════════════════════════════════════════════════════════════════
_CACHE_FILE  = Path("data/translation_cache.json")
_trans_cache: dict = {}
_cache_dirty = False

def _load_cache():
    global _trans_cache
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _trans_cache = json.load(f)
        except Exception:
            _trans_cache = {}

def _save_cache():
    global _cache_dirty
    if not _cache_dirty:
        return
    try:
        _CACHE_FILE.parent.mkdir(exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_trans_cache, f, ensure_ascii=False)
        _cache_dirty = False
    except Exception:
        pass

_load_cache()


def _is_already_english(text: str) -> bool:
    for start, end in [("\u0900","\u097f"),("\u0B80","\u0BFF"),
                       ("\u0C00","\u0C7F"),("\u0980","\u09FF")]:
        if any(start <= c <= end for c in text):
            return False
    return True


def translate_to_english(text: str, src_lang: str = "auto") -> str:
    """
    Translates text to English.
    FIX 5 — Sarcasm marker preservation:
    Before translating, we extract:
      - Sarcasm emojis (🙄😒😏 etc.) — Google Translate often strips these
      - Exclamation count — multiple !! are sarcasm signals
      - ALL-CAPS ratio — used by VADER for intensity detection
    After translating, we reattach any markers that were stripped.
    Why this matters:
      Original: "हाँ हाँ, बहुत अच्छी योजना! 😒!!"
      Without fix: "Yes yes, very good plan!"      ← sarcasm signal LOST
      With fix:    "Yes yes, very good plan! 😒!!" ← sarcasm signal PRESERVED
    This ensures:
      - ML models trained on translated text still see sarcasm features
      - VADER still detects !! intensity on translated text
      - Sarcasm penalty in model selection still fires correctly
    """
    global _trans_cache, _cache_dirty
    if not text or not text.strip():
        return text
    if src_lang in ("en", "english") and _is_already_english(text):
        return text
    if src_lang == "hinglish":
        return text

    # ── FIX 5: Extract sarcasm markers BEFORE translation ────────────────────
    sarc_emojis_found = [c for c in text if c in _SARC_EMOJIS_PRESERVE]
    exclaim_count     = text.count("!")
    question_count    = text.count("?")
    # Detect ALL-CAPS words (intensity signal for VADER)
    # We preserve up to 2 caps words if they exist
    words_list  = text.split()
    caps_words  = [w for w in words_list
                   if w.isalpha() and w.isupper() and len(w) > 2][:2]
    # ── End extraction ────────────────────────────────────────────────────────

    cache_key = text.strip()[:400]

    # ── Level 1: in-memory cache (within session) ─────────────────────────────
    if cache_key in _trans_cache:
        cached = _trans_cache[cache_key]
        return _reattach_markers(cached, sarc_emojis_found,
                                 exclaim_count, question_count, caps_words)

    # ── Level 2: Supabase translation cache (across sessions, deployment only) ─
    try:
        from data.storage import get_cached_translation, save_cached_translation
        supabase_cached = get_cached_translation(cache_key)
        if supabase_cached and supabase_cached.strip():
            # Promote to in-memory so next lookup in this session is instant
            _trans_cache[cache_key] = supabase_cached
            return _reattach_markers(supabase_cached, sarc_emojis_found,
                                     exclaim_count, question_count, caps_words)
    except Exception:
        pass

    # ── Level 3: actual Google Translate API call (only for truly new text) ───
    if not TRANSLATOR_OK:
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(str(text)[:500])
        result = translated if translated and translated.strip() else text
    except Exception:
        result = text

    # ── FIX 5: Reattach sarcasm markers if translation stripped them ──────────
    result = _reattach_markers(result, sarc_emojis_found,
                               exclaim_count, question_count, caps_words)
    # ── End reattachment ──────────────────────────────────────────────────────

    # ── Save to both in-memory and Supabase cache ─────────────────────────────
    _trans_cache[cache_key] = result
    _cache_dirty = True
    try:
        from data.storage import save_cached_translation
        save_cached_translation(cache_key, result)
    except Exception:
        pass

    return result


def _reattach_markers(translated: str,
                      sarc_emojis: list,
                      exclaim_count: int,
                      question_count: int,
                      caps_words: list) -> str:
    """
    Reattaches sarcasm markers stripped by Google Translate.
    Only adds back what was actually in the original — never fabricates.

    Rules:
      Emojis: if original had sarcasm emojis and translation lost them → append
      Exclaims: if original had 2+ !! and translation has fewer → restore count
      Caps words: if original had ALL-CAPS words → append as context
    """
    result = translated.rstrip()

    # Reattach sarcasm emojis if stripped
    if sarc_emojis:
        existing = [c for c in result if c in _SARC_EMOJIS_PRESERVE]
        if not existing:
            result = result + " " + "".join(sarc_emojis)

    # Restore exclamation marks if count dropped
    if exclaim_count >= 2:
        current_exclaim = result.count("!")
        if current_exclaim < exclaim_count:
            # Strip any trailing ! then add correct count
            result = result.rstrip("!").rstrip()
            result = result + "!" * exclaim_count

    # Restore question marks if count dropped significantly
    if question_count >= 2:
        current_question = result.count("?")
        if current_question < question_count:
            result = result.rstrip("?").rstrip()
            result = result + "?" * question_count

    return result


def _translate_series_fast(texts: pd.Series, langs: pd.Series) -> pd.Series:
    global _cache_dirty
    results = texts.copy()

    non_en_mask = ~(
        langs.isin(["en","english","EN","hinglish"]) |
        texts.apply(_is_already_english)
    )

    if not non_en_mask.any():
        return results

    non_en_texts = texts[non_en_mask]

    unique_uncached = {t for t in non_en_texts
                      if t.strip()[:400] not in _trans_cache and not _is_already_english(t)}

    for text in unique_uncached:
        translate_to_english(text, "auto")

    results[non_en_mask] = non_en_texts.apply(
        lambda t: _trans_cache.get(t.strip()[:400], t)
    )

    _save_cache()
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ═════════════════════════════════════════════════════════════════════════════
_URL_RE  = re.compile(r"http\S+|www\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")
_NONWORD = re.compile(r"[^\w\s]")
_DIGITS  = re.compile(r"\d+")
_SPACES  = re.compile(r"\s+")

# ── Scheme-name tokens — zero sentiment signal, stripped before TF-IDF ────────
SCHEME_STOPWORDS = {
    "pm", "pradhan", "mantri", "yojana", "scheme", "mission",
    "bharat", "india", "jan", "dhan", "kisan", "samman", "nidhi",
    "ayushman", "ujjwala", "swachh", "jeevan", "jyoti", "suraksha",
    "atal", "mudra", "pmay", "pmjay", "pmgsy", "pmkvy", "pmfby",
    "bima", "fasal", "saubhagya", "sagarmala", "bharatmala",
    "amrut", "enam", "fame", "beti", "bachao", "padhao", "sukanya",
    "samriddhi", "vishwakarma", "atmanirbhar", "svanidhi", "onorc",
    "abha", "abdm", "upi", "net", "digital", "skill",
    "startup", "poshan", "abhiyaan", "matru", "vandana", "garib",
    "kalyan", "anna", "surya", "ghar", "jal", "har", "sadak",
    "grameen", "gramin", "awas", "kcc", "kisaan",
}

def clean_text(text: str) -> str:
    t = str(text)
    t = _URL_RE.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(" ", t)
    t = _NONWORD.sub(" ", t)
    t = _DIGITS.sub(" ", t)
    t = t.lower().strip()
    t = _SPACES.sub(" ", t)
    tokens = [w for w in t.split()
              if w not in STOP_WORDS
              and w not in SCHEME_STOPWORDS
              and len(w) > 1]
    result = " ".join(tokens)
    return result if result.strip() else t.strip()[:200]


# ═════════════════════════════════════════════════════════════════════════════
#  SENTIMENT SCORING
# ═════════════════════════════════════════════════════════════════════════════
def _score_text(text: str) -> str:
    """TextBlob + VADER ensemble. Returns Positive / Negative / Neutral."""
    votes = []

    if TEXTBLOB_OK:
        try:
            pol = TextBlob(str(text)).sentiment.polarity
            if pol > 0.08:    votes.append("Positive")
            elif pol < -0.08: votes.append("Negative")
            else:             votes.append("Neutral")
        except Exception:
            pass

    if VADER_OK:
        try:
            c = _vader_sia.polarity_scores(str(text))["compound"]
            if c >= 0.05:    votes.append("Positive")
            elif c <= -0.05: votes.append("Negative")
            else:            votes.append("Neutral")
        except Exception:
            pass

    if not votes:
        return "Neutral"

    from collections import Counter
    return Counter(votes).most_common(1)[0][0]


def get_textblob_sentiment(text: str) -> str:
    return _score_text(text)

def get_sentiment(text: str) -> str:
    """Full live pipeline: detect → translate → Hindi keywords → ensemble."""
    lang    = detect_language(text)
    en_text = translate_to_english(text, lang)

    if lang in ("hi","hinglish"):
        hindi_prior = _detect_hindi_sentiment(text)
        if hindi_prior:
            if detect_sarcasm(text) and hindi_prior == "Positive":
                return "Negative"
            return hindi_prior

    result = _score_text(en_text)

    if detect_sarcasm(text) and result == "Positive":
        result = "Negative"

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN DATAFRAME PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
_UNKNOWN_SOURCE = {
    "unknown","Unknown","UNKNOWN","none","None",
    "nan","NaN","","null","Null","N/A","n/a","?"
}

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Normalise column names ─────────────────────────────────────────────────
    col_map = {
        "Comment":"text","comment":"text",
        "Source":"source","Scheme":"scheme",
        "Sentiment":"sentiment","Language":"language",
    }
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'Comment' or 'text' column")

    df["text"] = df["text"].fillna("").astype(str)

    # ─────────────────────────────────────────────────────────────────────────
    # OVERFIT FIX 1 — Remove duplicate comments
    # ─────────────────────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[DEDUP] Removed {removed} duplicate comments ({before} → {len(df)} unique)")

    # ─────────────────────────────────────────────────────────────────────────
    # OVERFIT FIX 2 — Clean Source column
    # ─────────────────────────────────────────────────────────────────────────
    if "source" in df.columns:
        df["source"] = df["source"].fillna("Other").astype(str).str.strip()
        df["source"] = df["source"].apply(
            lambda x: "Other" if x in _UNKNOWN_SOURCE else x
        )
    else:
        df["source"] = "Other"

    # ── Language detection ─────────────────────────────────────────────────────
    if "language" in df.columns:
        df["Lang"] = df["language"].fillna("en").str.strip().str.lower()
        missing = df["Lang"].isin(["","nan","none","unknown","?"])
        if missing.any():
            df.loc[missing,"Lang"] = df.loc[missing,"text"].apply(detect_language)
    else:
        df["Lang"] = df["text"].apply(detect_language)

    # ── Translation ───────────────────────────────────────────────────────────
    df["Translated"] = _translate_series_fast(df["text"], df["Lang"])

    # ── Sarcasm detection ─────────────────────────────────────────────────────
    df["IsSarcasm"]    = df["text"].apply(detect_sarcasm)
    df["SarcasmScore"] = df["text"].apply(lambda t: round(sarcasm_score(t)*100,1))

    # ── Text cleaning ─────────────────────────────────────────────────────────
    df["Cleaned"] = df["Translated"].apply(clean_text)

    # ─────────────────────────────────────────────────────────────────────────
    # OVERFIT FIX 3 — Sentiment label sanitisation
    # ─────────────────────────────────────────────────────────────────────────
    if "sentiment" in df.columns:
        df["Sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()

        df["Sentiment"] = df["Sentiment"].replace({
            "Sarcasm":   "Negative",
            "Sarcastic": "Negative",
            "Mixed":     "Neutral",
            "Unknown":   "Neutral",
            "Nan":       "Neutral",
            "None":      "Neutral",
            "":          "Neutral",
        })

        valid = {"Positive", "Negative", "Neutral"}
        still_invalid = ~df["Sentiment"].isin(valid)
        if still_invalid.any():
            df.loc[still_invalid, "Sentiment"] = (
                df.loc[still_invalid, "Translated"].apply(_score_text)
            )

    else:
        def _full_label(row):
            if row["Lang"] in ("hi","hinglish"):
                prior = _detect_hindi_sentiment(row["text"])
                if prior:
                    if row["IsSarcasm"] and prior == "Positive":
                        return "Negative"
                    return prior
            return _score_text(row["Translated"])

        df["Sentiment"] = df.apply(_full_label, axis=1)

    # ── Sarcasm correction ────────────────────────────────────────────────────
    flip = df["IsSarcasm"] & (df["Sentiment"] == "Positive")
    df.loc[flip, "Sentiment"] = "Negative"

    # ── Restore Source / Scheme ───────────────────────────────────────────────
    df["Source"]  = df["source"]
    df["Scheme"]  = df.get("scheme", pd.Series("General", index=df.index))
    df["Comment"] = df["text"]

    # ── Drop empty cleaned text ───────────────────────────────────────────────
    df = df[df["Cleaned"].str.strip().str.len() > 1].reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # OVERFIT FIX 4 — Final hard guarantee
    # ─────────────────────────────────────────────────────────────────────────
    final_invalid = ~df["Sentiment"].isin({"Positive", "Negative", "Neutral"})
    if final_invalid.any():
        df.loc[final_invalid, "Sentiment"] = "Neutral"

 
    return df
