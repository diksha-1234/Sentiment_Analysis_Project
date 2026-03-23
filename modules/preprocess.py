"""
modules/preprocess.py — Pulse Sentiment AI · Optimised for 14,000+ rows
════════════════════════════════════════════════════════════════════════
PERFORMANCE FIXES FOR LARGE DATASETS:
  ✅ FIX A: Uses pre-saved "translated" column from Supabase — skips API calls
             for rows that were already translated at fetch time. 14k rows that
             are already translated cost 0 API calls.
  ✅ FIX B: Language detection uses lru_cache(maxsize=16384) — never re-detects
             the same text twice within a session.
  ✅ FIX C: Sarcasm detection vectorised using pandas .apply() with early exit
             so short texts (< 4 chars) skip all regex immediately.
  ✅ FIX D: Translation batches only the truly un-translated rows, not the whole
             DataFrame. Skips hinglish entirely (untranslatable anyway).
  ✅ FIX E: preprocess_dataframe() prints progress so you can see it working
             in Streamlit Cloud logs.

ORIGINAL OVERFITTING FIXES (all unchanged):
  ✅ FIX 1: Duplicate comments removed before processing
  ✅ FIX 2: Source "Unknown"/nan/empty → "Other"
  ✅ FIX 3: Sentiment "Unknown"/NaN/Sarcasm/Mixed → remapped to valid 3-class labels
  ✅ FIX 4: Final sanity guard — nothing outside Positive/Negative/Neutral reaches model

TRANSLATION FIXES (unchanged):
  ✅ FIX 5: translate_to_english() — preserves sarcasm markers through translation
  ✅ FIX 6: preprocess_dataframe() — stores df["Original"] BEFORE translation
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
#  SARCASM ENGINE
# ═════════════════════════════════════════════════════════════════════════════
_SARC_EMOJIS = {"🙄","😒","😏","🤨","🙃","😤","😑","🤡","💀","😂","🤣","😆"}
_SARC_EMOJIS_PRESERVE = {"🙄","😒","😏","🤨","🙃","😤","😑"}

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


def detect_sarcasm(text: str) -> bool:
    return sarcasm_score(text) > 0.45


def sarcasm_score(text: str) -> float:
    """Returns 0.0 → 1.0 sarcasm confidence. Optimised with early exits."""
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
#  HINDI / REGIONAL KEYWORD DETECTOR
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


def _detect_hindi_sentiment(text: str):
    neg = sum(1 for w in _HINDI_NEGATIVE_WORDS if w in text)
    pos = sum(1 for w in _HINDI_POSITIVE_WORDS if w in text)
    if neg > 0 and neg >= pos: return "Negative"
    if pos > 0 and pos > neg:  return "Positive"
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  LANGUAGE DETECTION
#  FIX B — lru_cache raised to 16384 to cover 14k unique texts in one session
# ═════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=16384)
def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 4:
        return "en"

    hindi_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    if hindi_chars > 2:
        return "hi"

    hinglish_markers = {
        "nahi","koi","kuch","hai","hota","mila","yojana","sarkar",
        "paisa","paise","scheme","sarkaar","log","acha","badhiya",
        "bekar","bakwaas","kaam","gaya","hua","tha","thi",
    }
    words = set(text.lower().split())
    if len(words & hinglish_markers) >= 2:
        return "hinglish"

    if not LANGDETECT_OK:
        return "en"
    try:
        code = _langdetect(str(text))
        # Remap codes that langdetect misidentifies for short Indian text
        if code in ("so","tl","id","ms","af","cy","sw","sk","sl","hr"):
            return "en"
        return code
    except Exception:
        return "en"


# ═════════════════════════════════════════════════════════════════════════════
#  TRANSLATION — disk-persisted + Supabase cache, never re-hits API
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
            print(f"[Preprocess] Loaded {len(_trans_cache)} cached translations")
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
    Translates text to English with sarcasm marker preservation (FIX 5).
    3-level cache: in-memory → disk JSON → Google Translate API.
    """
    global _trans_cache, _cache_dirty
    if not text or not text.strip():
        return text
    if src_lang in ("en","english") and _is_already_english(text):
        return text
    if src_lang == "hinglish":
        return text  # Hinglish is Romanised — leave as-is for TF-IDF

    # Extract sarcasm markers before translation
    sarc_emojis_found = [c for c in text if c in _SARC_EMOJIS_PRESERVE]
    exclaim_count     = text.count("!")
    question_count    = text.count("?")
    words_list        = text.split()
    caps_words        = [w for w in words_list
                         if w.isalpha() and w.isupper() and len(w) > 2][:2]

    cache_key = text.strip()[:400]

    # Level 1: in-memory
    if cache_key in _trans_cache:
        return _reattach_markers(_trans_cache[cache_key], sarc_emojis_found,
                                 exclaim_count, question_count, caps_words)

    # Level 2: Supabase translation cache
    try:
        from data.storage import get_cached_translation, save_cached_translation
        supabase_cached = get_cached_translation(cache_key)
        if supabase_cached and supabase_cached.strip():
            _trans_cache[cache_key] = supabase_cached
            return _reattach_markers(supabase_cached, sarc_emojis_found,
                                     exclaim_count, question_count, caps_words)
    except Exception:
        pass

    # Level 3: Google Translate API (only for truly new text)
    if not TRANSLATOR_OK:
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(str(text)[:500])
        result = translated if translated and translated.strip() else text
    except Exception:
        result = text

    result = _reattach_markers(result, sarc_emojis_found,
                               exclaim_count, question_count, caps_words)

    _trans_cache[cache_key] = result
    _cache_dirty = True

    try:
        from data.storage import save_cached_translation
        save_cached_translation(cache_key, result)
    except Exception:
        pass

    return result


def _reattach_markers(translated: str, sarc_emojis: list,
                      exclaim_count: int, question_count: int,
                      caps_words: list) -> str:
    result = translated.rstrip()
    if sarc_emojis:
        existing = [c for c in result if c in _SARC_EMOJIS_PRESERVE]
        if not existing:
            result = result + " " + "".join(sarc_emojis)
    if exclaim_count >= 2:
        current_exclaim = result.count("!")
        if current_exclaim < exclaim_count:
            result = result.rstrip("!").rstrip() + "!" * exclaim_count
    if question_count >= 2:
        current_question = result.count("?")
        if current_question < question_count:
            result = result.rstrip("?").rstrip() + "?" * question_count
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  FIX A + FIX D — SMART TRANSLATION SERIES
#  Uses pre-saved Supabase "Translated" column first.
#  Only calls API for rows that truly have no translation yet.
# ═════════════════════════════════════════════════════════════════════════════
def _translate_series_fast(texts: pd.Series, langs: pd.Series,
                           existing_translated: pd.Series = None) -> pd.Series:
    """
    FIX A — If the DataFrame already has a "Translated" column from Supabase,
    use those values directly. Only translate rows where that column is empty.

    For 14,000 rows already fetched with translations saved:
      → 0 API calls (all served from existing_translated)

    For new rows without translations:
      → Only those rows hit the API
    """
    global _cache_dirty
    results = texts.copy()

    # FIX A: Use pre-existing translations from Supabase where available
    if existing_translated is not None:
        has_translation = (
            existing_translated.notna() &
            (existing_translated.str.strip() != "") &
            (existing_translated.str.strip() != texts.str.strip())
        )
        if has_translation.any():
            results[has_translation] = existing_translated[has_translation]
            already_count = has_translation.sum()
            print(f"[Preprocess] ✓ {already_count} rows used pre-saved translations "
                  f"(saved {already_count} API calls)")

        # Only translate rows that don't have a pre-saved translation
        needs_translation_mask = ~has_translation
    else:
        needs_translation_mask = pd.Series([True] * len(texts), index=texts.index)

    # Further filter: skip English and Hinglish
    non_en_mask = needs_translation_mask & ~(
        langs.isin(["en","english","EN","hinglish"]) |
        texts.apply(_is_already_english)
    )

    if not non_en_mask.any():
        _save_cache()
        return results

    non_en_texts = texts[non_en_mask]
    print(f"[Preprocess] Translating {len(non_en_texts)} rows that need API translation...")

    # Batch unique texts to minimise API calls
    unique_uncached = {
        t for t in non_en_texts
        if t.strip()[:400] not in _trans_cache and not _is_already_english(t)
    }

    if unique_uncached:
        print(f"[Preprocess] {len(unique_uncached)} unique texts need Google Translate API")
        for i, text in enumerate(unique_uncached):
            translate_to_english(text, "auto")
            # Log progress every 100 texts
            if (i + 1) % 100 == 0:
                print(f"[Preprocess] Translated {i+1}/{len(unique_uncached)} texts...")
    else:
        print("[Preprocess] All translations served from cache — 0 API calls")

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

SCHEME_STOPWORDS = {
    "pm","pradhan","mantri","yojana","scheme","mission",
    "bharat","india","jan","dhan","kisan","samman","nidhi",
    "ayushman","ujjwala","swachh","jeevan","jyoti","suraksha",
    "atal","mudra","pmay","pmjay","pmgsy","pmkvy","pmfby",
    "bima","fasal","saubhagya","sagarmala","bharatmala",
    "amrut","enam","fame","beti","bachao","padhao","sukanya",
    "samriddhi","vishwakarma","atmanirbhar","svanidhi","onorc",
    "abha","abdm","upi","net","digital","skill",
    "startup","poshan","abhiyaan","matru","vandana","garib",
    "kalyan","anna","surya","ghar","jal","har","sadak",
    "grameen","gramin","awas","kcc","kisaan",
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
#  MAIN DATAFRAME PIPELINE — OPTIMISED FOR 14,000 ROWS
# ═════════════════════════════════════════════════════════════════════════════
_UNKNOWN_SOURCE = {
    "unknown","Unknown","UNKNOWN","none","None",
    "nan","NaN","","null","Null","N/A","n/a","?"
}


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline, optimised for 14,000+ rows.

    Key optimisations vs original:
    - FIX A: Reuses pre-saved "Translated" column from Supabase
    - FIX B: lru_cache on detect_language (never re-detects same text)
    - FIX C: Sarcasm vectorised over entire Series in one pass
    - FIX D: Only un-translated rows hit Google Translate API
    - FIX E: Progress logging throughout
    """
    total_start = len(df)
    print(f"[Preprocess] Starting pipeline with {total_start} rows...")

    df = df.copy()

    # ── Normalise column names ─────────────────────────────────────────────────
    # Handle both Supabase lowercase and app Title Case
    col_map = {
        # Supabase lowercase → Title Case
        "comment":    "text",
        "source":     "source",
        "scheme":     "scheme",
        "sentiment":  "sentiment",
        "language":   "language",
        "translated": "translated_existing",
        # Already Title Case
        "Comment":    "text",
        "Source":     "source",
        "Scheme":     "scheme",
        "Sentiment":  "sentiment",
        "Language":   "language",
        "Translated": "translated_existing",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'Comment'/'comment'/'text' column. "
                         f"Got columns: {list(df.columns)}")

    df["text"] = df["text"].fillna("").astype(str)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 1 — Remove duplicate comments
    # ─────────────────────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[Preprocess] Dedup: removed {removed} duplicates ({before} → {len(df)})")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 2 — Clean Source column
    # ─────────────────────────────────────────────────────────────────────────
    if "source" in df.columns:
        df["source"] = df["source"].fillna("Other").astype(str).str.strip()
        df["source"] = df["source"].apply(
            lambda x: "Other" if x in _UNKNOWN_SOURCE else x
        )
    else:
        df["source"] = "Other"

    # ── Language detection ─────────────────────────────────────────────────────
    # FIX B: lru_cache means each unique text is only detected once
    print("[Preprocess] Detecting languages...")
    if "language" in df.columns:
        df["Lang"] = df["language"].fillna("en").str.strip().str.lower()
        missing = df["Lang"].isin(["","nan","none","unknown","?"])
        if missing.any():
            df.loc[missing, "Lang"] = df.loc[missing, "text"].apply(detect_language)
    else:
        df["Lang"] = df["text"].apply(detect_language)

    lang_counts = df["Lang"].value_counts().to_dict()
    print(f"[Preprocess] Language breakdown: {lang_counts}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 6 — Store Original text BEFORE translation
    # ─────────────────────────────────────────────────────────────────────────
    df["Original"] = df["text"].copy()

    # ── Get pre-existing translations (FIX A) ─────────────────────────────────
    existing_translated = None
    if "translated_existing" in df.columns:
        existing_translated = df["translated_existing"].fillna("").astype(str)
        pre_translated_count = (
            existing_translated.str.strip() != ""
        ).sum()
        print(f"[Preprocess] Found {pre_translated_count} pre-saved translations "
              f"in 'translated' column — these skip Google Translate API")

    # ── Translation — FIX A + FIX D ───────────────────────────────────────────
    print("[Preprocess] Running translation step...")
    df["Translated"] = _translate_series_fast(
        df["text"], df["Lang"],
        existing_translated=existing_translated
    )

    # ── FIX C — Sarcasm detection vectorised ──────────────────────────────────
    # Run sarcasm on ORIGINAL text (before translation)
    print("[Preprocess] Running sarcasm detection...")
    df["IsSarcasm"]    = df["text"].apply(detect_sarcasm)
    df["SarcasmScore"] = df["text"].apply(lambda t: round(sarcasm_score(t) * 100, 1))
    sarc_count = df["IsSarcasm"].sum()
    print(f"[Preprocess] Sarcasm detected in {sarc_count} rows "
          f"({round(sarc_count/len(df)*100,1)}%)")

    # ── Text cleaning — on translated text ───────────────────────────────────
    print("[Preprocess] Cleaning text...")
    df["Cleaned"] = df["Translated"].apply(clean_text)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 3 — Sentiment label sanitisation
    # ─────────────────────────────────────────────────────────────────────────
    print("[Preprocess] Sanitising sentiment labels...")
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
        valid = {"Positive","Negative","Neutral"}
        still_invalid = ~df["Sentiment"].isin(valid)
        if still_invalid.any():
            count_invalid = still_invalid.sum()
            print(f"[Preprocess] Re-scoring {count_invalid} rows with invalid labels...")
            df.loc[still_invalid, "Sentiment"] = (
                df.loc[still_invalid, "Translated"].apply(_score_text)
            )
    else:
        print("[Preprocess] No sentiment column — scoring all rows from scratch...")
        def _full_label(row):
            if row["Lang"] in ("hi","hinglish"):
                prior = _detect_hindi_sentiment(row["text"])
                if prior:
                    if row["IsSarcasm"] and prior == "Positive":
                        return "Negative"
                    return prior
            return _score_text(row["Translated"])
        df["Sentiment"] = df.apply(_full_label, axis=1)

    # ── Sarcasm correction on labels ──────────────────────────────────────────
    flip = df["IsSarcasm"] & (df["Sentiment"] == "Positive")
    flip_count = flip.sum()
    if flip_count > 0:
        df.loc[flip, "Sentiment"] = "Negative"
        print(f"[Preprocess] Flipped {flip_count} sarcastic Positive → Negative")

    # ── Restore output columns ─────────────────────────────────────────────────
    df["Source"]  = df["source"]
    df["Scheme"]  = df.get("scheme", pd.Series("General", index=df.index))
    df["Comment"] = df["text"]

    # ── Drop rows with empty cleaned text ─────────────────────────────────────
    before_drop = len(df)
    df = df[df["Cleaned"].str.strip().str.len() > 1].reset_index(drop=True)
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"[Preprocess] Dropped {dropped} rows with empty cleaned text")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4 — Final hard guarantee — nothing outside 3 classes reaches model
    # ─────────────────────────────────────────────────────────────────────────
    final_invalid = ~df["Sentiment"].isin({"Positive","Negative","Neutral"})
    if final_invalid.any():
        df.loc[final_invalid, "Sentiment"] = "Neutral"

    sentiment_dist = df["Sentiment"].value_counts().to_dict()
    print(f"[Preprocess] ✓ Done. {len(df)} rows ready. "
          f"Sentiment: {sentiment_dist}")

    return df
