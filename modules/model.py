"""
modules/model.py — Pulse Sentiment AI · Adaptive Model Engine
══════════════════════════════════════════════════════════════
ORIGINAL OVERFITTING FIXES (all unchanged):
  ✅ FIX 1: Duplicate removal in train_models
  ✅ FIX 2: TF-IDF min_df=3, max_df=0.95
  ✅ FIX 3: Naive Bayes alpha=1.0
  ✅ FIX 4: Unknown/Mixed/NaN labels remapped before encoding
  ✅ FIX 5: Retrain best model on full data uses fit_transform

SMART SELECTION (new):
  ✅ FIX 6: _analyze_data() — sarcasm ratio, lang diversity, social media score
  ✅ FIX 7: _select_candidate_types() — volume + avg_len + lang gates
  ✅ FIX 8: _compute_lexicon_penalties() — TextBlob vs VADER different penalties
  ✅ FIX 9: train_models() accepts df_meta for richer profiling
  ✅ BUG FIX: X_te_orig was referenced but never defined in original code

NOTE: Deep Learning (LSTM/BiLSTM/CNN) and Transformer (ALBERT/DistilBERT/mBERT)
      sections have been removed. Classical ML + NLP/Lexicon models run only.
      This keeps the app deployable on Streamlit Cloud without tensorflow/transformers.
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.calibration             import CalibratedClassifierCV
from sklearn.ensemble                import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.tree                    import DecisionTreeClassifier
from sklearn.metrics                 import (accuracy_score, f1_score, precision_score,
                                             recall_score, confusion_matrix, classification_report)
from sklearn.preprocessing           import LabelEncoder


def _get_sarcasm_score(text: str) -> float:
    try:
        from modules.preprocess import sarcasm_score
        return sarcasm_score(text)
    except Exception:
        return _sarcasm_score_standalone(text)

def _get_hindi_sentiment(text: str):
    try:
        from modules.preprocess import _detect_hindi_sentiment
        return _detect_hindi_sentiment(text)
    except Exception:
        return None

def _get_lang(text: str) -> str:
    try:
        from modules.preprocess import detect_language
        return detect_language(text)
    except Exception:
        return "en"

def _translate(text: str, lang: str) -> str:
    try:
        from modules.preprocess import translate_to_english
        return translate_to_english(text, lang)
    except Exception:
        return text

def _clean(text: str) -> str:
    try:
        from modules.preprocess import clean_text
        return clean_text(text)
    except Exception:
        return text.lower().strip()

import re as _re
_SARC_STANDALONE = [_re.compile(p, _re.IGNORECASE) for p in [
    r"\boh\s+(wow|great|sure|brilliant|perfect|fantastic|amazing)\b",
    r"yeah\s+right", r"sure\s+sure",
    r"as\s+if\s+it\s+(ever|will|would)",
    r"what\s+a\s+(joke|surprise|shocker)",
]]
_SARC_EMOJIS = {"🙄","😒","😏","🤨","🙃","😤","😑","🤡","💀","😂","🤣","😆","❌","🔄"}

def _sarcasm_score_standalone(text: str) -> float:
    score = 0.0
    tl = text.lower()
    for e in _SARC_EMOJIS:
        if e in text: score += 0.45; break
    for pat in _SARC_STANDALONE:
        if pat.search(tl): score += 0.35; break
    alpha = [c for c in text if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper())/len(alpha) > 0.45 and len(text) > 8:
        score += 0.20
    if text.count("!") >= 2: score += 0.15
    return min(score, 1.0)

_vectorizer     = None
_label_encoder  = LabelEncoder()
BEST_MODEL_OBJ  = None
BEST_MODEL_NAME = None
ALL_RESULTS     = {}


def _build_classical_models():
    return {
        "Naive Bayes":          MultinomialNB(alpha=1.0),
        "Logistic Regression":  LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "SVM (LinearSVC)":      CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0)),
        "Random Forest":        RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Decision Tree":        DecisionTreeClassifier(max_depth=15, random_state=42),
    }


class _VADERModel:
    name = "VADER (NLTK)"
    model_type = "NLP/Lexicon"
    def __init__(self):
        self.sia = None
        self._available = False
        try:
            import nltk
            try:    nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError: nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
            self._available = True
        except Exception: pass
    def predict_bulk(self, texts, label_map):
        if not self._available: return None
        reverse = {v:k for k,v in label_map.items()}
        preds = []
        for t in texts:
            c = self.sia.polarity_scores(str(t))["compound"]
            if c >= 0.05:    preds.append(reverse.get("Positive",2))
            elif c <= -0.05: preds.append(reverse.get("Negative",0))
            else:            preds.append(reverse.get("Neutral",1))
        return np.array(preds)
    def is_available(self): return self._available


class _TextBlobModel:
    name = "TextBlob"
    model_type = "NLP/Lexicon"
    def __init__(self):
        self._available = False
        try:
            from textblob import TextBlob
            self._tb = TextBlob
            self._available = True
        except Exception: pass
    def predict_bulk(self, texts, label_map):
        if not self._available: return None
        reverse = {v:k for k,v in label_map.items()}
        preds = []
        for t in texts:
            try:
                pol = self._tb(str(t)).sentiment.polarity
                if pol > 0.05:    preds.append(reverse.get("Positive",2))
                elif pol < -0.05: preds.append(reverse.get("Negative",0))
                else:             preds.append(reverse.get("Neutral",1))
            except Exception:
                preds.append(reverse.get("Neutral",1))
        return np.array(preds)
    def is_available(self): return self._available


def detect_sarcasm_advanced(text: str) -> float:
    return _get_sarcasm_score(text)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 6 — DATA PROFILE ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
def _analyze_data(X: pd.Series, y: pd.Series, df_meta: pd.DataFrame = None) -> dict:
    n_rows     = len(X)
    class_dist = y.value_counts(normalize=True)
    imbalance  = float(class_dist.max() - class_dist.min())
    all_tokens  = " ".join(X.dropna()).split()
    vocab_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)
    avg_len     = float(X.str.split().str.len().mean())

    if vocab_ratio > 0.3 or avg_len > 20:    complexity = "high"
    elif vocab_ratio > 0.15 or avg_len > 10: complexity = "medium"
    else:                                     complexity = "low"

    if df_meta is not None and "IsSarcasm" in df_meta.columns:
        sarcasm_ratio = float(df_meta["IsSarcasm"].mean())
    else:
        sample = X.sample(min(300, len(X)), random_state=42)
        sarc_count = sum(
            1 for t in sample
            if any(e in str(t) for e in {"🙄","😒","😏","🤨"})
            or str(t).count("!") >= 2
            or "yeah right" in str(t).lower()
            or "sure sure" in str(t).lower()
        )
        sarcasm_ratio = sarc_count / max(len(sample), 1)

    if df_meta is not None and "Lang" in df_meta.columns:
        lang_diversity = float(df_meta["Lang"].isin(["hi","hinglish","ta","te","bn"]).mean())
    else:
        sample = X.sample(min(300, len(X)), random_state=42)
        non_en_count = sum(
            1 for t in sample
            if any("\u0900" <= c <= "\u097f" for c in str(t))
            or len({"hai","nahi","kuch","yaar","bhai"} & set(str(t).lower().split())) >= 2
        )
        lang_diversity = non_en_count / max(len(sample), 1)

    sample = X.sample(min(300, len(X)), random_state=42)
    social_count = 0
    for t in sample:
        t = str(t)
        alpha = [c for c in t if c.isalpha()]
        caps_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
        if sum([t.count("!") >= 2, t.count("?") >= 2,
                caps_ratio > 0.35 and len(t) > 8,
                any(e in t for e in {"😂","🤣","😆","😤","😑","🤡","💀"})]) >= 1:
            social_count += 1
    social_media_score = social_count / max(len(sample), 1)

    profile = {
        "n_rows": n_rows, "imbalance": round(imbalance, 3),
        "vocab_ratio": round(vocab_ratio, 3), "avg_len": round(avg_len, 1),
        "complexity": complexity, "gpu_available": False,   # GPU not used — DL removed
        "sarcasm_ratio": round(sarcasm_ratio, 3),
        "lang_diversity": round(lang_diversity, 3),
        "social_media_score": round(social_media_score, 3),
    }
    print(f"[DATA PROFILE] rows={n_rows} | complexity={complexity} | "
          f"avg_len={profile['avg_len']} | sarcasm={profile['sarcasm_ratio']:.1%} | "
          f"non-english={profile['lang_diversity']:.1%} | "
          f"social_media={profile['social_media_score']:.1%}")
    return profile


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 7 — CANDIDATE TYPE SELECTOR
# ═════════════════════════════════════════════════════════════════════════════
def _select_candidate_types(profile: dict, use_dl: bool = False, use_transformers: bool = False) -> set:
    """
    DL and Transformer types removed — always returns Classical ML + NLP/Lexicon.
    Parameters kept for API compatibility but ignored.
    """
    n    = profile["n_rows"]
    comp = profile["complexity"]

    eligible = {"Classical ML", "NLP/Lexicon"}

    if n < 1000:
        reason = f"Small dataset ({n} rows). Classical ML + Lexicon eligible."
    elif n < 10000:
        reason = f"Medium dataset ({n} rows). Classical ML preferred."
    else:
        reason = f"Large dataset ({n} rows). Classical ML + NLP/Lexicon eligible."

    comp_notes = {
        "low":    "Low complexity → LR, NB likely to lead.",
        "medium": "Medium complexity → RF, GB likely to lead.",
        "high":   "High complexity → SVM or ensemble likely to lead.",
    }
    print(f"[CANDIDATE TYPES] {reason}")
    print(f"[CANDIDATE TYPES] {comp_notes.get(comp,'')}")
    print(f"[CANDIDATE TYPES] Eligible: {eligible}")
    return eligible


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 8 — PER-MODEL DYNAMIC LEXICON PENALTIES
# ═════════════════════════════════════════════════════════════════════════════
def _compute_lexicon_penalties(profile: dict) -> dict:
    n                  = profile["n_rows"]
    sarcasm_ratio      = profile["sarcasm_ratio"]
    lang_diversity     = profile["lang_diversity"]
    social_media_score = profile["social_media_score"]

    if n < 1000:    base = 5.0
    elif n < 5000:  base = 10.0
    elif n < 10000: base = 15.0
    else:           base = 20.0

    if sarcasm_ratio > 0.15:   sarcasm_penalty = 15.0
    elif sarcasm_ratio > 0.05: sarcasm_penalty = 8.0
    else:                      sarcasm_penalty = 0.0

    if lang_diversity > 0.3:   lang_penalty = 15.0
    elif lang_diversity > 0.1: lang_penalty = 8.0
    else:                      lang_penalty = 0.0

    shared = base + sarcasm_penalty + lang_penalty

    if social_media_score > 0.2:
        textblob_adj, vader_adj = +5.0, -3.0
    else:
        textblob_adj, vader_adj = 0.0, +3.0

    penalties = {
        "TextBlob":     round(max(shared + textblob_adj, 0.0), 1),
        "VADER (NLTK)": round(max(shared + vader_adj,    0.0), 1),
    }
    print(f"[PENALTIES] Base={base}% | Sarcasm={sarcasm_penalty}% | "
          f"Language={lang_penalty}% | "
          f"TextBlob_adj={textblob_adj:+}% VADER_adj={vader_adj:+}%")
    print(f"[PENALTIES] Final → TextBlob={penalties['TextBlob']}% | "
          f"VADER={penalties['VADER (NLTK)']}%")
    return penalties


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def train_models(X: pd.Series, y: pd.Series,
                 use_dl=False, use_transformers=False,
                 df_meta: pd.DataFrame = None):
    """
    FIX 9: Added df_meta parameter.
    X       : df["Cleaned"] — translated + cleaned English text
    y       : sentiment labels
    df_meta : full preprocessed DataFrame (optional) — provides:
                IsSarcasm → accurate sarcasm ratio
                Lang      → accurate language diversity

    NOTE: use_dl and use_transformers parameters are kept for API compatibility
          but are completely ignored — DL and Transformer models are not available.
          Only Classical ML + VADER + TextBlob models are trained.
    """
    global _vectorizer, _label_encoder, BEST_MODEL_OBJ, BEST_MODEL_NAME, ALL_RESULTS

    mask = X.notna() & y.notna() & (X.str.strip() != "")
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    if df_meta is not None:
        df_meta = df_meta.loc[mask].reset_index(drop=True)

    # FIX 1: Deduplicate
    df_temp = pd.DataFrame({"text": X, "label": y})
    before  = len(df_temp)
    df_temp = df_temp.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(df_temp) < before:
        print(f"[MODEL DEDUP] {before - len(df_temp)} cleaned-text duplicates removed")
    X = df_temp["text"].reset_index(drop=True)
    y = df_temp["label"].reset_index(drop=True)
    if df_meta is not None:
        df_meta = df_meta.iloc[df_temp.index].reset_index(drop=True)

    if len(X) < 10:
        return {"Insufficient data": {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                      "speed_ms":0,"type":"Classical ML","available":False}}, "N/A"

    # FIX 4: Remap labels
    y = y.replace({
        "Sarcasm":"Negative","Sarcastic":"Negative","Mixed":"Neutral",
        "Unknown":"Neutral","unknown":"Neutral","Nan":"Neutral","None":"Neutral","":"Neutral",
    })
    valid = {"Positive","Negative","Neutral"}
    mask2 = y.isin(valid)
    dropped = (~mask2).sum()
    if dropped > 0:
        print(f"[LABELS] Dropped {dropped} rows: {y[~mask2].unique().tolist()}")
    X, y = X[mask2].reset_index(drop=True), y[mask2].reset_index(drop=True)
    if df_meta is not None:
        df_meta = df_meta.loc[mask2].reset_index(drop=True)

    y_enc       = _label_encoder.fit_transform(y)
    num_classes = len(_label_encoder.classes_)
    label_map   = {i:cls for i,cls in enumerate(_label_encoder.classes_)}
    counts      = pd.Series(y_enc).value_counts()
    do_stratify = (counts >= 2).all()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42,
        stratify=y_enc if do_stratify else None, shuffle=True,
    )

    # FIX 2: TF-IDF
    n_docs = len(X_tr)
    _vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1 if n_docs < 500 else (2 if n_docs < 2000 else 3),
        max_df=0.98 if n_docs < 500 else 0.95,
        strip_accents="unicode",
    )
    X_tr_v = _vectorizer.fit_transform(X_tr)
    X_te_v = _vectorizer.transform(X_te)

    results = {}
    fitted  = {}

    def _record(name, preds, duration_ms, model_type, model_obj=None):
        if preds is None or len(preds) == 0: return
        n = min(len(preds), len(y_te))
        acc  = round(accuracy_score(y_te[:n], preds[:n]) * 100, 2)
        f1   = round(f1_score(y_te[:n], preds[:n], average="weighted", zero_division=0) * 100, 2)
        prec = round(precision_score(y_te[:n], preds[:n], average="weighted", zero_division=0) * 100, 2)
        rec  = round(recall_score(y_te[:n], preds[:n], average="weighted", zero_division=0) * 100, 2)
        results[name] = {"accuracy":acc,"f1":f1,"precision":prec,"recall":rec,
                         "speed_ms":round(duration_ms,1),"type":model_type,"available":True}
        if model_obj is not None:
            fitted[name] = model_obj

    # ── Classical ML ──────────────────────────────────────────────────────────
    for name, clf in _build_classical_models().items():
        try:
            t0 = time.time()
            clf.fit(X_tr_v, y_tr)
            preds = clf.predict(X_te_v)
            _record(name, preds, (time.time()-t0)*1000, "Classical ML", clf)
        except Exception:
            results[name] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                             "speed_ms":9999,"type":"Classical ML","available":False}

    # ── VADER ─────────────────────────────────────────────────────────────────
    vader = _VADERModel()
    if vader.is_available():
        try:
            t0    = time.time()
            preds = vader.predict_bulk(X_te.tolist(), label_map)
            if preds is not None:
                _record("VADER (NLTK)", preds, (time.time()-t0)*1000, "NLP/Lexicon")
        except Exception: pass
    else:
        results["VADER (NLTK)"] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                   "speed_ms":0,"type":"NLP/Lexicon","available":False}

    # ── TextBlob ──────────────────────────────────────────────────────────────
    tb = _TextBlobModel()
    if tb.is_available():
        try:
            t0    = time.time()
            preds = tb.predict_bulk(X_te.tolist(), label_map)
            if preds is not None:
                _record("TextBlob", preds, (time.time()-t0)*1000, "NLP/Lexicon")
        except Exception: pass
    else:
        results["TextBlob"] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                               "speed_ms":0,"type":"NLP/Lexicon","available":False}

    # ── Smart model selection ─────────────────────────────────────────────────
    available = {k:v for k,v in results.items()
                 if v.get("available",False) and v["accuracy"] > 0}
    if not available:
        available = results

    profile        = _analyze_data(X, y, df_meta)
    eligible_types = _select_candidate_types(profile)
    penalties      = _compute_lexicon_penalties(profile)

    smart_pool = {k:v for k,v in available.items()
                  if v.get("type") in eligible_types}
    if not smart_pool:
        print("[MODEL SELECTION] Warning: smart pool empty — falling back to all available")
        smart_pool = available

    def _selection_score(k: str) -> tuple:
        acc = smart_pool[k]["accuracy"]
        typ = smart_pool[k].get("type", "")
        spd = smart_pool[k].get("speed_ms", 9999)

        text_suitability_penalty = 0.0
        if k == "Decision Tree":        text_suitability_penalty = 5.0
        if k == "K-Nearest Neighbours": text_suitability_penalty = 3.0

        if typ == "NLP/Lexicon":
            penalty = penalties.get(k, penalties.get("TextBlob", 10.0))
            adj_acc = acc - penalty - text_suitability_penalty
        else:
            adj_acc = acc - text_suitability_penalty

        return (adj_acc, -spd)

    best_name = max(smart_pool, key=_selection_score)
    real_acc  = smart_pool[best_name]["accuracy"]
    btype     = smart_pool[best_name].get("type","")
    adj       = real_acc - penalties.get(best_name, 0.0) if btype == "NLP/Lexicon" else real_acc
    print(f"[MODEL SELECTION] Winner: {best_name} | Real: {real_acc}% | "
          f"Score: {round(adj,2)}% | Type: {btype} | Rows: {profile['n_rows']}")

    # FIX 5: Retrain on full data
    if best_name in fitted:
        n_docs     = len(X_tr)
        min_df_val = 2 if n_docs < 500 else 3
        max_df_val = 0.98 if n_docs < 500 else 0.95
        _vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
            min_df=min_df_val, max_df=max_df_val, strip_accents="unicode",
        )
        X_full_v = _vectorizer.fit_transform(X)
        fitted[best_name].fit(X_full_v, y_enc)
        BEST_MODEL_OBJ = fitted[best_name]

    BEST_MODEL_NAME = best_name
    ALL_RESULTS     = results
    return results, best_name


def get_detailed_metrics(X: pd.Series, y: pd.Series):
    if BEST_MODEL_OBJ is None: return None, None, None
    mask  = X.notna() & y.notna() & (X.str.strip() != "")
    X, y  = X[mask], y[mask]
    y     = y.replace({"Sarcasm":"Negative","Sarcastic":"Negative"})
    y_enc = _label_encoder.transform(y)
    X_vec = _vectorizer.transform(X)
    _, X_te, _, y_te = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)
    preds  = BEST_MODEL_OBJ.predict(X_te)
    cm     = confusion_matrix(y_te, preds)
    report = classification_report(y_te, preds, target_names=_label_encoder.classes_,
                                   output_dict=True, zero_division=0)
    return cm, report, _label_encoder.classes_


# ═════════════════════════════════════════════════════════════════════════════
#  DOMAIN KEYWORD ENGINE
#  Handles what TextBlob + VADER miss on Indian govt scheme comments
# ═════════════════════════════════════════════════════════════════════════════

# ── Positive signals ──────────────────────────────────────────────────────────
_DOMAIN_POSITIVE = {
    # English
    "life changing","changed everything","finally got","zero bribe","no middleman",
    "directly credited","actually helped","genuinely helped","real change",
    "actually works","came through","exceeded expectations","smooth process",
    "got the keys","saved my life","paid zero","golden card","truly helpful",
    "on time","timely support","without corruption","without bribe",
    "reached directly","no extra charges","free of cost","actually delivered",
    "improved conditions","better now","real difference","made a difference",
    "transformed","revolutionary","life saver","game changer",

    # Hindi positive
    "सच में मिला","सही समय पर","सीधे खाते में","बिना रिश्वत","काम आया",
    "फायदा हुआ","बदलाव आया","बहुत अच्छा","सही में मिला","राहत मिली",

    # Hinglish positive
    "seedha account mein","bina rishwat","sach mein kaam aaya","real benefit mila",
    "time pe mila","direct credit","actually mila","sach mein helpful",
    "genuine help","real fayda","actually kaam aaya",

    # Tamil positive
    "romba nalla","upyogamaana","help aagudhu","reach aagudhu","nandri",

    # Bengali positive
    "khub valo","onek upokar","bhalo hocche","valo initiative",
}

# ── Negative signals ──────────────────────────────────────────────────────────
_DOMAIN_NEGATIVE = {
    # English — direct
    "corruption","bribe","extra charges","middlemen","only on paper",
    "just a slogan","ground reality","still waiting","never received",
    "rejected without","not reached","not received","no benefit",
    "server is down","no beds","technical glitch","portal crashes",
    "impossible to register","complicated process","documents impossible",
    "political connections","only connected people","politically connected",
    "money not received","not credited","disappeared from list",
    "no one at office","nobody knows","no response","no reply",
    "useless implementation","completely useless","total failure",
    "only announcement","just announcement","zero impact","zero effect",
    "still no response","still pending","still waiting","years of waiting",
    "filed complaint","no action","ignored","bounced back",
    "name disappeared","name removed","data mismatch","aadhaar mismatch",

    # English — indirect negative (the hardest ones)
    "still stuck","stuck at verification","3 years ago","two years ago",
    "three years ago","been waiting","waiting since","still same",
    "nothing changed","nothing has changed","ground reality is different",
    "only in news","only in ads","only announcements",
    "used as storage","no water connection","no electricity connection",
    "broken computers","never touched","no tools","no equipment",
    "certificate but no job","trained but no job","certified but useless",
    "placement hasn't","no placement","no one called",

    # Hindi negative
    "नहीं मिला","नहीं आया","कोई जवाब नहीं","चक्कर लगाने पड़े","भ्रष्टाचार",
    "सिर्फ कागजों पर","जमीन पर कुछ नहीं","रिश्वत","बिचौलिए","घोटाला",
    "नाकाम","विफल","बेकार","फर्जी","झूठ","धोखा","नुकसान","खराब",
    "सिर्फ नाम","बस घोषणा","काम नहीं","लाभ नहीं","फायदा नहीं",

    # Hinglish negative
    "nahi mila","nahi aaya","kuch nahi mila","bekar hai","bakwaas hai",
    "koi fayda nahi","paisa nahi aaya","naam list se hata diya",
    "data match nahi","chakkar lagate rahe","office ke chakkar",
    "sirf kaagaz pe","ground pe kuch nahi","announcement hi hai",
    "kaam nahi hua","nahi pahuncha","sirf naam ka","faltu",

    # Tamil negative
    "onnum illa","reach aagalai","server down","payanilla","vela illai",
    "actual la onnum illai","soldra mattume dhan","nadakkalai",

    # Bengali negative
    "kono kaj hocche na","kono upokar painchi na","sudhu kotha",
    "shob kagoje","kichu nei","reply nei","kono bodol nei",
}

# ── Neutral signals ───────────────────────────────────────────────────────────
_DOMAIN_NEUTRAL = {
    "let's see","dekhte hain","mixed results","depends on location",
    "too early to say","still in progress","concept is good execution average",
    "some areas improved","not everywhere","partially helpful","average",
    "time will tell","not sure yet","thik thak","kuch khaas nahi",
    "okay i guess","not bad not good","mixed feedback","varies by area",
    "paakkalaam","konjam improvement","kichu jaygay valo",
}

# ── Sarcasm domain signals ────────────────────────────────────────────────────
_DOMAIN_SARCASM = {
    # Amount irony — tiny amount + big claim
    "₹2000 every 4 months","₹2000 every four months","6000 a year retire",
    "2000 rupees retire","luxury villa","farmer is king","rich now thanks to",
    "5 lakh card but","insurance card but hospital","100km away",
    "empanelled hospital is 100","hold my breath",

    # Bureaucracy irony
    "apply online then visit office","visit office to verify applied online",
    "4 times to verify","five photocopies of online","photocopies of my online",
    "online application at office","digital counter server down",
    "server down soul of","soul of digital india","powerless but digital",
    "5 star clean city award","clean city award garbage",
    "certified plumber never touched","never touched a pipe",
    "certificate but learned nothing","google earth from 2014",
    "corporator relative got","randomly got allotted",
    "such a fair system","so fair",

    # Indirect sarcasm markers
    "what a surprise","what a coincidence","oh what a system",
    "surely that's why","that explains why","makes total sense",
    "i love how","isn't it great","isnt it wonderful",
    "must have used google earth","jury must have",
}

# ── Emoji sentiment map ───────────────────────────────────────────────────────
_EMOJI_POSITIVE = {"👍","🙌","🔥","💪","✅","🎉","😊","🥰","❤️","💚","🌟","⭐","🏆"}
_EMOJI_NEGATIVE = {"👎","😡","😤","💔","❌","🚫","😔","😢","😭","🤦","😠","😞"}
_EMOJI_SARCASM  = {"🙄","😒","😏","🤨","🙃","😤","😑","🤡","💀","😂","🤣","😆","🔄"}
_EMOJI_NEUTRAL  = {"🤔","😐","🙂","😶","💭","🤷"}


# ═════════════════════════════════════════════════════════════════════════════
#  DOMAIN KEYWORD SCORER
# ═════════════════════════════════════════════════════════════════════════════
def _domain_score(text: str) -> dict:
    """
    Scores text using domain-specific Indian govt scheme knowledge.

    Returns dict with:
      pos_score    — positive signal strength 0.0-1.0
      neg_score    — negative signal strength 0.0-1.0
      sarc_score   — sarcasm signal strength 0.0-1.0
      neu_score    — neutral signal strength 0.0-1.0
      emoji_signal — "positive"/"negative"/"sarcasm"/"neutral"/None
      reasons      — list of matched signals (for debugging)
    """
    tl      = text.lower()
    words   = set(tl.split())
    reasons = []

    pos_score  = 0.0
    neg_score  = 0.0
    sarc_score = 0.0
    neu_score  = 0.0

    # ── Check domain positive phrases ────────────────────────────────────────
    for phrase in _DOMAIN_POSITIVE:
        if phrase in tl:
            pos_score += 0.4
            reasons.append(f"pos_phrase:{phrase}")
            break

    # ── Check domain negative phrases ────────────────────────────────────────
    neg_hits = 0
    for phrase in _DOMAIN_NEGATIVE:
        if phrase in tl:
            neg_score += 0.35
            neg_hits  += 1
            reasons.append(f"neg_phrase:{phrase}")
            if neg_hits >= 2:
                break

    # ── Check domain sarcasm phrases ─────────────────────────────────────────
    for phrase in _DOMAIN_SARCASM:
        if phrase in tl:
            sarc_score += 0.5
            reasons.append(f"sarc_phrase:{phrase}")
            break

    # ── Check domain neutral phrases ─────────────────────────────────────────
    for phrase in _DOMAIN_NEUTRAL:
        if phrase in tl:
            neu_score += 0.4
            reasons.append(f"neu_phrase:{phrase}")
            break

    # ── Emoji analysis ────────────────────────────────────────────────────────
    emoji_signal = None
    for e in _EMOJI_SARCASM:
        if e in text:
            sarc_score  += 0.35
            emoji_signal = "sarcasm"
            reasons.append(f"sarc_emoji:{e}")
            break
    if emoji_signal is None:
        for e in _EMOJI_POSITIVE:
            if e in text:
                pos_score   += 0.3
                emoji_signal = "positive"
                reasons.append(f"pos_emoji:{e}")
                break
        for e in _EMOJI_NEGATIVE:
            if e in text:
                neg_score   += 0.3
                emoji_signal = "negative"
                reasons.append(f"neg_emoji:{e}")
                break
        if emoji_signal is None:
            for e in _EMOJI_NEUTRAL:
                if e in text:
                    neu_score   += 0.2
                    emoji_signal = "neutral"
                    break

    # ── Indirect negative patterns ────────────────────────────────────────────
    import re
    if re.search(r"\b(applied|waiting|pending)\b.{0,60}\b(year|month|ago|since)\b", tl):
        neg_score += 0.35
        reasons.append("indirect_neg:waiting_time")

    if re.search(r"\bonly\s+(on\s+paper|in\s+news|in\s+ads|announcements?|for\s+show)\b", tl):
        neg_score += 0.4
        reasons.append("indirect_neg:only_on_paper")

    if re.search(r"\bnothing\s+(has\s+)?(changed|happened|improved|worked)\b", tl):
        neg_score += 0.4
        reasons.append("indirect_neg:nothing_changed")

    if re.search(r"\bname\b.{0,30}\b(removed|disappeared|deleted|missing|hata|gaya)\b", tl):
        neg_score += 0.4
        reasons.append("indirect_neg:name_removed")

    if re.search(r"\bstill\s+(no\s+\w+|waiting|pending|same|stuck|facing|nothing)\b", tl):
        neg_score += 0.3
        reasons.append("indirect_neg:still_no")

    # ── Amount irony detection ────────────────────────────────────────────────
    if re.search(r"(₹|rs\.?|rupee)\s*\d+", tl):
        if re.search(r"\b(retire|villa|luxury|rich|wealthy|afford|king|queen)\b", tl):
            sarc_score += 0.5
            reasons.append("amount_irony:tiny_amount_big_claim")

    if re.search(r"\b\d+\s*lakh\b.{0,60}\b(but|however|except|still|yet)\b", tl):
        sarc_score += 0.3
        reasons.append("amount_irony:lakh_card_but_problem")

    # ── "I love how [problem]" sarcasm ───────────────────────────────────────
    if re.search(r"\bi\s+love\s+how\b", tl):
        sarc_score += 0.45
        reasons.append("sarc_pattern:i_love_how")

    # ── Exclamation + downgrade pattern ──────────────────────────────────────
    if re.search(r"(amazing|great|wonderful|fantastic|brilliant)[!.]+.{0,80}(still|same|nothing|zero|no\s)", tl):
        sarc_score += 0.4
        reasons.append("sarc_pattern:praise_then_problem")

    # ── Raw social media short text ───────────────────────────────────────────
    word_count = len(text.split())
    if word_count <= 5:
        if any(w in tl for w in ["bekaar","bekar","bakwaas","useless","terrible","worst"]):
            neg_score += 0.5
            reasons.append("raw_social:short_negative")
        if any(w in tl for w in ["acha","badhiya","good","great","helpful","mast","nalla"]):
            pos_score += 0.5
            reasons.append("raw_social:short_positive")

    return {
        "pos_score":    round(min(pos_score, 1.0), 3),
        "neg_score":    round(min(neg_score, 1.0), 3),
        "sarc_score":   round(min(sarc_score, 1.0), 3),
        "neu_score":    round(min(neu_score, 1.0), 3),
        "emoji_signal": emoji_signal,
        "reasons":      reasons,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  5-WAY ENSEMBLE VOTER
#  Combines: ML model + TextBlob + VADER + Domain keywords + Sarcasm engine
# ═════════════════════════════════════════════════════════════════════════════
def _ensemble_vote(
    ml_label,
    ml_conf: float,
    tb_label,
    vader_label,
    domain: dict,
    sarc_score: float,
    hindi_prior,
    lang: str,
):
    """
    Combines all signals into final sentiment + confidence + model_used.

    Voting weights:
      Domain keywords   : 0.35  (domain-specific, handles indirect patterns)
      ML model          : 0.30  (trained, but needs enough vocabulary)
      Hindi prior       : 0.25  (for hi/hinglish text, very accurate)
      TextBlob          : 0.20  (good on formal English polarity words)
      VADER             : 0.20  (good on social media, !!!, CAPS)

    Sarcasm override:
      If sarc_score > 0.45 AND winning sentiment is Positive → flip to Negative
    """
    from collections import defaultdict
    votes  = defaultdict(float)
    agents = []

    # ── Domain keyword vote ───────────────────────────────────────────────────
    d_scores = {
        "Positive": domain["pos_score"],
        "Negative": domain["neg_score"],
        "Neutral":  domain["neu_score"],
    }
    if domain["sarc_score"] > 0.45:
        d_scores["Negative"] = max(d_scores["Negative"], domain["sarc_score"])

    d_winner = max(d_scores, key=d_scores.get)
    d_conf   = d_scores[d_winner]

    if d_conf > 0.25:
        votes[d_winner] += 0.35 * d_conf
        agents.append(f"Domain({d_winner},{d_conf:.2f})")

    # ── ML model vote ─────────────────────────────────────────────────────────
    if ml_label and ml_conf > 0:
        ml_weight = 0.30 * (ml_conf / 100.0)
        votes[ml_label] += ml_weight
        agents.append(f"ML({ml_label},{ml_conf:.0f}%)")

    # ── Hindi keyword prior vote ──────────────────────────────────────────────
    if hindi_prior and lang in ("hi", "hinglish"):
        votes[hindi_prior] += 0.25
        agents.append(f"Hindi({hindi_prior})")

    # ── TextBlob vote ─────────────────────────────────────────────────────────
    if tb_label:
        votes[tb_label] += 0.20
        agents.append(f"TextBlob({tb_label})")

    # ── VADER vote ────────────────────────────────────────────────────────────
    if vader_label:
        votes[vader_label] += 0.20
        agents.append(f"VADER({vader_label})")

    if not votes:
        return "Neutral", 50.0, "No signals detected"

    # ── Determine winner ──────────────────────────────────────────────────────
    winner      = max(votes, key=votes.get)
    total_score = sum(votes.values())
    win_score   = votes[winner]
    confidence  = min((win_score / max(total_score, 0.01)) * 100, 97.0)
    confidence  = max(confidence, 52.0)

    model_used = " + ".join(agents[:4])
    if sarc_score > 0.45 and winner == "Positive":
        winner     = "Negative"
        confidence = max(confidence, 70.0)
        model_used += " [Sarcasm Override]"

    return winner, round(confidence, 1), model_used


# ═════════════════════════════════════════════════════════════════════════════
#  TEXTBLOB + VADER STANDALONE PREDICTORS
# ═════════════════════════════════════════════════════════════════════════════
def _get_textblob_label(text: str):
    try:
        from textblob import TextBlob
        pol = TextBlob(str(text)).sentiment.polarity
        if pol > 0.08:    return "Positive"
        elif pol < -0.08: return "Negative"
        else:             return "Neutral"
    except Exception:
        return None


def _get_vader_label(text: str):
    try:
        import nltk
        try:    nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError: nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        c = SentimentIntensityAnalyzer().polarity_scores(str(text))["compound"]
        if c >= 0.05:    return "Positive"
        elif c <= -0.05: return "Negative"
        else:            return "Neutral"
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LIVE PROBE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
def predict_live(cleaned_text: str) -> str:
    return predict_live_with_confidence(cleaned_text, cleaned_text)["sentiment"]


def predict_live_with_confidence(text: str, cleaned_text: str = None) -> dict:
    """
    Full 5-way ensemble live probe.

    Pipeline:
      1. Language detection
      2. Hindi keyword prior (for hi/hinglish)
      3. Translation (for non-English)
      4. Sarcasm scoring on ORIGINAL text
      5. Text cleaning for ML
      6. ML model prediction (with sparsity guard)
      7. TextBlob prediction on translated text
      8. VADER prediction on translated text
      9. Domain keyword scoring on original + translated
      10. 5-way ensemble vote → final sentiment + confidence
      11. Sarcasm override if needed
    """
    if text is None:
        text = ""

    # ── Step 1: Language detection ────────────────────────────────────────────
    lang        = _get_lang(text)
    hindi_prior = None
    if lang in ("hi", "hinglish"):
        hindi_prior = _get_hindi_sentiment(text)

    # ── Step 2: Translation ───────────────────────────────────────────────────
    translated = _translate(text, lang) if lang != "en" else text

    # ── Step 3: Sarcasm score on ORIGINAL text ────────────────────────────────
    sc           = _get_sarcasm_score(text)
    is_sarcastic = sc > 0.45

    # ── Step 4: Clean text for ML ─────────────────────────────────────────────
    if cleaned_text and len(cleaned_text.strip()) > 1:
        c_text = cleaned_text
    else:
        c_text = _clean(translated)
        if not c_text.strip():
            c_text = translated[:200]

    # ── Step 5: Domain keyword scoring ────────────────────────────────────────
    domain_orig  = _domain_score(text)
    domain_trans = _domain_score(translated)
    domain = {
        "pos_score":    max(domain_orig["pos_score"],  domain_trans["pos_score"]),
        "neg_score":    max(domain_orig["neg_score"],  domain_trans["neg_score"]),
        "sarc_score":   max(domain_orig["sarc_score"], domain_trans["sarc_score"]),
        "neu_score":    max(domain_orig["neu_score"],  domain_trans["neu_score"]),
        "emoji_signal": domain_orig["emoji_signal"] or domain_trans["emoji_signal"],
        "reasons":      domain_orig["reasons"] + domain_trans["reasons"],
    }

    # ── Step 6: ML model prediction ───────────────────────────────────────────
    ml_label = None
    ml_conf  = 0.0
    ml_name  = "ML Model"

    if BEST_MODEL_OBJ is not None and _vectorizer is not None:
        try:
            vec      = _vectorizer.transform([c_text])
            non_zero = vec.nnz
            proba    = BEST_MODEL_OBJ.predict_proba(vec)[0]
            pred     = int(np.argmax(proba))
            label    = _label_encoder.inverse_transform([pred])[0]
            conf     = float(max(proba)) * 100

            if non_zero >= 3 and conf >= 45.0:
                ml_label = label
                ml_conf  = conf
                ml_name  = BEST_MODEL_NAME or "ML Model"
        except Exception:
            pass

    # ── Step 7: TextBlob prediction ───────────────────────────────────────────
    tb_label = _get_textblob_label(translated)

    # ── Step 8: VADER prediction ──────────────────────────────────────────────
    vader_label = _get_vader_label(translated)

    # ── Step 9: 5-way ensemble vote ───────────────────────────────────────────
    sentiment, confidence, model_used = _ensemble_vote(
        ml_label    = ml_label,
        ml_conf     = ml_conf,
        tb_label    = tb_label,
        vader_label = vader_label,
        domain      = domain,
        sarc_score  = sc,
        hindi_prior = hindi_prior,
        lang        = lang,
    )

    # ── Step 10: Final validation ─────────────────────────────────────────────
    if sentiment not in ("Positive", "Negative", "Neutral"):
        sentiment = "Neutral"

    return {
        "sentiment":     sentiment,
        "confidence":    round(confidence, 1),
        "sarcasm_score": round(sc * 100, 1),
        "is_sarcastic":  is_sarcastic,
        "model_used":    model_used,
        "language":      lang,
        "translated":    translated if translated != text else "",
    }


def _fallback_predict(text: str, sc: float = 0.0) -> str:
    """Legacy fallback — kept for compatibility but ensemble is now used instead."""
    from collections import Counter
    votes = []
    tb = _get_textblob_label(text)
    if tb: votes.append(tb)
    vd = _get_vader_label(text)
    if vd: votes.append(vd)
    if not votes: return "Neutral"
    result = Counter(votes).most_common(1)[0][0]
    if sc > 0.45 and result == "Positive":
        return "Negative"
    return result
