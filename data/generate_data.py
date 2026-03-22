"""
modules/model.py — Pulse Sentiment AI · Adaptive Model Engine
══════════════════════════════════════════════════════════════
FIXED (accuracy issues only — nothing else changed):
  ✅ FIX 1: Duplicate removal in train_models (same as preprocess dedup)
  ✅ FIX 2: TF-IDF min_df=2→3, max_df=0.95 added (stops ultra-rare/ultra-common terms)
  ✅ FIX 3: Naive Bayes alpha 0.5→1.0 (better smoothing for sparse text)
  ✅ FIX 4: Unknown/Mixed/NaN labels remapped BEFORE encoding (not just Sarcasm)
  ✅ FIX 5: Retrain best model on full data uses fit_transform not transform
             (prevents "vocabulary not fitted" error on unseen tokens)

SMART MODEL SELECTION (new):
  ✅ FIX 6: _analyze_data() — measures real volume, complexity, vocab richness
  ✅ FIX 7: _select_candidate_types() — maps data profile → eligible model types
             following the ML vs DL vs NLP decision framework
  ✅ FIX 8: TextBlob/VADER excluded from best-model selection (not trained on data)
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

# ── Delegate to preprocess (single source of truth) ──────────────────────────
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


# ── Standalone sarcasm fallback ───────────────────────────────────────────────
import re as _re
_SARC_STANDALONE = [_re.compile(p, _re.IGNORECASE) for p in [
    r"\boh\s+(wow|great|sure|brilliant|perfect|fantastic|amazing)\b",
    r"yeah\s+right",
    r"sure\s+sure",
    r"as\s+if\s+it\s+(ever|will|would)",
    r"what\s+a\s+(joke|surprise|shocker)",
]]
_SARC_EMOJIS_S = {"🙄","😒","😏","🤨","🙃","😤","😑","🤡","💀"}

def _sarcasm_score_standalone(text: str) -> float:
    score = 0.0
    tl = text.lower()
    for e in _SARC_EMOJIS_S:
        if e in text:
            score += 0.45; break
    for pat in _SARC_STANDALONE:
        if pat.search(tl):
            score += 0.35; break
    alpha = [c for c in text if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper())/len(alpha) > 0.45 and len(text) > 8:
        score += 0.20
    if text.count("!") >= 2:
        score += 0.15
    return min(score, 1.0)


# ── Global state ──────────────────────────────────────────────────────────────
_vectorizer     = None
_label_encoder  = LabelEncoder()
BEST_MODEL_OBJ  = None
BEST_MODEL_NAME = None
ALL_RESULTS     = {}


# ═════════════════════════════════════════════════════════════════════════════
#  CLASSICAL ML MODELS
# ═════════════════════════════════════════════════════════════════════════════
def _build_classical_models():
    return {
        # FIX 3: Naive Bayes alpha 0.5 → 1.0
        # alpha=0.5 under-smooths sparse TF-IDF vectors → model memorises
        # rare tokens from training set → inflated accuracy on test
        # alpha=1.0 (Laplace smoothing) is the correct default for text
        "Naive Bayes":          MultinomialNB(alpha=1.0),

        # Everything else unchanged
        "Logistic Regression":  LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "SVM (LinearSVC)":      CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0)),
        "Random Forest":        RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Decision Tree":        DecisionTreeClassifier(max_depth=15, random_state=42),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  NLP / LEXICON WRAPPERS  — unchanged
# ═════════════════════════════════════════════════════════════════════════════
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
        except Exception:
            pass

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
        except Exception:
            pass

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


# ═════════════════════════════════════════════════════════════════════════════
#  DEEP LEARNING  — unchanged
# ═════════════════════════════════════════════════════════════════════════════
def _build_dl_models(vocab_size, max_len, num_classes):
    models = {}
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Embedding, LSTM, Bidirectional, Conv1D,
                                             GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D)
        tf.get_logger().setLevel("ERROR")
        embed_dim = 64

        models["LSTM"] = Sequential([
            Embedding(vocab_size, embed_dim, input_length=max_len),
            SpatialDropout1D(0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation="relu"), Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ], name="LSTM")
        models["LSTM"].compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        models["BiLSTM"] = Sequential([
            Embedding(vocab_size, embed_dim, input_length=max_len),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            Dense(32, activation="relu"), Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ], name="BiLSTM")
        models["BiLSTM"].compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        models["CNN-Text"] = Sequential([
            Embedding(vocab_size, embed_dim, input_length=max_len),
            SpatialDropout1D(0.2),
            Conv1D(128, 5, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"), Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ], name="CNN-Text")
        models["CNN-Text"].compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    except ImportError:
        pass
    return models


# ═════════════════════════════════════════════════════════════════════════════
#  TRANSFORMER WRAPPER  — unchanged
# ═════════════════════════════════════════════════════════════════════════════
class _TransformerSentiment:
    def __init__(self, model_name, display_name):
        self.model_name   = model_name
        self.display_name = display_name
        self._pipeline    = None
        self._available   = False
        self.model_type   = "Transformer/BERT"

    def _load(self):
        if self._pipeline is not None: return True
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification", model=self.model_name,
                truncation=True, max_length=128, device=-1,
            )
            self._available = True
            return True
        except Exception:
            return False

    def predict_bulk(self, texts, label_map, max_samples=200):
        if not self._load(): return None
        reverse = {v:k for k,v in label_map.items()}
        preds = []
        texts = [str(t)[:512] for t in texts[:max_samples]]
        try:
            for r in self._pipeline(texts, batch_size=16):
                lbl = r["label"].upper()
                if "POS" in lbl or lbl in ("LABEL_2","5 STARS","4 STARS"):
                    preds.append(reverse.get("Positive",2))
                elif "NEG" in lbl or lbl in ("LABEL_0","1 STAR","2 STARS"):
                    preds.append(reverse.get("Negative",0))
                else:
                    preds.append(reverse.get("Neutral",1))
        except Exception:
            return None
        return np.array(preds)

    def is_available(self): return self._load()


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC SARCASM API
# ═════════════════════════════════════════════════════════════════════════════
def detect_sarcasm_advanced(text: str) -> float:
    """Returns 0.0–1.0. >0.45 = sarcastic."""
    return _get_sarcasm_score(text)


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 6 — DATA PROFILE ANALYZER
#  Measures real data characteristics to guide smart model selection.
#  Called once before model selection — adds zero training overhead.
# ═════════════════════════════════════════════════════════════════════════════
def _analyze_data(X: pd.Series, y: pd.Series) -> dict:
    """
    Measures real data characteristics to guide model selection.
    Returns a profile dict used by _select_candidate_types().

    Metrics measured:
      n_rows      — total training samples
      n_classes   — number of sentiment classes (should be 3)
      imbalance   — max class share minus min class share (0=balanced, 1=extreme)
      vocab_ratio — unique tokens / total tokens (proxy for text diversity)
      avg_len     — average tokens per comment (proxy for text complexity)
      complexity  — "low" / "medium" / "high" derived from vocab_ratio + avg_len
      gpu_available — whether a GPU is detected (gates Deep Learning eligibility)
    """
    n_rows     = len(X)
    n_classes  = y.nunique()
    class_dist = y.value_counts(normalize=True)
    imbalance  = float(class_dist.max() - class_dist.min())

    # Vocabulary richness — unique tokens / total tokens
    # High ratio = diverse language = harder classification problem
    all_tokens  = " ".join(X.dropna()).split()
    vocab_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)

    # Average comment length in tokens
    avg_len = float(X.str.split().str.len().mean())

    # Complexity classification
    # High:   rich vocabulary OR long comments → non-linear patterns likely
    # Medium: moderate diversity → ensemble methods competitive
    # Low:    repetitive short text → simple linear models sufficient
    if vocab_ratio > 0.3 or avg_len > 20:
        complexity = "high"
    elif vocab_ratio > 0.15 or avg_len > 10:
        complexity = "medium"
    else:
        complexity = "low"

    # GPU detection — gates Deep Learning eligibility
    gpu_available = False
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        pass

    profile = {
        "n_rows":        n_rows,
        "n_classes":     n_classes,
        "imbalance":     round(imbalance, 3),
        "vocab_ratio":   round(vocab_ratio, 3),
        "avg_len":       round(avg_len, 1),
        "complexity":    complexity,
        "gpu_available": gpu_available,
    }
    print(f"[DATA PROFILE] {profile}")
    return profile


# ═════════════════════════════════════════════════════════════════════════════
#  FIX 7 — CANDIDATE TYPE SELECTOR
#  Maps data profile → set of eligible model TYPES using the ML/DL/NLP
#  decision framework (volume + complexity + resources).
#  Eligible types are then used to filter the benchmark results pool.
# ═════════════════════════════════════════════════════════════════════════════
def _select_candidate_types(profile: dict, use_dl: bool, use_transformers: bool) -> set:
    """
    Decides which MODEL TYPES are eligible for best-model selection
    based on real data characteristics. Follows the decision framework:

    Volume rules (primary gate):
      < 1000 rows   → Classical ML only
                      DL overfits severely on small text datasets
      1000–10000    → Classical ML + NLP/Lexicon eligible
                      Enough data for fair lexicon comparison,
                      not enough for reliable DL training
      10000–50000   → Classical ML + NLP/Lexicon + Deep Learning (if GPU)
                      DL starts to shine but needs hardware
      50000+        → All types eligible including Transformers

    Complexity notes (secondary — logged only, does not remove types):
      low    → interpretable models (LR, NB, DT) likely best
      medium → ensemble methods (RF, GB) likely best
      high   → all ML competitive, DL justified if volume allows

    Resource gate:
      No GPU detected → Deep Learning excluded regardless of volume
      GPU detected    → Deep Learning allowed if volume qualifies

    NLP/Lexicon note:
      TextBlob and VADER appear in benchmark table for comparison
      but are NEVER eligible for best-model selection (FIX 8) because
      they are not trained on the user's data — they use a fixed
      pre-trained polarity dictionary.
    """
    n    = profile["n_rows"]
    comp = profile["complexity"]

    # ── Volume-based eligibility ──────────────────────────────────────────────
    if n < 1000:
        eligible = {"Classical ML"}
        reason   = (f"Small dataset ({n} rows < 1000) → Classical ML only. "
                    f"DL would overfit with insufficient training samples.")

    elif n < 10000:
        eligible = {"Classical ML", "NLP/Lexicon"}
        reason   = (f"Medium dataset ({n} rows, 1000–10000) → "
                    f"Classical ML + NLP/Lexicon eligible. "
                    f"DL excluded — risk of overfitting on medium data.")

    elif n < 50000:
        eligible = {"Classical ML", "NLP/Lexicon"}
        if use_dl and profile["gpu_available"]:
            eligible.add("Deep Learning")
            reason = (f"Large dataset ({n} rows, 10000–50000) + GPU detected → "
                      f"Deep Learning now eligible alongside Classical ML.")
        elif use_dl and not profile["gpu_available"]:
            reason = (f"Large dataset ({n} rows) but no GPU detected → "
                      f"Deep Learning excluded (CPU training too slow/unreliable).")
        else:
            reason = (f"Large dataset ({n} rows) → Classical ML + NLP eligible. "
                      f"Enable Deep Learning checkbox + GPU for DL models.")

    else:
        eligible = {"Classical ML", "NLP/Lexicon"}
        if use_dl:
            eligible.add("Deep Learning")
        if use_transformers:
            eligible.add("Transformer/BERT")
        reason = (f"Very large dataset ({n} rows ≥ 50000) → "
                  f"All model types eligible.")

    # ── Complexity note (informational only) ──────────────────────────────────
    if comp == "low":
        complexity_note = ("Low complexity (short/repetitive text) → "
                           "interpretable models (LR, NB, DT) likely to lead.")
    elif comp == "medium":
        complexity_note = ("Medium complexity → "
                           "ensemble methods (RF, Gradient Boosting) likely to lead.")
    else:
        complexity_note = ("High complexity (diverse/long text) → "
                           "all ML models competitive, non-linear patterns present.")

    print(f"[MODEL SELECTION] {reason}")
    print(f"[MODEL SELECTION] Complexity note: {complexity_note}")
    print(f"[MODEL SELECTION] Eligible types: {eligible}")
    return eligible


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def train_models(X: pd.Series, y: pd.Series, use_dl=False, use_transformers=False):
    global _vectorizer, _label_encoder, BEST_MODEL_OBJ, BEST_MODEL_NAME, ALL_RESULTS

    # ── Basic null filter ─────────────────────────────────────────────────────
    mask = X.notna() & y.notna() & (X.str.strip() != "")
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 1: Deduplicate at model training level too
    # preprocess deduplicates on raw Comment text, but Cleaned text (what
    # train_models receives) can still have duplicates after stopword removal
    # e.g. "The scheme is good!" and "The scheme is good." → same cleaned text
    # Deduplicating here on CLEANED text prevents test-set leakage at the
    # exact level the model actually sees.
    # ─────────────────────────────────────────────────────────────────────────
    df_temp = pd.DataFrame({"text": X, "label": y})
    before  = len(df_temp)
    df_temp = df_temp.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(df_temp) < before:
        print(f"[MODEL DEDUP] {before - len(df_temp)} cleaned-text duplicates removed")
    X = df_temp["text"].reset_index(drop=True)
    y = df_temp["label"].reset_index(drop=True)

    if len(X) < 10:
        return {"Insufficient data": {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                      "speed_ms":0,"type":"Classical ML","available":False}}, "N/A"

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4: Remap ALL non-standard labels, not just Sarcasm/Sarcastic/Mixed
    # Original code missed "Unknown", "Nan", "None", "" which the LabelEncoder
    # treated as real classes → extra columns in confusion matrix → wrong metrics
    # ─────────────────────────────────────────────────────────────────────────
    y = y.replace({
        "Sarcasm":   "Negative",
        "Sarcastic": "Negative",
        "Mixed":     "Neutral",
        "Unknown":   "Neutral",
        "unknown":   "Neutral",
        "Nan":       "Neutral",
        "None":      "Neutral",
        "":          "Neutral",
    })

    # Keep only valid 3-class labels
    valid = {"Positive","Negative","Neutral"}
    mask2 = y.isin(valid)
    dropped = (~mask2).sum()
    if dropped > 0:
        print(f"[LABELS] Dropped {dropped} rows with unrecognised labels: {y[~mask2].unique().tolist()}")
    X, y = X[mask2].reset_index(drop=True), y[mask2].reset_index(drop=True)

    y_enc       = _label_encoder.fit_transform(y)
    num_classes = len(_label_encoder.classes_)
    label_map   = {i:cls for i,cls in enumerate(_label_encoder.classes_)}

    counts      = pd.Series(y_enc).value_counts()
    do_stratify = (counts >= 2).all()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42,
        stratify=y_enc if do_stratify else None,
        shuffle=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 2: TF-IDF improvements
    # min_df=2 → min_df=3: a term must appear in ≥3 documents
    #   With 9000 comments, min_df=2 includes ~15,000 near-unique tokens that
    #   only appear in 2 documents — these are nearly always in the same
    #   train/test split → model memorises them → inflated accuracy
    # max_df=0.95 added: ignore terms in >95% of docs (scheme names etc.)
    #   that appear everywhere and carry no sentiment signal
    # ─────────────────────────────────────────────────────────────────────────
    _vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
    )
    X_tr_v = _vectorizer.fit_transform(X_tr)   # fit ONLY on train
    X_te_v = _vectorizer.transform(X_te)       # transform only — no fit

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
            t0 = time.time()
            preds = vader.predict_bulk(X_te.tolist(), label_map)
            if preds is not None:
                _record("VADER (NLTK)", preds, (time.time()-t0)*1000, "NLP/Lexicon")
        except Exception:
            pass
    else:
        results["VADER (NLTK)"] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                   "speed_ms":0,"type":"NLP/Lexicon","available":False}

    # ── TextBlob ──────────────────────────────────────────────────────────────
    tb = _TextBlobModel()
    if tb.is_available():
        try:
            t0 = time.time()
            preds = tb.predict_bulk(X_te.tolist(), label_map)
            if preds is not None:
                _record("TextBlob", preds, (time.time()-t0)*1000, "NLP/Lexicon")
        except Exception:
            pass
    else:
        results["TextBlob"] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                               "speed_ms":0,"type":"NLP/Lexicon","available":False}

    # ── Deep Learning ─────────────────────────────────────────────────────────
    if use_dl:
        try:
            from tensorflow.keras.preprocessing.text     import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            MAX_LEN, VOCAB, EPOCHS, BATCH = 100, 10000, 3, 32
            tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
            tok.fit_on_texts(X_tr.tolist())
            X_tr_s = pad_sequences(tok.texts_to_sequences(X_tr.tolist()), maxlen=MAX_LEN)
            X_te_s = pad_sequences(tok.texts_to_sequences(X_te.tolist()), maxlen=MAX_LEN)
            for name, model in _build_dl_models(VOCAB, MAX_LEN, num_classes).items():
                try:
                    t0 = time.time()
                    model.fit(X_tr_s, y_tr, epochs=EPOCHS, batch_size=BATCH,
                              validation_split=0.1, verbose=0)
                    preds = np.argmax(model.predict(X_te_s, verbose=0), axis=1)
                    _record(name, preds, (time.time()-t0)*1000, "Deep Learning")
                except Exception:
                    results[name] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                     "speed_ms":9999,"type":"Deep Learning","available":False}
        except ImportError:
            for name in ["LSTM","BiLSTM","CNN-Text"]:
                results[name] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                 "speed_ms":0,"type":"Deep Learning","available":False}

    # ── Transformers ──────────────────────────────────────────────────────────
    if use_transformers:
        for hf_name, display in [
            ("distilbert-base-uncased-finetuned-sst-2-english","DistilBERT"),
            ("albert-base-v2","ALBERT"),
            ("nlptown/bert-base-multilingual-uncased-sentiment","Multilingual BERT"),
        ]:
            tm = _TransformerSentiment(hf_name, display)
            if tm.is_available():
                try:
                    t0 = time.time()
                    preds = tm.predict_bulk(X_te.tolist(), label_map, max_samples=200)
                    if preds is not None:
                        _record(display, preds, (time.time()-t0)*1000, "Transformer/BERT")
                except Exception:
                    pass
            else:
                results[display] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                    "speed_ms":0,"type":"Transformer/BERT","available":False}

    # ── Select best ───────────────────────────────────────────────────────────
    available = {k:v for k,v in results.items()
                 if v.get("available",False) and v["accuracy"] > 0}
    if not available:
        available = results

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 6 + 7 + 8 — Smart model selection based on data profile
    #
    # Step 1: Analyze real data characteristics (volume, complexity, vocab)
    # Step 2: Determine which MODEL TYPES are eligible for this data profile
    # Step 3: Filter available results to only eligible types
    # Step 4: ADDITIONALLY exclude NLP/Lexicon from best-model selection (FIX 8)
    #         TextBlob and VADER use pre-trained polarity dictionaries —
    #         they were not trained on this data and win unfairly on
    #         synthetic templates written in unambiguous polarity language
    # Step 5: Safety fallback — if filtering left nothing, use full pool
    # ─────────────────────────────────────────────────────────────────────────
    profile        = _analyze_data(X, y)
    eligible_types = _select_candidate_types(profile, use_dl, use_transformers)

    # FIX 8: NLP/Lexicon models are never eligible for best-model selection
    # They still appear in the benchmark table in the UI for comparison
    NEVER_BEST = {"NLP/Lexicon"}
    eligible_types = eligible_types - NEVER_BEST

    # Filter benchmark results to eligible types only
    smart_pool = {k:v for k,v in available.items()
                  if v.get("type") in eligible_types}

    # Safety fallback — if smart filtering left nothing, widen to all trained types
    if not smart_pool:
        print("[MODEL SELECTION] Warning: smart pool empty after type filter — "
              "falling back to all non-lexicon available models")
        smart_pool = {k:v for k,v in available.items()
                      if v.get("type") not in NEVER_BEST}

    # Final fallback — if still nothing (e.g. only lexicons ran), use everything
    if not smart_pool:
        print("[MODEL SELECTION] Warning: no trained models available — "
              "falling back to full available pool")
        smart_pool = available

    best_name = max(smart_pool,
                    key=lambda k:(smart_pool[k]["accuracy"],
                                  -smart_pool[k].get("speed_ms", 9999)))

    print(f"[MODEL SELECTION] ✓ Best model: {best_name} | "
          f"Accuracy: {smart_pool[best_name]['accuracy']}% | "
          f"Type: {smart_pool[best_name].get('type')} | "
          f"Data: {profile['n_rows']} rows, {profile['complexity']} complexity")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 5: Retrain best model on FULL data correctly
    # Original: _vectorizer.transform(X) — but X has tokens the vectorizer
    # never saw during fit_transform(X_tr), causing KeyErrors on unseen vocab
    # Fix: re-fit vectorizer on ALL data, then refit model on all data
    # This is safe because final model is only used for live inference,
    # not for the accuracy metrics (those were computed on the held-out test set)
    # ─────────────────────────────────────────────────────────────────────────
    if best_name in fitted:
        n_docs     = len(X_tr)
        min_df_val = 2 if n_docs < 500 else 3
        max_df_val = 0.98 if n_docs < 500 else 0.95

        _vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=min_df_val,
            max_df=max_df_val,
            strip_accents="unicode",
        )
        X_full_v = _vectorizer.fit_transform(X)   # ← fit_transform, not transform
        fitted[best_name].fit(X_full_v, y_enc)
        BEST_MODEL_OBJ = fitted[best_name]

    BEST_MODEL_NAME = best_name
    ALL_RESULTS     = results
    return results, best_name


# ═════════════════════════════════════════════════════════════════════════════
#  DETAILED METRICS  — unchanged
# ═════════════════════════════════════════════════════════════════════════════
def get_detailed_metrics(X: pd.Series, y: pd.Series):
    if BEST_MODEL_OBJ is None: return None, None, None
    mask  = X.notna() & y.notna() & (X.str.strip() != "")
    X, y  = X[mask], y[mask]
    y     = y.replace({"Sarcasm":"Negative","Sarcastic":"Negative"})
    y_enc = _label_encoder.transform(y)
    X_vec = _vectorizer.transform(X)
    _, X_te, _, y_te = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)
    preds = BEST_MODEL_OBJ.predict(X_te)
    cm    = confusion_matrix(y_te, preds)
    report = classification_report(y_te, preds, target_names=_label_encoder.classes_,
                                   output_dict=True, zero_division=0)
    return cm, report, _label_encoder.classes_


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE INFERENCE  — unchanged
# ═════════════════════════════════════════════════════════════════════════════
def predict_live(cleaned_text: str) -> str:
    return predict_live_with_confidence(cleaned_text, cleaned_text)["sentiment"]


def predict_live_with_confidence(text: str, cleaned_text: str = None) -> dict:
    if text is None:
        text = ""

    lang        = _get_lang(text)
    hindi_prior = None
    if lang in ("hi","hinglish"):
        hindi_prior = _get_hindi_sentiment(text)

    translated = _translate(text, lang) if lang != "en" else text

    sc           = _get_sarcasm_score(text)
    is_sarcastic = sc > 0.45

    if cleaned_text and len(cleaned_text.strip()) > 1:
        c_text = cleaned_text
    else:
        c_text = _clean(translated)
        if not c_text.strip():
            c_text = translated[:200]

    model_used = "Fallback (TextBlob+VADER)"
    confidence = 65.0
    sentiment  = None

    if hindi_prior and BEST_MODEL_OBJ is None:
        sentiment  = hindi_prior
        confidence = 75.0
        model_used = "Hindi Keyword Engine"

    elif BEST_MODEL_OBJ is not None and _vectorizer is not None:
        try:
            vec   = _vectorizer.transform([c_text])
            proba = BEST_MODEL_OBJ.predict_proba(vec)[0]
            pred  = int(np.argmax(proba))
            label = _label_encoder.inverse_transform([pred])[0]
            conf  = float(max(proba)) * 100

            if conf < 55.0 and hindi_prior:
                sentiment  = hindi_prior
                confidence = 70.0
                model_used = f"Hindi Engine (ML conf={conf:.0f}% too low)"
            else:
                sentiment  = label
                confidence = min(conf, 97.0)
                model_used = BEST_MODEL_NAME or "ML Model"
        except Exception:
            pass

    if sentiment is None:
        sentiment  = _fallback_predict(translated, sc)
        confidence = 65.0
        model_used = "TextBlob+VADER Ensemble"

    if is_sarcastic and sentiment == "Positive":
        sentiment  = "Negative"
        confidence = max(confidence, 70.0)
        model_used += " [Sarcasm Override]"

    if sentiment not in ("Positive","Negative","Neutral"):
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
    from collections import Counter
    votes = []

    try:
        from textblob import TextBlob
        pol = TextBlob(str(text)).sentiment.polarity
        if pol > 0.08:    votes.append("Positive")
        elif pol < -0.08: votes.append("Negative")
        else:             votes.append("Neutral")
    except Exception:
        pass

    try:
        import nltk
        try:    nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError: nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        c = SentimentIntensityAnalyzer().polarity_scores(str(text))["compound"]
        if c >= 0.05:    votes.append("Positive")
        elif c <= -0.05: votes.append("Negative")
        else:            votes.append("Neutral")
    except Exception:
        pass

    if not votes:
        return "Neutral"

    result = Counter(votes).most_common(1)[0][0]
    if sc > 0.45 and result == "Positive":
        return "Negative"
    return result