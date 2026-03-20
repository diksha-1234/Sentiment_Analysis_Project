"""
modules/model.py — Pulse Sentiment AI · Adaptive Model Engine
══════════════════════════════════════════════════════════════
Models included:

── CLASSICAL ML ──────────────────────────────────────────────
  • Naive Bayes (MultinomialNB)        — fast, text baseline
  • Logistic Regression                — strong linear classifier
  • SVM (LinearSVC + calibration)      — best for sparse text
  • Random Forest                      — ensemble, handles noise
  • Gradient Boosting (XGBoost style)  — strong non-linear
  • K-Nearest Neighbours               — instance-based
  • Decision Tree                      — interpretable

── NLP / LEXICON ─────────────────────────────────────────────
  • TextBlob                           — rule-based NLP sentiment
  • VADER (NLTK)                       — social media optimised
  • Pattern (if available)             — linguistic pattern matching

── DEEP LEARNING ─────────────────────────────────────────────
  • LSTM (Keras/TF)                    — sequence model
  • BiLSTM (Keras/TF)                  — bidirectional sequence
  • CNN-Text (Keras/TF)                — n-gram feature learning

── TRANSFORMER / BERT ────────────────────────────────────────
  • ALBERT (albert-base-v2)            — lightweight BERT variant
  • DistilBERT                         — fast BERT distillation
  • Multilingual BERT                  — handles Hindi + English

Gap addressed: Random algorithm selection. System trains ALL models,
benchmarks them, and selects the BEST by accuracy. If two models
tie on accuracy, the FASTER one wins.
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
from sklearn.metrics                 import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing           import LabelEncoder

# ── Global state ──────────────────────────────────────────────────────────────
_vectorizer    = None
_label_encoder = LabelEncoder()
BEST_MODEL_OBJ  = None
BEST_MODEL_NAME = None
ALL_RESULTS     = {}   # {name: {accuracy, f1, precision, recall, speed_ms, type}}


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSICAL ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
def _build_classical_models():
    return {
        "Naive Bayes":           MultinomialNB(alpha=0.5),
        "Logistic Regression":   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "SVM (LinearSVC)":       CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0)),
        "Random Forest":         RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1, random_state=42),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "K-Nearest Neighbours":  KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Decision Tree":         DecisionTreeClassifier(max_depth=15, random_state=42),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NLP / LEXICON MODELS
# ─────────────────────────────────────────────────────────────────────────────
class _VADERModel:
    """VADER — NLTK Valence Aware Dictionary and sEntiment Reasoner."""
    name = "VADER (NLTK)"
    model_type = "NLP/Lexicon"

    def __init__(self):
        self.sia = None
        self._available = False
        try:
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
            self._available = True
        except Exception:
            pass

    def predict_proba_single(self, text):
        if not self._available:
            return None
        scores = self.sia.polarity_scores(text)
        # Map to [neg, neu, pos] probabilities
        return [scores["neg"], scores["neu"], scores["pos"]]

    def predict_bulk(self, texts, label_map):
        """label_map: {0:'Negative',1:'Neutral',2:'Positive'}"""
        if not self._available:
            return None
        preds = []
        reverse = {v: k for k, v in label_map.items()}
        for t in texts:
            scores = self.sia.polarity_scores(str(t))
            c = scores["compound"]
            if c >= 0.05:
                preds.append(reverse.get("Positive", 2))
            elif c <= -0.05:
                preds.append(reverse.get("Negative", 0))
            else:
                preds.append(reverse.get("Neutral", 1))
        return np.array(preds)

    def is_available(self):
        return self._available


class _TextBlobModel:
    """TextBlob — pattern-based NLP sentiment."""
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
        if not self._available:
            return None
        reverse = {v: k for k, v in label_map.items()}
        preds = []
        for t in texts:
            try:
                pol = self._tb(str(t)).sentiment.polarity
                if pol > 0.05:
                    preds.append(reverse.get("Positive", 2))
                elif pol < -0.05:
                    preds.append(reverse.get("Negative", 0))
                else:
                    preds.append(reverse.get("Neutral", 1))
            except Exception:
                preds.append(reverse.get("Neutral", 1))
        return np.array(preds)

    def is_available(self):
        return self._available


# ─────────────────────────────────────────────────────────────────────────────
#  DEEP LEARNING MODELS (Keras / TensorFlow)
# ─────────────────────────────────────────────────────────────────────────────
def _build_dl_models(vocab_size, max_len, num_classes):
    """Build LSTM, BiLSTM, CNN text models. Returns dict or empty if TF unavailable."""
    models = {}
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, LSTM, Bidirectional, Conv1D,
            GlobalMaxPooling1D, Dense, Dropout, SpatialDropout1D
        )
        tf.get_logger().setLevel("ERROR")

        embed_dim = 64

        # ── LSTM ──────────────────────────────────────────────────────────────
        lstm_model = Sequential([
            Embedding(vocab_size, embed_dim, input_length=max_len),
            SpatialDropout1D(0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ], name="LSTM")
        lstm_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        models["LSTM"] = lstm_model

        # ── BiLSTM ────────────────────────────────────────────────────────────
        bilstm_model = Sequential([
            Embedding(vocab_size, embed_dim, input_length=max_len),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ], name="BiLSTM")
        bilstm_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        models["BiLSTM"] = bilstm_model

        # ── CNN-Text ──────────────────────────────────────────────────────────
        cnn_model = Sequential([
            Embedding(vocab_size, embed_dim, input_length=max_len),
            SpatialDropout1D(0.2),
            Conv1D(128, 5, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ], name="CNN-Text")
        cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        models["CNN-Text"] = cnn_model

    except ImportError:
        pass   # TF not installed — skip DL models gracefully
    return models


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORMER / BERT MODELS
# ─────────────────────────────────────────────────────────────────────────────
class _TransformerSentiment:
    """
    Wrapper for HuggingFace transformer models.
    Uses zero-shot / pre-trained inference — no fine-tuning needed.
    Falls back gracefully if transformers not installed.
    """
    def __init__(self, model_name, display_name):
        self.model_name   = model_name
        self.display_name = display_name
        self._pipeline    = None
        self._available   = False
        self.model_type   = "Transformer/BERT"

    def _load(self):
        if self._available or self._pipeline is not None:
            return True
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
                max_length=128,
                device=-1,   # CPU
            )
            self._available = True
            return True
        except Exception:
            return False

    def predict_bulk(self, texts, label_map, max_samples=200):
        """Run inference on up to max_samples texts."""
        if not self._load():
            return None
        reverse = {v: k for k, v in label_map.items()}
        preds = []
        texts = [str(t)[:512] for t in texts[:max_samples]]
        try:
            results = self._pipeline(texts, batch_size=16)
            for r in results:
                label_raw = r["label"].upper()
                if "POS" in label_raw or label_raw in ("LABEL_2", "5 STARS", "4 STARS"):
                    preds.append(reverse.get("Positive", 2))
                elif "NEG" in label_raw or label_raw in ("LABEL_0", "1 STAR", "2 STARS"):
                    preds.append(reverse.get("Negative", 0))
                else:
                    preds.append(reverse.get("Neutral", 1))
        except Exception:
            return None
        return np.array(preds)

    def is_available(self):
        return self._load()


# ─────────────────────────────────────────────────────────────────────────────
#  SARCASM DETECTION — enhanced
# ─────────────────────────────────────────────────────────────────────────────
import re

_SARCASM_PATTERNS = [
    r"\boh\s+(wow|great|sure|brilliant|perfect|fantastic|amazing|excellent)\b",
    r"\b(great|excellent|wonderful|brilliant|amazing)\b.{0,40}\b(another|again|really|definitely)\b",
    r"\b(sure|yeah|right|totally|absolutely|obviously)\b.{0,30}\b(work|happen|help|reach|benefit)\b",
    r"as\s+if\s+it\s+(ever|will|would|could)",
    r"\bdefinitely\b.{0,30}\b(not|never|won.t|can.t)\b",
    r"what\s+a\s+(joke|surprise|shock|shocker)",
    r"\bthank\s+(you|god).{0,30}\b(nothing|zero|nobody|no one)\b",
    r"हाँ\s+हाँ|बिलकुल\s+सही|ज़रूर\s+होगा|वाह\s+क्या\s+बात",
]
_SARCASM_EMOJIS = ["🙄", "😒", "😏", "🤨", "🙃", "😤", "😑", "🤣", "😂"]
_SARCASM_MARKERS = [
    "as if", "yeah right", "oh sure", "totally works", "great job",
    "wow amazing", "definitely helping", "so helpful NOT",
    "working perfectly", "sure it did", "oh brilliant",
]

def detect_sarcasm_advanced(text: str) -> float:
    """
    Returns a sarcasm confidence score 0.0 → 1.0.
    > 0.5 = sarcastic
    """
    score = 0.0
    t = text.lower()

    # Emoji signals — strong indicator
    for emoji in _SARCASM_EMOJIS:
        if emoji in text:
            score += 0.4
            break

    # Pattern matching
    for pat in _SARCASM_PATTERNS:
        if re.search(pat, t):
            score += 0.3
            break

    # Marker phrases
    for marker in _SARCASM_MARKERS:
        if marker in t:
            score += 0.25
            break

    # Excessive capitalisation (shouting) — mild signal
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.3:
        score += 0.15

    # Exclamation overuse
    if text.count("!") >= 2:
        score += 0.1

    # Punctuation irony: "great!!!" or "amazing..."
    if re.search(r"(great|amazing|excellent|wonderful|fantastic)[!.]{2,}", t):
        score += 0.2

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_models(X: pd.Series, y: pd.Series, use_dl=False, use_transformers=False):
    """
    Train ALL model families.
    Returns: results, best_name

    results = {
      model_name: {
        accuracy, f1, precision, recall,
        speed_ms, type, available
      }
    }
    """
    global _vectorizer, _label_encoder, BEST_MODEL_OBJ, BEST_MODEL_NAME, ALL_RESULTS

    # ── Prep data ─────────────────────────────────────────────────────────────
    mask = X.notna() & y.notna() & (X.str.strip() != "")
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    if len(X) < 10:
        return {"Insufficient data": 0.0}, "N/A"

    y_enc = _label_encoder.fit_transform(y)
    num_classes = len(_label_encoder.classes_)
    label_map = {i: cls for i, cls in enumerate(_label_encoder.classes_)}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42,
        stratify=y_enc if len(set(y_enc)) > 1 else None
    )

    # ── TF-IDF vectoriser ─────────────────────────────────────────────────────
    _vectorizer = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2,
    )
    X_tr_v = _vectorizer.fit_transform(X_tr)
    X_te_v = _vectorizer.transform(X_te)

    results = {}
    fitted  = {}

    def _record(name, preds, duration_ms, model_type, model_obj=None):
        if preds is None or len(preds) == 0:
            return
        acc  = round(accuracy_score(y_te[:len(preds)], preds) * 100, 2)
        f1   = round(f1_score(y_te[:len(preds)], preds, average="weighted", zero_division=0) * 100, 2)
        prec = round(precision_score(y_te[:len(preds)], preds, average="weighted", zero_division=0) * 100, 2)
        rec  = round(recall_score(y_te[:len(preds)], preds, average="weighted", zero_division=0) * 100, 2)
        results[name] = {
            "accuracy":   acc,
            "f1":         f1,
            "precision":  prec,
            "recall":     rec,
            "speed_ms":   round(duration_ms, 1),
            "type":       model_type,
            "available":  True,
        }
        if model_obj is not None:
            fitted[name] = model_obj

    # ── 1. Classical ML ───────────────────────────────────────────────────────
    classical = _build_classical_models()
    for name, clf in classical.items():
        try:
            t0 = time.time()
            clf.fit(X_tr_v, y_tr)
            preds = clf.predict(X_te_v)
            ms = (time.time() - t0) * 1000
            _record(name, preds, ms, "Classical ML", clf)
        except Exception as e:
            results[name] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                             "speed_ms":9999,"type":"Classical ML","available":False}

    # ── 2. VADER ──────────────────────────────────────────────────────────────
    vader = _VADERModel()
    if vader.is_available():
        try:
            t0 = time.time()
            preds = vader.predict_bulk(X_te.tolist(), label_map)
            ms = (time.time() - t0) * 1000
            if preds is not None:
                _record("VADER (NLTK)", preds, ms, "NLP/Lexicon")
        except Exception:
            pass
    else:
        results["VADER (NLTK)"] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                   "speed_ms":0,"type":"NLP/Lexicon","available":False}

    # ── 3. TextBlob ───────────────────────────────────────────────────────────
    tb = _TextBlobModel()
    if tb.is_available():
        try:
            t0 = time.time()
            preds = tb.predict_bulk(X_te.tolist(), label_map)
            ms = (time.time() - t0) * 1000
            if preds is not None:
                _record("TextBlob", preds, ms, "NLP/Lexicon")
        except Exception:
            pass
    else:
        results["TextBlob"] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                               "speed_ms":0,"type":"NLP/Lexicon","available":False}

    # ── 4. Deep Learning (optional — slow on CPU) ────────────────────────────
    if use_dl:
        try:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            MAX_LEN   = 100
            VOCAB     = 10000
            EPOCHS    = 3
            BATCH     = 32

            tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
            tok.fit_on_texts(X_tr.tolist())
            X_tr_seq = pad_sequences(tok.texts_to_sequences(X_tr.tolist()), maxlen=MAX_LEN)
            X_te_seq = pad_sequences(tok.texts_to_sequences(X_te.tolist()), maxlen=MAX_LEN)

            dl_models = _build_dl_models(VOCAB, MAX_LEN, num_classes)
            for name, model in dl_models.items():
                try:
                    t0 = time.time()
                    model.fit(X_tr_seq, y_tr, epochs=EPOCHS, batch_size=BATCH,
                              validation_split=0.1, verbose=0)
                    proba = model.predict(X_te_seq, verbose=0)
                    preds = np.argmax(proba, axis=1)
                    ms = (time.time() - t0) * 1000
                    _record(name, preds, ms, "Deep Learning")
                except Exception:
                    results[name] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                     "speed_ms":9999,"type":"Deep Learning","available":False}
        except ImportError:
            for name in ["LSTM","BiLSTM","CNN-Text"]:
                results[name] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                 "speed_ms":0,"type":"Deep Learning","available":False}

    # ── 5. Transformers (optional — heavy) ───────────────────────────────────
    if use_transformers:
        transformer_models = [
            ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT"),
            ("albert-base-v2",                                   "ALBERT"),
            ("nlptown/bert-base-multilingual-uncased-sentiment", "Multilingual BERT"),
        ]
        for hf_name, display in transformer_models:
            t_model = _TransformerSentiment(hf_name, display)
            if t_model.is_available():
                try:
                    t0 = time.time()
                    preds = t_model.predict_bulk(X_te.tolist(), label_map, max_samples=200)
                    ms = (time.time() - t0) * 1000
                    if preds is not None:
                        _record(display, preds, ms, "Transformer/BERT")
                except Exception:
                    pass
            else:
                results[display] = {"accuracy":0,"f1":0,"precision":0,"recall":0,
                                    "speed_ms":0,"type":"Transformer/BERT","available":False}

    # ── Select best: highest accuracy; tie → fastest ──────────────────────────
    available = {k: v for k, v in results.items() if v.get("available", False) and v["accuracy"] > 0}
    if not available:
        available = results

    best_name = max(
        available,
        key=lambda k: (available[k]["accuracy"], -available[k].get("speed_ms", 9999))
    )

    # Retrain best classical model on full data
    if best_name in fitted:
        X_full_v = _vectorizer.transform(X)
        fitted[best_name].fit(X_full_v, y_enc)
        BEST_MODEL_OBJ = fitted[best_name]
    BEST_MODEL_NAME = best_name
    ALL_RESULTS     = results

    return results, best_name


# ─────────────────────────────────────────────────────────────────────────────
#  DETAILED METRICS
# ─────────────────────────────────────────────────────────────────────────────
def get_detailed_metrics(X: pd.Series, y: pd.Series):
    global BEST_MODEL_OBJ, _vectorizer, _label_encoder
    if BEST_MODEL_OBJ is None:
        return None, None, None
    mask  = X.notna() & y.notna() & (X.str.strip() != "")
    X, y  = X[mask], y[mask]
    y_enc = _label_encoder.transform(y)
    X_vec = _vectorizer.transform(X)
    _, X_te, _, y_te = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)
    preds  = BEST_MODEL_OBJ.predict(X_te)
    cm     = confusion_matrix(y_te, preds)
    report = classification_report(y_te, preds,
                                   target_names=_label_encoder.classes_,
                                   output_dict=True, zero_division=0)
    return cm, report, _label_encoder.classes_


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_live(cleaned_text: str) -> str:
    global BEST_MODEL_OBJ, _vectorizer, _label_encoder
    if BEST_MODEL_OBJ is None:
        return "Model not trained yet"
    try:
        vec   = _vectorizer.transform([cleaned_text])
        label = BEST_MODEL_OBJ.predict(vec)[0]
        result = _label_encoder.inverse_transform([label])[0]
        # Apply sarcasm correction
        if detect_sarcasm_advanced(cleaned_text) > 0.5 and result == "Positive":
            return "Negative"
        return result
    except Exception:
        return "Neutral"


def predict_live_with_confidence(text: str, cleaned_text: str) -> dict:
    """Full prediction with sarcasm score and confidence."""
    global BEST_MODEL_OBJ, _vectorizer, _label_encoder

    sarcasm_score = detect_sarcasm_advanced(text)

    if BEST_MODEL_OBJ is None:
        # Fallback to TextBlob + VADER ensemble
        sentiment = _fallback_predict(cleaned_text, sarcasm_score)
        return {
            "sentiment":     sentiment,
            "confidence":    70.0,
            "sarcasm_score": round(sarcasm_score * 100, 1),
            "is_sarcastic":  sarcasm_score > 0.5,
            "model_used":    "TextBlob + VADER (fallback)",
        }

    try:
        vec   = _vectorizer.transform([cleaned_text])
        proba = BEST_MODEL_OBJ.predict_proba(vec)[0]
        pred  = np.argmax(proba)
        label = _label_encoder.inverse_transform([pred])[0]
        conf  = float(max(proba)) * 100

        # Sarcasm override
        if sarcasm_score > 0.5 and label == "Positive":
            label = "Negative"
            conf  = max(conf, 65.0)

        return {
            "sentiment":     label,
            "confidence":    round(conf, 1),
            "sarcasm_score": round(sarcasm_score * 100, 1),
            "is_sarcastic":  sarcasm_score > 0.5,
            "model_used":    BEST_MODEL_NAME or "Unknown",
        }
    except Exception:
        return {
            "sentiment":     "Neutral",
            "confidence":    50.0,
            "sarcasm_score": round(sarcasm_score * 100, 1),
            "is_sarcastic":  sarcasm_score > 0.5,
            "model_used":    "Fallback",
        }


def _fallback_predict(text: str, sarcasm_score: float = 0.0) -> str:
    """TextBlob + VADER ensemble when no ML model trained yet."""
    scores = []

    # TextBlob
    try:
        from textblob import TextBlob
        pol = TextBlob(text).sentiment.polarity
        if pol > 0.05:   scores.append("Positive")
        elif pol < -0.05: scores.append("Negative")
        else:            scores.append("Neutral")
    except Exception:
        pass

    # VADER
    try:
        import nltk
        try: nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError: nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        c = sia.polarity_scores(text)["compound"]
        if c >= 0.05:    scores.append("Positive")
        elif c <= -0.05: scores.append("Negative")
        else:            scores.append("Neutral")
    except Exception:
        pass

    if not scores:
        return "Neutral"

    # Majority vote
    from collections import Counter
    result = Counter(scores).most_common(1)[0][0]

    # Sarcasm override
    if sarcasm_score > 0.5 and result == "Positive":
        return "Negative"
    return result