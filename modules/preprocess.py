"""
modules/preprocess.py
─────────────────────
Handles:
  • Text cleaning / normalisation
  • Language detection  (langdetect)
  • Translation to English (deep-translator)
  • Sarcasm signal detection
  • Sentiment labelling via TextBlob (fallback)
"""

import re
import pandas as pd

# ── Optional imports with graceful fallback ───────────────────────────────
try:
    from langdetect import detect as _detect_lang
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
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = set()

# ── Sarcasm signals ────────────────────────────────────────────────────────
SARCASM_EMOJIS  = ['🙄', '😒', '😏', '🤡', '💀']
SARCASM_PHRASES = [
    'oh great', 'wow amazing', 'yeah right', 'sure sure', 'oh sure',
    'obviously', 'brilliant idea', 'great job government',
    'another amazing', 'haan bilkul', 'wah wah', 'as if',
    'totally working', 'best scheme ever lol',
]


# ─────────────────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    if not LANGDETECT_OK or not text or len(str(text).strip()) < 5:
        return 'en'
    try:
        return _detect_lang(str(text))
    except Exception:
        return 'en'


def translate_to_english(text: str, src_lang: str = 'auto') -> str:
    if not TRANSLATOR_OK or src_lang == 'en':
        return text
    try:
        result = GoogleTranslator(source=src_lang, target='en').translate(str(text))
        return result if result else text
    except Exception:
        return text


def detect_sarcasm(text: str) -> bool:
    text_lower = str(text).lower()
    for e in SARCASM_EMOJIS:
        if e in text:
            return True
    for phrase in SARCASM_PHRASES:
        if phrase in text_lower:
            return True
    # Over-enthusiastic positivity flag
    if text.count('!') >= 2:
        for word in ['great', 'amazing', 'wonderful', 'excellent', 'fantastic']:
            if word in text_lower:
                return True
    return False


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def get_textblob_sentiment(text: str) -> str:
    if not TEXTBLOB_OK:
        return 'Neutral'
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    return 'Neutral'


def get_sentiment(text: str) -> str:
    """Live single-comment sentiment — full pipeline."""
    lang    = detect_language(text)
    en_text = translate_to_english(text, lang)
    cleaned = clean_text(en_text)
    result  = get_textblob_sentiment(cleaned)
    if result == 'Positive' and detect_sarcasm(text):
        result = 'Negative (Sarcasm Detected)'
    return result


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rename columns to standard names if they exist
    col_map = {
        'Comment': 'text',
        'comment': 'text',
        'Source':  'source',
        'Scheme':  'scheme',
        'Sentiment': 'sentiment',
        'Language': 'language',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Language detection
    if 'language' in df.columns:
        df['Lang'] = df['language']
    else:
        df['Lang'] = df['text'].apply(detect_language)

    # Translate to English
    df['Translated'] = df.apply(
        lambda r: translate_to_english(r['text'], r['Lang'])
        if r['Lang'] != 'en' else r['text'], axis=1
    )

    # Sarcasm flag
    df['IsSarcasm'] = df['text'].apply(detect_sarcasm)

    # Clean text
    df['Cleaned'] = df['Translated'].apply(clean_text)

    # Sentiment label
    if 'sentiment' in df.columns:
        df['Sentiment'] = df['sentiment'].str.capitalize()
    else:
        df['Sentiment'] = df['Translated'].apply(get_textblob_sentiment)

    # Sarcasm overrides Positive → Negative
    df.loc[df['IsSarcasm'] & (df['Sentiment'] == 'Positive'), 'Sentiment'] = 'Negative'

    # Keep Source and Scheme columns
    df['Source'] = df['source'] if 'source' in df.columns else 'Unknown'
    df['Scheme'] = df['scheme'] if 'scheme' in df.columns else 'General'

    return df
