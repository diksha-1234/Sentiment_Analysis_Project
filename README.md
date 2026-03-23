# 🧠 Pulse Sentiment AI

> **Multilingual Multi-Source Sentiment Analysis Framework for Indian Government Schemes Using Adaptive Model Selection**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 📌 Overview

**Pulse Sentiment AI** is a final year research project that analyses public sentiment toward Indian government welfare schemes (PMAY, PM Kisan, Ayushman Bharat, and 37 more) using real data fetched from multiple platforms.

The system addresses **5 critical research gaps** that existing sentiment analysis papers fail to solve:

| # | Research Gap | How Pulse Solves It |
|---|---|---|
| 01 | English-only analysis | Multilingual pipeline — Hindi, Hinglish, Tamil, Telugu, Bengali |
| 02 | Single platform bias | 4 source types — YouTube, NewsAPI, Google News RSS, Hindi News RSS |
| 03 | Random model selection | Adaptive engine profiles data and selects best model automatically |
| 04 | Binary classification only | 3-class — Positive / Neutral / Negative with per-class metrics |
| 05 | Sarcasm ignored | 30+ rule sarcasm engine with automatic positive→negative flip |

---

## 🚀 Live Demo

> **[Deploy your own instance — instructions below](#-deployment)**

---

## 📁 Project Structure

```
sentiment_analysis_project/
│
├── app.py                      # Main Streamlit application
│
├── modules/
│   ├── preprocess.py           # Language detection, translation, cleaning, sarcasm
│   ├── model.py                # ML training, smart model selection, live probe
│   └── scraper.py              # Data fetchers — YouTube, NewsAPI, Google News, Hindi RSS
│
├── auth/
│   ├── auth_manager.py         # Login, signup, Google OAuth — Supabase-backed
│   └── users.json              # Local dev user store (not used in deployment)
│
├── data/
│   ├── storage.py              # Supabase ↔ CSV abstraction layer + translation cache
│   └── data.csv                # Local development dataset (not used in deployment)
│
├── requirements.txt            # Python dependencies
└── .streamlit/
    └── secrets.toml            # API keys (never commit this)
```

---

## ✨ Features

### 🔍 Analysis Tab
- Select from **40 Indian government schemes** or analyse all at once
- Trains **7 Classical ML models** simultaneously and benchmarks them
- **Adaptive model selection** — picks the best model based on data profile (size, sarcasm ratio, language diversity)
- Visualisations: sentiment pie chart, platform bar chart, language breakdown, sarcasm detection, Platform × Sentiment heatmap
- Dataset preview sorted by source with source distribution summary

### ⚡ Live Probe Tab
- Real-time sentiment analysis of any text in any language
- **5-way ensemble voting** — ML model + TextBlob + VADER + Domain keywords + Sarcasm engine
- Displays language detected, sarcasm score, confidence, translation
- Works even before running full analysis (fallback mode)

### 🗂️ Platforms Tab
- Cross-platform comparison table and heatmap
- Source row count chart
- Shows all fetched sources: YouTube, News App, Google News, Dainik Bhaskar, Amar Ujala, NDTV Hindi, ABP Live, Navbharat Times, Jagran

### 🔒 Data Tab *(Admin only)*
- Fetch live data from all 4 source types
- Progress log during fetch
- Dataset status with live Supabase row counts
- Source distribution chart (current vs legacy sources)

### ℹ️ About Tab
- Full research context with all 5 gaps explained

---

## 🧠 How the ML Engine Works

### Models Trained
| Model | Type |
|---|---|
| Naive Bayes | Classical ML |
| Logistic Regression | Classical ML |
| SVM (LinearSVC) | Classical ML |
| Random Forest | Classical ML |
| Gradient Boosting | Classical ML |
| K-Nearest Neighbours | Classical ML |
| Decision Tree | Classical ML |
| VADER (NLTK) | NLP/Lexicon |
| TextBlob | NLP/Lexicon |

### Adaptive Selection Logic
The engine profiles your dataset before selecting:
- **Data volume** → decides eligible model types
- **Sarcasm ratio** → penalises TextBlob/VADER if high sarcasm detected
- **Language diversity** → penalises lexicon models if multilingual content
- **Avg comment length** → guides complexity assessment
- **Social media score** → adjusts VADER vs TextBlob preference

### Live Probe Ensemble
```
Input text
   ↓
Language detection → Hindi keyword prior (if hi/hinglish)
   ↓
Translation to English (3-level cache: memory → Supabase → API)
   ↓
Sarcasm scoring on ORIGINAL text (preserves emojis)
   ↓
Domain keyword scoring (handles indirect negatives, amount irony)
   ↓
ML model + TextBlob + VADER predictions
   ↓
5-way weighted ensemble vote
   ↓
Sarcasm override (if score > 45% and winner = Positive → flip to Negative)
   ↓
Final: Sentiment + Confidence + Model chain
```

---

## 🌐 Data Sources

| Source | Method | API Key Required |
|---|---|---|
| YouTube | Official Data API v3 | ✅ Yes |
| News App | NewsAPI.org | ✅ Yes |
| Google News | RSS Feed | ❌ No |
| Dainik Bhaskar | RSS Feed | ❌ No |
| Amar Ujala | RSS Feed | ❌ No |
| Navbharat Times | RSS Feed | ❌ No |
| Jagran | RSS Feed | ❌ No |
| NDTV Hindi | RSS Feed | ❌ No |
| ABP Live | RSS Feed | ❌ No |

---

## ⚙️ Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sentiment_analysis_project.git
cd sentiment_analysis_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root:
```env
YOUTUBE_API_KEY=your_youtube_api_key
NEWS_API_KEY=your_newsapi_key

# Optional — Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
REDIRECT_URI=http://localhost:8501

# Optional — Supabase (leave empty to use local CSV)
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

### 5. Generate seed data (first time only)
```bash
python data/generate_data.py
```

### 6. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

**Demo login:** username `admin` / password `1234`

---

## ☁️ Deployment (Streamlit Cloud)

### Step 1 — Set up Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run:

```sql
-- Sentiment data table
CREATE TABLE sentiment_data (
    id         BIGSERIAL PRIMARY KEY,
    scheme     TEXT,
    source     TEXT,
    language   TEXT DEFAULT 'en',
    comment    TEXT UNIQUE,
    sentiment  TEXT DEFAULT 'Neutral',
    translated TEXT DEFAULT ''
);

-- Users table
CREATE TABLE users (
    username   TEXT PRIMARY KEY,
    name       TEXT,
    email      TEXT,
    password   TEXT,
    role       TEXT DEFAULT 'user',
    joined     TEXT,
    avatar     TEXT DEFAULT '👤'
);
```

### Step 2 — Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app.py`

### Step 3 — Add secrets

In Streamlit Cloud → your app → **Settings → Secrets**, add:

```toml
YOUTUBE_API_KEY = "your_youtube_api_key"
NEWS_API_KEY = "your_newsapi_key"
SUPABASE_URL = "https://yourproject.supabase.co"
SUPABASE_KEY = "your_supabase_anon_key"
REDIRECT_URI = "https://yourapp.streamlit.app"

# Optional
GOOGLE_CLIENT_ID = "your_google_client_id"
GOOGLE_CLIENT_SECRET = "your_google_client_secret"
```

### Step 4 — First time use

1. Log in as `admin` / `1234`
2. Go to **Data tab** → click **Fetch →**
3. Wait for data to be fetched from all sources
4. Go to **Analysis tab** → click **Run Analysis →**
5. All charts, models and platform comparisons will populate

> ⚠️ `data.csv` and `generate_data.py` are **not used in deployment**. Supabase is the only data store when deployed.

---

## 🔐 Authentication

| Feature | Local Dev | Deployment |
|---|---|---|
| User store | `auth/users.json` | Supabase `users` table |
| Admin account | Auto-created | Auto-created on first login |
| New registrations | Saved to JSON | Saved to Supabase (persist forever) |
| Google OAuth | Optional | Optional |
| Data tab access | Admin only | Admin only |

---

## 💾 Storage Architecture

```
Local Development:
  app.py → data/storage.py → data/data.csv

Deployment (Supabase active):
  app.py → data/storage.py → Supabase sentiment_data table

Translation Cache (3 levels):
  Level 1: st.cache_data    → in-memory, within session (instant)
  Level 2: Supabase         → across sessions, DB lookup (fast)
  Level 3: Google Translate → API call only for truly new text (saves quota)
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
nltk
textblob
deep-translator
langdetect
supabase
python-dotenv
bcrypt
feedparser
google-api-python-client
requests
beautifulsoup4
```

> ❌ **Do NOT add** `tensorflow` or `transformers` — they exceed Streamlit Cloud's size limits. The app uses Classical ML + NLP/Lexicon models which perform excellently for this domain.

---

## 🔑 API Keys

### YouTube Data API v3
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable **YouTube Data API v3**
3. Create credentials → API Key
4. Free tier: 10,000 units/day

### NewsAPI
1. Register at [newsapi.org](https://newsapi.org)
2. Free tier: 100 requests/day, 1 month history

### Google News RSS + Hindi News RSS
- **No API key required** — public RSS feeds
- Always available, no rate limits

---

## 🏗️ Architecture Decisions

**Why Supabase over a file store?**
Streamlit Cloud's filesystem is ephemeral — any file written to disk vanishes on every restart. Supabase provides a permanent PostgreSQL database that survives restarts, redeployments, and scaling.

**Why Classical ML over Deep Learning?**
TensorFlow/Transformers exceed Streamlit Cloud's 1GB RAM/slug limit. Classical ML (SVM, Logistic Regression, Random Forest) achieves strong performance on this domain with millisecond inference time and zero deployment issues.

**Why a 5-way ensemble for live probe?**
Single models have blind spots — TextBlob misses sarcasm, ML misses sparse vocabulary, VADER misses domain-specific patterns. The ensemble compensates: domain keywords handle indirect negatives, Hindi prior handles transliterated text, sarcasm override catches irony that all other models miss.

---

## 📊 40 Government Schemes Covered

PMAY, Ayushman Bharat PM-JAY, Poshan Abhiyaan, Ayushman Bharat Digital Mission, PM Kisan Samman Nidhi, Fasal Bima, Kisan Credit Card, e-NAM, Digital India, BharatNet, UPI, Jan Dhan Yojana, Mudra Yojana, Stand Up India, Atal Pension Yojana, PM Jeevan Jyoti Bima, PM Suraksha Bima, Ujjwala Yojana, Saubhagya, Solar Rooftop PM Surya Ghar, FAME EV Scheme, Swachh Bharat Mission, Jal Jeevan Mission, AMRUT, Skill India PMKVY, Startup India, Make in India, PM eVIDYA, Beti Bachao Beti Padhao, Sukanya Samriddhi Yojana, PM Matru Vandana, Pradhan Mantri Gram Sadak Yojana, Bharatmala, Smart Cities Mission, Sagarmala, One Nation One Ration Card, PM Garib Kalyan Anna Yojana, PM SVANidhi, Vishwakarma Yojana, Atmanirbhar Bharat

---

## 👤 Author

**Final Year Research Project — 2025–26**

---

## 📄 License

This project is for academic and research purposes.

---

<div align="center">
  <sub>Built with ❤️ using Streamlit · Supabase · scikit-learn · NLTK · deep-translator</sub>
</div>
