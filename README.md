# 🧠 Pulse Sentiment AI
### Multilingual · Multi-Source · Sarcasm-Aware · Adaptive ML
**Final Year Major Research Project**

---

## 📁 Project Structure

```
Sentiment_Project/
│
├── app.py                    ← Main Streamlit application (run this)
├── requirements.txt          ← All Python dependencies
├── .env.example              ← Copy to .env, add Google OAuth keys
│
├── data/
│   ├── generate_data.py      ← Run once to generate data.csv
│   └── data.csv              ← Auto-generated dataset (800 rows)
│
├── modules/
│   ├── __init__.py
│   ├── preprocess.py         ← Language detection, translation, sarcasm, cleaning
│   └── model.py              ← 5 ML models + adaptive selection
    ├── scraper.py
│
├── auth/
│   ├── __init__.py
│   ├── auth_manager.py       ← Login, Signup, Google OAuth
│   └── users.json            ← Auto-created user store
│
└── README.md
```

---

## ⚙️ Setup (Step by Step)

### Step 1 — Create Virtual Environment
```bash
cd Sentiment_Project
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Step 4 — Generate Dataset
```bash
python data/generate_data.py
```
This creates `data/data.csv` with 800 multilingual comments.

### Step 5 — Run the App
```bash
streamlit run app.py
```
Open your browser at: **http://localhost:8501**

**Demo Login:** username: `admin` | password: `1234`

---

## 🔑 Google OAuth Setup (Optional but Recommended)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project → **APIs & Services** → **Credentials**
3. Click **Create Credentials** → **OAuth 2.0 Client ID**
4. Application type: **Web Application**
5. Authorized redirect URIs: add `http://localhost:8501`
6. Copy your **Client ID** and **Client Secret**
7. Create `.env` file:
```
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret
```
8. Install dotenv: `pip install python-dotenv`
9. Add to top of `app.py` (already included):
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 🔬 Research Gaps Addressed

| # | Gap | Solution |
|---|-----|----------|
| 1 | Language Barrier | LangDetect + GoogleTranslator → multilingual support |
| 2 | Only Twitter Data | Multi-source: Twitter, YouTube, Instagram, News, Forums |
| 3 | Random Algorithm Selection | 5 models benchmarked → best auto-selected |
| 4 | Binary Classification Only | 3-class: Positive / Neutral / Negative |
| 5 | Sarcasm Ignored | Rule-based + emoji sarcasm detection + label correction |

---

## 🤖 ML Models Compared

- Naive Bayes (MultinomialNB)
- Logistic Regression
- SVM (LinearSVC with calibration)
- Random Forest
- Gradient Boosting

The system trains all 5 on your data, evaluates accuracy/F1/precision/recall,
and **automatically selects the best model** for live predictions.

---

## 📊 Features

- ✅ Multi-scheme selector (PMAY, Ayushman Bharat, Digital India, PM Kisan, Swachh Bharat)
- ✅ Login / Signup with bcrypt password hashing
- ✅ Google OAuth integration
- ✅ Multilingual comment analysis (EN/HI/TA/Hinglish)
- ✅ Automatic language detection & translation
- ✅ Sarcasm detection with emoji signals
- ✅ Adaptive model selection dashboard
- ✅ Source-wise sentiment breakdown (5 platforms)
- ✅ Live real-time probe with confidence scores
- ✅ Stunning animated dark UI

---

## 👥 Team Roles

| Member | Module | File |
|--------|--------|------|
| Member 1 | Data & Preprocessing | `data/`, `modules/preprocess.py` |
| Member 2 | ML Models | `modules/model.py` |
| Member 3 | Dashboard & UI | `app.py`, `auth/` |

---

*Built as a Final Year Research Project · Addressing 5 gaps from 15+ reviewed papers*