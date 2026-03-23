"""
app.py — Pulse Sentiment AI
Run:  streamlit run app.py
"""
import nltk
for _pkg, _path in [
    ("vader_lexicon", "sentiment/vader_lexicon.zip"),
    ("stopwords",     "corpora/stopwords"),
    ("punkt",         "tokenizers/punkt"),
]:
    try:    nltk.data.find(_path)
    except LookupError: nltk.download(_pkg, quiet=True)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.preprocess  import preprocess_dataframe, detect_language, translate_to_english, detect_sarcasm, clean_text, get_sentiment
from modules.model       import train_models, get_detailed_metrics, predict_live, predict_live_with_confidence, detect_sarcasm_advanced
from modules.scraper     import ALL_SCHEMES, fetch_all
from auth.auth_manager   import login, signup, get_google_auth_url

# ── Storage layer (works locally via CSV and deployed via Supabase) ───────────
try:
    from data.storage import load_data as _storage_load_data, get_stats as _storage_get_stats, _is_deployed
except ImportError:
    def _storage_load_data():
        try:    return pd.read_csv("data/data.csv")
        except: return pd.DataFrame()
    def _storage_get_stats(): return {"total_rows": 0}
    def _is_deployed():       return False

# ── Cache helpers ─────────────────────────────────────────────────────────────
def _get_data_hash() -> str:
    """
    Hash that changes whenever data changes.
    Deployed (Supabase): uses row count + a time bucket so stale cache is
                         busted after the first minute following a fetch.
    Local: uses data/data.csv mtime.
    """
    try:
        if _is_deployed():
            import time
            stats = _storage_get_stats()
            bucket = str(int(time.time() // 60))
            return f"{stats.get('total_rows', 0)}-{bucket}"
        else:
            return str(os.path.getmtime("data/data.csv"))
    except Exception:
        return "0"

@st.cache_data(show_spinner=False)
def _cached_preprocess(data_hash: str, scheme: str):
    """Load from storage (Supabase or CSV) then preprocess."""
    df_raw = _storage_load_data()
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    if scheme != "All Schemes":
        key   = scheme.split("—")[0].strip().split(" ")[0]
        df_f  = df_raw[df_raw["Scheme"].str.contains(key, case=False, na=False)]
        df_raw = df_f if len(df_f) >= 10 else df_raw
    return preprocess_dataframe(df_raw)

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pulse · Sentiment AI", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

def _get_secret(key: str) -> str:
    try:    return st.secrets.get(key, os.getenv(key, ""))
    except: return os.getenv(key, "")

GOOGLE_CLIENT_ID     = _get_secret("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = _get_secret("GOOGLE_CLIENT_SECRET")

def _get_redirect_uri() -> str:
    try:
        cloud_url = st.secrets.get("REDIRECT_URI", "")
        if cloud_url: return cloud_url
    except Exception:
        pass
    return os.getenv("REDIRECT_URI", "http://localhost:8501")

REDIRECT_URI = _get_redirect_uri()

# All valid source names — must match scraper.py exactly
VALID_SOURCES = {
    "YouTube", "News App", "Google News",
    "Dainik Bhaskar", "Amar Ujala", "Navbharat Times",
    "Jagran", "NDTV Hindi", "ABP Live",
}

SCHEME_EMOJI = {
    "PMAY — Pradhan Mantri Awas Yojana":"🏘️","Ayushman Bharat — PM-JAY":"🏥",
    "Poshan Abhiyaan — Nutrition Mission":"🥗","Ayushman Bharat Digital Mission":"💊",
    "PM Kisan Samman Nidhi":"🌾","Fasal Bima — PM Crop Insurance":"🌱",
    "Kisan Credit Card":"💳","e-NAM — National Agriculture Market":"🛒",
    "Digital India Initiative":"💡","BharatNet — Rural Internet":"🌐",
    "UPI — Unified Payments Interface":"📱","Jan Dhan Yojana — Financial Inclusion":"🏦",
    "Mudra Yojana — MSME Loans":"💰","Stand Up India Scheme":"📈",
    "Atal Pension Yojana":"👴","PM Jeevan Jyoti Bima":"🛡️","PM Suraksha Bima":"🔐",
    "Ujjwala Yojana — LPG for Poor":"🔥","Saubhagya — Household Electrification":"⚡",
    "Solar Rooftop — PM Surya Ghar":"☀️","FAME — Electric Vehicle Scheme":"🚗",
    "Swachh Bharat Mission":"♻️","Jal Jeevan Mission — Har Ghar Jal":"💧",
    "AMRUT — Urban Development":"🏙️","Skill India — PMKVY":"🎓",
    "Startup India":"🚀","Make in India":"🏭","PM eVIDYA — Digital Education":"📚",
    "Beti Bachao Beti Padhao":"👧","Sukanya Samriddhi Yojana":"🌸",
    "PM Matru Vandana — Maternity Benefit":"🤱","Pradhan Mantri Gram Sadak Yojana":"🛣️",
    "Bharatmala — Highway Project":"🛤️","Smart Cities Mission":"🏢",
    "Sagarmala — Port Development":"⚓","One Nation One Ration Card":"🍚",
    "PM Garib Kalyan Anna Yojana":"🌽","PM SVANidhi — Street Vendor Loan":"🛍️",
    "Vishwakarma Yojana":"🔨","Atmanirbhar Bharat":"🇮🇳",
}

MODEL_TYPE_COLORS = {
    "Classical ML":"#38bdf8","NLP/Lexicon":"#34d399",
    "Deep Learning":"#818cf8","Transformer/BERT":"#f59e0b",
}

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #080c14;
    --bg2:       #0d1420;
    --bg3:       #121a28;
    --bg4:       #1a2235;
    --surface:   rgba(13,20,32,0.95);
    --surface2:  rgba(18,26,40,0.9);
    --border:    rgba(99,179,237,0.08);
    --border2:   rgba(99,179,237,0.16);
    --border3:   rgba(99,179,237,0.28);
    --txt:       #e8eef8;
    --txt2:      #8fa8c8;
    --txt3:      #4a6380;
    --txt4:      #2a3a50;
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --accent3:   #34d399;
    --green:     #34d399;
    --red:       #fb7185;
    --amber:     #fbbf24;
    --purple:    #a78bfa;
    --pink:      #f472b6;
    --shadow:    0 2px 8px rgba(0,0,0,0.4), 0 8px 32px rgba(0,0,0,0.3);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.5), 0 24px 64px rgba(0,0,0,0.4);
    --glow:      0 0 20px rgba(56,189,248,0.15);
    --glow-lg:   0 0 40px rgba(56,189,248,0.2);
    --radius:    14px;
    --radius-sm: 8px;
    --radius-xs: 5px;
}

html, body, .stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
section[data-testid="stSidebar"],
.main, .block-container {
    background-color: var(--bg) !important;
    color: var(--txt) !important;
    font-family: 'Inter', sans-serif !important;
}

header, footer, #MainMenu, .stDeployButton,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stSidebar"] { display: none !important; }

.stApp::before {
    content: "";
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none;
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 5%,  rgba(56,189,248,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 90% 90%,  rgba(129,140,248,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 35% at 55% 45%,  rgba(52,211,153,0.03) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    background-size: cover, cover, cover, 200px 200px;
    animation: ambientShift 20s ease-in-out infinite alternate;
}
@keyframes ambientShift { 0%{opacity:0.6} 50%{opacity:1.0} 100%{opacity:0.7} }

.stApp::after {
    content: "";
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);
}

.block-container {
    position: relative; z-index: 2;
    padding: 2rem 2.5rem 6rem !important;
    max-width: 1360px !important;
}

.block-container { animation: pageIn .5s cubic-bezier(.16,1,.3,1) both; }
@keyframes pageIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }

.card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 24px 28px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: var(--shadow);
    transition: border-color .3s ease, box-shadow .3s ease, transform .2s ease;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.4), transparent);
    opacity: 0; transition: opacity .3s;
}
.card:hover { border-color:var(--border3); box-shadow:var(--shadow-lg),var(--glow); transform:translateY(-1px); }
.card:hover::before { opacity: 1; }

.label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px; letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); display: flex; align-items: center; gap: 10px;
    margin-bottom: 18px; opacity: 0.9;
}
.label::before { content:""; width:3px; height:3px; border-radius:50%; background:var(--accent); box-shadow:0 0 6px var(--accent); }
.label::after  { content:""; flex:1; height:1px; background:linear-gradient(90deg,rgba(56,189,248,0.25),transparent); }

h1, h2, h3 { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:var(--txt) !important; letter-spacing:-.5px !important; }
p, li, span, div { color: var(--txt2); }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background:var(--bg3) !important; border:1px solid var(--border2) !important;
    border-radius:var(--radius-sm) !important; color:var(--txt) !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important;
    padding:12px 16px !important; transition:border-color .2s,box-shadow .2s !important;
    box-shadow:inset 0 1px 3px rgba(0,0,0,0.3) !important; caret-color:var(--accent) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px rgba(56,189,248,0.1),inset 0 1px 3px rgba(0,0,0,0.3) !important;
    outline:none !important; background:var(--bg4) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color:var(--txt3) !important; }
.stTextInput > label,.stTextArea > label,.stSelectbox > label,.stNumberInput > label {
    font-family:'IBM Plex Mono',monospace !important; font-size:9px !important;
    letter-spacing:2.5px !important; text-transform:uppercase !important; color:var(--txt3) !important;
}

.stSelectbox > div > div {
    background:var(--bg3) !important; border:1px solid var(--border2) !important;
    border-radius:var(--radius-sm) !important; color:var(--txt) !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important;
    box-shadow:inset 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stSelectbox > div > div:hover { border-color:var(--border3) !important; }
[data-baseweb="select"] > div { background:var(--bg3) !important; border-color:var(--border2) !important; color:var(--txt) !important; }
[data-baseweb="popover"] { background:var(--bg3) !important; border:1px solid var(--border2) !important; border-radius:var(--radius-sm) !important; box-shadow:var(--shadow-lg) !important; }
[data-baseweb="menu"] { background:var(--bg3) !important; }
[data-baseweb="option"]:hover { background:var(--bg4) !important; }

.stNumberInput > div > div > input {
    background:var(--bg3) !important; border:1px solid var(--border2) !important;
    border-radius:var(--radius-sm) !important; color:var(--txt) !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important;
    box-shadow:inset 0 1px 3px rgba(0,0,0,0.3) !important;
}

.stButton > button {
    background:var(--bg3) !important; border:1px solid var(--border2) !important;
    border-radius:var(--radius-sm) !important; color:var(--txt2) !important;
    font-family:'IBM Plex Mono',monospace !important; font-size:10px !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    padding:12px 20px !important; width:100% !important;
    transition:all .25s cubic-bezier(.16,1,.3,1) !important;
    box-shadow:var(--shadow) !important; position:relative !important; overflow:hidden !important;
}
.stButton > button::before {
    content:"" !important; position:absolute !important; inset:0 !important;
    background:linear-gradient(135deg,rgba(56,189,248,0.05),rgba(129,140,248,0.05)) !important;
    opacity:0 !important; transition:opacity .25s !important;
}
.stButton > button:hover {
    background:var(--bg4) !important; border-color:var(--accent) !important;
    color:var(--accent) !important; transform:translateY(-1px) !important;
    box-shadow:var(--shadow),0 0 16px rgba(56,189,248,0.15) !important;
}
.stButton > button:hover::before { opacity:1 !important; }
.stButton > button:active { transform:translateY(0) !important; }

div[data-testid="column"]:first-child .stButton > button {
    background:linear-gradient(135deg,#0ea5e9,#6366f1) !important; border:none !important;
    color:#ffffff !important; font-weight:500 !important;
    box-shadow:0 4px 20px rgba(14,165,233,0.3),0 2px 8px rgba(0,0,0,0.4) !important;
    text-shadow:0 1px 2px rgba(0,0,0,0.2) !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    background:linear-gradient(135deg,#38bdf8,#818cf8) !important; color:#ffffff !important;
    box-shadow:0 6px 28px rgba(56,189,248,0.4),0 2px 8px rgba(0,0,0,0.4) !important;
    transform:translateY(-2px) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background:var(--bg2) !important; border-radius:10px !important; padding:4px !important;
    gap:2px !important; border:1px solid var(--border) !important;
    box-shadow:inset 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family:'IBM Plex Mono',monospace !important; font-size:10px !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    color:var(--txt3) !important; border-radius:7px !important;
    padding:9px 20px !important; transition:all .2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--txt2) !important; background:var(--bg3) !important; }
.stTabs [aria-selected="true"] {
    background:var(--bg4) !important; color:var(--accent) !important;
    box-shadow:0 1px 6px rgba(0,0,0,0.4),0 0 12px rgba(56,189,248,0.1) !important;
    border:1px solid var(--border2) !important;
}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] { display:none !important; }

.stSuccess > div { background:rgba(52,211,153,0.07) !important; border:1px solid rgba(52,211,153,0.25) !important; border-radius:var(--radius-sm) !important; color:#34d399 !important; font-family:'Inter',sans-serif !important; }
.stError > div   { background:rgba(251,113,133,0.07) !important; border:1px solid rgba(251,113,133,0.25) !important; border-radius:var(--radius-sm) !important; color:#fb7185 !important; }
.stWarning > div { background:rgba(251,191,36,0.07) !important;  border:1px solid rgba(251,191,36,0.25) !important;  border-radius:var(--radius-sm) !important; color:#fbbf24 !important; }
.stInfo > div    { background:rgba(56,189,248,0.07) !important;  border:1px solid rgba(56,189,248,0.25) !important;  border-radius:var(--radius-sm) !important; color:#38bdf8 !important; }

.mcard {
    background:var(--surface2); border:1px solid var(--border2);
    border-radius:var(--radius); padding:22px 16px; text-align:center;
    transition:all .25s cubic-bezier(.16,1,.3,1); position:relative; overflow:hidden;
}
.mcard::before { content:""; position:absolute; inset:0; background:radial-gradient(ellipse at 50% 0%,rgba(56,189,248,0.06),transparent 70%); opacity:0; transition:opacity .3s; }
.mcard::after  { content:""; position:absolute; bottom:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--accent),var(--accent2)); transform:scaleX(0); transform-origin:center; transition:transform .35s cubic-bezier(.16,1,.3,1); }
.mcard:hover { border-color:var(--border3); box-shadow:var(--glow); transform:translateY(-2px); }
.mcard:hover::before { opacity:1; }
.mcard:hover::after  { transform:scaleX(1); }
.mval { font-family:'Syne',sans-serif; font-size:32px; font-weight:800; line-height:1.1; background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:-1px; }
.mlbl { font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:2.5px; text-transform:uppercase; color:var(--txt3); margin-top:6px; }

.stDataFrame { border:1px solid var(--border2) !important; border-radius:var(--radius) !important; overflow:hidden !important; background:var(--bg2) !important; }
.stDataFrame [data-testid="stDataFrameResizable"] { background:var(--bg2) !important; }

.stCheckbox > label { font-family:'Inter',sans-serif !important; font-size:13px !important; color:var(--txt2) !important; }
.stCheckbox > label:hover { color:var(--txt) !important; }
.stSpinner > div { border-color:var(--accent) transparent transparent transparent !important; }

::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--bg4); border-radius:4px; transition:background .2s; }
::-webkit-scrollbar-thumb:hover { background:var(--accent); }

.div { height:1px; background:linear-gradient(90deg,transparent,var(--border2),transparent); margin:20px 0; }

#pulse-bar {
    position:fixed; bottom:0; left:0; right:0; height:2px; z-index:9999;
    background:linear-gradient(90deg,#38bdf8,#818cf8,#34d399,#fbbf24,#fb7185,#38bdf8);
    background-size:400% auto; animation:barFlow 8s linear infinite;
}
@keyframes barFlow { 0%{background-position:0% center} 100%{background-position:400% center} }

.auth-card {
    background:var(--surface); border:1px solid var(--border2); border-radius:18px;
    padding:36px 40px; backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px);
    box-shadow:var(--shadow-lg),0 0 60px rgba(56,189,248,0.05);
    animation:authIn .5s cubic-bezier(.16,1,.3,1) both; position:relative; overflow:hidden;
}
.auth-card::before { content:""; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(56,189,248,0.5),transparent); }
@keyframes authIn { from{opacity:0;transform:translateY(24px) scale(.97)} to{opacity:1;transform:translateY(0) scale(1)} }

.g-btn {
    display:flex; align-items:center; justify-content:center; gap:10px;
    background:var(--bg3); border:1px solid var(--border2); border-radius:var(--radius-sm);
    padding:13px 20px; color:var(--txt2); font-family:'Inter',sans-serif; font-size:14px;
    font-weight:500; cursor:pointer; transition:all .25s cubic-bezier(.16,1,.3,1);
    text-decoration:none; width:100%; text-align:center; box-shadow:var(--shadow);
}
.g-btn:hover { background:var(--bg4); color:var(--txt); border-color:var(--border3); box-shadow:var(--shadow-lg); text-decoration:none; transform:translateY(-1px); }

.hero-title { animation:heroIn .7s cubic-bezier(.16,1,.3,1) both; }
.hero-sub   { animation:heroIn .7s cubic-bezier(.16,1,.3,1) .2s both; }
.hero-badge { animation:heroIn .7s cubic-bezier(.16,1,.3,1) .05s both; }
@keyframes heroIn { from{opacity:0;transform:translateY(-16px)} to{opacity:1;transform:translateY(0)} }

.js-plotly-plot .plotly,.plot-container { background:transparent !important; }
.js-plotly-plot .plotly .bg { fill:transparent !important; }
::selection { background:rgba(56,189,248,0.2); color:var(--txt); }
*:focus-visible { outline:2px solid rgba(56,189,248,0.4) !important; outline-offset:2px !important; }
</style>
<div id="pulse-bar"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("logged_in",False),("user_info",{}),("auth_mode","login"),
              ("analysis_done",False),("df_store",None),
              ("results_store",None),("best_name_store",None),("metrics_store",None),
              ("used_dl",False),("used_tr",False),("fetch_done",False)]:
    if k not in st.session_state: st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:

    params = st.query_params
    if "code" in params and GOOGLE_CLIENT_ID:
        from auth.auth_manager import exchange_google_code
        ui = exchange_google_code(params["code"], GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI)
        if ui:
            st.session_state.logged_in = True
            st.session_state.user_info = {"name":ui.get("name","Google User"),"email":ui.get("email",""),"role":"user","avatar":"🌐"}
            st.query_params.clear(); st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:60px 20px 48px;">
        <div class="hero-badge" style="display:inline-flex;align-items:center;gap:8px;
             background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.18);
             border-radius:100px;padding:6px 18px;font-family:'IBM Plex Mono',monospace;
             font-size:9.5px;letter-spacing:3px;color:#38bdf8;margin-bottom:24px;
             text-transform:uppercase;box-shadow:0 0 20px rgba(56,189,248,0.08);">
            <span style="width:5px;height:5px;border-radius:50%;background:#34d399;
                  box-shadow:0 0 8px #34d399;animation:livePulse 2s ease infinite;
                  display:inline-block;"></span>
            Pulse Sentiment AI
        </div>
        <div class="hero-title" style="font-family:'Syne',sans-serif;font-size:clamp(32px,5vw,62px);
             font-weight:800;line-height:1.05;letter-spacing:-2px;
             background:linear-gradient(135deg,#e8eef8 0%,#38bdf8 40%,#818cf8 70%,#a78bfa 100%);
             background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin-bottom:14px;animation:gradShift 6s ease infinite alternate;">
            Read the Nation.<br>Understand Every Voice.
        </div>
        <div class="hero-sub" style="font-family:'IBM Plex Mono',monospace;font-size:10px;
             color:#4a6380;letter-spacing:3px;text-transform:uppercase;">
            Multilingual · Multi-Source · Sarcasm-Aware · Adaptive ML
        </div>
    </div>
    <style>
    @keyframes livePulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.6)} }
    @keyframes gradShift { 0%{background-position:0% 50%} 100%{background-position:100% 50%} }
    </style>
    """, unsafe_allow_html=True)

    _, cc, _ = st.columns([1, 1.2, 1])
    with cc:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Sign In", key="tab_login"):
                st.session_state.auth_mode = "login"
        with col_b:
            if st.button("Register", key="tab_register"):
                st.session_state.auth_mode = "signup"

        st.markdown("<div class='div'></div>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            st.markdown("""
            <div style="margin-bottom:22px;">
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;
                     color:#e8eef8;margin-bottom:4px;letter-spacing:-.5px;">Welcome back</div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;color:#4a6380;">
                     Sign in to access your dashboard</div>
            </div>""", unsafe_allow_html=True)

            uname = st.text_input("Username", placeholder="Enter username", key="li_user", label_visibility="collapsed")
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:#4a6380;margin-bottom:4px;">USERNAME</div>', unsafe_allow_html=True)
            passw = st.text_input("Password", type="password", placeholder="Enter password", key="li_pass", label_visibility="collapsed")
            st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:#4a6380;margin-bottom:18px;">PASSWORD</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Sign In →", key="btn_login"):
                    if uname.strip() and passw.strip():
                        ok, msg, info = login(uname.strip(), passw)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_info  = info
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill in all fields.")
            with c2:
                if st.button("Forgot Password", key="btn_forgot"):
                    st.info("Contact admin@pulse.ai")

        else:
            st.markdown("""
            <div style="margin-bottom:22px;">
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;
                     color:#e8eef8;margin-bottom:4px;letter-spacing:-.5px;">Create account</div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;color:#4a6380;">
                     Join Pulse Sentiment AI</div>
            </div>""", unsafe_allow_html=True)

            su_name  = st.text_input("Full Name",  placeholder="Full name",         key="su_name")
            su_email = st.text_input("Email",       placeholder="Email address",     key="su_email")
            su_user  = st.text_input("Username",    placeholder="Choose a username", key="su_user")
            su_pass  = st.text_input("Password",    type="password", placeholder="Min 4 characters", key="su_pass")

            c1, _ = st.columns(2)
            with c1:
                if st.button("Create Account →", key="btn_signup"):
                    if su_user.strip() and su_pass.strip() and su_name.strip():
                        ok, msg = signup(su_user.strip(), su_pass, su_name.strip(), su_email.strip())
                        if ok:
                            st.success("Account created. Sign in now.")
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill in all required fields.")

        if GOOGLE_CLIENT_ID:
            st.markdown("<div class='div'></div>", unsafe_allow_html=True)
            g_url = get_google_auth_url(GOOGLE_CLIENT_ID, REDIRECT_URI)
            st.markdown(f"""
            <a href="{g_url}" class="g-btn">
                <svg width="16" height="16" viewBox="0 0 48 48">
                  <path fill="#EA4335" d="M24 9.5c3.3 0 5.9 1.4 7.7 2.6l5.7-5.7C33.9 3.5 29.3 1.5 24 1.5 14.8 1.5 7 7.4 3.7 15.5l6.7 5.2C12 15.1 17.5 9.5 24 9.5z"/>
                  <path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h12.4c-.5 2.8-2.1 5.2-4.5 6.8l7 5.4c4.1-3.8 6.2-9.4 6.2-16.2z"/>
                  <path fill="#FBBC05" d="M10.4 28.4A14.3 14.3 0 0 1 9.5 24c0-1.5.3-3 .7-4.3L3.7 14.5A22.5 22.5 0 0 0 1.5 24c0 3.6.9 7 2.2 10l6.7-5.6z"/>
                  <path fill="#34A853" d="M24 46.5c5.3 0 9.7-1.7 12.9-4.7l-7-5.4c-1.7 1.1-3.9 1.8-5.9 1.8-6.4 0-11.9-4.3-13.8-10.1l-6.8 5.3C7 41 15 46.5 24 46.5z"/>
                </svg>
                Continue with Google
            </a>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;font-family:'IBM Plex Mono',monospace;font-size:9px;
         color:#2a3a50;margin-top:14px;letter-spacing:2px;">
         DEMO &nbsp;·&nbsp; username: admin &nbsp;·&nbsp; password: 1234
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
else:
    user   = st.session_state.user_info
    uname  = user.get("name", "User")
    avatar = user.get("avatar", "👤")

    col_t, col_u = st.columns([6, 1])
    with col_t:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;padding:4px 0 22px;">
            <div style="width:40px;height:40px;border-radius:50%;flex-shrink:0;
                 background:linear-gradient(135deg,#0ea5e9,#6366f1);
                 display:flex;align-items:center;justify-content:center;font-size:17px;
                 box-shadow:0 0 16px rgba(14,165,233,0.3);border:1px solid rgba(56,189,248,0.2);">
                 {avatar}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                     color:#e8eef8;line-height:1.2;letter-spacing:-.3px;">Pulse Dashboard</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                     color:#4a6380;letter-spacing:2.5px;">{uname.upper()}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_u:
        st.markdown("<div style='padding-top:6px;'>", unsafe_allow_html=True)
        if st.button("Sign Out", key="logout"):
            for k in ["logged_in","user_info","analysis_done","df_store","results_store","best_name_store","metrics_store"]:
                st.session_state[k] = False if k=="logged_in" else ({} if k=="user_info" else None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    is_admin = user.get("role") == "admin"

    if is_admin:
       t1, t2, t3, t4, t5 = st.tabs(["Analysis","Live Probe","Platforms","Data","About"])
    else:
       t1, t2, t3, t5 = st.tabs(["Analysis","Live Probe","Platforms","About"])
       t4 = None

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with t1:
        # Banner when new data was fetched and analysis is stale
        if st.session_state.get("fetch_done"):
            st.markdown("""
            <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.28);
                 border-radius:10px;padding:12px 18px;margin-bottom:14px;display:flex;
                 align-items:center;gap:10px;">
              <span style="font-size:16px;">🔄</span>
              <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#fbbf24;letter-spacing:1.5px;">
                NEW DATA FETCHED — Click <b>Run Analysis →</b> to include it in charts and models.
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Select Scheme</div>", unsafe_allow_html=True)

        scheme_options = ["All Schemes"] + ALL_SCHEMES
        sc1, sc2, sc3 = st.columns([3, 1, 1])
        with sc1:
            scheme = st.selectbox("Scheme", scheme_options, label_visibility="collapsed", key="sel_scheme")
        with sc2:
            use_dl = st.checkbox("Deep Learning", value=False, key="use_dl")
        with sc3:
            use_tr = st.checkbox("Transformers", value=False, key="use_tr")

        r1, _ = st.columns([1, 5])
        with r1:
            run = st.button("Run Analysis →", key="run_analysis")
        st.markdown("</div>", unsafe_allow_html=True)

        if run:
            # Clear all caches so fresh data is loaded from storage
            st.cache_data.clear()
            st.cache_resource.clear()

            raw_check = _storage_load_data()
            if raw_check is None or raw_check.empty:
                st.error("Dataset not found. Run: python data/generate_data.py  (or fetch data from the Data tab)")
                st.stop()

            data_hash = _get_data_hash()

            with st.spinner("Step 1 / 2 — Preprocessing & labelling…"):
                df = _cached_preprocess(data_hash, scheme)

            @st.cache_resource
            def _cached_train(data_hash, scheme, use_dl, use_tr):
                df_inner = _cached_preprocess(data_hash, scheme)
                return train_models(df_inner["Cleaned"], df_inner["Sentiment"],
                        use_dl=use_dl, use_transformers=use_tr, df_meta=df_inner)

            with st.spinner("Step 2 / 2 — Training & benchmarking all models…"):
                results, best_name = _cached_train(data_hash, scheme, use_dl, use_tr)
            metrics = results

            st.session_state.df_store        = df
            st.session_state.results_store   = results
            st.session_state.best_name_store = best_name
            st.session_state.metrics_store   = metrics
            st.session_state.analysis_done   = True
            st.session_state.used_dl         = use_dl
            st.session_state.used_tr         = use_tr
            st.session_state.fetch_done      = False   # banner consumed

        if st.session_state.analysis_done and st.session_state.df_store is not None:
            df        = st.session_state.df_store
            results   = st.session_state.results_store
            best_name = st.session_state.best_name_store
            metrics   = st.session_state.metrics_store

            counts   = df["Sentiment"].value_counts()
            n_pos    = counts.get("Positive", 0)
            n_neg    = counts.get("Negative", 0)
            n_neu    = counts.get("Neutral",  0)
            n_sar    = int(df["IsSarcasm"].sum()) if "IsSarcasm" in df.columns else 0
            best_acc = metrics[best_name]["accuracy"] if best_name in metrics else 0

            k1,k2,k3,k4,k5,k6 = st.columns(6)
            for col, val, lbl in zip([k1,k2,k3,k4,k5,k6],[
                len(df), n_pos, n_neg, n_neu, n_sar, f"{best_acc}%"
            ],["Comments","Positive","Negative","Neutral","Sarcasm","Best Accuracy"]):
                col.markdown(f"""<div class="mcard">
                    <div class="mval">{val}</div>
                    <div class="mlbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='div'></div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='label'>Sentiment Distribution</div>", unsafe_allow_html=True)
                fig_pie = px.pie(values=counts.values, names=counts.index, hole=0.6,
                    color=counts.index,
                    color_discrete_map={"Positive":"#34d399","Negative":"#fb7185","Neutral":"#fbbf24","Sarcasm":"#a78bfa"})
                fig_pie.update_traces(textfont_color="white", textfont_size=12)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8fa8c8", legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8fa8c8")),
                    margin=dict(t=10,b=10,l=10,r=10), height=300)
                st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie")
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='label'>By Platform</div>", unsafe_allow_html=True)
                if "Source" in df.columns:
                    # Show every valid source that is actually present — no silent fallback
                    df_plot = df[df["Source"].isin(VALID_SOURCES)].copy()
                    if df_plot.empty:
                        df_plot = df.copy()
                    ss = df_plot.groupby(["Source","Sentiment"]).size().reset_index(name="Count")
                    fig_bar = px.bar(ss, x="Source", y="Count", color="Sentiment", barmode="group",
                        color_discrete_map={"Positive":"#34d399","Negative":"#fb7185","Neutral":"#fbbf24","Sarcasm":"#a78bfa"})
                    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8fa8c8", legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8fa8c8")),
                        xaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380", tickangle=-30),
                        yaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380"),
                        margin=dict(t=10,b=60,l=10,r=10), height=320)
                    st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar")
                st.markdown("</div>", unsafe_allow_html=True)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='label'>Language Breakdown</div>", unsafe_allow_html=True)
                if "Lang" in df.columns:
                    lc   = df["Lang"].value_counts()
                    lmap = {"en":"English","hi":"Hindi","hinglish":"Hinglish","ta":"Tamil","te":"Telugu","bn":"Bengali"}
                    lc   = lc.rename(index=lambda x: lmap.get(x, x.upper()))
                    fig_lang = px.bar(x=lc.index, y=lc.values, color=lc.values,
                        color_continuous_scale=["#38bdf8","#818cf8","#34d399"],
                        labels={"x":"","y":"Count","color":"Count"})
                    fig_lang.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8fa8c8", showlegend=False,
                        xaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380"),
                        yaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380"),
                        margin=dict(t=10,b=10,l=10,r=10), height=260)
                    st.plotly_chart(fig_lang, use_container_width=True, key="fig_lang")
                st.markdown("</div>", unsafe_allow_html=True)

            with c4:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='label'>Sarcasm Detection</div>", unsafe_allow_html=True)
                sd = pd.DataFrame({"Type":["Genuine","Sarcastic"],"Count":[len(df)-n_sar, n_sar]})
                fig_sar = px.bar(sd, x="Type", y="Count", color="Type",
                    color_discrete_map={"Genuine":"#38bdf8","Sarcastic":"#fb7185"})
                fig_sar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8fa8c8", showlegend=False,
                    xaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380"),
                    yaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380"),
                    margin=dict(t=10,b=10,l=10,r=10), height=260)
                st.plotly_chart(fig_sar, use_container_width=True, key="fig_sar")
                st.markdown("</div>", unsafe_allow_html=True)

            # Platform × Sentiment Heatmap — all sources in analysed df
            if "Source" in df.columns:
                df_heat = df[df["Source"].isin(VALID_SOURCES)].copy()
                if df_heat.empty:
                    df_heat = df.copy()
                if df_heat["Source"].nunique() >= 1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='label'>Platform × Sentiment Heatmap</div>", unsafe_allow_html=True)
                    srcs  = sorted(df_heat["Source"].unique().tolist())
                    sents = ["Positive","Negative","Neutral"]
                    z, t  = [], []
                    for s in srcs:
                        sub = df_heat[df_heat["Source"]==s]["Sentiment"].value_counts(normalize=True).mul(100)
                        row = [round(sub.get(x,0),1) for x in sents]
                        z.append(row); t.append([f"{v}%" for v in row])
                    fig_heat = go.Figure(go.Heatmap(z=z, x=sents, y=srcs, text=t, texttemplate="%{text}",
                        colorscale=[[0,"#080c14"],[0.4,"#0ea5e9"],[1,"#34d399"]], showscale=True))
                    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8fa8c8", margin=dict(t=10,b=10,l=10,r=10),
                        height=max(280, 50 * len(srcs) + 60))
                    st.plotly_chart(fig_heat, use_container_width=True, key="fig_heat")
                    st.markdown("</div>", unsafe_allow_html=True)

            # ── Model Benchmarks ──────────────────────────────────────────────
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='label'>Model Benchmarks</div>", unsafe_allow_html=True)

            failed_models = {k:v for k,v in metrics.items() if v.get("accuracy",0) == 0 and not v.get("available",True)}
            used_dl = st.session_state.get("used_dl", False)
            used_tr = st.session_state.get("used_tr", False)
            if used_dl or used_tr:
                libs = []
                if used_dl: libs.append("TensorFlow (LSTM · BiLSTM · CNN)")
                if used_tr: libs.append("Transformers (ALBERT · DistilBERT · mBERT)")
                if failed_models:
                    st.info(f"Some models could not run — library may not be installed: {' · '.join(libs)}. Install with: pip install tensorflow transformers")

            mhtml = ""
            for mn, md in sorted(metrics.items(), key=lambda x: -x[1].get("accuracy", 0)):
                acc = md.get("accuracy", 0)
                if acc == 0 and not md.get("available", True):
                    continue
                ib = mn == best_name
                mt = md.get("type", "Classical ML")
                TYPE_COLORS_HEX = {
                    "Classical ML":"#38bdf8","NLP/Lexicon":"#34d399",
                    "Deep Learning":"#818cf8","Transformer/BERT":"#fbbf24",
                }
                tc  = TYPE_COLORS_HEX.get(mt, "#38bdf8")
                spd = f"{md.get('speed_ms',0):.0f}ms" if md.get("speed_ms",0) > 0 else "—"

                if acc == 0:
                    mhtml += (
                        f'<div style="display:flex;align-items:center;justify-content:space-between;'
                        f'background:#0d1420;border:1px solid rgba(56,189,248,0.06);'
                        f'border-radius:10px;padding:13px 18px;margin-bottom:8px;opacity:0.45;">'
                        f'<div><div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">'
                        f'<span style="font-family:Syne,sans-serif;font-size:14px;font-weight:600;color:#4a6380;">{mn}</span>'
                        f'<span style="background:{tc}15;border:1px solid {tc}25;color:{tc};border-radius:100px;padding:1px 9px;font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:1px;">{mt}</span>'
                        f'<span style="background:rgba(251,113,133,0.08);border:1px solid rgba(251,113,133,0.2);color:#fb7185;border-radius:100px;padding:1px 9px;font-family:IBM Plex Mono,monospace;font-size:9px;">NOT INSTALLED</span>'
                        f'</div><div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#2a3a50;">Library not available — install to enable</div></div>'
                        f'<div style="font-size:18px;font-weight:700;color:#2a3a50;font-family:Syne,sans-serif;">—</div></div>'
                    )
                    continue

                bdr = ("border-color:rgba(52,211,153,0.3);background:rgba(52,211,153,0.03);"
                       if ib else "border-color:rgba(56,189,248,0.1);background:#0d1420;")
                bdg = ("<span style='background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.25);color:#34d399;border-radius:100px;padding:2px 10px;font-family:IBM Plex Mono,monospace;font-size:9px;letter-spacing:1px;'>BEST</span>"
                       if ib else "")

                mhtml += (
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'border:1px solid;border-radius:10px;padding:14px 20px;margin-bottom:8px;{bdr}transition:all .25s;cursor:default;">'
                    f'<div><div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                    f'<span style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#e8eef8;">{mn}</span>'
                    f'<span style="background:{tc}12;border:1px solid {tc}22;color:{tc};border-radius:100px;padding:2px 10px;font-family:IBM Plex Mono,monospace;font-size:8.5px;letter-spacing:1px;">{mt}</span>'
                    f'</div><div style="font-family:IBM Plex Mono,monospace;font-size:9.5px;color:#4a6380;">'
                    f'F1 {md.get("f1",0)}% &nbsp;&middot;&nbsp; Precision {md.get("precision",0)}% &nbsp;&middot;&nbsp; Recall {md.get("recall",0)}% &nbsp;&middot;&nbsp; {spd}'
                    f'</div></div>'
                    f'<div style="display:flex;align-items:center;gap:10px;">{bdg}'
                    f'<div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;'
                    f'background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{acc}%</div>'
                    f'</div></div>'
                )

            st.markdown(mhtml, unsafe_allow_html=True)

            avail = {k:v for k,v in metrics.items() if v.get("available",True) and v.get("accuracy",0)>0}
            if avail:
                fig_cmp = px.bar(x=list(avail.keys()), y=[v["accuracy"] for v in avail.values()],
                    color=[v.get("type","Classical ML") for v in avail.values()],
                    color_discrete_map={"Classical ML":"#38bdf8","NLP/Lexicon":"#34d399","Deep Learning":"#818cf8","Transformer/BERT":"#fbbf24"},
                    labels={"x":"","y":"Accuracy (%)","color":"Type"})
                fig_cmp.add_hline(y=max(v["accuracy"] for v in avail.values()),
                    line_dash="dot", line_color="#34d399", opacity=0.5)
                fig_cmp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8fa8c8", legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8fa8c8")),
                    xaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380"),
                    yaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380", range=[0,108]),
                    margin=dict(t=16,b=10,l=10,r=10), height=280)
                st.plotly_chart(fig_cmp, use_container_width=True, key="fig_cmp")
            st.markdown("</div>", unsafe_allow_html=True)

            # Dataset preview — sorted so all sources appear, not just the first 50 rows
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='label'>Dataset Preview</div>", unsafe_allow_html=True)
            show = [c for c in ["Scheme","Source","Lang","Comment","Translated","IsSarcasm","Sentiment"] if c in df.columns]
            df_preview = df[show].copy()
            if "Source" in df_preview.columns:
                df_preview = df_preview.sort_values("Source").reset_index(drop=True)
            st.dataframe(df_preview.head(100), use_container_width=True, height=320)
            # Source distribution summary
            if "Source" in df.columns:
                src_counts = df["Source"].value_counts().reset_index()
                src_counts.columns = ["Source","Rows"]
                src_counts["% of total"] = (src_counts["Rows"] / len(df) * 100).round(1).astype(str) + "%"
                st.dataframe(src_counts, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif not st.session_state.analysis_done:
            st.markdown("""
            <div style="text-align:center;padding:80px 20px;">
                <div style="font-size:36px;margin-bottom:14px;opacity:.4;">📊</div>
                <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#8fa8c8;margin-bottom:8px;">Select a scheme and run analysis</div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;color:#4a6380;">Choose from 40 government schemes and click Run Analysis to begin</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — LIVE PROBE
    # ══════════════════════════════════════════════════════════════════════════
    with t2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Live Sentiment Analysis</div>", unsafe_allow_html=True)

        import modules.model as _model_module
        if st.session_state.get("analysis_done") and st.session_state.get("best_name_store"):
            st.markdown(f"""
            <div style="display:inline-flex;align-items:center;gap:8px;
                 background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);
                 border-radius:8px;padding:7px 16px;margin-bottom:16px;">
              <span style="width:6px;height:6px;border-radius:50%;background:#34d399;
                    box-shadow:0 0 8px #34d399;display:inline-block;animation:livePulse 2s ease infinite;"></span>
              <span style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#34d399;letter-spacing:2px;">
                ACTIVE MODEL: {st.session_state.best_name_store.upper()}
              </span>
            </div>
            <style>@keyframes livePulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(.6)}}}}</style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:inline-flex;align-items:center;gap:8px;
                 background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);
                 border-radius:8px;padding:7px 16px;margin-bottom:16px;">
              <span style="width:6px;height:6px;border-radius:50%;background:#fbbf24;display:inline-block;"></span>
              <span style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#fbbf24;letter-spacing:2px;">
                USING FALLBACK — Run Analysis first to activate best ML model
              </span>
            </div>""", unsafe_allow_html=True)

        comment_input = st.text_area(
            "Comment", height=120, label_visibility="collapsed", key="live_comment",
            placeholder='Enter a comment in any language — English, Hindi, Tamil, Hinglish…'
        )

        p1, _ = st.columns([1, 5])
        with p1:
            probe = st.button("Analyse →", key="btn_probe")

        if probe and comment_input.strip():
            result     = predict_live_with_confidence(comment_input.strip())
            sent       = result["sentiment"]
            conf       = result["confidence"]
            sarc       = result["is_sarcastic"]
            ss         = result["sarcasm_score"]
            model_used = result["model_used"]
            lang       = result.get("language", "en")
            translated = result.get("translated", "")

            cmap = {"Positive":"#34d399","Negative":"#fb7185","Neutral":"#fbbf24"}
            imap = {"Positive":"↑","Negative":"↓","Neutral":"→"}
            clr  = cmap.get(sent, "#8fa8c8")
            ico  = imap.get(sent, "·")

            st.markdown(f"""
            <div style="margin:20px 0;padding:32px;border-radius:14px;
                 background:linear-gradient(135deg,rgba(13,20,32,0.9),rgba(18,26,40,0.7));
                 border:1px solid {clr}30;text-align:center;box-shadow:0 0 30px {clr}10;">
              <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:800;
                   color:{clr};letter-spacing:-1.5px;margin-bottom:10px;text-shadow:0 0 30px {clr}40;">
                {ico} {sent}
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#4a6380;letter-spacing:2px;">
                {conf}% CONFIDENCE &nbsp;·&nbsp; {model_used[:80]}
              </div>
            </div>""", unsafe_allow_html=True)

            d1, d2, d3, d4 = st.columns(4)
            lang_display = lang.upper() if lang else "EN"
            sarc_display = f"Detected ({ss}%)" if sarc else f"None ({ss}%)"
            sarc_color   = "#fb7185" if sarc else "#34d399"

            for col, (lbl, val, c) in zip([d1,d2,d3,d4],[
                ("Language",   lang_display,               "#38bdf8"),
                ("Sarcasm",    sarc_display,               sarc_color),
                ("Confidence", f"{conf}%",                 "#a78bfa"),
                ("Signal",     "Sarcasm ✓" if sarc else "Clean ✓", "#34d399"),
            ]):
                col.markdown(f"""<div class="mcard">
                    <div style="font-size:14px;font-weight:600;color:{c};font-family:'Inter',sans-serif;margin-bottom:5px;">{val}</div>
                    <div class="mlbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            if translated and translated.strip() and translated.strip().lower() != comment_input.strip().lower():
                st.markdown(f"""
                <div style="margin-top:14px;background:rgba(56,189,248,0.04);
                     border:1px solid rgba(56,189,248,0.12);border-radius:10px;padding:16px 20px;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#38bdf8;
                       letter-spacing:2.5px;text-transform:uppercase;margin-bottom:7px;">
                       Translated ({lang.upper()} → EN)</div>
                  <div style="font-family:'Inter',sans-serif;font-size:14px;color:#8fa8c8;font-style:italic;">"{translated}"</div>
                </div>""", unsafe_allow_html=True)

        elif probe:
            st.warning("Enter some text to analyse.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Quick Test Examples</div>", unsafe_allow_html=True)
        exs = [
            ("Positive",  "The PMAY scheme has genuinely helped millions of rural families."),
            ("Negative",  "यह योजना सिर्फ नाम की है, जमीन पर कुछ नहीं होता।"),
            ("Neutral",   "Scheme ke baare mein suna hai, apply nahi kiya abhi."),
            ("Sarcasm",   "Oh wow, another GREAT scheme that will DEFINITELY help everyone! 🙄"),
            ("Sarcasm",   "Sure sure, PM Kisan money definitely reached the farmers 😒"),
        ]
        ec    = st.columns(len(exs))
        cmap2 = {"Positive":"#34d399","Negative":"#fb7185","Neutral":"#fbbf24","Sarcasm":"#a78bfa"}
        for col, (lbl, txt) in zip(ec, exs):
            c = cmap2.get(lbl, "#8fa8c8")
            col.markdown(f"""
            <div style="background:var(--bg3);border:1px solid var(--border);border-top:2px solid {c};
                 border-radius:10px;padding:14px;transition:border-color .2s;cursor:default;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:8.5px;color:{c};letter-spacing:2px;text-transform:uppercase;margin-bottom:7px;">{lbl}</div>
              <div style="font-family:'Inter',sans-serif;font-size:12px;color:#8fa8c8;line-height:1.6;">{txt[:70]}{"…" if len(txt)>70 else ""}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — PLATFORMS
    # ══════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Cross-Platform Sentiment Comparison</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
          <div style="background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.2);border-radius:8px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#38bdf8;letter-spacing:1.5px;">▶ YouTube — Official API</div>
          <div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);border-radius:8px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#34d399;letter-spacing:1.5px;">▶ News App — Official API</div>
          <div style="background:rgba(129,140,248,0.07);border:1px solid rgba(129,140,248,0.2);border-radius:8px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#818cf8;letter-spacing:1.5px;">▶ Google News — RSS Feed</div>
          <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);border-radius:8px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#fbbf24;letter-spacing:1.5px;">▶ Dainik Bhaskar — RSS</div>
          <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);border-radius:8px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#fbbf24;letter-spacing:1.5px;">▶ Amar Ujala — RSS</div>
          <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);border-radius:8px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#fbbf24;letter-spacing:1.5px;">▶ NDTV Hindi — RSS</div>
          <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);border-radius:8px;padding:8px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#fbbf24;letter-spacing:1.5px;">▶ ABP Live — RSS</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.df_store is not None:
            df = st.session_state.df_store

            df_plat = df[df["Source"].isin(VALID_SOURCES)].copy()
            if df_plat.empty:
                df_plat = df.copy()

            if "Source" in df_plat.columns and df_plat["Source"].nunique() >= 1:
                # Percentage breakdown table
                ps = (df_plat.groupby("Source")["Sentiment"]
                      .value_counts(normalize=True).mul(100).round(1)
                      .unstack(fill_value=0).reset_index())
                st.dataframe(ps, use_container_width=True, hide_index=True)

                # Heatmap — auto-height scales to number of sources
                srcs  = sorted(df_plat["Source"].unique().tolist())
                sents = ["Positive","Negative","Neutral"]
                z, t  = [], []
                for s in srcs:
                    sub = df_plat[df_plat["Source"]==s]["Sentiment"].value_counts(normalize=True).mul(100)
                    row = [round(sub.get(x,0),1) for x in sents]
                    z.append(row); t.append([f"{v}%" for v in row])
                fh = go.Figure(go.Heatmap(z=z, x=sents, y=srcs, text=t, texttemplate="%{text}",
                    colorscale=[[0,"#080c14"],[0.4,"#0ea5e9"],[1,"#34d399"]], showscale=True))
                fh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8fa8c8", margin=dict(t=10,b=10,l=10,r=10),
                    height=max(300, 50 * len(srcs) + 80))
                st.plotly_chart(fh, use_container_width=True, key="fig_plat_heat")

                # Grouped bar chart — all sources
                ss2 = df_plat.groupby(["Source","Sentiment"]).size().reset_index(name="Count")
                fs  = px.bar(ss2, x="Source", y="Count", color="Sentiment", barmode="group",
                    color_discrete_map={"Positive":"#34d399","Negative":"#fb7185","Neutral":"#fbbf24","Sarcasm":"#a78bfa"})
                fs.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8fa8c8", legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8fa8c8")),
                    xaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380", tickangle=-30),
                    yaxis=dict(gridcolor="rgba(56,189,248,0.06)", color="#4a6380"),
                    margin=dict(t=10,b=60,l=10,r=10), height=320)
                st.plotly_chart(fs, use_container_width=True, key="fig_src")

                # Row count per source
                st.markdown("<div class='label' style='margin-top:18px;'>Source Row Count</div>", unsafe_allow_html=True)
                src_cnt = df_plat["Source"].value_counts().reset_index()
                src_cnt.columns = ["Source","Rows"]
                fc = px.bar(src_cnt, x="Source", y="Rows",
                    color_discrete_sequence=["#38bdf8"],
                    labels={"Source":"","Rows":"Rows"})
                fc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#8fa8c8", showlegend=False,
                    xaxis=dict(showgrid=False, color="#4a6380", tickangle=-30),
                    yaxis=dict(showgrid=False, color="#4a6380"),
                    margin=dict(t=10,b=60,l=5,r=5), height=200)
                st.plotly_chart(fc, use_container_width=True, key="fig_src_cnt")
            else:
                st.info("Run analysis first — or fetch data from at least one source to see platform comparison.")
        else:
            st.info("Run analysis first to see platform comparison.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — DATA
    # ══════════════════════════════════════════════════════════════════════════
    with t4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Fetch Live Data</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:18px;">
          <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:18px;margin-bottom:4px;">📺</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#38bdf8;letter-spacing:1.5px;">YOUTUBE</div>
            <div style="font-family:'Inter',sans-serif;font-size:10px;color:#4a6380;margin-top:3px;">Official API · Real-time</div>
          </div>
          <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:18px;margin-bottom:4px;">📰</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#34d399;letter-spacing:1.5px;">NEWS APP</div>
            <div style="font-family:'Inter',sans-serif;font-size:10px;color:#4a6380;margin-top:3px;">Official API · Real-time</div>
          </div>
          <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:18px;margin-bottom:4px;">🌐</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#818cf8;letter-spacing:1.5px;">GOOGLE NEWS</div>
            <div style="font-family:'Inter',sans-serif;font-size:10px;color:#4a6380;margin-top:3px;">RSS Feed · No key needed</div>
          </div>
          <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:18px;margin-bottom:4px;">🗞️</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#fbbf24;letter-spacing:1.5px;">HINDI NEWS RSS</div>
            <div style="font-family:'Inter',sans-serif;font-size:10px;color:#4a6380;margin-top:3px;">Bhaskar · Ujala · NDTV · ABP</div>
          </div>
        </div>""", unsafe_allow_html=True)

        f1, f2, f3 = st.columns([3, 1, 1])
        with f1:
            fs = st.selectbox("Source scheme", ["All Schemes"] + ALL_SCHEMES, key="fetch_scheme", label_visibility="collapsed")
        with f2:
            mx = st.number_input("Max / source", 50, 500, 200, 50, key="max_src", label_visibility="collapsed")
        with f3:
            fb = st.button("Fetch →", key="btn_fetch")

        if fb:
            ll = []; lb = st.empty()
            def _lg(m):
                ll.append(m)
                lb.markdown(
                    "<div style='background:var(--bg3);border:1px solid var(--border2);border-radius:10px;"
                    "padding:14px 18px;font-family:IBM Plex Mono,monospace;font-size:11px;color:#8fa8c8;"
                    "line-height:1.9;max-height:180px;overflow-y:auto;'>"
                    + "<br>".join(f"› {x}" for x in ll[-10:]) + "</div>", unsafe_allow_html=True)

            with st.spinner("Fetching from YouTube, NewsAPI, Google News RSS, Hindi News RSS…"):
                cnts = fetch_all(scheme="All" if "All" in fs else fs,
                                 max_per_source=int(mx), progress_callback=_lg)

            tot = sum(cnts.values())

            source_icons = {
                "YouTube":"📺","News App":"📰","Google News":"🌐",
                "Hindi News":"🗞️","Dainik Bhaskar":"🗞️","Amar Ujala":"🗞️",
                "Navbharat Times":"🗞️","Jagran":"🗞️","NDTV Hindi":"🗞️","ABP Live":"🗞️",
            }
            cols = st.columns(max(len(cnts), 1))
            for col, (src, cnt) in zip(cols, cnts.items()):
                icon = source_icons.get(src, "📡")
                col.markdown(f"""<div class="mcard" style="margin-top:10px;">
                    <div style="font-size:22px;margin-bottom:4px;">{icon}</div>
                    <div class="mval" style="font-size:24px;">{cnt}</div>
                    <div class="mlbl">{src}</div>
                </div>""", unsafe_allow_html=True)

            if tot > 0:
                # ── KEY FIX: bust all caches and mark analysis stale ──────────
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state.analysis_done   = False
                st.session_state.df_store        = None
                st.session_state.results_store   = None
                st.session_state.best_name_store = None
                st.session_state.metrics_store   = None
                st.session_state.fetch_done      = True
                st.success(f"✓ {tot} total items fetched and saved. Go to the Analysis tab and click Run Analysis → to include new data.")
            else:
                st.warning("No data fetched — check YOUTUBE_API_KEY and NEWS_API_KEY in secrets/env. Google News RSS and Hindi News RSS require no keys.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Dataset Status — reads LIVE from storage every time ───────────────
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Dataset Status</div>", unsafe_allow_html=True)

        try:
            ds = _storage_load_data()   # always fresh — no cache here

            if ds is None or ds.empty:
                st.info("No dataset found. Run: python data/generate_data.py  (or fetch data above)")
            else:
                old_sources = set(ds["Source"].unique()) - VALID_SOURCES if "Source" in ds.columns else set()
                active_src_count = len(VALID_SOURCES & set(ds["Source"].unique())) if "Source" in ds.columns else 0
                lang_count = ds["Language"].nunique() if "Language" in ds.columns else "—"

                d1, d2, d3, d4 = st.columns(4)
                for col, (v, l, c) in zip([d1,d2,d3,d4],[
                    (len(ds),               "Total Rows",    "#38bdf8"),
                    (ds["Scheme"].nunique(), "Scheme Values", "#34d399"),
                    (active_src_count,      "Active Sources","#a78bfa"),
                    (lang_count,            "Languages",     "#fbbf24"),
                ]):
                    col.markdown(f"""<div class="mcard">
                        <div class="mval" style="font-size:26px;">{v}</div>
                        <div class="mlbl">{l}</div>
                    </div>""", unsafe_allow_html=True)

                if ds["Scheme"].nunique() > 40:
                    st.info(f"ℹ️ Scheme count shows {ds['Scheme'].nunique()} because data.csv contains both short names (from generate_data.py) and full names (from scraper.py). Run data/fix_data.py to normalise.")

                if old_sources:
                    st.info(f"ℹ️ Legacy source(s) still in dataset: {', '.join(sorted(old_sources))}. Run data/fix_data.py to remove them.")

                # Source distribution chart — current sources highlighted
                if "Source" in ds.columns:
                    sc2 = ds["Source"].value_counts().reset_index()
                    sc2.columns = ["Source", "Count"]
                    sc2["Status"] = sc2["Source"].apply(
                        lambda x: "Current" if x in VALID_SOURCES else "Legacy"
                    )
                    fs2 = px.bar(sc2, x="Source", y="Count", color="Status",
                        color_discrete_map={"Current":"#38bdf8","Legacy":"#2a3a50"},
                        labels={"Source":"","Count":"Count","Status":"Status"})
                    fs2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#8fa8c8",
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8fa8c8")),
                        xaxis=dict(showgrid=False, color="#4a6380", tickangle=-30),
                        yaxis=dict(showgrid=False, color="#4a6380"),
                        margin=dict(t=10,b=70,l=5,r=5), height=220)
                    st.plotly_chart(fs2, use_container_width=True, key="fig_ds_src")

        except Exception as e:
            st.error(f"Could not load dataset: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — ABOUT
    # ══════════════════════════════════════════════════════════════════════════
    with t5:
        st.markdown("""
        <div class="card" style="text-align:center;padding:40px 36px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:#38bdf8;letter-spacing:3px;text-transform:uppercase;margin-bottom:16px;">Final Year Research Project</div>
            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#e8eef8;line-height:1.4;margin-bottom:12px;letter-spacing:-.5px;">
                Multilingual Multi-Source Sentiment Analysis Framework<br>
                for Indian Government Schemes Using Adaptive Model Selection
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#4a6380;letter-spacing:1.5px;">
                ML · DL · NLP · Transformers &nbsp;·&nbsp; 5 Research Gaps Addressed
            </div>
        </div>""", unsafe_allow_html=True)

        gaps = [
            ("01","Language Barrier","#38bdf8",
             "Most research processes only English text, ignoring India's 22 official languages and Hinglish.",
             "LangDetect + deep-translator pipeline supports English, Hindi, Tamil, Telugu, Bengali, Hinglish, and more. Non-English text is auto-translated before analysis."),
            ("02","Single Platform","#818cf8",
             "95% of published papers analyse a single source, missing the broader public discourse across platforms.",
             "Multi-source architecture covers YouTube (Official API), NewsAPI (Official API), Google News RSS (real-time English headlines), and Hindi News RSS — Dainik Bhaskar, Amar Ujala, Navbharat Times, Jagran, NDTV Hindi, ABP Live. Cross-platform comparison in the Platforms tab."),
            ("03","Random Model Selection","#34d399",
             "Researchers arbitrarily select algorithms without benchmarking them against the specific dataset characteristics.",
             "Adaptive engine profiles data volume, sarcasm ratio, language diversity, and avg comment length. Trains 7 classical ML models + TextBlob + VADER. Best model selected by data-aware scoring. Optional LSTM, BiLSTM, CNN, ALBERT, DistilBERT."),
            ("04","Binary Classification","#fbbf24",
             "Most studies output only Positive / Negative, discarding the Neutral class that represents a large portion of real public opinion.",
             "Three-class classification: Positive / Neutral / Negative with per-class F1, Precision, and Recall metrics for each model."),
            ("05","Sarcasm Ignored","#fb7185",
             "Standard sentiment models misclassify sarcastic comments as positive because they read literal word polarity.",
             "Advanced sarcasm engine with 30+ pattern rules including Digital India irony, bureaucracy loop detection, amount irony (₹2000 retire), and multilingual Hindi sarcasm phrases. Returns 0–100% confidence. Sarcastic positives auto-flipped to Negative."),
        ]

        for gid, title, clr, problem, solution in gaps:
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {clr};padding:24px 28px;margin-bottom:12px;">
              <div style="display:flex;align-items:flex-start;gap:18px;">
                <div style="min-width:38px;height:38px;border-radius:9px;background:{clr}10;border:1px solid {clr}22;
                     display:flex;align-items:center;justify-content:center;font-family:'IBM Plex Mono',monospace;
                     font-size:11px;font-weight:500;color:{clr};flex-shrink:0;box-shadow:0 0 12px {clr}15;">{gid}</div>
                <div style="flex:1;">
                  <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#e8eef8;margin-bottom:12px;letter-spacing:-.3px;">{title}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">
                    <div>
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:8.5px;color:#4a6380;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:6px;">Problem</div>
                      <div style="font-family:'Inter',sans-serif;font-size:13px;color:#8fa8c8;line-height:1.7;">{problem}</div>
                    </div>
                    <div>
                      <div style="font-family:'IBM Plex Mono',monospace;font-size:8.5px;color:{clr};letter-spacing:2.5px;text-transform:uppercase;margin-bottom:6px;opacity:0.8;">Solution</div>
                      <div style="font-family:'Inter',sans-serif;font-size:13px;color:#e8eef8;line-height:1.7;">{solution}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px;font-family:'IBM Plex Mono',monospace;
         font-size:9px;color:#2a3a50;letter-spacing:3px;text-transform:uppercase;">
        Pulse Sentiment AI &nbsp;·&nbsp; Research Edition &nbsp;·&nbsp; 2025–26
    </div>""", unsafe_allow_html=True)
