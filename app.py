"""
app.py — Pulse Sentiment AI · Main Application
═══════════════════════════════════════════════
Run:  streamlit run app.py
"""
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
from modules.scraper     import ALL_SCHEMES, fetch_all, fetch_instagram_post
from auth.auth_manager   import login, signup, get_google_auth_url

st.set_page_config(page_title="Pulse · Sentiment AI", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI         = "http://localhost:8501"

# ── Scheme emoji map ──────────────────────────────────────────────────────────
SCHEME_EMOJI = {
    "PMAY — Pradhan Mantri Awas Yojana":"🏘️",
    "Ayushman Bharat — PM-JAY":"🏥",
    "Poshan Abhiyaan — Nutrition Mission":"🥗",
    "Ayushman Bharat Digital Mission":"💊",
    "PM Kisan Samman Nidhi":"🌾",
    "Fasal Bima — PM Crop Insurance":"🌱",
    "Kisan Credit Card":"💳",
    "e-NAM — National Agriculture Market":"🛒",
    "Digital India Initiative":"💡",
    "BharatNet — Rural Internet":"🌐",
    "UPI — Unified Payments Interface":"📱",
    "Jan Dhan Yojana — Financial Inclusion":"🏦",
    "Mudra Yojana — MSME Loans":"💰",
    "Stand Up India Scheme":"📈",
    "Atal Pension Yojana":"👴",
    "PM Jeevan Jyoti Bima":"🛡️",
    "PM Suraksha Bima":"🔐",
    "Ujjwala Yojana — LPG for Poor":"🔥",
    "Saubhagya — Household Electrification":"⚡",
    "Solar Rooftop — PM Surya Ghar":"☀️",
    "FAME — Electric Vehicle Scheme":"🚗",
    "Swachh Bharat Mission":"♻️",
    "Jal Jeevan Mission — Har Ghar Jal":"💧",
    "AMRUT — Urban Development":"🏙️",
    "Skill India — PMKVY":"🎓",
    "Startup India":"🚀",
    "Make in India":"🏭",
    "PM eVIDYA — Digital Education":"📚",
    "Beti Bachao Beti Padhao":"👧",
    "Sukanya Samriddhi Yojana":"🌸",
    "PM Matru Vandana — Maternity Benefit":"🤱",
    "Pradhan Mantri Gram Sadak Yojana":"🛣️",
    "Bharatmala — Highway Project":"🛤️",
    "Smart Cities Mission":"🏢",
    "Sagarmala — Port Development":"⚓",
    "One Nation One Ration Card":"🍚",
    "PM Garib Kalyan Anna Yojana":"🌽",
    "PM SVANidhi — Street Vendor Loan":"🛍️",
    "Vishwakarma Yojana":"🔨",
    "Atmanirbhar Bharat":"🇮🇳",
}

MODEL_TYPE_COLORS = {
    "Classical ML":     "#38bdf8",
    "NLP/Lexicon":      "#34d399",
    "Deep Learning":    "#818cf8",
    "Transformer/BERT": "#f59e0b",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Outfit:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,.stApp{background:#05080f!important;color:#dde3ef;font-family:'Outfit',sans-serif}
header,footer{visibility:hidden!important}
#MainMenu,.stDeployButton{display:none!important}
[data-testid="stSidebar"]{display:none!important}
.stApp::before{content:"";position:fixed;inset:0;
  background:radial-gradient(ellipse 80% 60% at 20% 10%,rgba(14,165,233,.10) 0%,transparent 60%),
             radial-gradient(ellipse 70% 50% at 80% 80%,rgba(99,102,241,.10) 0%,transparent 60%),
             radial-gradient(ellipse 50% 40% at 50% 50%,rgba(16,185,129,.05) 0%,transparent 70%),
             linear-gradient(135deg,#05080f 0%,#0a0f1e 100%);
  animation:bgPulse 12s ease-in-out infinite alternate;pointer-events:none;z-index:0}
@keyframes bgPulse{0%{opacity:1}100%{opacity:.85}}
.stApp::after{content:"";position:fixed;inset:0;
  background-image:linear-gradient(rgba(56,189,248,.025) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(56,189,248,.025) 1px,transparent 1px);
  background-size:64px 64px;pointer-events:none;z-index:0}
.block-container{position:relative;z-index:2;padding:1.5rem 2.5rem 4rem!important;max-width:1400px!important}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:#f1f5f9!important;letter-spacing:-.5px}
.g-panel{background:rgba(255,255,255,.028);border:1px solid rgba(255,255,255,.07);border-radius:20px;
  padding:28px 32px;backdrop-filter:blur(20px);position:relative;overflow:hidden;
  transition:border-color .3s,box-shadow .3s;margin-bottom:20px}
.g-panel::before{content:"";position:absolute;top:0;left:10%;right:10%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(56,189,248,.4),transparent)}
.g-panel:hover{border-color:rgba(56,189,248,.18);box-shadow:0 0 50px rgba(56,189,248,.06),0 24px 60px rgba(0,0,0,.45)}
.eyebrow{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;
  color:#38bdf8;display:flex;align-items:center;gap:10px;margin-bottom:18px}
.eyebrow::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,rgba(56,189,248,.35),transparent)}
.hero-wrap{text-align:center;padding:52px 20px 36px}
.hero-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(56,189,248,.08);
  border:1px solid rgba(56,189,248,.22);border-radius:100px;padding:6px 20px;font-family:'DM Mono',monospace;
  font-size:11px;letter-spacing:2.5px;color:#38bdf8;margin-bottom:22px;text-transform:uppercase}
.live-dot{width:6px;height:6px;border-radius:50%;background:#34d399;animation:livePulse 2s ease infinite;display:inline-block}
@keyframes livePulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.6;transform:scale(.7)}}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(38px,5.5vw,72px);font-weight:800;
  line-height:1.0;letter-spacing:-2px;
  background:linear-gradient(130deg,#f8fafc 10%,#38bdf8 40%,#818cf8 65%,#34d399 90%);
  background-size:250% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;animation:textShimmer 7s linear infinite;margin-bottom:14px}
@keyframes textShimmer{0%{background-position:0% center}100%{background-position:250% center}}
.hero-sub{font-family:'DM Mono',monospace;font-size:12px;color:#475569;letter-spacing:2px;text-transform:uppercase}
.h-div{height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.25),rgba(99,102,241,.25),transparent);margin:24px 0}
.metric-card{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:18px;
  padding:24px 20px;text-align:center;position:relative;overflow:hidden;transition:all .3s}
.metric-card::after{content:"";position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#0ea5e9,#6366f1);transform:scaleX(0);transform-origin:left;transition:transform .3s}
.metric-card:hover{border-color:rgba(56,189,248,.2)}
.metric-card:hover::after{transform:scaleX(1)}
.metric-val{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;
  background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;line-height:1.1}
.metric-lbl{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
  text-transform:uppercase;color:#475569;margin-top:6px}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{
  background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.10)!important;
  border-radius:14px!important;color:#e2e8f0!important;font-family:'Outfit',sans-serif!important;
  font-size:14px!important;padding:13px 18px!important;transition:border-color .3s,box-shadow .3s!important}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{
  border-color:rgba(56,189,248,.5)!important;box-shadow:0 0 0 3px rgba(56,189,248,.09)!important}
.stTextInput>label,.stTextArea>label,.stSelectbox>label{
  font-family:'DM Mono',monospace!important;font-size:10px!important;
  letter-spacing:2.5px!important;text-transform:uppercase!important;color:#64748b!important}
.stSelectbox>div>div{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.10)!important;
  border-radius:14px!important;color:#e2e8f0!important;font-family:'Outfit',sans-serif!important}
.stButton>button{background:rgba(14,165,233,.12)!important;border:1px solid rgba(14,165,233,.35)!important;
  border-radius:12px!important;color:#e2e8f0!important;font-family:'DM Mono',monospace!important;
  font-size:11px!important;letter-spacing:2px!important;text-transform:uppercase!important;
  padding:14px 24px!important;width:100%!important;transition:all .25s!important}
.stButton>button:hover{background:rgba(14,165,233,.22)!important;border-color:rgba(14,165,233,.65)!important;
  box-shadow:0 0 28px rgba(14,165,233,.18),0 8px 24px rgba(0,0,0,.3)!important;
  transform:translateY(-2px)!important;color:#fff!important}
div[data-testid="column"]:first-child .stButton>button{
  background:linear-gradient(135deg,#0ea5e9,#6366f1)!important;border:none!important;
  color:white!important;font-weight:600!important;box-shadow:0 6px 24px rgba(14,165,233,.28)!important}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.03)!important;border-radius:14px!important;
  padding:4px!important;gap:4px!important;border:1px solid rgba(255,255,255,.07)!important}
.stTabs [data-baseweb="tab"]{font-family:'DM Mono',monospace!important;font-size:11px!important;
  letter-spacing:1.5px!important;text-transform:uppercase!important;color:#64748b!important;
  border-radius:10px!important;padding:10px 20px!important;transition:all .2s!important}
.stTabs [aria-selected="true"]{background:rgba(14,165,233,.15)!important;color:#38bdf8!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important}
::-webkit-scrollbar{width:4px;background:#05080f}
::-webkit-scrollbar-thumb{background:linear-gradient(#38bdf8,#818cf8);border-radius:4px}
#status-bar{position:fixed;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#0ea5e9,#6366f1,#34d399,#0ea5e9);
  background-size:300% auto;animation:shimBar 4s linear infinite;z-index:9999}
@keyframes shimBar{0%{background-position:0% center}100%{background-position:300% center}}
.google-btn{display:flex;align-items:center;justify-content:center;gap:12px;
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.14);border-radius:12px;
  padding:13px 24px;color:#e2e8f0;font-family:'Outfit',sans-serif;font-size:14px;font-weight:500;
  cursor:pointer;transition:all .25s;text-decoration:none;width:100%;text-align:center}
.google-btn:hover{background:rgba(255,255,255,.10);transform:translateY(-2px);color:white;text-decoration:none}
.stDataFrame{border:1px solid rgba(255,255,255,.07)!important;border-radius:14px!important;overflow:hidden!important}
</style>
<div id="status-bar"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, val in [
    ("logged_in", False), ("user_info", {}), ("auth_mode", "login"),
    ("analysis_done", False), ("model_store", None), ("df_store", None),
    ("results_store", None), ("best_name_store", None), ("metrics_store", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge"><span class="live-dot"></span> Pulse Sentiment AI &nbsp;·&nbsp; Research Edition</div>
  <div class="hero-title">Read the Nation.<br>Understand Every Voice.</div>
  <div class="hero-sub">Multilingual · Multi-Platform · Sarcasm-Aware · Adaptive ML + DL + NLP</div>
</div>
<div class="h-div"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    params = st.query_params
    if "code" in params and GOOGLE_CLIENT_ID:
        from auth.auth_manager import exchange_google_code
        user_info = exchange_google_code(params["code"], GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI)
        if user_info:
            st.session_state.logged_in = True
            st.session_state.user_info = {"name": user_info.get("name","Google User"),
                                           "email": user_info.get("email",""), "role":"user","avatar":"🌐"}
            st.query_params.clear(); st.rerun()

    lc, cc, rc = st.columns([1, 1.4, 1])
    with cc:
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Sign In"):  st.session_state.auth_mode = "login"
        with col_b:
            if st.button("Create Account"): st.session_state.auth_mode = "signup"
        st.markdown("<div class='h-div'></div>", unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:4px;">Welcome back</div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-bottom:22px;">Sign in to your account</div>""", unsafe_allow_html=True)
            uname = st.text_input("Username", placeholder="admin", key="li_user")
            passw = st.text_input("Password", type="password", placeholder="••••••", key="li_pass")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("→  Sign In", key="btn_login"):
                    ok, msg, info = login(uname, passw)
                    if ok:
                        st.session_state.logged_in = True; st.session_state.user_info = info; st.rerun()
                    else: st.error(f"⟨ {msg} ⟩")
            with c2:
                if st.button("Forgot Password?", key="btn_forgot"):
                    st.info("Please contact admin@pulse.ai to reset your password.")
        else:
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:4px;">Create account</div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-bottom:22px;">Join Pulse Sentiment AI</div>""", unsafe_allow_html=True)
            su_name  = st.text_input("Full Name",  placeholder="Riya Sharma",  key="su_name")
            su_email = st.text_input("Email",       placeholder="you@email.com", key="su_email")
            su_user  = st.text_input("Username",    placeholder="riya123",       key="su_user")
            su_pass  = st.text_input("Password",    type="password", placeholder="min 4 chars", key="su_pass")
            c1, _ = st.columns(2)
            with c1:
                if st.button("→  Create Account", key="btn_signup"):
                    ok, msg = signup(su_user, su_pass, su_name, su_email)
                    if ok: st.success(f"✅ {msg}"); st.session_state.auth_mode="login"; st.rerun()
                    else:  st.error(f"⟨ {msg} ⟩")

        st.markdown("<div class='h-div'></div>", unsafe_allow_html=True)
        if GOOGLE_CLIENT_ID:
            g_url = get_google_auth_url(GOOGLE_CLIENT_ID, REDIRECT_URI)
            st.markdown(f"""<a href="{g_url}" class="google-btn">
                <svg width="18" height="18" viewBox="0 0 48 48">
                  <path fill="#EA4335" d="M24 9.5c3.3 0 5.9 1.4 7.7 2.6l5.7-5.7C33.9 3.5 29.3 1.5 24 1.5 14.8 1.5 7 7.4 3.7 15.5l6.7 5.2C12 15.1 17.5 9.5 24 9.5z"/>
                  <path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h12.4c-.5 2.8-2.1 5.2-4.5 6.8l7 5.4c4.1-3.8 6.2-9.4 6.2-16.2z"/>
                  <path fill="#FBBC05" d="M10.4 28.4A14.3 14.3 0 0 1 9.5 24c0-1.5.3-3 .7-4.3L3.7 14.5A22.5 22.5 0 0 0 1.5 24c0 3.6.9 7 2.2 10l6.7-5.6z"/>
                  <path fill="#34A853" d="M24 46.5c5.3 0 9.7-1.7 12.9-4.7l-7-5.4c-1.7 1.1-3.9 1.8-5.9 1.8-6.4 0-11.9-4.3-13.8-10.1l-6.8 5.3C7 41 15 46.5 24 46.5z"/>
                </svg>
                Continue with Google
            </a>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center;font-family:'DM Mono',monospace;font-size:10px;color:#334155;letter-spacing:1px;margin-top:8px;">
                Set GOOGLE_CLIENT_ID in .env to enable Google login</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;font-family:'DM Mono',monospace;font-size:10px;color:#1e3a4a;margin-top:8px;letter-spacing:1px;">
         Demo credentials → username: admin &nbsp;·&nbsp; password: 1234</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
else:
    user          = st.session_state.user_info
    uname_display = user.get("name", "User")
    avatar        = user.get("avatar", "👤")

    col_t, col_u = st.columns([5, 1])
    with col_t:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:14px;">
          <div style="width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,#0ea5e9,#6366f1);
               display:flex;align-items:center;justify-content:center;font-size:20px;border:2px solid rgba(56,189,248,.3);">{avatar}</div>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#f1f5f9;letter-spacing:-.3px;">Sentiment Dashboard</div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;color:#475569;letter-spacing:2px;">WELCOME, {uname_display.upper()} · ACTIVE SESSION</div>
          </div></div>""", unsafe_allow_html=True)
    with col_u:
        if st.button("Disconnect", key="logout"):
            for k in ["logged_in","user_info","analysis_done","model_store","df_store","results_store","best_name_store","metrics_store"]:
                st.session_state[k] = False if k=="logged_in" else ({} if k=="user_info" else None)
            st.rerun()

    st.markdown("<div class='h-div'></div>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📊  Analysis Dashboard",
        "⚡  Live Probe",
        "🌍  Multi-Source View",
        "🌐  Fetch Data",
        "📄  About & Gaps",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — ANALYSIS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with t1:
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>⬡ Target Scheme</div>", unsafe_allow_html=True)

        scheme_options = ["All Schemes (Combined)"] + ALL_SCHEMES
        scheme = st.selectbox("Government Scheme", scheme_options, label_visibility="collapsed")
        emoji  = SCHEME_EMOJI.get(scheme, "🇮🇳")
        color  = "#38bdf8"
        st.markdown(f"""<div style="display:inline-flex;align-items:center;gap:8px;
             background:rgba(255,255,255,.04);border:1px solid {color}30;border-radius:100px;
             padding:6px 16px;margin:8px 0 12px;"><span style="font-size:16px;">{emoji}</span>
          <span style="font-family:'DM Mono',monospace;font-size:11px;color:{color};letter-spacing:1px;">{scheme}</span></div>""",
            unsafe_allow_html=True)

        # Model options
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            use_dl = st.checkbox("Include Deep Learning models (LSTM, BiLSTM, CNN)", value=False,
                                  help="Requires TensorFlow. Slower but more powerful.")
        with adv_col2:
            use_transformers = st.checkbox("Include Transformer models (BERT, ALBERT)", value=False,
                                            help="Requires transformers library. Very slow on CPU.")

        rc1, _ = st.columns([1, 4])
        with rc1:
            run = st.button("⟶  Run Analysis", key="run_analysis")
        st.markdown("</div>", unsafe_allow_html=True)

        if run:
            try:
                df_raw = pd.read_csv("data/data.csv")
            except FileNotFoundError:
                st.error("data/data.csv not found. Run: python data/generate_data.py")
                st.stop()

            if scheme != "All Schemes (Combined)":
                scheme_key = scheme.split("—")[0].strip().split(" ")[0]
                df_filt = df_raw[df_raw["Scheme"].str.contains(scheme_key, case=False, na=False)]
                df_raw  = df_filt if len(df_filt) >= 10 else df_raw

            with st.spinner("Running full ML + NLP pipeline…"):
                df = preprocess_dataframe(df_raw)
                results, best_name = train_models(
                    df["Cleaned"], df["Sentiment"],
                    use_dl=use_dl, use_transformers=use_transformers
                )
                metrics = results  # already has accuracy, f1, precision, recall, speed_ms, type

            st.session_state.df_store         = df
            st.session_state.results_store    = results
            st.session_state.best_name_store  = best_name
            st.session_state.metrics_store    = metrics
            st.session_state.analysis_done    = True

        # ── Display results ───────────────────────────────────────────────────
        if st.session_state.analysis_done and st.session_state.df_store is not None:
            df       = st.session_state.df_store
            results  = st.session_state.results_store
            best_name = st.session_state.best_name_store
            metrics  = st.session_state.metrics_store

            counts   = df["Sentiment"].value_counts()
            n_pos    = counts.get("Positive", 0)
            n_neg    = counts.get("Negative", 0)
            n_neu    = counts.get("Neutral",  0)
            n_sar    = int(df["IsSarcasm"].sum()) if "IsSarcasm" in df.columns else 0
            best_acc = metrics[best_name]["accuracy"] if best_name in metrics else 0

            # ── Metric cards ─────────────────────────────────────────────────
            mc = st.columns(6)
            for col, (val, lbl) in zip(mc, [
                (len(df), "Total Comments"), (n_pos, "Positive"),
                (n_neg, "Negative"), (n_neu, "Neutral"),
                (n_sar, "Sarcasm Detected"), (f"{best_acc}%", "Best Accuracy"),
            ]):
                col.markdown(f"""<div class="metric-card">
                  <div class="metric-val">{val}</div>
                  <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

            st.markdown("<div class='h-div'></div>", unsafe_allow_html=True)

            # ── Charts row 1 ─────────────────────────────────────────────────
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='eyebrow'>Sentiment Distribution</div>", unsafe_allow_html=True)
                fig_pie = px.pie(values=counts.values, names=counts.index,
                    color=counts.index,
                    color_discrete_map={"Positive":"#34d399","Negative":"#f87171","Neutral":"#fbbf24","Sarcasm":"#818cf8"},
                    hole=0.55)
                fig_pie.update_traces(textfont_color="white", textfont_size=13)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8", legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
                    margin=dict(t=10,b=10,l=10,r=10), height=320)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with ch2:
                st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='eyebrow'>Source-wise Breakdown (Gap 2 ✅)</div>", unsafe_allow_html=True)
                if "Source" in df.columns:
                    src_sent = df.groupby(["Source","Sentiment"]).size().reset_index(name="Count")
                    fig_bar = px.bar(src_sent, x="Source", y="Count", color="Sentiment",
                        color_discrete_map={"Positive":"#34d399","Negative":"#f87171","Neutral":"#fbbf24","Sarcasm":"#818cf8"},
                        barmode="group")
                    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8", legend=dict(bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                        margin=dict(t=10,b=10,l=10,r=10), height=320)
                    st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Charts row 2 ─────────────────────────────────────────────────
            ch3, ch4 = st.columns(2)
            with ch3:
                st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='eyebrow'>Language Detection (Gap 1 ✅)</div>", unsafe_allow_html=True)
                if "Lang" in df.columns:
                    lang_counts = df["Lang"].value_counts()
                    lang_labels = {"en":"English","hi":"Hindi/Hinglish","ta":"Tamil","bn":"Bengali","te":"Telugu"}
                    lang_display = lang_counts.rename(index=lambda x: lang_labels.get(x, x.upper()))
                    fig_lang = px.bar(x=lang_display.index, y=lang_display.values,
                        color=lang_display.values,
                        color_continuous_scale=["#0ea5e9","#6366f1","#34d399"],
                        labels={"x":"Language","y":"Count","color":"Count"})
                    fig_lang.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8", showlegend=False,
                        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                        margin=dict(t=10,b=10,l=10,r=10), height=280)
                    st.plotly_chart(fig_lang, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with ch4:
                st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='eyebrow'>Sarcasm Detection (Gap 5 ✅)</div>", unsafe_allow_html=True)
                sar_data = pd.DataFrame({"Type":["Non-Sarcastic","Sarcastic"],
                                          "Count":[len(df)-n_sar, n_sar]})
                fig_sar = px.bar(sar_data, x="Type", y="Count", color="Type",
                    color_discrete_map={"Non-Sarcastic":"#38bdf8","Sarcastic":"#f87171"})
                fig_sar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8", showlegend=False,
                    xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                    margin=dict(t=10,b=10,l=10,r=10), height=280)
                st.plotly_chart(fig_sar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Platform Comparison Heatmap ───────────────────────────────────
            if "Source" in df.columns and df["Source"].nunique() > 1:
                st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='eyebrow'>Platform Sentiment Comparison (Gap 2 ✅)</div>", unsafe_allow_html=True)
                sources    = df["Source"].unique().tolist()
                sentiments = ["Positive","Negative","Neutral"]
                z_data, text_data = [], []
                for src in sources:
                    sub = df[df["Source"]==src]["Sentiment"].value_counts(normalize=True).mul(100)
                    row_z = [round(sub.get(s,0),1) for s in sentiments]
                    row_t = [f"{v}%" for v in row_z]
                    z_data.append(row_z); text_data.append(row_t)
                fig_heat = go.Figure(go.Heatmap(
                    z=z_data, x=sentiments, y=sources,
                    text=text_data, texttemplate="%{text}",
                    colorscale=[[0,"#0f172a"],[0.5,"#0ea5e9"],[1,"#34d399"]], showscale=True))
                fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8", margin=dict(t=10,b=10,l=10,r=10), height=300)
                st.plotly_chart(fig_heat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Model Comparison ─────────────────────────────────────────────
            st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>Adaptive Model Selection (Gap 3 ✅)</div>", unsafe_allow_html=True)
            st.markdown("""<div style="font-family:'Outfit',sans-serif;font-size:13px;color:#64748b;margin-bottom:18px;">
            All models trained and benchmarked. Best selected by accuracy — ties broken by speed. TextBlob, VADER, LSTM, ALBERT included.</div>""",
                unsafe_allow_html=True)

            # ── KEY FIX: build ONE html string, render once ───────────────────
            model_html = ""
            for mname, mdata in sorted(metrics.items(), key=lambda x: -x[1].get("accuracy",0)):
                if not mdata.get("available", True) and mdata.get("accuracy",0) == 0:
                    continue
                is_best    = mname == best_name
                mtype      = mdata.get("type", "Classical ML")
                type_color = MODEL_TYPE_COLORS.get(mtype, "#38bdf8")
                border     = f"border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.05);" if is_best else ""
                badge      = f"<span style='background:rgba(52,211,153,.15);border:1px solid rgba(52,211,153,.3);color:#34d399;border-radius:100px;padding:3px 12px;font-family:DM Mono,monospace;font-size:10px;letter-spacing:1.5px;'>★ BEST</span>" if is_best else ""
                speed_text = f"{mdata.get('speed_ms',0):.0f}ms" if mdata.get("speed_ms",0) > 0 else "—"
                model_html += f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                     background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);
                     border-radius:12px;padding:14px 20px;margin-bottom:10px;{border}transition:all .2s;">
                  <div>
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-family:'DM Mono',monospace;font-size:13px;color:#94a3b8;">{mname}</span>
                      <span style="background:{type_color}18;border:1px solid {type_color}30;color:{type_color};
                           border-radius:100px;padding:2px 10px;font-family:'DM Mono',monospace;font-size:9px;
                           letter-spacing:1px;">{mtype}</span>
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#334155;margin-top:4px;">
                      F1: {mdata.get('f1',0)}% &nbsp;·&nbsp; Precision: {mdata.get('precision',0)}% &nbsp;·&nbsp;
                      Recall: {mdata.get('recall',0)}% &nbsp;·&nbsp; Speed: {speed_text}
                    </div>
                  </div>
                  <div style="display:flex;align-items:center;gap:12px;">
                    {badge}
                    <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#38bdf8;">{mdata.get('accuracy',0)}%</div>
                  </div>
                </div>"""
            st.markdown(model_html, unsafe_allow_html=True)

            # Bar chart
            avail = {k:v for k,v in metrics.items() if v.get("available",True) and v.get("accuracy",0)>0}
            if avail:
                fig_cmp = px.bar(x=list(avail.keys()), y=[v["accuracy"] for v in avail.values()],
                    color=[v.get("type","Classical ML") for v in avail.values()],
                    color_discrete_map=MODEL_TYPE_COLORS,
                    labels={"x":"Model","y":"Accuracy (%)","color":"Type"})
                fig_cmp.add_hline(y=max(v["accuracy"] for v in avail.values()),
                    line_dash="dot", line_color="#34d399", opacity=0.5)
                fig_cmp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
                    xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,.04)", range=[0,105]),
                    margin=dict(t=20,b=10,l=10,r=10), height=300)
                st.plotly_chart(fig_cmp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Data preview ─────────────────────────────────────────────────
            st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>Processed Dataset Preview</div>", unsafe_allow_html=True)
            show_cols = [c for c in ["Scheme","Source","Lang","Comment","Translated","IsSarcasm","Sentiment"] if c in df.columns]
            st.dataframe(df[show_cols].head(50), use_container_width=True, height=320)
            st.markdown("</div>", unsafe_allow_html=True)

        elif not st.session_state.analysis_done:
            st.markdown("""<div style="text-align:center;padding:60px 20px;color:#334155;
                font-family:'DM Mono',monospace;font-size:12px;letter-spacing:2px;">
                SELECT A SCHEME AND CLICK RUN ANALYSIS TO BEGIN</div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — LIVE PROBE
    # ══════════════════════════════════════════════════════════════════════════
    with t2:
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>⚡ Live Sentiment Probe — All 5 Gaps Active</div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Outfit',sans-serif;font-size:14px;color:#64748b;margin-bottom:20px;line-height:1.7;">
        Enter any comment in <strong style="color:#94a3b8;">any language</strong> (English, Hindi, Tamil, Hinglish…).
        System detects language, translates, runs sarcasm detection, and classifies sentiment.</div>""", unsafe_allow_html=True)

        comment_input = st.text_area("Enter comment",
            placeholder='Try: "यह योजना बहुत अच्छी है!" or "Great scheme... as if it ever works 🙄"',
            height=130, label_visibility="collapsed", key="live_comment")

        pc, _ = st.columns([1,4])
        with pc:
            probe_btn = st.button("⟶  Analyse Now", key="btn_probe")

        if probe_btn and comment_input.strip():
            lang       = detect_language(comment_input)
            translated = translate_to_english(comment_input, lang) if lang != "en" else comment_input
            cleaned    = clean_text(translated)
            result     = predict_live_with_confidence(comment_input, cleaned)

            sentiment  = result["sentiment"]
            confidence = result["confidence"]
            sarcasm    = result["is_sarcastic"]
            sarc_score = result["sarcasm_score"]
            model_used = result["model_used"]

            color_map = {"Positive":"#34d399","Negative":"#f87171","Neutral":"#fbbf24","Unknown":"#94a3b8"}
            icon_map  = {"Positive":"🟢","Negative":"🔴","Neutral":"🟡","Unknown":"⚪"}
            clr  = color_map.get(sentiment, "#94a3b8")
            icon = icon_map.get(sentiment, "⚪")

            st.markdown(f"""<div style="margin:20px 0;padding:24px 30px;border-radius:18px;
                 background:{clr}0d;border:1.5px solid {clr}40;text-align:center;">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:#64748b;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">
                   Sentiment Classification Result</div>
              <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:{clr};letter-spacing:1px;">{icon} &nbsp; {sentiment.upper()}</div>
              <div style="font-family:'DM Mono',monospace;font-size:12px;color:#475569;margin-top:8px;">
                   Confidence: {confidence}% &nbsp;·&nbsp; Model: {model_used}</div>
            </div>""", unsafe_allow_html=True)

            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
            for col, (lbl, val, c) in zip([dc1,dc2,dc3,dc4,dc5], [
                ("Language", lang.upper(), "#38bdf8"),
                ("Sarcasm", f"⚠️ YES ({sarc_score}%)" if sarcasm else f"No ({sarc_score}%)", "#f87171" if sarcasm else "#34d399"),
                ("Confidence", f"{confidence}%", "#818cf8"),
                ("Model Type", result.get("model_used","—")[:15], "#f59e0b"),
                ("Gap 5", "Sarcasm ✅" if sarcasm else "Clean ✅", "#34d399"),
            ]):
                col.markdown(f"""<div class="metric-card">
                  <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:{c};">{val}</div>
                  <div class="metric-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

            if lang != "en" and translated != comment_input:
                st.markdown(f"""<div style="margin-top:16px;background:rgba(56,189,248,.05);
                     border:1px solid rgba(56,189,248,.15);border-radius:14px;padding:16px 20px;">
                  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#38bdf8;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">Auto-Translated (Gap 1 ✅)</div>
                  <div style="font-family:'Outfit',sans-serif;font-size:14px;color:#94a3b8;font-style:italic;">"{translated}"</div>
                </div>""", unsafe_allow_html=True)

        elif probe_btn:
            st.warning("⟨ Please enter some text to analyse ⟩")
        st.markdown("</div>", unsafe_allow_html=True)

        # Examples
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Test Examples</div>", unsafe_allow_html=True)
        examples = [
            ("Positive · English",  "The PMAY scheme has truly helped millions of rural families get proper housing."),
            ("Negative · Hindi",    "यह योजना सिर्फ नाम की है, जमीन पर कुछ नहीं होता।"),
            ("Neutral · Hinglish",  "Scheme ke baare mein suna hai, apply nahi kiya abhi."),
            ("Sarcasm 🙄",           "Oh wow, another GREAT government scheme that will DEFINITELY help everyone! 🙄"),
            ("Sarcasm 😒",           "Sure sure, the PM Kisan money definitely reached the farmers 😒"),
        ]
        ex_cols = st.columns(len(examples))
        for col, (lbl, txt) in zip(ex_cols, examples):
            col.markdown(f"""<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);
                 border-radius:12px;padding:12px;font-family:'DM Mono',monospace;">
              <div style="font-size:9px;color:#38bdf8;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">{lbl}</div>
              <div style="font-size:12px;color:#94a3b8;line-height:1.5;">{txt[:80]}{"…" if len(txt)>80 else ""}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — MULTI-SOURCE VIEW
    # ══════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>🌍 Multi-Platform Opinion Analysis (Gap 2 ✅)</div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Outfit',sans-serif;font-size:14px;color:#64748b;margin-bottom:20px;line-height:1.7;">
        Unlike existing research that analyzes <em>only Twitter</em>, Pulse collects and compares sentiment across
        <strong style="color:#94a3b8;">4 platforms</strong>: YouTube, Instagram, NewsAPI, and Twitter.</div>""", unsafe_allow_html=True)

        if st.session_state.df_store is not None:
            df = st.session_state.df_store
            if "Source" in df.columns and df["Source"].nunique() > 1:
                # Stats table
                platform_stats = df.groupby("Source")["Sentiment"].value_counts(normalize=True).mul(100).round(1)
                platform_stats = platform_stats.unstack(fill_value=0).reset_index()
                st.dataframe(platform_stats, use_container_width=True)

                # Heatmap
                sources    = df["Source"].unique().tolist()
                sentiments = ["Positive","Negative","Neutral"]
                z_data, text_data = [], []
                for src in sources:
                    sub    = df[df["Source"]==src]["Sentiment"].value_counts(normalize=True).mul(100)
                    row_z  = [round(sub.get(s,0),1) for s in sentiments]
                    z_data.append(row_z); text_data.append([f"{v}%" for v in row_z])
                fig_heat = go.Figure(go.Heatmap(z=z_data, x=sentiments, y=sources,
                    text=text_data, texttemplate="%{text}",
                    colorscale=[[0,"#0f172a"],[0.5,"#0ea5e9"],[1,"#34d399"]], showscale=True))
                fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8", margin=dict(t=10,b=10,l=10,r=10), height=340)
                st.plotly_chart(fig_heat, use_container_width=True)

                # Grouped bar
                src_sent = df.groupby(["Source","Sentiment"]).size().reset_index(name="Count")
                fig_src = px.bar(src_sent, x="Source", y="Count", color="Sentiment", barmode="group",
                    color_discrete_map={"Positive":"#34d399","Negative":"#f87171","Neutral":"#fbbf24","Sarcasm":"#818cf8"})
                fig_src.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8", legend=dict(bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                    margin=dict(t=10,b=10,l=10,r=10), height=320)
                st.plotly_chart(fig_src, use_container_width=True)
            else:
                st.info("Run analysis first, then fetch real data from multiple sources to see platform comparison.")
        else:
            st.info("ℹ️ Run analysis in the Dashboard tab first.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — FETCH DATA
    # ══════════════════════════════════════════════════════════════════════════
    with t4:
        # ── YouTube + News + Twitter ──────────────────────────────────────────
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>🌐 Fetch — YouTube · NewsAPI · Twitter</div>", unsafe_allow_html=True)

        fc1, fc2 = st.columns([3,1])
        with fc1:
            fetch_scheme = st.selectbox("Scheme", ["All Schemes"] + ALL_SCHEMES, key="fetch_scheme_sel", label_visibility="collapsed")
        with fc2:
            max_per = st.number_input("Max per source", min_value=50, max_value=500, value=200, step=50, key="max_per_src")

        fc3, fc4 = st.columns(2)
        with fc3:
            fetch_btn = st.button("⟶ Fetch Now", key="btn_fetch_all")
        with fc4:
            st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:10px;color:#334155;line-height:1.8;padding-top:6px;">
            YOUTUBE_API_KEY · NEWS_API_KEY · TWITTER_BEARER_TOKEN<br>Add these to your .env file</div>""", unsafe_allow_html=True)

        if fetch_btn:
            scheme_to_fetch = "All" if "All" in fetch_scheme else fetch_scheme
            log_lines = []
            log_box   = st.empty()

            def _log(msg):
                log_lines.append(msg)
                log_box.markdown(
                    "<div style='background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.06);"
                    "border-radius:12px;padding:14px 18px;font-family:DM Mono,monospace;font-size:11px;"
                    "color:#64748b;line-height:2;max-height:200px;overflow-y:auto;'>"
                    + "<br>".join(f"› {m}" for m in log_lines[-12:]) + "</div>",
                    unsafe_allow_html=True)

            with st.spinner("Fetching from APIs..."):
                counts = fetch_all(scheme=scheme_to_fetch, max_per_source=int(max_per), progress_callback=_log)

            total = sum(counts.values())
            cols = st.columns(4)
            for col, (src, cnt) in zip(cols, counts.items()):
                col.markdown(f"""<div class="metric-card" style="margin-top:12px;">
                  <div class="metric-val">{cnt}</div>
                  <div class="metric-lbl">{src}</div></div>""", unsafe_allow_html=True)

            if total > 0:
                st.success(f"✅ {total} new items saved to data.csv")
            else:
                st.warning("No data fetched. Check API keys in .env file.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Instagram URL ─────────────────────────────────────────────────────
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>📸 Instagram — Paste Reel or Post URL</div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Outfit',sans-serif;font-size:13px;color:#4a5a72;line-height:1.8;margin-bottom:16px;">
        Find any Instagram reel or post about a government scheme from
        <b style="color:#e879f9">@narendramodi, @mygovindia, @pibindia</b> or hashtags like
        <b style="color:#e879f9">#PMAY #AyushmanBharat #DigitalIndia</b>. Paste URL below.</div>""", unsafe_allow_html=True)

        ig_url = st.text_input("Instagram URL",
            placeholder="https://www.instagram.com/reel/ABC123xyz/",
            key="ig_url_input", label_visibility="collapsed")
        ig1, ig2, ig3 = st.columns([3,2,1])
        with ig1:
            ig_scheme = st.selectbox("Scheme for this post", ALL_SCHEMES, key="ig_scheme_sel")
        with ig2:
            ig_max = st.number_input("Max comments", min_value=50, max_value=1000, value=300, step=50, key="ig_max")
        with ig3:
            ig_btn = st.button("⟶ Fetch", key="btn_ig_fetch")

        if ig_btn:
            if not ig_url.strip():
                st.warning("Please paste an Instagram URL first.")
            elif "instagram.com" not in ig_url:
                st.error("That does not look like an Instagram URL.")
            else:
                ig_logs = []; ig_log_box = st.empty()
                def ig_log(msg):
                    ig_logs.append(msg)
                    ig_log_box.markdown(
                        "<div style='background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.06);"
                        "border-radius:12px;padding:14px 18px;font-family:DM Mono,monospace;font-size:11px;"
                        "color:#64748b;line-height:2;'>"
                        + "<br>".join(f"› {m}" for m in ig_logs[-8:]) + "</div>",
                        unsafe_allow_html=True)
                with st.spinner("Fetching Instagram comments..."):
                    rows = fetch_instagram_post(ig_url.strip(), ig_scheme, int(ig_max), ig_log)
                if rows:
                    st.success(f"✅ {len(rows)} comments fetched from Instagram!")
                    st.dataframe(pd.DataFrame(rows[:10])[["Scheme","Language","Comment"]], use_container_width=True, height=200)
                else:
                    st.error("No comments fetched. Check INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env")


        # ── Dataset status ────────────────────────────────────────────────────
        st.markdown("<div class='g-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>📊 Current Dataset Status</div>", unsafe_allow_html=True)
        try:
            df_s = pd.read_csv("data/data.csv")
            s1,s2,s3,s4 = st.columns(4)
            for col, (val, lbl, c) in zip([s1,s2,s3,s4],[
                (len(df_s),"Total Rows","#38bdf8"),
                (df_s["Scheme"].nunique(),"Schemes","#34d399"),
                (df_s["Source"].nunique(),"Sources","#818cf8"),
                (df_s["Language"].nunique() if "Language" in df_s.columns else "—","Languages","#f59e0b"),
            ]):
                col.markdown(f"""<div style="text-align:center;">
                  <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:700;color:{c};">{val}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#475569;">{lbl}</div></div>""",
                    unsafe_allow_html=True)
            src_counts = df_s["Source"].value_counts()
            fig_s = px.bar(x=src_counts.index, y=src_counts.values, color=src_counts.values,
                color_continuous_scale=["#1e3a4a","#38bdf8"], labels={"x":"Source","y":"Count"})
            fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#64748b", showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(family="DM Mono",size=10,color="#64748b")),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(t=10,b=10,l=5,r=10), height=180)
            st.plotly_chart(fig_s, use_container_width=True)
        except FileNotFoundError:
            st.info("No data yet. Run: python data/generate_data.py")
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — ABOUT & GAPS
    # ══════════════════════════════════════════════════════════════════════════
    with t5:
        gaps = [
            ("Gap 1","Language Barrier","🌐","Most research analyzes only English tweets. Regional languages like Hindi, Tamil, Hinglish are ignored.",
             "Multilingual pipeline: LangDetect → Google Translator → English NLP. Supports EN/HI/TA/Hinglish.","#38bdf8"),
            ("Gap 2","Only Twitter Data","🐦","Existing papers analyze Twitter exclusively, missing public opinion on YouTube, Instagram, News apps.",
             "Multi-source: YouTube, Instagram, NewsAPI, Twitter — all compared in Platform Comparison tab.","#818cf8"),
            ("Gap 3","Random Algorithm Selection","🎲","Researchers randomly pick algorithms without knowing which suits the dataset.",
             "Adaptive AutoML: ML (7 models) + NLP (TextBlob, VADER) + DL (LSTM, BiLSTM, CNN) + Transformers (ALBERT, DistilBERT). Best auto-selected by accuracy; ties broken by speed.","#34d399"),
            ("Gap 4","Binary Classification Only","⚖️","Most studies classify only Positive/Negative, ignoring the Neutral class.",
             "Three-class classification: Positive / Neutral / Negative with per-class F1/Precision/Recall.","#f59e0b"),
            ("Gap 5","Sarcasm Ignored","🙄","Sarcasm flips sentiment. Most papers ignore it, causing misclassification.",
             "Advanced sarcasm detection: emoji signals + regex pattern matching + capitalisation ratio + punctuation irony + Hindi sarcasm markers. Score 0–100%. Overrides Positive → Negative.","#f87171"),
        ]
        for gap_id, title, icon, problem, solution, clr in gaps:
            st.markdown(f"""<div class="g-panel" style="border-left:3px solid {clr}50;">
              <div style="display:flex;align-items:flex-start;gap:18px;">
                <div style="min-width:48px;height:48px;border-radius:12px;background:{clr}15;border:1px solid {clr}30;
                     display:flex;align-items:center;justify-content:center;font-size:22px;">{icon}</div>
                <div style="flex:1;">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <span style="font-family:'DM Mono',monospace;font-size:10px;color:{clr};letter-spacing:2px;text-transform:uppercase;
                         background:{clr}15;border:1px solid {clr}30;border-radius:100px;padding:3px 12px;">{gap_id}</span>
                    <span style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#f1f5f9;">{title}</span>
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:11px;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">Problem</div>
                  <div style="font-family:'Outfit',sans-serif;font-size:14px;color:#94a3b8;margin-bottom:14px;line-height:1.7;">{problem}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:11px;color:{clr};letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">Our Solution ✅</div>
                  <div style="font-family:'Outfit',sans-serif;font-size:14px;color:#cbd5e1;line-height:1.7;">{solution}</div>
                </div></div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class='g-panel' style='text-align:center;'>
          <div style='font-family:"DM Mono",monospace;font-size:10px;color:#38bdf8;letter-spacing:3px;text-transform:uppercase;margin-bottom:12px;'>Research Title</div>
          <div style='font-family:"Syne",sans-serif;font-size:20px;font-weight:700;color:#f1f5f9;line-height:1.4;'>
            Multilingual Multi-Source Sentiment Analysis Framework<br>for Indian Government Schemes Using Adaptive Model Selection
          </div>
          <div style='font-family:"DM Mono",monospace;font-size:11px;color:#475569;margin-top:10px;letter-spacing:1px;'>
            Addressing 5 Research Gaps · ML + DL + NLP + Transformers · Final Year Major Project
          </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;font-family:'DM Mono',monospace;font-size:10px;
         color:#1e293b;margin-top:50px;padding-bottom:20px;letter-spacing:2px;text-transform:uppercase;">
      Pulse Sentiment AI · ML + DL + NLP + Transformers · Multilingual · Multi-Source · Sarcasm-Aware
    </div>""", unsafe_allow_html=True)