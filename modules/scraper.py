"""
modules/scraper.py — Pulse Sentiment AI · Real-Time Data Fetcher
═══════════════════════════════════════════════════════════════════
Sources:
  ✅ YouTube        — comments from scheme-related videos (Official API)
  ✅ NewsAPI         — news headlines + descriptions (Official API)
  ✅ Google News RSS — real-time headlines, no API key needed
  ✅ Hindi News RSS  — Dainik Bhaskar, Amar Ujala, NBT (no API key needed)
  ❌ Twitter         — removed (API paid, approval issues)
  ❌ Instagram       — removed (login issues, scraping unreliable)
  ❌ Reddit          — removed (API approval issues)

FIXES INTACT:
  ✅ Dedup: _normalise() case-insensitive comparison
  ✅ Dedup: _save_rows() deduplicates batch + CSV
  ✅ Dedup: each fetcher deduplicates its own output
  ✅ Min comment length 15 chars
  ✅ Comments labelled at fetch time (not saved as Unknown)
  ✅ Hinglish language detection
  ✅ VADER + TextBlob loaded ONCE at module level (not per comment)
"""

import os, csv, re, time
import urllib.parse
import pandas as pd
from pathlib     import Path
from dotenv      import load_dotenv
from collections import Counter
load_dotenv()


# ── Secret loader — works both locally (.env) and on Streamlit Cloud ──────────
def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")


# ── API Keys ──────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = _get_secret("YOUTUBE_API_KEY")
NEWS_API_KEY    = _get_secret("NEWS_API_KEY")

DATA_CSV = Path("data/data.csv")


# ── Sentiment tools — loaded ONCE at module level ─────────────────────────────
try:
    import nltk
    try:    nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError: nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _SIA
    _VADER   = _SIA()
    VADER_OK = True
except Exception:
    _VADER   = None
    VADER_OK = False

try:
    from textblob import TextBlob as _TB
    TEXTBLOB_OK = True
except Exception:
    _TB         = None
    TEXTBLOB_OK = False


# ── Hindi keyword sets ────────────────────────────────────────────────────────
_NEG_HI = {
    "नहीं","बेकार","फर्जी","झूठ","घोटाला","भ्रष्टाचार","धोखा",
    "नाकाम","विफल","समस्या","परेशानी","खराब","बुरा","गलत","अन्याय",
    "लूट","ठगी","काम नहीं","नहीं मिला","नहीं होता","नाराज","निराश",
    "bakwaas","bekar","faltu","fraud","nahi mila","kaam nahi",
}
_POS_HI = {
    "अच्छा","बढ़िया","शानदार","फायदेमंद","मदद","खुश","संतुष्ट",
    "धन्यवाद","सुधार","सफल","कामयाब",
    "accha","badhiya","helpful","fayda","khush","shukriya","safal",
}
_SARC_EMOJIS = {"🙄","😒","😏","🤨","🙃","😤","😑"}


# ─────────────────────────────────────────────────────────────────────────────
#  ALL MODI GOVERNMENT SCHEMES SINCE 2014
# ─────────────────────────────────────────────────────────────────────────────
SCHEME_KEYWORDS = {
    "PMAY — Pradhan Mantri Awas Yojana": [
        "PMAY scheme review", "Pradhan Mantri Awas Yojana", "PMAY housing India",
        "pm awas yojana gramin", "PMAY urban rural",
    ],
    "Ayushman Bharat — PM-JAY": [
        "Ayushman Bharat scheme", "PM-JAY health card", "ayushman bharat hospital",
        "ayushman card benefits", "pmjay insurance india",
    ],
    "Poshan Abhiyaan — Nutrition Mission": [
        "Poshan Abhiyaan scheme", "national nutrition mission india",
        "poshan mission malnutrition", "anganwadi nutrition scheme",
    ],
    "PM Kisan Samman Nidhi": [
        "PM Kisan scheme", "PM Kisan Samman Nidhi", "pm kisan money farmers",
        "kisan nidhi 6000", "pm kisan yojana review",
    ],
    "Fasal Bima — PM Crop Insurance": [
        "Pradhan Mantri Fasal Bima Yojana", "PMFBY crop insurance",
        "fasal bima scheme farmers", "kisan crop insurance india",
    ],
    "Kisan Credit Card": [
        "Kisan Credit Card scheme", "KCC loan farmers india",
        "kisan credit card benefits", "kcc agriculture loan",
    ],
    "e-NAM — National Agriculture Market": [
        "eNAM agriculture market", "e-NAM scheme farmers",
        "national agriculture market online", "enam mandi platform",
    ],
    "Digital India Initiative": [
        "Digital India scheme", "digital india government initiative",
        "digital india internet rural", "bharat net digital india",
    ],
    "BharatNet — Rural Internet": [
        "BharatNet scheme rural internet", "bharat net broadband village",
        "rural broadband india scheme", "optical fibre village india",
    ],
    "UPI — Unified Payments Interface": [
        "UPI digital payment india", "unified payments interface review",
        "upi transaction india scheme", "digital payment modi",
    ],
    "Jan Dhan Yojana — Financial Inclusion": [
        "Jan Dhan Yojana scheme", "Pradhan Mantri Jan Dhan Yojana",
        "pmjdy bank account poor", "jan dhan account benefits review",
    ],
    "Mudra Yojana — MSME Loans": [
        "Mudra Yojana loan scheme", "PMMY mudra loan india",
        "mudra loan small business", "pradhan mantri mudra yojana review",
    ],
    "Stand Up India Scheme": [
        "Stand Up India scheme", "standup india loan SC ST women",
        "standup india bank loan", "women entrepreneur loan india",
    ],
    "Atal Pension Yojana": [
        "Atal Pension Yojana scheme", "APY pension unorganised workers",
        "atal pension yojana review", "apy pension india",
    ],
    "PM Jeevan Jyoti Bima": [
        "PM Jeevan Jyoti Bima Yojana", "PMJJBY life insurance scheme",
        "jeevan jyoti bima review", "pm life insurance 330",
    ],
    "PM Suraksha Bima": [
        "PM Suraksha Bima Yojana", "PMSBY accident insurance",
        "suraksha bima 12 rupees", "accidental insurance india scheme",
    ],
    "Ujjwala Yojana — LPG for Poor": [
        "Ujjwala Yojana scheme", "PM Ujjwala LPG cylinder",
        "ujjwala yojana BPL women", "free gas cylinder scheme india",
    ],
    "Saubhagya — Household Electrification": [
        "Saubhagya scheme electricity", "PM saubhagya yojana",
        "har ghar bijli scheme", "rural electrification india modi",
    ],
    "Solar Rooftop — PM Surya Ghar": [
        "PM Surya Ghar scheme", "solar rooftop india scheme",
        "free solar panel india government", "pm surya ghar bijli review",
    ],
    "FAME — Electric Vehicle Scheme": [
        "FAME scheme electric vehicle india", "EV subsidy india government",
        "electric vehicle policy india", "fame 2 ev incentive",
    ],
    "Swachh Bharat Mission": [
        "Swachh Bharat Mission scheme", "swachh bharat toilet india",
        "open defecation free india", "swachh bharat review",
    ],
    "Jal Jeevan Mission — Har Ghar Jal": [
        "Jal Jeevan Mission scheme", "har ghar nal jal yojana",
        "tap water every house india", "jal jeevan mission review",
    ],
    "AMRUT — Urban Development": [
        "AMRUT scheme urban", "atal mission rejuvenation urban",
        "amrut 2.0 city water india", "urban infrastructure india scheme",
    ],
    "Skill India — PMKVY": [
        "Skill India Mission scheme", "PMKVY skill development",
        "pradhan mantri kaushal vikas yojana", "skill india training review",
    ],
    "Startup India": [
        "Startup India scheme", "startup india initiative modi",
        "startup india fund recognition", "startup india review 2023 2024",
    ],
    "Make in India": [
        "Make in India scheme", "make in india manufacturing",
        "atma nirbhar bharat make in india", "make in india review",
    ],
    "PM eVIDYA — Digital Education": [
        "PM eVIDYA scheme", "digital education india covid",
        "diksha platform education india", "one nation one digital platform",
    ],
    "Beti Bachao Beti Padhao": [
        "Beti Bachao Beti Padhao scheme", "bbbp scheme girl child india",
        "beti bachao review impact", "save daughter educate daughter india",
    ],
    "Sukanya Samriddhi Yojana": [
        "Sukanya Samriddhi Yojana scheme", "ssy girl child savings india",
        "sukanya samriddhi account benefits", "girl child investment scheme india",
    ],
    "PM Matru Vandana — Maternity Benefit": [
        "PM Matru Vandana Yojana", "PMMVY maternity benefit scheme",
        "maternity cash benefit pregnant women india", "pmmvy 5000 rupees",
    ],
    "Pradhan Mantri Gram Sadak Yojana": [
        "PMGSY rural road scheme", "Pradhan Mantri Gram Sadak Yojana",
        "village road connectivity india", "pmgsy road construction review",
    ],
    "Bharatmala — Highway Project": [
        "Bharatmala highway scheme", "bharatmala project india roads",
        "national highway development india modi", "bharatmala review",
    ],
    "Smart Cities Mission": [
        "Smart Cities Mission india", "smart city scheme india",
        "smart city development modi", "smart city mission review",
    ],
    "Sagarmala — Port Development": [
        "Sagarmala scheme port india", "sagarmala project coastal",
        "port development india modi", "sagarmala review",
    ],
    "One Nation One Ration Card": [
        "One Nation One Ration Card scheme", "ONORC ration portability",
        "one nation ration card migrant worker", "onorc scheme review",
    ],
    "PM Garib Kalyan Anna Yojana": [
        "PM Garib Kalyan Anna Yojana", "PMGKAY free food grain scheme",
        "free ration covid india", "garib kalyan anna yojana review",
    ],
    "PM SVANidhi — Street Vendor Loan": [
        "PM SVANidhi scheme", "street vendor loan india",
        "svanidhi micro credit vendor", "pm svanidhi review",
    ],
    "Vishwakarma Yojana": [
        "PM Vishwakarma Yojana scheme", "vishwakarma yojana artisan",
        "vishwakarma skill loan india", "pm vishwakarma review 2023 2024",
    ],
    "Atmanirbhar Bharat": [
        "Atmanirbhar Bharat scheme", "self reliant india initiative",
        "atmanirbhar bharat package", "vocal for local india modi",
    ],
    "Ayushman Bharat Digital Mission": [
        "Ayushman Bharat Digital Mission", "ABDM health ID india",
        "digital health id abha card", "health stack india scheme",
    ],
}

ALL_SCHEMES = list(SCHEME_KEYWORDS.keys())


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace. For dedup comparison only."""
    return " ".join(text.lower().split())


def _detect_lang(text: str) -> str:
    """Detect language including Hinglish."""
    hindi   = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    tamil   = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
    telugu  = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    bengali = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
    if hindi   > 3: return "hi"
    if tamil   > 3: return "ta"
    if telugu  > 3: return "te"
    if bengali > 3: return "bn"
    hinglish_markers = {
        "hai","hain","nahi","kuch","yeh","toh","bhai","tha","gaya","mila",
        "wala","sab","bahut","acha","achi","bhi","se","ko","ka","haan",
        "koi","yaar","bilkul","zaroor","accha","bekar","bakwaas","kaam",
        "sarkar","paisa","paise","log","kaise","kyun","agar","lekin",
        "par","aur","mein","pe","iske","scheme",
    }
    if len(set(text.lower().split()) & hinglish_markers) >= 2:
        return "hinglish"
    return "en"


def _quick_sentiment(text: str, lang: str) -> str:
    """Fast sentiment using module-level singletons — no re-imports."""
    if lang in ("hi", "hinglish"):
        neg = sum(1 for w in _NEG_HI if w in text)
        pos = sum(1 for w in _POS_HI if w in text)
        if neg > 0 and neg >= pos: return "Negative"
        if pos > 0 and pos > neg:  return "Positive"

    if any(e in text for e in _SARC_EMOJIS):
        return "Negative"

    tb_vote = None
    if TEXTBLOB_OK:
        try:
            pol = _TB(str(text)).sentiment.polarity
            if pol > 0.08:    tb_vote = "Positive"
            elif pol < -0.08: tb_vote = "Negative"
            else:             tb_vote = "Neutral"
        except Exception:
            pass

    vader_vote = None
    if VADER_OK:
        try:
            c = _VADER.polarity_scores(str(text))["compound"]
            if c >= 0.05:    vader_vote = "Positive"
            elif c <= -0.05: vader_vote = "Negative"
            else:            vader_vote = "Neutral"
        except Exception:
            pass

    votes = [v for v in [tb_vote, vader_vote] if v is not None]
    if not votes:
        return "Neutral"
    return Counter(votes).most_common(1)[0][0]


def _make_row(scheme: str, source: str, lang: str,
              comment: str, sentiment: str = None) -> dict:
    """Build a data row — always labelled, never saved as Unknown."""
    text = comment.strip()
    if not sentiment or sentiment == "Unknown":
        sentiment = _quick_sentiment(text, lang)
    return {
        "ID": "", "Scheme": scheme, "Source": source,
        "Language": lang, "Comment": text, "Sentiment": sentiment,
    }


def _load_existing_normalised() -> tuple[set, int]:
    """Load normalised existing comments from CSV."""
    existing_norm = set()
    next_id = 1
    if DATA_CSV.exists():
        try:
            df = pd.read_csv(DATA_CSV, encoding="utf-8", usecols=["ID","Comment"])
            existing_norm = set(df["Comment"].dropna().apply(_normalise).tolist())
            next_id = int(df["ID"].max()) + 1 if len(df) else 1
        except Exception:
            pass
    return existing_norm, next_id


def _save_rows(rows: list) -> int:
    """Save unique rows to CSV. Deduplicates within batch AND vs CSV."""
    if not rows:
        return 0
    DATA_CSV.parent.mkdir(exist_ok=True)
    existing_norm, next_id = _load_existing_normalised()
    seen_in_batch = set()
    deduped_batch = []
    for r in rows:
        text = r.get("Comment", "").strip()
        norm = _normalise(text)
        if len(text) < 15:        continue
        if norm in seen_in_batch: continue
        if norm in existing_norm: continue
        seen_in_batch.add(norm)
        existing_norm.add(norm)
        deduped_batch.append(r)
    if not deduped_batch:
        return 0
    header = not DATA_CSV.exists() or DATA_CSV.stat().st_size == 0
    with open(DATA_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID","Scheme","Source","Language","Comment","Sentiment"])
        if header:
            w.writeheader()
        for i, r in enumerate(deduped_batch):
            r["ID"] = next_id + i
            w.writerow(r)
    return len(deduped_batch)


# ═════════════════════════════════════════════════════════════════════════════
#  YOUTUBE FETCHER  — Official API, real-time, unchanged
# ═════════════════════════════════════════════════════════════════════════════
def fetch_youtube(scheme, limit=300, cb=None):
    if not YOUTUBE_API_KEY:
        if cb: cb("YouTube: No API key — add YOUTUBE_API_KEY to .env")
        return []
    try:
        from googleapiclient.discovery import build
        yt = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except ImportError:
        if cb: cb("YouTube: Run → pip install google-api-python-client")
        return []

    rows = []
    seen = set()
    for kw in SCHEME_KEYWORDS.get(scheme, [scheme])[:2]:
        if len(rows) >= limit: break
        try:
            vids = yt.search().list(
                q=kw, part="id", type="video",
                maxResults=8, regionCode="IN", order="relevance"
            ).execute()
            for v in vids.get("items", []):
                if len(rows) >= limit: break
                try:
                    comments = yt.commentThreads().list(
                        part="snippet", videoId=v["id"]["videoId"],
                        maxResults=100, textFormat="plainText"
                    ).execute()
                    for c in comments.get("items", []):
                        t    = c["snippet"]["topLevelComment"]["snippet"]["textDisplay"].strip()
                        norm = _normalise(t)
                        if len(t) >= 15 and norm not in seen:
                            seen.add(norm)
                            lang = _detect_lang(t)
                            rows.append(_make_row(scheme, "YouTube", lang, t))
                        if len(rows) >= limit: break
                    time.sleep(0.3)
                except Exception:
                    continue
        except Exception as e:
            if cb: cb(f"YouTube error: {e}")
    if cb: cb(f"YouTube: {len(rows)} unique comments fetched")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  NEWS API FETCHER  — Official API, real-time, unchanged
# ═════════════════════════════════════════════════════════════════════════════
def fetch_news(scheme, limit=200, cb=None):
    if not NEWS_API_KEY:
        if cb: cb("News: No API key — add NEWS_API_KEY to .env")
        return []
    try:
        from newsapi import NewsApiClient
        api = NewsApiClient(api_key=NEWS_API_KEY)
    except ImportError:
        if cb: cb("News: Run → pip install newsapi-python")
        return []

    rows = []
    seen = set()
    for kw in SCHEME_KEYWORDS.get(scheme, [scheme])[:2]:
        if len(rows) >= limit: break
        try:
            resp = api.get_everything(
                q=kw, language="en",
                sort_by="publishedAt", page_size=100
            )
            for a in resp.get("articles", []):
                for text in [a.get("title",""), a.get("description","")]:
                    text = text.strip()
                    norm = _normalise(text)
                    if (len(text) >= 15
                            and "[Removed]" not in text
                            and norm not in seen):
                        seen.add(norm)
                        rows.append(_make_row(scheme, "News App", "en", text[:400]))
                if len(rows) >= limit: break
            time.sleep(0.3)
        except Exception as e:
            if cb: cb(f"News error: {e}")
    if cb: cb(f"News: {len(rows)} unique articles fetched")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  GOOGLE NEWS RSS FETCHER
#  ✅ No API key needed
#  ✅ Real-time — updates every few hours
#  ✅ Official RSS feed from Google — not scraping
#  ✅ English headlines and descriptions about Indian govt schemes
# ═════════════════════════════════════════════════════════════════════════════
def fetch_google_news_rss(scheme: str, limit: int = 200, cb=None) -> list:
    """
    Fetches Google News RSS feed for Indian government schemes.

    How it works:
      Google provides an official RSS feed at news.google.com/rss
      RSS is a machine-readable format designed to be read by code
      No scraping — Google publishes this data intentionally
      Updates in real-time as news is published

    No API key required.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        if cb: cb("Google News RSS: Run → pip install requests beautifulsoup4 lxml")
        return []

    rows    = []
    seen    = set()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RSSReader/1.0)"}

    search_terms = SCHEME_KEYWORDS.get(scheme, [scheme])[:3]

    for term in search_terms:
        if len(rows) >= limit:
            break
        try:
            # Google News RSS — India edition, English
            query   = urllib.parse.quote(f"{term} India government scheme")
            rss_url = (f"https://news.google.com/rss/search"
                       f"?q={query}&hl=en-IN&gl=IN&ceid=IN:en")

            resp = requests.get(rss_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                if cb: cb(f"Google News RSS: HTTP {resp.status_code} for {term}")
                continue

            # Parse RSS XML
            soup = BeautifulSoup(resp.content, "xml")

            for item in soup.find_all("item"):
                if len(rows) >= limit:
                    break

                # Extract title
                title_tag = item.find("title")
                if title_tag:
                    text = title_tag.get_text(strip=True)
                    # Remove source suffix " - Source Name"
                    if " - " in text:
                        text = text.rsplit(" - ", 1)[0].strip()
                    norm = _normalise(text)
                    if len(text) >= 20 and norm not in seen:
                        seen.add(norm)
                        rows.append(_make_row(scheme, "Google News", "en", text))

                # Extract description
                desc_tag = item.find("description")
                if desc_tag and len(rows) < limit:
                    # Description is often HTML — extract plain text
                    desc_soup = BeautifulSoup(desc_tag.get_text(), "html.parser")
                    text = desc_soup.get_text(strip=True)
                    if " - " in text:
                        text = text.rsplit(" - ", 1)[0].strip()
                    norm = _normalise(text)
                    if len(text) >= 20 and norm not in seen:
                        seen.add(norm)
                        rows.append(_make_row(scheme, "Google News", "en", text[:400]))

            time.sleep(0.5)  # polite delay between requests

        except Exception as e:
            if cb: cb(f"Google News RSS error for '{term}': {e}")
            continue

    if cb: cb(f"Google News RSS: {len(rows)} items fetched for {scheme}")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  HINDI NEWS RSS FETCHER
#  ✅ No API key needed
#  ✅ Real-time — RSS feeds update every few hours
#  ✅ Official RSS feeds — Dainik Bhaskar, Amar Ujala, Navbharat Times
#  ✅ Best source for Hindi content about Indian government schemes
# ═════════════════════════════════════════════════════════════════════════════

# Hindi news RSS feeds — all official, no API key needed
_HINDI_RSS_FEEDS = [
    # Dainik Bhaskar — India's largest Hindi newspaper
    ("https://www.bhaskar.com/rss-feed/1061/",           "Dainik Bhaskar"),
    # Amar Ujala — Major Hindi newspaper
    ("https://www.amarujala.com/rss/india-news.xml",      "Amar Ujala"),
    # Navbharat Times — Hindi edition of Times of India
    ("https://navbharattimes.indiatimes.com/rssfeeds/1564454837.cms", "Navbharat Times"),
    # Jagran — Major Hindi newspaper
    ("https://www.jagran.com/rss/news-national.xml",      "Jagran"),
]


def fetch_hindi_news_rss(scheme: str, limit: int = 150, cb=None) -> list:
    """
    Fetches Hindi news from official RSS feeds of major Hindi newspapers.

    How it works:
      RSS feeds are official, machine-readable XML files published by newspapers
      They update automatically as new articles are published
      We filter articles related to the scheme using keyword matching

    Sources: Dainik Bhaskar, Amar Ujala, Navbharat Times, Jagran
    No API key required.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        if cb: cb("Hindi News RSS: Run → pip install requests beautifulsoup4 lxml")
        return []

    rows    = []
    seen    = set()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RSSReader/1.0)"}

    # Build keyword list for filtering — scheme name + related Hindi terms
    search_terms = SCHEME_KEYWORDS.get(scheme, [scheme])
    # Extract short keywords for matching (first word of each term)
    keywords = set()
    for term in search_terms[:3]:
        for word in term.lower().split():
            if len(word) > 3:
                keywords.add(word)

    # Also add scheme short name (e.g. "PMAY", "PM Kisan", "Ayushman")
    scheme_short = scheme.split("—")[0].strip().lower()
    for word in scheme_short.split():
        if len(word) > 2:
            keywords.add(word)

    for rss_url, source_name in _HINDI_RSS_FEEDS:
        if len(rows) >= limit:
            break
        try:
            resp = requests.get(rss_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                if cb: cb(f"Hindi RSS: HTTP {resp.status_code} from {source_name}")
                continue

            soup = BeautifulSoup(resp.content, "xml")

            for item in soup.find_all("item"):
                if len(rows) >= limit:
                    break

                # Get title and description
                title_tag = item.find("title")
                desc_tag  = item.find("description")

                title_text = title_tag.get_text(strip=True) if title_tag else ""
                desc_text  = desc_tag.get_text(strip=True)  if desc_tag  else ""

                # Combine for keyword matching
                combined = (title_text + " " + desc_text).lower()

                # Only include if related to the scheme
                # Check if any keyword appears in the article
                if not any(kw in combined for kw in keywords):
                    continue

                # Process title
                if title_text:
                    norm = _normalise(title_text)
                    if len(title_text) >= 20 and norm not in seen:
                        seen.add(norm)
                        lang = _detect_lang(title_text)
                        rows.append(_make_row(scheme, source_name, lang, title_text))

                # Process description
                if desc_text and len(rows) < limit:
                    # Strip HTML tags from description
                    desc_clean = BeautifulSoup(desc_text, "html.parser").get_text(strip=True)
                    norm = _normalise(desc_clean)
                    if (len(desc_clean) >= 20
                            and len(desc_clean) <= 500
                            and norm not in seen):
                        seen.add(norm)
                        lang = _detect_lang(desc_clean)
                        rows.append(_make_row(scheme, source_name, lang,
                                              desc_clean[:400]))

            time.sleep(0.5)

        except Exception as e:
            if cb: cb(f"Hindi RSS error from {source_name}: {e}")
            continue

    if cb: cb(f"Hindi News RSS: {len(rows)} items fetched for {scheme}")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  FETCH ALL — main entry point
#  Now uses 4 sources: YouTube + NewsAPI + Google News RSS + Hindi News RSS
# ═════════════════════════════════════════════════════════════════════════════
def fetch_all(scheme="All", max_per_source=200, progress_callback=None):
    cb      = progress_callback
    schemes = ALL_SCHEMES if scheme == "All" else [scheme]
    totals  = {
        "YouTube":     0,
        "News App":    0,
        "Google News": 0,
        "Hindi News":  0,
    }

    for s in schemes:
        if cb: cb(f"━━ Fetching: {s} ━━")

        # Official API sources
        yt   = fetch_youtube(s, max_per_source, cb)
        news = fetch_news(s,   max_per_source, cb)

        # Free RSS sources — no API key needed
        gnews = fetch_google_news_rss(s, max_per_source, cb)
        hindi = fetch_hindi_news_rss(s,  min(150, max_per_source), cb)

        all_rows = yt + news + gnews + hindi
        saved    = _save_rows(all_rows)

        totals["YouTube"]     += len(yt)
        totals["News App"]    += len(news)
        totals["Google News"] += len(gnews)
        totals["Hindi News"]  += len(hindi)

        if cb: cb(f"✓ {saved} new unique rows saved for {s}")

    return totals


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Pulse Sentiment AI — Live Data Fetcher")
    print("  Sources: YouTube + NewsAPI + Google News RSS + Hindi News RSS")
    print("="*60)

    print("\nWhich scheme?")
    for i, s in enumerate(ALL_SCHEMES, 1):
        print(f"  {i:>2}. {s}")
    print(f"  {len(ALL_SCHEMES)+1:>2}. All schemes")

    c = input("\nEnter number: ").strip()
    scheme = "All"
    if c.isdigit() and 1 <= int(c) <= len(ALL_SCHEMES):
        scheme = ALL_SCHEMES[int(c)-1]

    mx = input("Max per source? (default 200): ").strip()
    mx = int(mx) if mx.isdigit() else 200

    print(f"\nFetching '{scheme}'...\n")
    counts = fetch_all(scheme, mx, lambda m: print(f"  {m}"))

    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    for src, cnt in counts.items():
        bar = "█" * min(cnt // 5, 40)
        print(f"  {src:<22} {cnt:>5}  {bar}")
    print(f"\n  Total fetched : {sum(counts.values())} items")

    if DATA_CSV.exists():
        df        = pd.read_csv(DATA_CSV)
        dupe_rate = round((1 - df["Comment"].nunique() / len(df)) * 100, 1)
        print(f"  CSV total rows  : {len(df)}")
        print(f"  Unique comments : {df['Comment'].nunique()}")
        print(f"  Duplicate rate  : {dupe_rate}%  "
              f"({'✓ healthy' if dupe_rate < 3 else '⚠ check data'})")
        print(f"\n  Sentiment breakdown:")
        for s, c in df["Sentiment"].value_counts().items():
            pct = round(c / len(df) * 100, 1)
            print(f"    {s:<12} {c:>5}  ({pct}%)")
        print(f"\n  Source breakdown:")
        for s, c in df["Source"].value_counts().items():
            print(f"    {s:<22} {c:>5}")
    print("="*60 + "\n")
