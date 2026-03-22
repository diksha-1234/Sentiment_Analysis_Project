"""
modules/scraper.py — Pulse Sentiment AI · Real-Time Data Fetcher
═══════════════════════════════════════════════════════════════════
Sources:
  ✅ YouTube   — comments from scheme-related videos
  ✅ NewsAPI   — news headlines + descriptions
  ✅ Twitter   — recent tweets
  ✅ Instagram — comments from any reel/post URL you paste
  ❌ Reddit    — removed (API approval issues)

FIXES APPLIED:
  ✅ Dedup: _normalise() case-insensitive comparison
  ✅ Dedup: _save_rows() deduplicates batch + CSV
  ✅ Dedup: each fetcher deduplicates its own output
  ✅ Min comment length 5→15 chars
  ✅ Comments labelled at fetch time (not saved as Unknown)
  ✅ Hinglish language detection added
  ✅ VADER + TextBlob loaded ONCE at module level (not per comment)
     Your version re-initialised VADER on every single comment
     = ~0.3s × 300 comments = 90 seconds wasted per fetch
"""

import os, csv, re, time
import pandas as pd
from pathlib     import Path
from dotenv      import load_dotenv
from collections import Counter
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except:
        return os.getenv(key, "")

YOUTUBE_API_KEY      = _get_secret("YOUTUBE_API_KEY")
NEWS_API_KEY         = _get_secret("NEWS_API_KEY")
TWITTER_BEARER_TOKEN = _get_secret("TWITTER_BEARER_TOKEN")

DATA_CSV = Path("data/data.csv")

# ── Sentiment tools — loaded ONCE here, reused for every comment ──────────────
# Your previous version did `from nltk.sentiment.vader import ...` inside
# _quick_sentiment() which ran on every comment = extremely slow
try:
    import nltk
    try:    nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError: nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _SIA
    _VADER   = _SIA()    # ← single instance, reused for all comments
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

# ── Hindi keyword sets — defined ONCE at module level ─────────────────────────
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
    """
    Fast sentiment label using module-level singletons.
    No imports inside this function — everything loaded at startup.
    """
    # Hindi/Hinglish keyword prior
    if lang in ("hi", "hinglish"):
        neg = sum(1 for w in _NEG_HI if w in text)
        pos = sum(1 for w in _POS_HI if w in text)
        if neg > 0 and neg >= pos: return "Negative"
        if pos > 0 and pos > neg:  return "Positive"

    # Sarcasm emoji → Negative
    if any(e in text for e in _SARC_EMOJIS):
        return "Negative"

    # TextBlob
    tb_vote = None
    if TEXTBLOB_OK:
        try:
            pol = _TB(str(text)).sentiment.polarity
            if pol > 0.08:    tb_vote = "Positive"
            elif pol < -0.08: tb_vote = "Negative"
            else:             tb_vote = "Neutral"
        except Exception:
            pass

    # VADER — uses pre-loaded _VADER instance, not a new one
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
    """Build a data row — always labels, never saves 'Unknown'."""
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
        if len(text) < 15:         continue
        if norm in seen_in_batch:  continue
        if norm in existing_norm:  continue
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


def _extract_shortcode(url):
    for p in [r"instagram\.com/reel/([A-Za-z0-9_-]+)",
              r"instagram\.com/p/([A-Za-z0-9_-]+)",
              r"instagram\.com/tv/([A-Za-z0-9_-]+)"]:
        m = re.search(p, url)
        if m: return m.group(1)
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  INSTAGRAM FETCHER
# ═════════════════════════════════════════════════════════════════════════════
def fetch_instagram_post(url, scheme, max_comments=500, cb=None):
    if not INSTAGRAM_USERNAME or not INSTAGRAM_PASSWORD:
        if cb: cb("Instagram: Add INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD to .env")
        return []
    try:
        import instaloader
    except ImportError:
        if cb: cb("Instagram: Run → pip install instaloader")
        return []

    shortcode = _extract_shortcode(url)
    if not shortcode:
        if cb: cb(f"Instagram: Could not extract shortcode from URL: {url}")
        return []

    if cb: cb(f"Instagram: Loading post {shortcode}...")
    try:
        L = instaloader.Instaloader(
            download_pictures=False, download_videos=False,
            download_video_thumbnails=False, download_geotags=False,
            download_comments=True, save_metadata=False,
            compress_json=False, quiet=True,
            request_timeout=10, max_connection_attempts=3,
        )
        L.context.sleep = True
        L.context.max_connection_attempts = 3
        try:
            L.login(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD)
            if cb: cb("Instagram: Logged in successfully")
        except Exception as e:
            if cb: cb(f"Instagram: Login failed — {e}")
            return []

        post  = instaloader.Post.from_shortcode(L.context, shortcode)
        if cb: cb("Instagram: Post found — fetching comments...")
        rows  = []
        count = 0
        for comment in post.get_comments():
            text = comment.text.strip()
            if len(text) < 15: continue
            lang = _detect_lang(text)
            rows.append(_make_row(scheme, "Instagram", lang, text))
            count += 1
            if hasattr(comment, "answers"):
                for reply in comment.answers:
                    rt = reply.text.strip()
                    if len(rt) >= 15:
                        rows.append(_make_row(scheme, "Instagram", _detect_lang(rt), rt))
                        count += 1
            if count >= max_comments: break
            if count % 50 == 0 and cb:
                cb(f"Instagram: {count} comments fetched so far...")
            time.sleep(0.3)

        saved = _save_rows(rows)
        if cb: cb(f"Instagram: {len(rows)} fetched, {saved} new unique saved")
        return rows
    except Exception as e:
        if cb: cb(f"Instagram: Error — {e}")
        return []


def fetch_instagram_multiple(urls_with_schemes, max_per_post=300, cb=None):
    all_rows = []
    for url, scheme in urls_with_schemes:
        if cb: cb(f"Instagram: Fetching {url[:50]}...")
        rows = fetch_instagram_post(url, scheme, max_per_post, cb)
        all_rows.extend(rows)
        time.sleep(2)
    return all_rows


# ═════════════════════════════════════════════════════════════════════════════
#  YOUTUBE FETCHER
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
#  NEWS API FETCHER
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
#  TWITTER FETCHER
# ═════════════════════════════════════════════════════════════════════════════
def fetch_twitter(scheme, limit=100, cb=None):
    if not TWITTER_BEARER_TOKEN:
        if cb: cb("Twitter: No Bearer Token — skipping")
        return []
    try:
        import tweepy
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
    except ImportError:
        if cb: cb("Twitter: Run → pip install tweepy")
        return []

    rows = []
    seen = set()
    for kw in SCHEME_KEYWORDS.get(scheme, [scheme])[:1]:
        if len(rows) >= limit: break
        try:
            for lang in ["en", "hi"]:
                if len(rows) >= limit: break
                resp = client.search_recent_tweets(
                    query=f"{kw} lang:{lang} -is:retweet",
                    max_results=min(100, limit - len(rows)),
                    tweet_fields=["text"],
                )
                if resp.data:
                    for tw in resp.data:
                        text = tw.text.strip()
                        norm = _normalise(text)
                        if len(text) >= 15 and norm not in seen:
                            seen.add(norm)
                            rows.append(_make_row(scheme, "Twitter", lang, text))
            time.sleep(1)
        except Exception as e:
            if cb: cb(f"Twitter error: {e}")
    if cb: cb(f"Twitter: {len(rows)} unique tweets fetched")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  FETCH ALL — main entry point
# ═════════════════════════════════════════════════════════════════════════════
def fetch_all(scheme="All", max_per_source=200, progress_callback=None):
    cb      = progress_callback
    schemes = ALL_SCHEMES if scheme == "All" else [scheme]
    totals  = {"YouTube": 0, "News App": 0, "Twitter": 0, "Instagram": 0}
    for s in schemes:
        if cb: cb(f"━━ Fetching: {s} ━━")
        yt   = fetch_youtube(s,  max_per_source,           cb)
        news = fetch_news(s,     max_per_source,           cb)
        twt  = fetch_twitter(s,  min(100, max_per_source), cb)
        saved = _save_rows(yt + news + twt)
        totals["YouTube"]  += len(yt)
        totals["News App"] += len(news)
        totals["Twitter"]  += len(twt)
        if cb: cb(f"✓ {saved} new unique rows saved for {s}")
    return totals


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Pulse Sentiment AI — Live Data Fetcher")
    print("  All Modi Government Schemes (2014 → Present)")
    print("="*60)

    print("\nFetch mode:")
    print("  1. Fetch by scheme (YouTube + News + Twitter)")
    print("  2. Fetch Instagram post/reel by URL")
    print("  3. Both")
    mode = input("\nEnter 1 / 2 / 3: ").strip()

    if mode in ("1", "3"):
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
            df = pd.read_csv(DATA_CSV)
            dupe_rate = round((1 - df["Comment"].nunique() / len(df)) * 100, 1)
            print(f"  CSV total rows  : {len(df)}")
            print(f"  Unique comments : {df['Comment'].nunique()}")
            print(f"  Duplicate rate  : {dupe_rate}%  "
                  f"({'✓ healthy' if dupe_rate < 3 else '⚠ check data'})")
            print(f"\n  Sentiment breakdown:")
            for s, c in df["Sentiment"].value_counts().items():
                pct = round(c / len(df) * 100, 1)
                print(f"    {s:<12} {c:>5}  ({pct}%)")

    if mode in ("2", "3"):
        print("\n── Instagram URL Fetcher ──")
        url = input("Paste Instagram reel/post URL: ").strip()
        print("\nWhich scheme does this post belong to?")
        for i, s in enumerate(ALL_SCHEMES, 1):
            print(f"  {i:>2}. {s}")
        c = input("\nEnter number: ").strip()
        scheme = (ALL_SCHEMES[int(c)-1]
                  if c.isdigit() and 1 <= int(c) <= len(ALL_SCHEMES)
                  else "General")
        mx = input("Max comments to fetch? (default 300): ").strip()
        mx = int(mx) if mx.isdigit() else 300
        print(f"\nFetching comments from {url[:60]}...\n")
        rows  = fetch_instagram_post(url, scheme, mx, lambda m: print(f"  {m}"))
        saved = _save_rows(rows)
        print(f"\n  Fetched : {len(rows)} comments")
        print(f"  Saved   : {saved} new unique rows")

    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        print(f"\n  data.csv total rows : {len(df)}")
        print(f"  Sources  : {df['Source'].value_counts().to_dict()}")
        print(f"  Schemes  : {df['Scheme'].nunique()} unique schemes")
    print("="*60 + "\n")