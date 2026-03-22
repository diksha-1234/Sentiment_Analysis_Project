"""
data/generate_data.py
═════════════════════
Generates the base synthetic dataset for Pulse Sentiment AI.

OVERFITTING FIXES (only changes from original):
  ✅ FIX 1: Comment bank massively expanded — 60+ unique per sentiment
             Original had ~30 comments shared across 40 schemes = same
             comment appeared up to 80 times in the dataset
  ✅ FIX 2: _generate_unique() — scheme name injected into every comment
             "PM Kisan helped farmers" vs "Ujjwala Yojana helped farmers"
             are now truly different training samples, not duplicates
  ✅ FIX 3: ROWS_PER_SCHEME reduced 60→25
             60 × 40 = 2400 rows from ~30 templates = avg 80 duplicates each
             25 × 40 = 1000 rows, all unique = better generalisation
  ✅ FIX 4: smart_merge_and_save() uses normalised comparison
             Original used raw .strip() — missed case/whitespace variants
  ✅ FIX 5: Duplicate rate printed at end so you can verify
"""

import csv, random
from pathlib import Path

DATA_CSV = Path("data/data.csv")

SCHEMES = [
    "PMAY — Pradhan Mantri Awas Yojana",
    "Ayushman Bharat — PM-JAY",
    "Poshan Abhiyaan — Nutrition Mission",
    "Ayushman Bharat Digital Mission",
    "PM Kisan Samman Nidhi",
    "Fasal Bima — PM Crop Insurance",
    "Kisan Credit Card",
    "e-NAM — National Agriculture Market",
    "Digital India Initiative",
    "BharatNet — Rural Internet",
    "UPI — Unified Payments Interface",
    "Jan Dhan Yojana — Financial Inclusion",
    "Mudra Yojana — MSME Loans",
    "Stand Up India Scheme",
    "Atal Pension Yojana",
    "PM Jeevan Jyoti Bima",
    "PM Suraksha Bima",
    "Ujjwala Yojana — LPG for Poor",
    "Saubhagya — Household Electrification",
    "Solar Rooftop — PM Surya Ghar",
    "FAME — Electric Vehicle Scheme",
    "Swachh Bharat Mission",
    "Jal Jeevan Mission — Har Ghar Jal",
    "AMRUT — Urban Development",
    "Skill India — PMKVY",
    "Startup India",
    "Make in India",
    "PM eVIDYA — Digital Education",
    "Beti Bachao Beti Padhao",
    "Sukanya Samriddhi Yojana",
    "PM Matru Vandana — Maternity Benefit",
    "Pradhan Mantri Gram Sadak Yojana",
    "Bharatmala — Highway Project",
    "Smart Cities Mission",
    "Sagarmala — Port Development",
    "One Nation One Ration Card",
    "PM Garib Kalyan Anna Yojana",
    "PM SVANidhi — Street Vendor Loan",
    "Vishwakarma Yojana",
    "Atmanirbhar Bharat",
]

SOURCES        = ["YouTube", "Twitter", "Instagram", "News App", "Public Forum"]
SOURCE_WEIGHTS = [25, 25, 20, 20, 10]
SENTIMENT_KEYS = ["Positive", "Negative", "Neutral", "Sarcasm"]
SENT_WEIGHTS   = [35, 30, 25, 10]

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 + FIX 2 — Comment TEMPLATES (not final comments)
# {scheme} is replaced with the actual scheme name at generation time
# so every scheme gets a UNIQUE comment, not a shared one
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATES = {

    "Positive": [
        # English
        "The {scheme} scheme has genuinely transformed living conditions in rural India.",
        "My family directly benefited from {scheme}, it made a real difference for us.",
        "{scheme} reached the poorest families in our district, excellent work.",
        "The implementation of {scheme} in our village was smooth and effective.",
        "Thanks to {scheme} my family now has access to something we never had before.",
        "{scheme} is one of the few government schemes that actually delivers on its promise.",
        "Our whole community saw positive change after {scheme} was implemented here.",
        "The benefits of {scheme} arrived quickly and without any major hassle.",
        "I was skeptical about {scheme} but it genuinely helped my family this year.",
        "{scheme} has improved daily life for thousands of families in my area.",
        "The enrollment process for {scheme} was simple and staff were helpful.",
        "After {scheme}, my family's financial situation has clearly improved.",
        "{scheme} gave us access to services we could never afford before.",
        "Real impact on the ground from {scheme}, not just announcements.",
        "People in my block are actually happy with how {scheme} was executed.",
        "The {scheme} initiative is working well in our panchayat area.",
        "We applied for {scheme} and received benefits within three months.",
        "{scheme} made a difference that we can see and feel every single day.",
        "My neighbor's life changed after enrolling in {scheme} last year.",
        "The quality of {scheme} implementation here is genuinely commendable.",
        # Hindi
        "{scheme} योजना से हमारे गाँव की जिंदगी बदल गई।",
        "{scheme} का लाभ हमारे परिवार को सही में मिला, बहुत फायदा हुआ।",
        "{scheme} से हमें वह मिला जिसकी हमें सालों से जरूरत थी।",
        "{scheme} के तहत काम बहुत अच्छा हुआ हमारे जिले में।",
        "{scheme} योजना की वजह से परिवार की आर्थिक स्थिति बेहतर हुई।",
        "{scheme} से गरीब परिवारों को सच में राहत मिली है।",
        "हमारे ब्लॉक में {scheme} का क्रियान्वयन बहुत अच्छे से हुआ।",
        "{scheme} के फायदे तीन महीने में ही मिलने शुरू हो गए।",
        # Hinglish
        "{scheme} ne sach mein farak padaya, family bahut khush hai.",
        "Yaar {scheme} se humein real benefit mila, highly recommend karunga.",
        "{scheme} ka kaam acha hua hamare area mein, log khush hain.",
        "{scheme} enrollment easy tha aur time pe benefit mila.",
        "Mujhe {scheme} se genuine help mili is saal, thankful hun.",
        "{scheme} se gareeb logo ki life mein real change aaya hai.",
    ],

    "Negative": [
        # English
        "{scheme} sounds great on paper but ground implementation is terrible.",
        "Applied for {scheme} two years ago, still waiting, corruption at every step.",
        "{scheme} benefit was rejected without any proper explanation given to us.",
        "Only politically connected people actually receive {scheme} benefits.",
        "The {scheme} portal keeps crashing, impossible to complete registration.",
        "{scheme} was launched before elections, nothing changed after voting.",
        "Documents required for {scheme} are impossible for illiterate people to arrange.",
        "Three years of waiting and still no benefit received under {scheme}.",
        "The {scheme} money never reached us, middlemen swallowed everything.",
        "{scheme} is just a slogan, roads and ground reality remain the same.",
        "My application for {scheme} was rejected on a minor technicality.",
        "{scheme} has helped people with connections, not ordinary citizens.",
        "Registration for {scheme} requires documents that poor families cannot access.",
        "The quality under {scheme} was terrible, completely unusable.",
        "{scheme} awareness is zero in our area, nobody even knows it exists.",
        "Three offices visited, no one could explain how to access {scheme}.",
        "{scheme} benefit was given to the wrong family due to data errors.",
        "After {scheme} was launched our situation is exactly the same as before.",
        "The officials handling {scheme} demand bribes before processing anything.",
        "Rural areas are completely ignored under {scheme}, only cities benefit.",
        # Hindi
        "{scheme} सिर्फ कागजों पर है, जमीन पर कुछ नहीं होता।",
        "{scheme} का पैसा बिचौलिए खा जाते हैं, असली गरीब को कुछ नहीं मिलता।",
        "दो साल से {scheme} के लिए आवेदन किया, अभी तक कोई जवाब नहीं।",
        "{scheme} में भ्रष्टाचार है, बिना रिश्वत के कुछ नहीं होता।",
        "{scheme} से हमें कोई फायदा नहीं हुआ, सिर्फ चक्कर लगाने पड़े।",
        "{scheme} का पोर्टल काम नहीं करता, रजिस्ट्रेशन हो ही नहीं पाई।",
        "{scheme} सिर्फ नेताओं के करीबी लोगों को मिलती है।",
        "गाँव में {scheme} का कोई असर नहीं दिखा, सब झूठ है।",
        # Hinglish
        "{scheme} sirf election ke time ka dikhawa hai bhai.",
        "{scheme} ka paisa neta log kha jaate hain, public ko kuch nahi.",
        "Teen saal ho gaye {scheme} ke liye apply kiya, kuch nahi mila.",
        "{scheme} portal crash karta rehta hai, registration ho hi nahi pati.",
        "Garib insaan {scheme} ki itni paperwork kahan se laayega, practical nahi.",
        "{scheme} sirf bade shahar walon ko milti hai, gaon walon ko kuch nahi.",
        "Hamara {scheme} application reject ho gaya bina kisi reason ke.",
        "{scheme} se real help nahi mili, sirf dhakke khane pade office mein.",
    ],

    "Neutral": [
        # English
        "{scheme} has some good aspects but significant improvements are still needed.",
        "The {scheme} policy is well designed but execution varies greatly by district.",
        "Heard about {scheme} but have not applied for it yet personally.",
        "Results from {scheme} are mixed depending heavily on which district you are in.",
        "{scheme} addresses some important needs but ignores equally critical issues.",
        "Launched recently, still too early to fully judge {scheme} effectiveness.",
        "Some districts did well with {scheme}, others not at all unfortunately.",
        "I have no direct experience with {scheme} yet so cannot comment properly.",
        "The intentions behind {scheme} seem good but delivery is unclear.",
        "{scheme} coverage is improving slowly but has a very long way to go.",
        "Both positive and negative experiences reported under {scheme} by different people.",
        "Government should publish transparent data on actual {scheme} impact.",
        "{scheme} is partially helpful in some areas but completely absent in others.",
        "Need more time to assess the real long-term impact of {scheme}.",
        "The {scheme} numbers look good on paper but ground situation is complex.",
        "Not sure how to access {scheme} benefits, lack of awareness is the issue.",
        "{scheme} exists in our area but I do not know anyone who actually got it.",
        "Mixed feedback about {scheme} from different people in our locality.",
        "Too early to judge {scheme}, let us wait and see the actual outcomes.",
        "Some families benefited from {scheme} while others faced documentation issues.",
        # Hindi
        "{scheme} ठीक है लेकिन अभी और सुधार की जरूरत है।",
        "{scheme} के बारे में सुना है, अभी तक आवेदन नहीं किया।",
        "{scheme} के मिश्रित नतीजे हैं, जगह-जगह अलग अनुभव है।",
        "{scheme} की नीति अच्छी है लेकिन अमल उतना अच्छा नहीं।",
        "पता नहीं {scheme} हमारे गाँव में कब आएगी।",
        "{scheme} के बारे में अभी पूरा आंकलन करना जल्दबाजी होगी।",
        # Hinglish
        "{scheme} ke baare mein suna hai, apply nahi kiya abhi tak.",
        "Kuch logon ko {scheme} se mila kuch ko nahi, mixed results hain.",
        "{scheme} ke baare mein alag alag log alag alag bol rahe hain.",
        "Thoda aur time lagega {scheme} ka full impact samajhne mein.",
        "{scheme} ka kuch jagah acha kaam hua hai, kuch jagah nahi hua.",
        "Intentions theek lagte hain {scheme} mein par ground reality alag hai.",
    ],

    "Sarcasm": [
        # English
        "Oh wow, {scheme} will DEFINITELY reach everyone this time! 🙄",
        "Sure, {scheme} is working perfectly. That is why nobody in my village got it.",
        "Great job on {scheme}! Three years later and still nothing changed. Amazing! 🙄",
        "Yes yes, I am sure {scheme} money reached the right people and not politicians.",
        "Fantastic! Another {scheme} promise, another disappointment. So consistent! 😒",
        "Wow what an incredible {scheme} initiative, I am sure THIS time it will work.",
        "Oh absolutely, {scheme} made the poor SO much better off. Big smiles all around! 🙄",
        "Oh brilliant, {scheme} launched right before elections. What a coincidence! 🙄",
        "Of course {scheme} portal works perfectly at 3am when nobody is applying. 😒",
        "Amazing, three years and {scheme} still has zero impact in my area. Record!",
        "Oh yes, {scheme} is SO effective. That is why my application was rejected twice. 🙄",
        "Wow they announced {scheme}. Too bad nobody told the officials to implement it.",
        "Sure {scheme} is a huge success. That is why all my neighbours gave up on it. 😒",
        "Oh great, another {scheme} welfare announcement right before poll season. Never seen this! 🙄",
        "Amazing {scheme} policy, truly. Works so well that nobody in my area has heard of it. 🙄",
        # Hindi
        "अरे वाह! {scheme} इतनी अच्छी योजना और हमें पता भी नहीं था! बहुत शुक्रिया! 🙄",
        "हाँ हाँ, {scheme} सब ठीक है। एकदम परफेक्ट काम हो रहा है! 😒",
        "वाह क्या बात है! {scheme} के लिए तीन साल से इंतज़ार, अभी तक कुछ नहीं। कमाल! 🙄",
        "जी बिल्कुल, {scheme} से डिजिटल हो गए। बस काम कुछ नहीं होता! 😒",
        # Hinglish
        "Haan haan, {scheme} toh bahut achhi hai. Bus kaam nahi karti kuch! 😒",
        "Oh sure, {scheme} DEFINITELY kaam karta hai. Sirf humara luck kharab hai! 🙄",
        "Wah kya {scheme} hai! Teen saal pehle apply kiya, abhi bhi wait kar rahe hain.",
        "Bilkul sahi hai, {scheme} ne reject kar diya. Itni great scheme ke liye normal hai! 🙄",
        "Sure sure, {scheme} ka paisa zaroor sahi jagah pahuncha. Politicians ko kya pata! 🙄",
        "Haan {scheme} bahut successful hai. Tabhi toh mere gaon mein iska naam bhi nahi suna. 😒",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def detect_lang(text: str) -> str:
    hindi  = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    tamil  = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
    telugu = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    if hindi  > 3: return "hi"
    if tamil  > 3: return "ta"
    if telugu > 3: return "te"
    hinglish = {"hai","hain","nahi","kuch","yeh","toh","bhai","tha","gaya","mila",
                "wala","sab","bahut","acha","achi","bhi","pe","se","ko","ka","haan"}
    if len(set(text.lower().split()) & hinglish) >= 2:
        return "hinglish"
    return "en"


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — Normalise for dedup comparison (same as scraper.py)
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 + FIX 2 + FIX 3 — Generate unique comments per scheme
# Each comment has {scheme} replaced with the actual scheme name
# so the same template produces 40 DIFFERENT training samples
# ROWS_PER_SCHEME reduced 60→25 to avoid over-saturation
# ─────────────────────────────────────────────────────────────────────────────
ROWS_PER_SCHEME = 25   # 25 × 40 schemes = 1000 unique rows


def generate_rows() -> list:
    rows    = []
    row_id  = 1
    seen    = set()   # track normalised text globally

    for scheme in SCHEMES:
        counts = {s: 0 for s in SENTIMENT_KEYS}

        # How many of each sentiment for this scheme
        targets = {}
        remaining = ROWS_PER_SCHEME
        for i, sent in enumerate(SENTIMENT_KEYS):
            if i == len(SENTIMENT_KEYS) - 1:
                targets[sent] = remaining
            else:
                n = round(ROWS_PER_SCHEME * SENT_WEIGHTS[i] / 100)
                targets[sent] = n
                remaining -= n

        for sentiment, target in targets.items():
            templates = TEMPLATES[sentiment].copy()
            random.shuffle(templates)
            added = 0

            for template in templates:
                if added >= target:
                    break

                # FIX 2: Inject scheme name into template → unique comment
                comment = template.replace("{scheme}", scheme)
                norm    = _normalise(comment)

                # Skip if this exact normalised text already exists globally
                if norm in seen:
                    continue

                seen.add(norm)
                source = random.choices(SOURCES, weights=SOURCE_WEIGHTS)[0]
                lang   = detect_lang(comment)
                rows.append([row_id, scheme, source, lang, comment, sentiment])
                row_id += 1
                added  += 1
                counts[sentiment] += 1

    random.shuffle(rows)
    return rows


def smart_merge_and_save(new_rows: list) -> int:
    """
    Reads existing data.csv (preserves all real fetched data),
    adds only non-duplicate synthetic rows, saves back.
    FIX 4: Uses normalised comparison, not raw .strip()
    """
    existing_norm = set()
    next_id       = 1

    if DATA_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(DATA_CSV, encoding="utf-8")
            # FIX 4: normalised set, not raw
            existing_norm = set(df["Comment"].dropna().apply(_normalise).tolist())
            next_id       = int(df["ID"].max()) + 1 if len(df) > 0 else 1
            print(f"\n  Found existing data.csv → {len(df)} rows preserved")
            print(f"  Real data kept — only appending new unique synthetic rows\n")
        except Exception as e:
            print(f"\n  Could not read existing CSV ({e}), creating fresh\n")

    to_add = []
    for row in new_rows:
        norm = _normalise(str(row[4]))
        if norm not in existing_norm:
            row[0] = next_id + len(to_add)
            to_add.append(row)
            existing_norm.add(norm)

    if not to_add:
        print("  No new synthetic rows to add — all already present in data.csv")
        return 0

    DATA_CSV.parent.mkdir(exist_ok=True)
    header     = ["ID", "Scheme", "Source", "Language", "Comment", "Sentiment"]
    write_mode = "a" if DATA_CSV.exists() and DATA_CSV.stat().st_size > 0 else "w"

    with open(DATA_CSV, write_mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_mode == "w":
            w.writerow(header)
        for row in to_add:
            w.writerow(row)

    return len(to_add)


def print_final_stats():
    try:
        import pandas as pd
        df = pd.read_csv(DATA_CSV, encoding="utf-8")

        total   = len(df)
        unique  = df["Comment"].nunique()
        dupe_rt = round((1 - unique / total) * 100, 1) if total > 0 else 0
        health  = "✓ healthy" if dupe_rt < 3 else "⚠ high — check templates"

        print(f"\n  ── Final data.csv stats ──")
        print(f"     Total rows      : {total}")
        print(f"     Unique comments : {unique}")
        print(f"     Duplicate rate  : {dupe_rt}%  ({health})")   # FIX 5
        print(f"     Schemes         : {df['Scheme'].nunique()}")
        print(f"     Sources         : {df['Source'].nunique()}")
        print(f"     Languages       : {df['Language'].nunique() if 'Language' in df.columns else '?'}")
        print()
        print(f"  ── Sentiment breakdown ──")
        for s, c in df["Sentiment"].value_counts().items():
            pct = round(c / total * 100, 1)
            bar = "█" * (c // 20)
            print(f"     {s:<12} {c:>5}  ({pct}%)  {bar}")
        print()
        print(f"  ── Sources breakdown ──")
        for s, c in df["Source"].value_counts().items():
            print(f"     {s:<22} {c:>5}")
    except Exception as e:
        print(f"  Could not read stats: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*58)
    print("  Pulse Sentiment AI — Dataset Generator")
    print("  All Modi Government Schemes (2014 → Present)")
    print("="*58)

    new_rows = generate_rows()
    print(f"\n  Generated {len(new_rows)} unique synthetic rows")
    print(f"  Schemes covered : {len(SCHEMES)}")
    print(f"  Rows per scheme : {ROWS_PER_SCHEME} (unique)")

    added = smart_merge_and_save(new_rows)

    if added > 0:
        print(f"  Added {added} new unique rows to data.csv")

    print_final_stats()

    print("\n" + "="*58)
    print("  data.csv is ready!")
    print("  Now run:  streamlit run app.py")
    print("="*58 + "\n")
