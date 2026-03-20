"""
data/generate_data.py
═════════════════════
Generates the base synthetic dataset for Pulse Sentiment AI.

Covers ALL major Modi government schemes since 2014.
Languages: English, Hindi, Hinglish, Tamil (sample), Telugu (sample)
Sources:   YouTube, Twitter, Instagram, News App, Public Forum

Run ONCE (or anytime) to populate data/data.csv:
    python data/generate_data.py

SMART MERGE: If data.csv already has real fetched data from YouTube,
Twitter, Instagram or NewsAPI — this script NEVER overwrites it.
It only APPENDS new synthetic rows, avoiding duplicates automatically.
"""

import csv, random
from pathlib import Path
from collections import Counter

DATA_CSV = Path("data/data.csv")

# ─────────────────────────────────────────────────────────────────────────────
#  ALL SCHEMES — matches scraper.py SCHEME_KEYWORDS exactly
# ─────────────────────────────────────────────────────────────────────────────
SCHEMES = [
    "PMAY",
    "Ayushman Bharat", "Poshan Abhiyaan", "Ayushman Bharat Digital Mission",
    "PM Kisan", "Fasal Bima", "Kisan Credit Card", "e-NAM",
    "Digital India", "BharatNet", "UPI",
    "Jan Dhan Yojana", "Mudra Yojana", "Stand Up India",
    "Atal Pension Yojana", "PM Jeevan Jyoti Bima", "PM Suraksha Bima",
    "Ujjwala Yojana", "Saubhagya", "Solar Rooftop", "FAME",
    "Swachh Bharat", "Jal Jeevan Mission", "AMRUT",
    "Skill India", "Startup India", "Make in India", "PM eVIDYA",
    "Beti Bachao Beti Padhao", "Sukanya Samriddhi", "PM Matru Vandana",
    "PMGSY", "Bharatmala", "Smart Cities", "Sagarmala",
    "One Nation One Ration Card", "PM Garib Kalyan", "PM SVANidhi",
    "Vishwakarma Yojana", "Atmanirbhar Bharat",
]

SOURCES         = ["YouTube", "Twitter", "Instagram", "News App", "Public Forum"]
SOURCE_WEIGHTS  = [25, 25, 20, 20, 10]
SENTIMENT_KEYS  = ["Positive", "Negative", "Neutral", "Sarcasm"]
SENT_WEIGHTS    = [35, 30, 25, 10]
ROWS_PER_SCHEME = 60   # 60 × 40 schemes = 2400 rows

# ─────────────────────────────────────────────────────────────────────────────
#  COMMENT BANK
# ─────────────────────────────────────────────────────────────────────────────
COMMENTS = {
    "Positive": [
        "This scheme has genuinely transformed living conditions in rural India.",
        "My family finally got a pucca house after years of struggle, PMAY really works.",
        "Ayushman Bharat saved my father's life, the free treatment was a real blessing.",
        "Digital India has made banking accessible even in remote villages.",
        "PM Kisan money arrives every season on time, very helpful for buying seeds.",
        "Swachh Bharat Mission made our entire village clean and hygienic.",
        "The scheme reached the poorest families in our district, excellent implementation.",
        "Jan Dhan account opened the door to banking for my whole family.",
        "Ujjwala Yojana gave us clean cooking gas, no more smoke-filled kitchen.",
        "PM SVANidhi loan helped me restart my street food business after COVID.",
        "Jal Jeevan Mission brought tap water to our home for the first time ever.",
        "Skill India training helped my son get a decent job in six months.",
        "The Mudra loan was processed quickly, helped me expand my small shop.",
        "Saubhagya scheme finally brought electricity to our village after 70 years.",
        "Kisan Credit Card made accessing agricultural credit so much simpler.",
        "Make in India has created real manufacturing jobs in our district.",
        "Startup India funding helped our team launch a product that is now profitable.",
        "Sukanya Samriddhi is the best investment I made for my daughter's future.",
        "PM Garib Kalyan free ration during COVID literally saved people from starvation.",
        "BharatNet brought broadband to our village, children can now study online.",
        "Solar rooftop scheme reduced our electricity bill by almost 80 percent.",
        "Beti Bachao Beti Padhao changed mindsets in our community for the better.",
        "FAME subsidy made it affordable for me to buy an electric scooter.",
        "One Nation One Ration Card helped my cousin access ration in another city.",
        "Atal Pension Yojana gives peace of mind for old age, affordable and reliable.",
        "The Poshan Abhiyaan has visibly reduced malnutrition in our village children.",
        "PM eVIDYA helped my kids continue their studies even during school closures.",
        "PMGSY road finally connected our village to the main highway after decades.",
        "Vishwakarma Yojana gave my carpenter neighbor the recognition and loan he needed.",
        "Fasal Bima insurance actually paid out when our crop was damaged by floods.",
        "यह योजना हमारे गाँव के लिए बहुत फायदेमंद साबित हुई है।",
        "सरकार ने गरीबों के लिए सच में अच्छा काम किया है।",
        "आयुष्मान भारत से हमारा मुफ्त इलाज हुआ, बहुत धन्यवाद।",
        "डिजिटल इंडिया से हमारे गाँव में भी इंटरनेट आ गया।",
        "पीएम किसान की राशि समय पर आती है, बहुत मदद मिलती है।",
        "उज्जवला योजना से घर में गैस मिली, अब धुएं से राहत मिली।",
        "जल जीवन मिशन से पहली बार घर में नल से पानी आया।",
        "जनधन खाते से पूरे परिवार की बैंकिंग शुरू हो गई।",
        "स्वच्छ भारत मिशन ने हमारे मोहल्ले को पूरी तरह बदल दिया।",
        "मुद्रा लोन से मेरी दुकान फिर से चल पड़ी।",
        "सौभाग्य योजना से गाँव में बिजली आई, बहुत खुशी हुई।",
        "बेटी बचाओ बेटी पढ़ाओ ने हमारे इलाके में सोच बदल दी।",
        "स्किल इंडिया ट्रेनिंग से बेटे को नौकरी मिल गई।",
        "आयुष्मान कार्ड से अस्पताल में मुफ्त ऑपरेशन हुआ।",
        "Yeh scheme bahut achi hai yaar, sach mein farak padta hai.",
        "PMAY ne humari life badal di, finally ghar mil gaya.",
        "Ayushman card se hospital mein free treatment mila, bahut achha laga.",
        "Digital India se sab kuch easy ho gaya hai, payments bhi online.",
        "PM Kisan ka paisa time pe aata hai, kisan ko bahut help hoti hai.",
        "Jan Dhan account khulne se bank se connection ho gaya finally.",
        "Ujjwala se gas cylinder mila, kitchen mein dhuan khatam ho gaya.",
        "Mudra loan process fast tha, business start karne mein help mili.",
        "Skill India training ke baad acchi job mil gayi bhai.",
        "இந்த திட்டம் மக்களுக்கு மிகவும் பயனுள்ளதாக உள்ளது.",
        "ஆயுஷ்மான் பாரத் திட்டம் என் குடும்பத்தை காப்பாற்றியது.",
        "ఈ పథకం నిజంగా పేద కుటుంబాలకు సహాయం చేసింది.",
        "పీఎం కిసాన్ డబ్బులు సకాలంలో వస్తున్నాయి చాలా మంచిది.",
    ],

    "Negative": [
        "The scheme sounds great on paper but implementation on ground is terrible.",
        "Applied 2 years ago, still waiting for PMAY benefit, corruption at every step.",
        "Ayushman Bharat card rejected at three hospitals, what is the point of it?",
        "Digital India but half the country still has no reliable internet at all.",
        "PM Kisan money never reached my account, middlemen swallowed everything.",
        "Swachh Bharat is just a photo-op slogan, roads in my area are still filthy.",
        "Only politically connected people actually receive these scheme benefits.",
        "The government portal keeps crashing, impossible to complete registration.",
        "Another scheme launched before elections, nothing will change after voting.",
        "Documents required are impossible for illiterate rural poor to arrange.",
        "Three years of waiting and still no gas cylinder under Ujjwala scheme.",
        "Jal Jeevan Mission pipes were laid but water hasn't come through once.",
        "Skill India training was completely useless, no placement support provided.",
        "The Mudra loan process is so complicated, most small businesses give up.",
        "Smart Cities project money went into pockets, our city looks the same.",
        "One Nation Ration Card doesn't work at shops in my area, shopkeepers refuse.",
        "PM Garib Kalyan ration quality was terrible, full of stones and insects.",
        "BharatNet cables were laid three years ago but service never activated.",
        "FAME subsidy applications take so long the scheme becomes practically useless.",
        "Startup India recognition means nothing without actual funding.",
        "Fasal Bima claim was rejected on a technicality after crop was destroyed.",
        "AMRUT funds were allocated but our city's water supply is worse than before.",
        "Poshan Abhiyaan anganwadi is closed most days, children get nothing.",
        "PM eVIDYA content is not in regional language, my children cannot understand.",
        "Vishwakarma Yojana registration portal is broken, artisans cannot enroll.",
        "यह योजना सिर्फ नाम की है, जमीन पर कुछ नहीं होता।",
        "बिचौलिए सारा पैसा खा जाते हैं, गरीब को कुछ नहीं मिलता।",
        "दो साल हो गए आवेदन किया, अभी तक घर नहीं मिला।",
        "सरकारी दफ्तर में सिर्फ रिश्वत चलती है।",
        "आयुष्मान कार्ड से कोई फायदा नहीं, हर जगह रिजेक्ट होता है।",
        "उज्जवला का सिलेंडर मिला पर रिफिल इतना महंगा है कि काम नहीं चलता।",
        "जल जीवन मिशन के पाइप लगे पर पानी एक बार भी नहीं आया।",
        "पोर्टल काम नहीं करता, हर बार एरर आता है।",
        "स्किल इंडिया ट्रेनिंग से कुछ नहीं मिला, समय बर्बाद हुआ।",
        "मुद्रा लोन के लिए बैंक वाले चक्कर पर चक्कर लगवाते हैं।",
        "Yeh sab sirf election ke time ka dikhaawa hai bhai.",
        "Scheme ka paisa neta log kha jaate hain, public ko kuch nahi milta.",
        "PMAY ke liye apply kiya tha, teen saal ho gaye kuch nahi mila.",
        "Portal crash karta rehta hai, registration complete hi nahi hoti.",
        "Garib insaan itni paperwork kahan se laayega, practical nahi hai.",
        "Ayushman card le ke gaya hospital mein, unhone reject kar diya.",
        "Ujjwala cylinder mila tha ek baar, refill ka paisa kahan se laayein.",
        "Ration card wali scheme dukaan waale accept hi nahi karte yahan.",
        "Sirf bade shahar walon ko faida hota hai, gaon walon ko kuch nahi.",
        "Smart City mein smart kuch nahi dikha abhi tak, sab bakwaas hai.",
    ],

    "Neutral": [
        "The scheme has some good aspects but significant improvements are still needed.",
        "Benefits exist but awareness is still completely lacking in remote areas.",
        "Some families benefited while others faced serious documentation issues.",
        "Policy is well-designed but execution quality varies wildly by state.",
        "Heard about this scheme but have not applied for it yet.",
        "Results are mixed, depends heavily on which district you are in.",
        "It addresses some important needs but ignores equally critical issues.",
        "Launched recently, still too early to fully judge its effectiveness.",
        "Some districts did well with implementation, others not at all.",
        "I have no direct experience with this scheme yet so cannot comment.",
        "The intentions seem good but whether it actually delivers is unclear.",
        "Coverage is improving slowly but has a very long way to go.",
        "Both positive and negative experiences reported by different people.",
        "The scheme exists but I am not sure how to practically access it.",
        "Government should evaluate and transparently publish actual impact data.",
        "Partially helpful in some areas but completely absent in others.",
        "Need more time to assess the real long-term impact of this initiative.",
        "The numbers look good but on-ground situation is more complicated.",
        "योजना ठीक है लेकिन और सुधार की जरूरत है।",
        "कुछ लोगों को फायदा हुआ, कुछ को नहीं हुआ।",
        "अभी तक आवेदन नहीं किया, पूरी जानकारी नहीं है।",
        "सरकार की नीति सही है लेकिन अमल उतना अच्छा नहीं।",
        "पता नहीं यह योजना हमारे गाँव में कब आएगी।",
        "मिश्रित नतीजे हैं, जगह-जगह अलग-अलग अनुभव है।",
        "देखना होगा, अभी पूरी तरह आंकलन करना जल्दबाजी होगी।",
        "Scheme ke baare mein suna hai, apply nahi kiya abhi tak.",
        "Kuch logon ko mila kuch ko nahi, mixed results hain.",
        "Dekhna padega, abhi judge karna thoda jaldi hai.",
        "Kuch jagah acha kaam hua hai, kuch jagah nahi hua.",
        "Thoda aur time lagega iska full impact samajhne mein.",
        "Intentions theek lagte hain par ground reality alag hai.",
        "Mujhe abhi tak is scheme ka koi experience nahi hai personally.",
        "Slowly slowly improve ho raha hai shayad, wait and watch.",
    ],

    "Sarcasm": [
        "Oh wow, another GREAT government scheme that will DEFINITELY reach everyone! 🙄",
        "Sure, Digital India works perfectly when half the country cannot afford smartphones.",
        "Great job! 5 years later and the road outside my house is still broken. Amazing! 🙄",
        "Yes yes, I am sure the PM Kisan money reached farmers and not the politicians.",
        "Fantastic! Another promise, another disappointment. At least they are consistent! 😒",
        "Wow what an incredible initiative, I am sure THIS time it will actually work.",
        "Oh absolutely, the poor are SO much better off now. Look at those big smiles! 🙄",
        "Sure, blame the farmer for not receiving PM Kisan money. System is perfect! 😒",
        "Oh brilliant, another scheme launched right before elections. What a coincidence! 🙄",
        "Great, the hospital rejected my Ayushman card again. Such a wonderful scheme.",
        "Of course the portal works perfectly at 3am when no one is applying. Very convenient! 😒",
        "Amazing, three years and the Jal Jeevan pipe still has zero water. Record!",
        "Oh yes, Smart Cities are SO smart. Same potholes from 10 years ago! 🙄",
        "Wow they built BharatNet cables in our village. Too bad no internet came through.",
        "Sure the scheme is a huge success. That is why all my neighbours gave up applying. 😒",
        "Oh great, another welfare announcement right before poll season. Never seen this before!",
        "Amazing policy, truly. Works so well that nobody in my village has heard of it. 🙄",
        "Fantastic, the toilet was built in front of the house. We use it as a storage room. 😒",
        "अरे वाह! इतनी अच्छी योजना और हमें पता भी नहीं था! बहुत शुक्रिया सरकार जी! 🙄",
        "हाँ हाँ, सब ठीक है। योजना भी, अमल भी। सब एकदम परफेक्ट है! 😒",
        "वाह क्या बात है! तीन साल में सिर्फ फॉर्म भरवाए, घर नहीं मिला। कमाल है!",
        "जी बिल्कुल, हमारे गाँव में डिजिटल इंडिया है। बस नेटवर्क नहीं आता! 😒",
        "Haan haan, scheme toh bahut achhi hai. Bus kaam nahi karti kuch! 😒",
        "Oh sure, portal DEFINITELY works. Sirf 500 baar refresh karna padta hai! 🙄",
        "Wah kya scheme hai! Teen saal pehle apply kiya, abhi bhi wait kar rahe hain.",
        "Bilkul sahi hai, hospital ne card reject kiya. Itni great scheme ke liye normal hai! 🙄",
        "Haan digital india ho gaya, bas mere gaon mein internet nahi aata. Chhoti si baat! 😒",
        "Sure sure, PM Kisan paisa zaroor farmers tak pahuncha. Politicians ko kya pata! 🙄",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def detect_lang(text):
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


def generate_rows():
    rows, row_id = [], 1
    for scheme in SCHEMES:
        for _ in range(ROWS_PER_SCHEME):
            sentiment = random.choices(SENTIMENT_KEYS, weights=SENT_WEIGHTS)[0]
            comment   = random.choice(COMMENTS[sentiment])
            source    = random.choices(SOURCES, weights=SOURCE_WEIGHTS)[0]
            lang      = detect_lang(comment)
            rows.append([row_id, scheme, source, lang, comment, sentiment])
            row_id += 1
    random.shuffle(rows)
    return rows


def smart_merge_and_save(new_rows):
    """
    Reads existing data.csv (preserves all real fetched data),
    adds only non-duplicate synthetic rows, saves back.
    """
    existing_comments = set()
    existing_rows     = []
    next_id           = 1

    if DATA_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(DATA_CSV, encoding="utf-8")
            existing_comments = set(df["Comment"].str.strip().tolist())
            existing_rows     = df.values.tolist()
            next_id           = int(df["ID"].max()) + 1 if len(df) > 0 else 1
            print(f"\n  📂 Found existing data.csv → {len(df)} rows (real + previous synthetic)")
            print(f"     Real data preserved — only adding new synthetic rows\n")
        except Exception as e:
            print(f"\n  Could not read existing CSV ({e}), creating fresh\n")

    # Deduplicate
    to_add = []
    for row in new_rows:
        if row[4].strip() not in existing_comments:
            row[0] = next_id + len(to_add)
            to_add.append(row)
            existing_comments.add(row[4].strip())

    if not to_add:
        print("  ℹ️  No new synthetic rows to add — all already present in data.csv")
        return 0

    DATA_CSV.parent.mkdir(exist_ok=True)
    header = ["ID", "Scheme", "Source", "Language", "Comment", "Sentiment"]

    if existing_rows:
        # Append only
        with open(DATA_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in to_add:
                w.writerow(row)
    else:
        # Fresh file
        with open(DATA_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in to_add:
                w.writerow(row)

    return len(to_add)


def print_final_stats():
    try:
        import pandas as pd
        df = pd.read_csv(DATA_CSV, encoding="utf-8")
        print(f"\n  ── Final data.csv stats ──")
        print(f"     Total rows     : {len(df)}")
        print(f"     Schemes        : {df['Scheme'].nunique()}")
        print(f"     Sources        : {df['Source'].nunique()}")
        print(f"     Languages      : {df['Language'].nunique() if 'Language' in df.columns else '?'}")
        print()
        print(f"  ── Sentiment breakdown ──")
        for s, c in df["Sentiment"].value_counts().items():
            pct = round(c / len(df) * 100, 1)
            bar = "█" * (c // 30)
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
    print(f"\n  Generated {len(new_rows)} candidate synthetic rows")
    print(f"  Schemes covered: {len(SCHEMES)}")

    added = smart_merge_and_save(new_rows)

    if added > 0:
        print(f"  ✅ Added {added} new synthetic rows to data.csv")

    print_final_stats()

    print("="*58)
    print("  ✅ data.csv is ready!")
    print("  👉 Now run:  streamlit run app.py")
    print("="*58 + "\n")