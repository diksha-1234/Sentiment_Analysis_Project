"""
data/storage.py — Pulse Sentiment AI · Fixed Version
══════════════════════════════════════════════════════
FIXES IN THIS VERSION:
  ✅ FIX 1: load_data() — proper pagination beyond 1000 row Supabase default cap
  ✅ FIX 2: load_data() — RPC fallback with manual pagination if RPC fails
  ✅ FIX 3: get_stats() — uses COUNT(*) SQL directly, never capped at 1000
  ✅ FIX 4: _df_from_records() — handles ALL Supabase column name variants
  ✅ FIX 5: _save_to_supabase() — accurate saved count, better error logging
  ✅ FIX 6: force_refresh() — wipes ALL caches including translation memory
  ✅ FIX 7: load_data() — prints actual row count so you can see in logs
"""

import os
import pandas as pd
from pathlib import Path

DATA_CSV = Path("data/data.csv")

# All expected output columns
COLUMNS = ["ID", "Scheme", "Source", "Language", "Comment", "Sentiment"]

# In-memory translation cache (within session)
_translation_memory: dict = {}


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")


def _is_deployed() -> bool:
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_KEY"))


def _get_client():
    from supabase import create_client
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    return create_client(url, key)


def _df_from_records(records: list) -> pd.DataFrame:
    """
    FIX 4 — Robust column renaming.

    Supabase returns lowercase column names.
    preprocess_dataframe() expects Title-Case names.
    This function maps ALL possible variants so nothing gets dropped.
    """
    if not records:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(records)

    # Comprehensive rename map — covers all Supabase column name variants
    rename_map = {
        # Supabase lowercase → app Title Case
        "id":          "ID",
        "scheme":      "Scheme",
        "source":      "Source",
        "language":    "Language",
        "comment":     "Comment",
        "sentiment":   "Sentiment",
        "translated":  "Translated",
        # Already correct (in case of future mixed case)
        "ID":          "ID",
        "Scheme":      "Scheme",
        "Source":      "Source",
        "Language":    "Language",
        "Comment":     "Comment",
        "Sentiment":   "Sentiment",
        "Translated":  "Translated",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure all required columns exist — fill missing ones with sensible defaults
    if "ID"         not in df.columns: df["ID"]         = range(len(df))
    if "Scheme"     not in df.columns: df["Scheme"]      = "General"
    if "Source"     not in df.columns: df["Source"]      = "Other"
    if "Language"   not in df.columns: df["Language"]    = "en"
    if "Comment"    not in df.columns:
        # Try to find comment column under any name
        for possible in ["text", "Text", "body", "Body", "content", "Content"]:
            if possible in df.columns:
                df["Comment"] = df[possible]
                break
        else:
            df["Comment"] = ""
    if "Sentiment"  not in df.columns: df["Sentiment"]   = "Neutral"
    if "Translated" not in df.columns: df["Translated"]  = ""

    return df[["ID", "Scheme", "Source", "Language", "Comment", "Sentiment", "Translated"]].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
#  FORCE REFRESH
# ═════════════════════════════════════════════════════════════════════════════
def force_refresh() -> None:
    """
    Hard-wipe every in-memory cache.
    After calling this, the next load_data() / get_stats() will
    query Supabase completely fresh.
    """
    global _translation_memory
    _translation_memory = {}

    try:
        import streamlit as st
        st.cache_data.clear()
        st.cache_resource.clear()
        print("[Storage] force_refresh: all caches cleared")
    except Exception as e:
        print(f"[Storage] force_refresh: could not clear st caches: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD DATA — FIX 1 + FIX 2: proper pagination, never capped at 1000
# ═════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Load all sentiment rows from Supabase with proper pagination.

    FIX 1: The default Supabase REST API caps at 1000 rows per request.
    This function paginates in batches of 1000 until ALL rows are fetched.

    FIX 2: Tries RPC first (fastest), then paginated SELECT as fallback.
    """
    if not _is_deployed():
        # Local CSV fallback
        if DATA_CSV.exists():
            try:
                df = pd.read_csv(DATA_CSV, encoding="utf-8")
                print(f"[Storage] Loaded {len(df)} rows from local CSV")
                return df
            except Exception as e:
                print(f"[Storage] CSV load failed: {e}")
        return pd.DataFrame(columns=COLUMNS)

    client = _get_client()

    # ── Method 1: RPC (designed to bypass row limits) ─────────────────────────
    try:
        response = client.rpc("get_all_sentiment_data").execute()
        if response.data:
            df = _df_from_records(response.data)
            print(f"[Storage] ✓ Loaded {len(df)} rows via RPC")
            return df
        else:
            print("[Storage] RPC returned empty — trying paginated SELECT")
    except Exception as e:
        print(f"[Storage] RPC failed: {e} — trying paginated SELECT")

    # ── Method 2: Paginated SELECT — fetches ALL rows beyond 1000 cap ─────────
    try:
        all_data   = []
        batch_size = 1000
        offset     = 0

        while True:
            response = (
                client.table("sentiment_data")
                .select("*")
                .order("id")
                .range(offset, offset + batch_size - 1)
                .execute()
            )

            batch = response.data or []
            if not batch:
                break

            all_data.extend(batch)
            print(f"[Storage] Fetched batch: offset={offset}, got={len(batch)}, total so far={len(all_data)}")

            # If we got fewer rows than requested, we've hit the end
            if len(batch) < batch_size:
                break

            offset += batch_size

        if all_data:
            df = _df_from_records(all_data)
            print(f"[Storage] ✓ Loaded {len(df)} total rows via paginated SELECT")
            return df
        else:
            print("[Storage] No data found in Supabase")
            return pd.DataFrame(columns=COLUMNS)

    except Exception as e:
        print(f"[Storage] Paginated SELECT failed: {e}")
        return pd.DataFrame(columns=COLUMNS)


# ═════════════════════════════════════════════════════════════════════════════
#  GET STATS — FIX 3: accurate count using SQL COUNT, never capped
# ═════════════════════════════════════════════════════════════════════════════
def get_stats() -> dict:
    """
    FIX 3 — Uses Supabase count="exact" with head=True.
    This sends a COUNT(*) query — returns the real number of rows,
    never capped at 1000, never reads from any cache.
    """
    if not _is_deployed():
        df = load_data()
        return {"total_rows": len(df)}

    try:
        client   = _get_client()
        response = (
            client.table("sentiment_data")
            .select("*", count="exact")
            .limit(1)           # Fetch only 1 row — we only need the count
            .execute()
        )
        count = response.count or 0
        print(f"[Storage] get_stats: {count} rows in Supabase")
        return {"total_rows": count}
    except Exception as e:
        print(f"[Storage] get_stats failed: {e}")
        # Fallback — count manually
        try:
            df = load_data()
            return {"total_rows": len(df)}
        except Exception:
            return {"total_rows": 0}


# ═════════════════════════════════════════════════════════════════════════════
#  SAVE ROWS
# ═════════════════════════════════════════════════════════════════════════════
def save_rows(rows: list) -> int:
    if not rows:
        return 0
    if _is_deployed():
        return _save_to_supabase(rows)
    else:
        return _save_to_csv(rows)


def _save_to_supabase(rows: list) -> int:
    """
    FIX 5 — Accurate saved count + better error logging.

    Uses upsert with on_conflict="comment" so:
    - New comments → inserted
    - Existing comments → updated (sentiment refreshed)
    - No duplicate errors
    """
    try:
        client    = _get_client()
        formatted = []

        for r in rows:
            text = r.get("Comment", "").strip()
            if len(text) < 15:
                continue
            formatted.append({
                "scheme":     r.get("Scheme",    "General"),
                "source":     r.get("Source",    "Other"),
                "language":   r.get("Language",  "en"),
                "comment":    text,
                "sentiment":  r.get("Sentiment", "Neutral"),
                "translated": r.get("Translated",""),
            })

        if not formatted:
            print("[Storage] No valid rows to save (all too short)")
            return 0

        saved      = 0
        batch_size = 100

        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i + batch_size]
            try:
                response = (
                    client.table("sentiment_data")
                    .upsert(batch, on_conflict="comment")
                    .execute()
                )
                # response.data contains the upserted rows
                batch_saved = len(response.data) if response.data else 0
                saved += batch_saved
                print(f"[Storage] Batch {i//batch_size + 1}: saved {batch_saved} rows")
            except Exception as e:
                print(f"[Storage] Batch {i//batch_size + 1} error: {e}")
                continue

        print(f"[Storage] ✓ Total saved to Supabase: {saved} rows")
        return saved

    except Exception as e:
        print(f"[Storage] Supabase save failed: {e}")
        return 0


def _save_to_csv(rows: list) -> int:
    import csv

    DATA_CSV.parent.mkdir(exist_ok=True)

    existing_norm = set()
    next_id       = 1

    if DATA_CSV.exists():
        try:
            df = pd.read_csv(DATA_CSV, encoding="utf-8", usecols=["ID", "Comment"])
            existing_norm = set(
                df["Comment"].dropna()
                .apply(lambda t: " ".join(str(t).lower().split()))
                .tolist()
            )
            next_id = int(df["ID"].max()) + 1 if len(df) else 1
        except Exception:
            pass

    seen_in_batch = set()
    deduped       = []

    for r in rows:
        text = r.get("Comment", "").strip()
        norm = " ".join(text.lower().split())
        if len(text) < 15:        continue
        if norm in seen_in_batch: continue
        if norm in existing_norm: continue
        seen_in_batch.add(norm)
        existing_norm.add(norm)
        deduped.append(r)

    if not deduped:
        return 0

    header = not DATA_CSV.exists() or DATA_CSV.stat().st_size == 0

    with open(DATA_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["ID", "Scheme", "Source", "Language", "Comment", "Sentiment"]
        )
        if header:
            w.writeheader()
        for i, r in enumerate(deduped):
            r["ID"] = next_id + i
            w.writerow(r)

    print(f"[Storage] ✓ Saved {len(deduped)} rows to local CSV")
    return len(deduped)


# ═════════════════════════════════════════════════════════════════════════════
#  TRANSLATION CACHE
# ═════════════════════════════════════════════════════════════════════════════
def get_cached_translation(text: str):
    text = text.strip()
    if not text:
        return None

    # Level 1: in-memory (instant)
    if text in _translation_memory:
        return _translation_memory[text]

    if not _is_deployed():
        return None

    # Level 2: Supabase lookup
    try:
        client = _get_client()
        res = (
            client.table("sentiment_data")
            .select("translated")
            .eq("comment", text)
            .limit(1)
            .execute()
        )
        if res.data:
            cached = res.data[0].get("translated", "").strip()
            if cached:
                _translation_memory[text] = cached
                return cached
    except Exception as e:
        print(f"[Storage] Translation cache lookup failed: {e}")

    return None


def save_cached_translation(text: str, translated: str) -> None:
    text       = text.strip()
    translated = translated.strip()

    if not text or not translated:
        return

    # Always save in-memory
    _translation_memory[text] = translated

    if not _is_deployed():
        return

    # Save to Supabase
    try:
        client = _get_client()
        client.table("sentiment_data") \
            .update({"translated": translated}) \
            .eq("comment", text) \
            .execute()
    except Exception as e:
        print(f"[Storage] Translation cache save failed: {e}")


def get_translation_cache_stats() -> dict:
    stats = {"in_memory_count": len(_translation_memory), "supabase_cached": 0}
    if _is_deployed():
        try:
            client = _get_client()
            res = (
                client.table("sentiment_data")
                .select("*", count="exact")
                .neq("translated", "")
                .limit(1)
                .execute()
            )
            stats["supabase_cached"] = res.count or 0
        except Exception:
            pass
    return stats
