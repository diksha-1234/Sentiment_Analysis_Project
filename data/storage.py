"""
data/storage.py — Persistent Storage for Pulse Sentiment AI
════════════════════════════════════════════════════════════
Uses Supabase (PostgreSQL) when deployed on Streamlit Cloud.
Falls back to local data/data.csv when running locally.

FEATURES:
  ✅ Sentiment data persistence (load/save rows)
  ✅ Translation cache — translations stored in Supabase `translated` column
     so the same Hindi/Tamil/Bengali text is never sent to API twice
  ✅ Stats helper
  ✅ Local CSV fallback for development

Supabase table setup (run once in SQL Editor):
  ── sentiment_data table ──────────────────────────────────
  CREATE TABLE sentiment_data (
      id         BIGSERIAL PRIMARY KEY,
      scheme     TEXT,
      source     TEXT,
      language   TEXT DEFAULT 'en',
      comment    TEXT UNIQUE,
      sentiment  TEXT DEFAULT 'Neutral',
      translated TEXT DEFAULT ''
  );

  ── Add translated column if table already exists ─────────
  ALTER TABLE sentiment_data
  ADD COLUMN IF NOT EXISTS translated TEXT DEFAULT '';

  ── RPC function to bypass row limit (run once) ───────────
  CREATE OR REPLACE FUNCTION get_all_sentiment_data()
  RETURNS SETOF sentiment_data
  LANGUAGE sql
  SECURITY DEFINER
  AS $$
    SELECT * FROM sentiment_data ORDER BY id;
  $$;
"""

import os
import pandas as pd
from pathlib import Path

DATA_CSV = Path("data/data.csv")

# ── Columns the app expects ───────────────────────────────────────────────────
COLUMNS = ["ID", "Scheme", "Source", "Language", "Comment", "Sentiment"]


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _get_secret(key: str) -> str:
    """Get secret from Streamlit Cloud secrets or local .env."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")


def _is_deployed() -> bool:
    """True when running on Streamlit Cloud with Supabase configured."""
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_KEY"))


def _get_client():
    """Get Supabase client."""
    from supabase import create_client
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    return create_client(url, key)


def _df_from_records(records: list) -> pd.DataFrame:
    """
    Convert Supabase records to a clean DataFrame
    with columns renamed to what the app expects.
    """
    df = pd.DataFrame(records)
    df = df.rename(columns={
        "id":         "ID",
        "scheme":     "Scheme",
        "source":     "Source",
        "language":   "Language",
        "comment":    "Comment",
        "sentiment":  "Sentiment",
        "translated": "Translated",
    })
    keep = [c for c in COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
#  Uses RPC to bypass Supabase's 1000-row REST limit.
#  Falls back to paginated select if RPC is not yet created.
# ═════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Load all sentiment rows.
    Deployed  → reads from Supabase via RPC (no row limit)
    Local     → reads from data/data.csv
    """
    if _is_deployed():
        client = _get_client()

        # ── Method 1: RPC — bypasses Supabase 1000-row REST limit ────────────
        # Run this SQL once in Supabase SQL Editor to enable:
        #   CREATE OR REPLACE FUNCTION get_all_sentiment_data()
        #   RETURNS SETOF sentiment_data LANGUAGE sql SECURITY DEFINER AS $$
        #     SELECT * FROM sentiment_data ORDER BY id;
        #   $$;
        try:
            response = client.rpc("get_all_sentiment_data").execute()
            if response.data:
                print(f"[Storage] Loaded {len(response.data)} rows via RPC")
                return _df_from_records(response.data)
        except Exception as e:
            print(f"[Storage] RPC load failed: {e} — falling back to paginated select")

        # ── Method 2: Paginated select — fallback if RPC not created yet ─────
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
                if not response.data:
                    break
                all_data.extend(response.data)
                if len(response.data) < batch_size:
                    break
                offset += batch_size

            if all_data:
                print(f"[Storage] Loaded {len(all_data)} rows via paginated select")
                return _df_from_records(all_data)
            else:
                return pd.DataFrame(columns=COLUMNS)

        except Exception as e:
            print(f"[Storage] Supabase load failed: {e}")
            return pd.DataFrame(columns=COLUMNS)

    # ── Local fallback ────────────────────────────────────────────────────────
    if DATA_CSV.exists():
        try:
            return pd.read_csv(DATA_CSV, encoding="utf-8")
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)


# ═════════════════════════════════════════════════════════════════════════════
#  SAVE NEW ROWS
# ═════════════════════════════════════════════════════════════════════════════
def save_rows(rows: list) -> int:
    """
    Save new unique rows.
    Deployed  → Supabase (duplicates silently skipped via UNIQUE on comment)
    Local     → data/data.csv
    Returns number of rows actually saved.
    """
    if not rows:
        return 0
    if _is_deployed():
        return _save_to_supabase(rows)
    else:
        return _save_to_csv(rows)


def _save_to_supabase(rows: list) -> int:
    """Insert rows into Supabase in batches of 100."""
    try:
        client    = _get_client()
        formatted = []
        for r in rows:
            text = r.get("Comment", "").strip()
            if len(text) < 15:
                continue
            formatted.append({
                "scheme":     r.get("Scheme", ""),
                "source":     r.get("Source", ""),
                "language":   r.get("Language", "en"),
                "comment":    text,
                "sentiment":  r.get("Sentiment", "Neutral"),
                "translated": r.get("Translated", ""),
            })

        if not formatted:
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
                if response.data:
                    saved += len(response.data)
            except Exception as e:
                print(f"[Storage] Batch insert error: {e}")
                continue

        print(f"[Storage] Saved {saved} rows to Supabase")
        return saved

    except Exception as e:
        print(f"[Storage] Supabase save failed: {e}")
        return 0


def _save_to_csv(rows: list) -> int:
    """Save rows to local CSV with deduplication."""
    import csv

    DATA_CSV.parent.mkdir(exist_ok=True)

    existing_norm = set()
    next_id       = 1
    if DATA_CSV.exists():
        try:
            df = pd.read_csv(DATA_CSV, encoding="utf-8", usecols=["ID","Comment"])
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
        w = csv.DictWriter(f, fieldnames=["ID","Scheme","Source","Language","Comment","Sentiment"])
        if header:
            w.writeheader()
        for i, r in enumerate(deduped):
            r["ID"] = next_id + i
            w.writerow(r)

    return len(deduped)


# ═════════════════════════════════════════════════════════════════════════════
#  TRANSLATION CACHE
#  Stores translations in the `translated` column of sentiment_data.
#  Same text is never sent to the translation API twice.
# ═════════════════════════════════════════════════════════════════════════════

# ── In-memory cache (within session) ─────────────────────────────────────────
_translation_memory: dict = {}


def get_cached_translation(text: str):
    """
    Look up a translation.
    1. Check in-memory dict first  (fastest — no network)
    2. Check Supabase              (fast — single DB query)
    3. Return None if not found    (caller must call translation API)
    """
    text = text.strip()
    if not text:
        return None

    # Level 1: in-memory
    if text in _translation_memory:
        return _translation_memory[text]

    # Level 2: Supabase
    if not _is_deployed():
        return None

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
        return None
    except Exception as e:
        print(f"[Storage] Translation cache lookup failed: {e}")
        return None


def save_cached_translation(text: str, translated: str) -> None:
    """
    Save a translation result back to Supabase so it persists forever.
    Also saves to in-memory cache for the current session.
    """
    text       = text.strip()
    translated = translated.strip()

    if not text or not translated:
        return

    _translation_memory[text] = translated

    if not _is_deployed():
        return

    try:
        client = _get_client()
        client.table("sentiment_data") \
            .update({"translated": translated}) \
            .eq("comment", text) \
            .execute()
    except Exception as e:
        print(f"[Storage] Translation cache save failed: {e}")


def get_translation_cache_stats() -> dict:
    """Returns stats about the translation cache."""
    stats = {"in_memory_count": len(_translation_memory), "supabase_cached": 0}
    if _is_deployed():
        try:
            client = _get_client()
            res = (
                client.table("sentiment_data")
                .select("*", count="exact")
                .neq("translated", "")
                .execute()
            )
            stats["supabase_cached"] = res.count or 0
        except Exception:
            pass
    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  STATS HELPER
# ═════════════════════════════════════════════════════════════════════════════
def get_stats() -> dict:
    """Returns quick stats without loading full dataset."""
    if _is_deployed():
        try:
            client = _get_client()
            count  = client.table("sentiment_data").select(
                "*", count="exact"
            ).execute()
            return {"total_rows": count.count or 0}
        except Exception:
            pass
    df = load_data()
    return {"total_rows": len(df)}
