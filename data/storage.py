"""
data/storage.py — Persistent Storage for Pulse Sentiment AI
════════════════════════════════════════════════════════════
Uses Supabase (PostgreSQL) when deployed on Streamlit Cloud.
Falls back to local data/data.csv when running locally.

FIXES:
  ✅ force_refresh() — hard-wipes all in-memory caches
  ✅ _translation_memory cleared on force_refresh
  ✅ load_data() always hits Supabase fresh (no @st.cache_data here)
  ✅ get_stats() always hits Supabase fresh

Supabase table setup (run once in SQL Editor):
  CREATE TABLE sentiment_data (
      id         BIGSERIAL PRIMARY KEY,
      scheme     TEXT,
      source     TEXT,
      language   TEXT DEFAULT 'en',
      comment    TEXT UNIQUE,
      sentiment  TEXT DEFAULT 'Neutral',
      translated TEXT DEFAULT ''
  );

  ALTER TABLE sentiment_data
  ADD COLUMN IF NOT EXISTS translated TEXT DEFAULT '';

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

COLUMNS = ["ID", "Scheme", "Source", "Language", "Comment", "Sentiment"]

# ── In-memory translation cache (within session) ──────────────────────────────
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
#  FORCE REFRESH — wipes ALL in-memory state so next load_data() call
#  hits Supabase completely fresh.  Call this from the app after any
#  manual Supabase delete / truncate.
# ═════════════════════════════════════════════════════════════════════════════
def force_refresh() -> None:
    """
    Hard-wipe every in-memory cache in this module.
    After calling this, the next load_data() / get_stats() will
    query Supabase fresh regardless of what was cached before.
    """
    global _translation_memory
    _translation_memory = {}

    # Also clear Streamlit's own function-level caches so
    # _cached_preprocess and _cached_train in app.py are invalidated.
    try:
        import streamlit as st
        st.cache_data.clear()
        st.cache_resource.clear()
        print("[Storage] force_refresh: all caches cleared")
    except Exception as e:
        print(f"[Storage] force_refresh: could not clear st caches: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD DATA  — always fresh, no @st.cache_data here on purpose
# ═════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Load all sentiment rows directly from Supabase (no caching).
    Caching is handled upstream in app.py by _cached_preprocess.
    """
    if _is_deployed():
        client = _get_client()

        # Method 1: RPC (bypasses 1000-row REST limit)
        try:
            response = client.rpc("get_all_sentiment_data").execute()
            if response.data:
                print(f"[Storage] Loaded {len(response.data)} rows via RPC")
                return _df_from_records(response.data)
        except Exception as e:
            print(f"[Storage] RPC load failed: {e} — trying paginated select")

        # Method 2: Paginated select fallback
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

    # Local fallback
    if DATA_CSV.exists():
        try:
            return pd.read_csv(DATA_CSV, encoding="utf-8")
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)


# ═════════════════════════════════════════════════════════════════════════════
#  STATS — always live from Supabase, never cached
# ═════════════════════════════════════════════════════════════════════════════
def get_stats() -> dict:
    """Always queries Supabase live — never returns a cached count."""
    if _is_deployed():
        try:
            client = _get_client()
            count  = client.table("sentiment_data").select(
                "*", count="exact"
            ).limit(1).execute()          # limit(1) so no rows transferred, just count
            return {"total_rows": count.count or 0}
        except Exception as e:
            print(f"[Storage] get_stats failed: {e}")
    df = load_data()
    return {"total_rows": len(df)}


# ═════════════════════════════════════════════════════════════════════════════
#  SAVE NEW ROWS
# ═════════════════════════════════════════════════════════════════════════════
def save_rows(rows: list) -> int:
    if not rows:
        return 0
    if _is_deployed():
        return _save_to_supabase(rows)
    else:
        return _save_to_csv(rows)


def _save_to_supabase(rows: list) -> int:
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
# ═════════════════════════════════════════════════════════════════════════════
def get_cached_translation(text: str):
    text = text.strip()
    if not text:
        return None

    if text in _translation_memory:
        return _translation_memory[text]

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
