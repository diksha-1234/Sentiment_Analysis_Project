"""
data/storage.py — Persistent Storage for Pulse Sentiment AI
════════════════════════════════════════════════════════════
Uses Supabase (PostgreSQL) when deployed on Streamlit Cloud.
Falls back to local data/data.csv when running locally.

This solves the data vanishing problem:
  Without this: data saved to Streamlit's disk → lost on restart
  With this:    data saved to Supabase database → persists forever
"""

import os
import pandas as pd
from pathlib import Path

DATA_CSV = Path("data/data.csv")

# ── Columns must match your CSV exactly ──────────────────────────────────────
COLUMNS = ["ID", "Scheme", "Source", "Language", "Comment", "Sentiment"]


def _get_secret(key: str) -> str:
    """Get secret from Streamlit Cloud or local .env."""
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


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Load all data.
    Deployed  → reads from Supabase database
    Local     → reads from data/data.csv
    """
    if _is_deployed():
        try:
            client   = _get_client()
            response = client.table("sentiment_data").select("*").execute()
            if response.data:
                df = pd.DataFrame(response.data)
                # Rename columns to match rest of app
                df = df.rename(columns={
                    "id":        "ID",
                    "scheme":    "Scheme",
                    "source":    "Source",
                    "language":  "Language",
                    "comment":   "Comment",
                    "sentiment": "Sentiment",
                })
                # Keep only columns the app needs
                keep = [c for c in COLUMNS if c in df.columns]
                return df[keep].reset_index(drop=True)
            else:
                return pd.DataFrame(columns=COLUMNS)
        except Exception as e:
            print(f"[Storage] Supabase load failed: {e}")
            # Fall through to local CSV

    # Local fallback
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
    Save new unique rows to storage.
    Deployed  → inserts into Supabase (duplicate comments auto-rejected
                because Comment column has UNIQUE constraint)
    Local     → appends to data/data.csv

    Returns number of rows actually saved.
    """
    if not rows:
        return 0

    if _is_deployed():
        return _save_to_supabase(rows)
    else:
        return _save_to_csv(rows)


def _save_to_supabase(rows: list) -> int:
    """Insert rows into Supabase. Duplicate comments silently ignored."""
    try:
        client = _get_client()

        # Format rows for Supabase — lowercase column names
        formatted = []
        for r in rows:
            text = r.get("Comment", "").strip()
            if len(text) < 15:
                continue
            formatted.append({
                "scheme":    r.get("Scheme", ""),
                "source":    r.get("Source", ""),
                "language":  r.get("Language", "en"),
                "comment":   text,
                "sentiment": r.get("Sentiment", "Neutral"),
            })

        if not formatted:
            return 0

        # Insert in batches of 100 to avoid timeout
        # on_conflict="comment" → duplicate comments silently skipped
        saved = 0
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
    """Save rows to local CSV — original logic preserved."""
    import csv

    DATA_CSV.parent.mkdir(exist_ok=True)

    # Load existing normalised comments for dedup
    existing_norm = set()
    next_id = 1
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
