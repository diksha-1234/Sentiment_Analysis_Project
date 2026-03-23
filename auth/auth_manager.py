"""
auth/auth_manager.py — Pulse Sentiment AI
==========================================
Handles login, signup, and Google OAuth.
Works with both local (users.json) and Supabase (deployed).

DEMO ACCOUNT: admin / 1234  — auto-created if not found
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

# ── Path for local dev fallback ───────────────────────────────────────────────
USERS_JSON = Path("auth/users.json")


# ── Secret loader ─────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")


def _is_deployed() -> bool:
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_KEY"))


def _get_supabase():
    from supabase import create_client
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    return create_client(url, key)


# ── Password hashing (simple sha256 — no bcrypt dependency needed) ────────────
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _check_password(plain: str, stored: str) -> bool:
    # Support both sha256 hashed and legacy plain text (for migration)
    if stored == plain:
        return True
    if stored == _hash_password(plain):
        return True
    # Try bcrypt if installed (for compatibility with existing data)
    try:
        import bcrypt
        if stored.startswith("$2b$") or stored.startswith("$2a$"):
            return bcrypt.checkpw(plain.encode(), stored.encode())
    except ImportError:
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL USER STORE (used when Supabase is not configured)
# ─────────────────────────────────────────────────────────────────────────────
def _load_local_users() -> dict:
    if USERS_JSON.exists():
        try:
            with open(USERS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Return default admin if file doesn't exist
    return {
        "admin": {
            "name":     "Admin",
            "email":    "admin@pulse.ai",
            "password": _hash_password("1234"),
            "role":     "admin",
            "joined":   "2025-01-01",
            "avatar":   "🔑",
        }
    }


def _save_local_users(users: dict) -> None:
    USERS_JSON.parent.mkdir(exist_ok=True)
    with open(USERS_JSON, "w") as f:
        json.dump(users, f, indent=2)


def _ensure_local_admin() -> None:
    """Make sure admin/1234 always exists in local store."""
    users = _load_local_users()
    if "admin" not in users:
        users["admin"] = {
            "name":     "Admin",
            "email":    "admin@pulse.ai",
            "password": _hash_password("1234"),
            "role":     "admin",
            "joined":   "2025-01-01",
            "avatar":   "🔑",
        }
        _save_local_users(users)


# ─────────────────────────────────────────────────────────────────────────────
#  SUPABASE USER STORE
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_supabase_admin() -> None:
    """Auto-creates admin/1234 in Supabase if it doesn't exist."""
    try:
        client = _get_supabase()
        res = client.table("users").select("username").eq("username", "admin").execute()
        if not res.data:
            client.table("users").insert({
                "username": "admin",
                "name":     "Admin",
                "email":    "admin@pulse.ai",
                "password": _hash_password("1234"),
                "role":     "admin",
                "joined":   "2025-01-01",
                "avatar":   "🔑",
            }).execute()
            print("[Auth] Admin user auto-created in Supabase")
    except Exception as e:
        print(f"[Auth] Could not ensure admin in Supabase: {e}")


def _get_supabase_user(username: str) -> dict | None:
    try:
        client = _get_supabase()
        res = client.table("users").select("*").eq("username", username).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"[Auth] Supabase user lookup failed: {e}")
    return None


def _save_supabase_user(user_data: dict) -> bool:
    try:
        client = _get_supabase()
        client.table("users").insert(user_data).execute()
        return True
    except Exception as e:
        print(f"[Auth] Supabase user save failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def login(username: str, password: str) -> tuple[bool, str, dict]:
    """
    Returns (success, message, user_info_dict)
    user_info_dict has: name, email, role, avatar
    """
    username = username.strip().lower()

    if _is_deployed():
        _ensure_supabase_admin()
        user = _get_supabase_user(username)
        if not user:
            return False, "User not found.", {}
        if not _check_password(password, user.get("password", "")):
            return False, "Incorrect password.", {}
        return True, "OK", {
            "name":   user.get("name", username),
            "email":  user.get("email", ""),
            "role":   user.get("role", "user"),
            "avatar": user.get("avatar", "👤"),
        }

    else:
        _ensure_local_admin()
        users = _load_local_users()
        if username not in users:
            return False, "User not found.", {}
        user = users[username]
        if not _check_password(password, user.get("password", "")):
            return False, "Incorrect password.", {}
        return True, "OK", {
            "name":   user.get("name", username),
            "email":  user.get("email", ""),
            "role":   user.get("role", "user"),
            "avatar": user.get("avatar", "👤"),
        }


def signup(username: str, password: str, name: str, email: str = "") -> tuple[bool, str]:
    """
    Returns (success, message)
    """
    username = username.strip().lower()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if not name.strip():
        return False, "Name is required."

    hashed = _hash_password(password)
    now    = datetime.now().strftime("%Y-%m-%d")

    if _is_deployed():
        existing = _get_supabase_user(username)
        if existing:
            return False, "Username already taken."
        ok = _save_supabase_user({
            "username": username,
            "name":     name.strip(),
            "email":    email.strip(),
            "password": hashed,
            "role":     "user",
            "joined":   now,
            "avatar":   "👤",
        })
        if ok:
            return True, "Account created successfully."
        return False, "Could not create account. Please try again."

    else:
        _ensure_local_admin()
        users = _load_local_users()
        if username in users:
            return False, "Username already taken."
        users[username] = {
            "name":     name.strip(),
            "email":    email.strip(),
            "password": hashed,
            "role":     "user",
            "joined":   now,
            "avatar":   "👤",
        }
        _save_local_users(users)
        return True, "Account created successfully."


# ─────────────────────────────────────────────────────────────────────────────
#  GOOGLE OAUTH
# ─────────────────────────────────────────────────────────────────────────────
def get_google_auth_url(client_id: str, redirect_uri: str) -> str:
    import urllib.parse
    params = {
        "client_id":     client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    base = "https://accounts.google.com/o/oauth2/v2/auth"
    return f"{base}?{urllib.parse.urlencode(params)}"


def exchange_google_code(code: str, client_id: str,
                         client_secret: str, redirect_uri: str) -> dict | None:
    try:
        import requests
        token_res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     client_id,
                "client_secret": client_secret,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
            timeout=10,
        )
        token_data = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            return None

        info_res = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        return info_res.json()
    except Exception as e:
        print(f"[Auth] Google OAuth exchange failed: {e}")
        return None
