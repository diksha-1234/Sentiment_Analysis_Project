"""
auth/auth_manager.py
────────────────────
Handles:
  • Local signup / login with bcrypt-hashed passwords
  • Google OAuth
  • Supabase persistence for deployment (users survive restarts)
  • Falls back to local users.json for local development

DEPLOYMENT:  users stored in Supabase `users` table  → survive forever
LOCAL DEV:   users stored in auth/users.json          → local only

Supabase table required (run once in Supabase SQL editor):

    CREATE TABLE users (
        username   TEXT PRIMARY KEY,
        name       TEXT,
        email      TEXT,
        password   TEXT,
        role       TEXT DEFAULT 'user',
        joined     TEXT,
        avatar     TEXT DEFAULT '👤'
    );

    INSERT INTO users (username, name, email, password, role, joined, avatar)
    VALUES ('admin', 'Admin User', 'admin@pulse.ai', '', 'admin', '2024-01-01', '🛡️')
    ON CONFLICT (username) DO NOTHING;
"""

import json
import os
import hashlib
from pathlib import Path
from datetime import datetime

# ── Optional bcrypt ──────────────────────────────────────────────────────────
try:
    import bcrypt
    BCRYPT_OK = True
except ImportError:
    BCRYPT_OK = False

USERS_FILE = Path(__file__).parent / "users.json"


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
    """True when Supabase credentials are present (Streamlit Cloud)."""
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_KEY"))


def _get_supabase():
    from supabase import create_client
    return create_client(_get_secret("SUPABASE_URL"), _get_secret("SUPABASE_KEY"))


def _hash_pw(pw: str) -> str:
    if BCRYPT_OK:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(pw.encode()).hexdigest()


def _verify_pw(pw: str, hashed: str) -> bool:
    if BCRYPT_OK:
        try:
            return bcrypt.checkpw(pw.encode(), hashed.encode())
        except Exception:
            pass
    return hashlib.sha256(pw.encode()).hexdigest() == hashed


# ═════════════════════════════════════════════════════════════════════════════
#  SUPABASE USER STORE
# ═════════════════════════════════════════════════════════════════════════════
def _supabase_get_user(username: str) -> dict | None:
    """Fetch one user row from Supabase. Returns dict or None."""
    try:
        client = _get_supabase()
        res = client.table("users").select("*").eq("username", username).execute()
        if res.data:
            return res.data[0]
        return None
    except Exception as e:
        print(f"[Auth] Supabase get_user error: {e}")
        return None


def _supabase_create_user(user: dict) -> bool:
    """Insert a new user row into Supabase. Returns True on success."""
    try:
        client = _get_supabase()
        client.table("users").insert(user).execute()
        return True
    except Exception as e:
        print(f"[Auth] Supabase create_user error: {e}")
        return False


def _supabase_ensure_admin():
    """
    Make sure the admin account exists in Supabase.
    Called once on first login attempt so admin is always available.
    """
    try:
        existing = _supabase_get_user("admin")
        if existing is None:
            client = _get_supabase()
            client.table("users").insert({
                "username": "admin",
                "name":     "Admin User",
                "email":    "admin@pulse.ai",
                "password": _hash_pw("1234"),
                "role":     "admin",
                "joined":   "2024-01-01",
                "avatar":   "🛡️",
            }).execute()
            print("[Auth] Admin account created in Supabase.")
        elif not existing.get("password"):
            # Password column was empty (from SQL seed) — set it now
            client = _get_supabase()
            client.table("users").update(
                {"password": _hash_pw("1234")}
            ).eq("username", "admin").execute()
            print("[Auth] Admin password set in Supabase.")
    except Exception as e:
        print(f"[Auth] ensure_admin error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  LOCAL JSON USER STORE  (used when not deployed)
# ═════════════════════════════════════════════════════════════════════════════
def _default_users() -> dict:
    return {
        "admin": {
            "name":     "Admin User",
            "email":    "admin@pulse.ai",
            "password": _hash_pw("1234"),
            "role":     "admin",
            "joined":   "2024-01-01",
            "avatar":   "🛡️",
        }
    }


def _load_users_local() -> dict:
    if not USERS_FILE.exists():
        defaults = _default_users()
        _save_users_local(defaults)
        return defaults
    try:
        with open(USERS_FILE, "r") as f:
            data = json.load(f)
        if "admin" not in data:
            data["admin"] = _default_users()["admin"]
            _save_users_local(data)
        return data
    except Exception:
        return _default_users()


def _save_users_local(users: dict):
    USERS_FILE.parent.mkdir(exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════
def login(username: str, password: str) -> tuple[bool, str, dict]:
    """
    Returns (success, message, user_info_dict).
    Routes to Supabase (deployed) or local JSON (dev).
    """
    un = username.strip().lower()

    if _is_deployed():
        # ── Supabase path ─────────────────────────────────────────────────────
        _supabase_ensure_admin()   # make sure admin always exists
        user = _supabase_get_user(un)
        if user is None:
            return False, "User not found.", {}
        if not _verify_pw(password, user.get("password", "")):
            return False, "Incorrect password.", {}
        return True, "Login successful!", {
            "name":   user.get("name", un),
            "email":  user.get("email", ""),
            "role":   user.get("role", "user"),
            "avatar": user.get("avatar", "👤"),
        }

    else:
        # ── Local JSON path ───────────────────────────────────────────────────
        users = _load_users_local()
        if un not in users:
            return False, "User not found.", {}
        user = users[un]
        if not _verify_pw(password, user["password"]):
            return False, "Incorrect password.", {}
        return True, "Login successful!", user


def signup(username: str, password: str, name: str, email: str) -> tuple[bool, str]:
    """
    Returns (success, message).
    Routes to Supabase (deployed) or local JSON (dev).
    """
    un = username.strip().lower()

    if len(un) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    if _is_deployed():
        # ── Supabase path ─────────────────────────────────────────────────────
        existing = _supabase_get_user(un)
        if existing is not None:
            return False, "Username already exists. Please choose another."
        ok = _supabase_create_user({
            "username": un,
            "name":     name.strip(),
            "email":    email.strip(),
            "password": _hash_pw(password),
            "role":     "user",
            "joined":   datetime.now().strftime("%Y-%m-%d"),
            "avatar":   "👤",
        })
        if ok:
            return True, "Account created successfully! You can now log in."
        else:
            return False, "Could not create account. Please try again."

    else:
        # ── Local JSON path ───────────────────────────────────────────────────
        users = _load_users_local()
        if un in users:
            return False, "Username already exists. Please choose another."
        users[un] = {
            "name":     name.strip(),
            "email":    email.strip(),
            "password": _hash_pw(password),
            "role":     "user",
            "joined":   datetime.now().strftime("%Y-%m-%d"),
            "avatar":   "👤",
        }
        _save_users_local(users)
        return True, "Account created successfully! You can now log in."


# ═════════════════════════════════════════════════════════════════════════════
#  GOOGLE OAUTH  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
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
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_google_code(code: str, client_id: str, client_secret: str,
                         redirect_uri: str) -> dict:
    import urllib.request
    import urllib.parse

    data = urllib.parse.urlencode({
        "code":          code,
        "client_id":     client_id,
        "client_secret": client_secret,
        "redirect_uri":  redirect_uri,
        "grant_type":    "authorization_code",
    }).encode()

    try:
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
        with urllib.request.urlopen(req) as resp:
            tokens = json.loads(resp.read())

        id_token    = tokens.get("id_token", "")
        import base64
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload     = json.loads(base64.urlsafe_b64decode(payload_b64))

        # Auto-register Google users in Supabase so they persist
        google_un = payload.get("email", "").split("@")[0].lower()
        if google_un and _is_deployed():
            existing = _supabase_get_user(google_un)
            if existing is None:
                _supabase_create_user({
                    "username": google_un,
                    "name":     payload.get("name", "Google User"),
                    "email":    payload.get("email", ""),
                    "password": "",   # Google users have no password
                    "role":     "user",
                    "joined":   datetime.now().strftime("%Y-%m-%d"),
                    "avatar":   "🌐",
                })

        return {
            "name":   payload.get("name", "Google User"),
            "email":  payload.get("email", ""),
            "avatar": "🌐",
            "sub":    payload.get("sub", ""),
        }
    except Exception:
        return {}
