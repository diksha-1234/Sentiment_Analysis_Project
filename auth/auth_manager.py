"""
auth/auth_manager.py — Pulse Sentiment AI
==========================================
Handles login, signup, and Google OAuth.
Works with both local (users.json) and Supabase (deployed).

DEMO ACCOUNT: admin / 1234  — auto-created if not found, no registration needed.

SUPABASE SETUP — run this SQL once in your Supabase SQL editor:
──────────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS users (
        id          BIGSERIAL PRIMARY KEY,
        username    TEXT UNIQUE NOT NULL,
        name        TEXT NOT NULL,
        email       TEXT,
        password    TEXT,
        google_id   TEXT UNIQUE,
        role        TEXT DEFAULT 'user',
        joined      TEXT,
        avatar      TEXT DEFAULT '👤'
    );
    CREATE INDEX IF NOT EXISTS idx_users_email     ON users(email);
    CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);
──────────────────────────────────────────────────────────────

STREAMLIT SECRETS (secrets.toml or Streamlit Cloud dashboard):
    SUPABASE_URL         = "https://xxxx.supabase.co"
    SUPABASE_KEY         = "your-anon-key"
    GOOGLE_CLIENT_ID     = "xxxx.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET = "GOCSPX-xxxx"
    REDIRECT_URI         = "https://your-app.streamlit.app/"
"""

from __future__ import annotations  # makes all type hints strings → no runtime issues

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# ── Path for local dev fallback ───────────────────────────────────────────────
USERS_JSON = Path("auth/users.json")

# ── In-memory flag so we only ensure-admin once per process ──────────────────
_admin_ensured: Dict[str, bool] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    """Read from st.secrets first, then environment variables."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, "")


def _is_deployed() -> bool:
    """True when both SUPABASE_URL and SUPABASE_KEY are present."""
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_KEY"))


def _get_supabase():
    """Return a Supabase client. Raises ImportError if supabase-py not installed."""
    try:
        from supabase import create_client  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "supabase-py is not installed. Add `supabase` to requirements.txt."
        ) from exc
    return create_client(_get_secret("SUPABASE_URL"), _get_secret("SUPABASE_KEY"))


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _check_password(plain: str, stored: str) -> bool:
    """Verify a plaintext password against a stored hash (SHA-256 or bcrypt)."""
    if not stored:
        return False
    if stored == _hash_password(plain):
        return True
    # bcrypt legacy compatibility
    try:
        import bcrypt  # type: ignore
        if stored.startswith(("$2b$", "$2a$")):
            return bcrypt.checkpw(plain.encode(), stored.encode())
    except ImportError:
        pass
    return False


def _username_from_email(email: str) -> str:
    """Derive a clean username from an email address."""
    base = email.split("@")[0].lower()
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in base)
    return cleaned[:30] or "user"


def _user_info(user: Dict) -> Dict:
    """Normalise a raw user record into a safe public dict."""
    return {
        "name":   user.get("name", "User"),
        "email":  user.get("email", ""),
        "role":   user.get("role", "user"),
        "avatar": user.get("avatar", "👤"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
_ADMIN_RECORD: Dict = {
    "name":      "Admin",
    "email":     "admin@pulse.ai",
    "password":  _hash_password("1234"),
    "role":      "admin",
    "joined":    "2025-01-01",
    "avatar":    "🔑",
    "google_id": None,
}


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL USER STORE
# ─────────────────────────────────────────────────────────────────────────────
def _load_local_users() -> Dict:
    if USERS_JSON.exists():
        try:
            with open(USERS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_local_users(users: Dict) -> None:
    USERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_JSON, "w") as f:
        json.dump(users, f, indent=2)


def _ensure_local_admin() -> None:
    if _admin_ensured.get("local"):
        return
    users   = _load_local_users()
    changed = False
    if "admin" not in users:
        users["admin"] = dict(_ADMIN_RECORD)
        changed = True
    else:
        if not users["admin"].get("password"):
            users["admin"]["password"] = _hash_password("1234")
            changed = True
    if changed:
        _save_local_users(users)
    _admin_ensured["local"] = True


# ─────────────────────────────────────────────────────────────────────────────
#  SUPABASE USER STORE
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_supabase_admin() -> None:
    """Upsert admin/1234 in Supabase — safe to call repeatedly, runs once per process."""
    if _admin_ensured.get("supabase"):
        return
    try:
        client = _get_supabase()
        client.table("users").upsert(
            {
                "username": "admin",
                "name":     "Admin",
                "email":    "admin@pulse.ai",
                "password": _hash_password("1234"),
                "role":     "admin",
                "joined":   "2025-01-01",
                "avatar":   "🔑",
            },
            on_conflict="username",
        ).execute()
        _admin_ensured["supabase"] = True
    except Exception as e:
        # Non-fatal: log and continue — admin may already exist
        print(f"[Auth] Could not ensure admin in Supabase: {e}")
        _admin_ensured["supabase"] = True  # Don't retry on every call


def _get_supabase_user(column: str, value: str) -> Optional[Dict]:
    """Generic single-row lookup by any unique column."""
    try:
        res = _get_supabase().table("users").select("*").eq(column, value).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"[Auth] Supabase lookup ({column}={value}) failed: {e}")
        return None


def _upsert_supabase_user(user_data: Dict, conflict_col: str = "username") -> bool:
    try:
        _get_supabase().table("users").upsert(
            user_data, on_conflict=conflict_col
        ).execute()
        return True
    except Exception as e:
        print(f"[Auth] Supabase upsert failed: {e}")
        return False


def _insert_supabase_user(user_data: Dict) -> bool:
    try:
        _get_supabase().table("users").insert(user_data).execute()
        return True
    except Exception as e:
        print(f"[Auth] Supabase insert failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — PASSWORD AUTH
# ─────────────────────────────────────────────────────────────────────────────
def login(username: str, password: str) -> Tuple[bool, str, Dict]:
    """
    Authenticate with username + password.
    Returns (success, message, user_info_dict).
    user_info_dict keys: name, email, role, avatar
    """
    username = username.strip().lower()

    if _is_deployed():
        _ensure_supabase_admin()
        user = _get_supabase_user("username", username)
        if not user:
            return False, "User not found.", {}
        if not user.get("password"):
            return False, "This account uses Google sign-in. Please continue with Google.", {}
        if not _check_password(password, user["password"]):
            return False, "Incorrect password.", {}
        return True, "OK", _user_info(user)

    else:
        _ensure_local_admin()
        users = _load_local_users()
        if username not in users:
            return False, "User not found.", {}
        user = users[username]
        if not user.get("password"):
            return False, "This account uses Google sign-in. Please continue with Google.", {}
        if not _check_password(password, user["password"]):
            return False, "Incorrect password.", {}
        return True, "OK", _user_info(user)


def signup(
    username: str,
    password: str,
    name: str,
    email: str = "",
) -> Tuple[bool, str]:
    """
    Register a new user.
    Returns (success, message).
    """
    username = username.strip().lower()
    email    = email.strip().lower()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not username.replace("_", "").isalnum():
        return False, "Username can only contain letters, numbers, and underscores."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if not name.strip():
        return False, "Name is required."

    hashed = _hash_password(password)
    now    = datetime.now().strftime("%Y-%m-%d")

    if _is_deployed():
        _ensure_supabase_admin()
        if _get_supabase_user("username", username):
            return False, "Username already taken."
        if email and _get_supabase_user("email", email):
            return False, "An account with this email already exists."
        ok = _insert_supabase_user({
            "username": username,
            "name":     name.strip(),
            "email":    email,
            "password": hashed,
            "role":     "user",
            "joined":   now,
            "avatar":   "👤",
        })
        if ok:
            return True, "Account created successfully! You can now sign in."
        return False, "Could not create account. Please try again."

    else:
        _ensure_local_admin()
        users = _load_local_users()
        if username in users:
            return False, "Username already taken."
        if email and any(u.get("email") == email for u in users.values()):
            return False, "An account with this email already exists."
        users[username] = {
            "name":      name.strip(),
            "email":     email,
            "password":  hashed,
            "role":      "user",
            "joined":    now,
            "avatar":    "👤",
            "google_id": None,
        }
        _save_local_users(users)
        return True, "Account created successfully! You can now sign in."


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — GOOGLE OAUTH
# ─────────────────────────────────────────────────────────────────────────────
def get_google_auth_url(
    client_id: Optional[str] = None,
    redirect_uri: Optional[str] = None,
) -> str:
    """
    Build the Google OAuth redirect URL.

    app.py calls this as:
        get_google_auth_url(GOOGLE_CLIENT_ID, REDIRECT_URI)

    Falls back to secrets if args are omitted / empty.
    Secret key for redirect URI is REDIRECT_URI (matches app.py).
    """
    import urllib.parse

    client_id    = client_id    or _get_secret("GOOGLE_CLIENT_ID")
    redirect_uri = redirect_uri or _get_secret("REDIRECT_URI")

    if not client_id:
        raise ValueError("GOOGLE_CLIENT_ID must be set in secrets / env vars.")
    if not redirect_uri:
        raise ValueError("REDIRECT_URI must be set in secrets / env vars.")

    params = {
        "client_id":     client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"


def exchange_google_code(code: str) -> Optional[Dict]:
    """
    Exchange an OAuth authorisation code for Google user info.
    Returns the raw Google profile dict, or None on failure.
    """
    try:
        import requests as _requests  # type: ignore
    except ImportError:
        print("[Auth] 'requests' library not installed — cannot exchange Google code.")
        return None

    client_id     = _get_secret("GOOGLE_CLIENT_ID")
    client_secret = _get_secret("GOOGLE_CLIENT_SECRET")
    redirect_uri  = _get_secret("REDIRECT_URI")

    try:
        token_res = _requests.post(
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
        token_data   = token_res.json()
        access_token = token_data.get("access_token")
        if not access_token:
            print(
                f"[Auth] Google token exchange failed: "
                f"{token_data.get('error_description', token_data)}"
            )
            return None

        info_res = _requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        return info_res.json()

    except Exception as e:
        print(f"[Auth] Google OAuth exchange failed: {e}")
        return None


def google_login_or_register(google_info: Dict) -> Tuple[bool, str, Dict]:
    """
    Takes the dict returned by exchange_google_code() and:
      1. Finds an existing account by google_id  (returning user).
      2. Links to an existing account by email   (email match).
      3. Auto-creates a new account              (new user).

    Returns (success, message, user_info_dict).
    """
    google_id = google_info.get("id") or google_info.get("sub")
    email     = (google_info.get("email") or "").strip().lower()
    name      = google_info.get("name") or google_info.get("given_name") or "User"
    avatar    = "🔵"
    now       = datetime.now().strftime("%Y-%m-%d")

    if not google_id:
        return False, "Google did not return a valid user ID.", {}

    # ── DEPLOYED (Supabase) ────────────────────────────────────────────────
    if _is_deployed():
        _ensure_supabase_admin()

        # 1. Known Google user — fast path
        user = _get_supabase_user("google_id", google_id)
        if user:
            return True, "OK", _user_info(user)

        # 2. Existing account with same email — link it
        if email:
            user = _get_supabase_user("email", email)
            if user:
                try:
                    _get_supabase().table("users") \
                        .update({"google_id": google_id}) \
                        .eq("username", user["username"]) \
                        .execute()
                except Exception as e:
                    print(f"[Auth] Failed to link google_id: {e}")
                return True, "OK", _user_info(user)

        # 3. Brand-new user
        base_username = _username_from_email(email) if email else "user"
        username      = base_username
        suffix        = 1
        while _get_supabase_user("username", username):
            username = f"{base_username}{suffix}"
            suffix  += 1

        new_user = {
            "username":  username,
            "name":      name,
            "email":     email,
            "password":  None,
            "google_id": google_id,
            "role":      "user",
            "joined":    now,
            "avatar":    avatar,
        }
        ok = _insert_supabase_user(new_user)
        if ok:
            return True, "Account created via Google.", _user_info(new_user)
        return False, "Could not create account. Please try again.", {}

    # ── LOCAL ──────────────────────────────────────────────────────────────
    else:
        _ensure_local_admin()
        users = _load_local_users()

        # 1. Known Google user
        for _uname, u in users.items():
            if u.get("google_id") == google_id:
                return True, "OK", _user_info(u)

        # 2. Link by email
        if email:
            for uname, u in users.items():
                if u.get("email") == email:
                    users[uname]["google_id"] = google_id
                    _save_local_users(users)
                    return True, "OK", _user_info(u)

        # 3. New user
        base_username = _username_from_email(email) if email else "user"
        username      = base_username
        suffix        = 1
        while username in users:
            username = f"{base_username}{suffix}"
            suffix  += 1

        users[username] = {
            "name":      name,
            "email":     email,
            "password":  None,
            "google_id": google_id,
            "role":      "user",
            "joined":    now,
            "avatar":    avatar,
        }
        _save_local_users(users)
        return True, "Account created via Google.", _user_info(users[username])
