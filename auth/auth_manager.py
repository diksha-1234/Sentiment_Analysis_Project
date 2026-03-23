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
        password    TEXT,                  -- NULL for Google-only accounts
        google_id   TEXT UNIQUE,           -- NULL for password accounts
        role        TEXT DEFAULT 'user',
        joined      TEXT,
        avatar      TEXT DEFAULT '👤'
    );
    CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
    CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);
──────────────────────────────────────────────────────────────

STREAMLIT SECRETS (secrets.toml or Supabase dashboard):
    SUPABASE_URL      = "https://xxxx.supabase.co"
    SUPABASE_KEY      = "your-anon-key"
    GOOGLE_CLIENT_ID     = "xxxx.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET = "GOCSPX-xxxx"
    GOOGLE_REDIRECT_URI  = "https://your-app.streamlit.app/"   # must match Google console
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

# ── Path for local dev fallback ───────────────────────────────────────────────
USERS_JSON = Path("auth/users.json")

# ── In-memory flag so we only ensure-admin once per process ──────────────────
_admin_ensured: dict[str, bool] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, "")


def _is_deployed() -> bool:
    return bool(_get_secret("SUPABASE_URL") and _get_secret("SUPABASE_KEY"))


def _get_supabase():
    from supabase import create_client
    return create_client(_get_secret("SUPABASE_URL"), _get_secret("SUPABASE_KEY"))


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _check_password(plain: str, stored: str) -> bool:
    """Check a plaintext password against a stored hash. Never compares plain == stored."""
    if not stored:
        return False
    # SHA-256 (primary)
    if stored == _hash_password(plain):
        return True
    # bcrypt (legacy compatibility — only if bcrypt is installed)
    try:
        import bcrypt
        if stored.startswith(("$2b$", "$2a$")):
            return bcrypt.checkpw(plain.encode(), stored.encode())
    except ImportError:
        pass
    return False


def _username_from_email(email: str) -> str:
    """Derive a clean username from an email address."""
    base = email.split("@")[0].lower()
    # Keep only alphanumeric + underscore
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in base)
    return cleaned[:30] or "user"


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL USER STORE
# ─────────────────────────────────────────────────────────────────────────────
_ADMIN_DEFAULTS = {
    "name":      "Admin",
    "email":     "admin@pulse.ai",
    "password":  None,           # set below
    "role":      "admin",
    "joined":    "2025-01-01",
    "avatar":    "🔑",
    "google_id": None,
}
_ADMIN_DEFAULTS["password"] = _hash_password("1234")


def _load_local_users() -> dict:
    if USERS_JSON.exists():
        try:
            with open(USERS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_local_users(users: dict) -> None:
    USERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_JSON, "w") as f:
        json.dump(users, f, indent=2)


def _ensure_local_admin() -> None:
    if _admin_ensured.get("local"):
        return
    users = _load_local_users()
    changed = False
    if "admin" not in users:
        users["admin"] = dict(_ADMIN_DEFAULTS)
        changed = True
    else:
        # Repair if password was wiped somehow
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
        # Use upsert so it never duplicates
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
            on_conflict="username",   # update if already exists
        ).execute()
        _admin_ensured["supabase"] = True
    except Exception as e:
        print(f"[Auth] Could not ensure admin in Supabase: {e}")


def _get_supabase_user_by_username(username: str) -> dict | None:
    try:
        res = _get_supabase().table("users").select("*").eq("username", username).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"[Auth] Supabase username lookup failed: {e}")
        return None


def _get_supabase_user_by_email(email: str) -> dict | None:
    try:
        res = _get_supabase().table("users").select("*").eq("email", email).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"[Auth] Supabase email lookup failed: {e}")
        return None


def _get_supabase_user_by_google_id(google_id: str) -> dict | None:
    try:
        res = _get_supabase().table("users").select("*").eq("google_id", google_id).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"[Auth] Supabase google_id lookup failed: {e}")
        return None


def _upsert_supabase_user(user_data: dict, conflict_col: str = "username") -> bool:
    try:
        _get_supabase().table("users").upsert(user_data, on_conflict=conflict_col).execute()
        return True
    except Exception as e:
        print(f"[Auth] Supabase upsert failed: {e}")
        return False


def _insert_supabase_user(user_data: dict) -> bool:
    try:
        _get_supabase().table("users").insert(user_data).execute()
        return True
    except Exception as e:
        print(f"[Auth] Supabase insert failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED RESULT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _user_info(user: dict) -> dict:
    return {
        "name":   user.get("name", "User"),
        "email":  user.get("email", ""),
        "role":   user.get("role", "user"),
        "avatar": user.get("avatar", "👤"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — PASSWORD AUTH
# ─────────────────────────────────────────────────────────────────────────────
def login(username: str, password: str) -> tuple[bool, str, dict]:
    """
    Returns (success, message, user_info_dict).
    user_info_dict keys: name, email, role, avatar
    """
    username = username.strip().lower()

    if _is_deployed():
        _ensure_supabase_admin()
        user = _get_supabase_user_by_username(username)
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


def signup(username: str, password: str, name: str, email: str = "") -> tuple[bool, str]:
    """
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
        if _get_supabase_user_by_username(username):
            return False, "Username already taken."
        if email and _get_supabase_user_by_email(email):
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
        return (True, "Account created successfully! You can now sign in.") if ok \
               else (False, "Could not create account. Please try again.")

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
def get_google_auth_url(client_id: str = None, redirect_uri: str = None) -> str:
    """
    Build the Google OAuth redirect URL.
    Args are optional — falls back to GOOGLE_CLIENT_ID / GOOGLE_REDIRECT_URI secrets.
    Both call styles work:
        get_google_auth_url()
        get_google_auth_url(GOOGLE_CLIENT_ID, REDIRECT_URI)
    """
    import urllib.parse

    client_id    = client_id    or _get_secret("GOOGLE_CLIENT_ID")
    redirect_uri = redirect_uri or _get_secret("GOOGLE_REDIRECT_URI")

    if not client_id or not redirect_uri:
        raise ValueError(
            "GOOGLE_CLIENT_ID and GOOGLE_REDIRECT_URI must be set in secrets / env vars."
        )

    params = {
        "client_id":     client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"


def exchange_google_code(code: str) -> dict | None:
    """
    Exchange an OAuth code for Google user info.
    Returns raw Google profile dict or None on failure.
    """
    import requests as req

    client_id     = _get_secret("GOOGLE_CLIENT_ID")
    client_secret = _get_secret("GOOGLE_CLIENT_SECRET")
    redirect_uri  = _get_secret("GOOGLE_REDIRECT_URI")

    try:
        token_res = req.post(
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
            print(f"[Auth] Google token exchange failed: {token_data.get('error_description', token_data)}")
            return None

        info_res = req.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        return info_res.json()
    except Exception as e:
        print(f"[Auth] Google OAuth exchange failed: {e}")
        return None


def google_login_or_register(google_info: dict) -> tuple[bool, str, dict]:
    """
    The missing bridge function.

    Takes the dict returned by exchange_google_code() and:
      1. Finds an existing account by google_id (returning user).
      2. Links to an existing account by email if google_id not found.
      3. Auto-creates a new account if neither match.

    Returns (success, message, user_info_dict).
    """
    google_id = google_info.get("id") or google_info.get("sub")
    email     = (google_info.get("email") or "").strip().lower()
    name      = google_info.get("name") or google_info.get("given_name") or "User"
    avatar    = "🔵"   # distinguishes Google accounts visually
    now       = datetime.now().strftime("%Y-%m-%d")

    if not google_id:
        return False, "Google did not return a valid user ID.", {}

    # ── DEPLOYED (Supabase) path ───────────────────────────────────────────
    if _is_deployed():
        _ensure_supabase_admin()

        # 1. Known Google user — fast path
        user = _get_supabase_user_by_google_id(google_id)
        if user:
            return True, "OK", _user_info(user)

        # 2. Existing account with same email — link it
        if email:
            user = _get_supabase_user_by_email(email)
            if user:
                # Link google_id to the existing account
                _get_supabase().table("users") \
                    .update({"google_id": google_id}) \
                    .eq("username", user["username"]) \
                    .execute()
                return True, "OK", _user_info(user)

        # 3. Brand-new user — create account
        base_username = _username_from_email(email) if email else "user"
        username      = base_username
        suffix        = 1
        # Avoid username collision
        while _get_supabase_user_by_username(username):
            username = f"{base_username}{suffix}"
            suffix  += 1

        new_user = {
            "username":  username,
            "name":      name,
            "email":     email,
            "password":  None,          # Google-only account — no password
            "google_id": google_id,
            "role":      "user",
            "joined":    now,
            "avatar":    avatar,
        }
        ok = _insert_supabase_user(new_user)
        if ok:
            return True, "Account created via Google.", _user_info(new_user)
        return False, "Could not create account. Please try again.", {}

    # ── LOCAL path ────────────────────────────────────────────────────────
    else:
        _ensure_local_admin()
        users = _load_local_users()

        # 1. Known Google user
        for uname, u in users.items():
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
