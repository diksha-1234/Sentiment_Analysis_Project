"""
auth/auth_manager.py
────────────────────
Handles:
  • Local signup / login with bcrypt-hashed passwords
  • Google OAuth  (uses google-auth library; add your Client ID in .env)
  • Session state helpers

USER STORE: auth/users.json  (simple JSON file – swap for DB in production)
"""

import json
import os
import hashlib
import hmac
from pathlib import Path
from datetime import datetime

# ── Optional bcrypt ──────────────────────────────────────────────────────────
try:
    import bcrypt
    BCRYPT_OK = True
except ImportError:
    BCRYPT_OK = False

USERS_FILE = Path(__file__).parent / "users.json"

# ── Demo admin account always present ────────────────────────────────────────
_DEFAULT_USERS = {
    "admin": {
        "name":     "Admin User",
        "email":    "admin@pulse.ai",
        "password": "",   # filled below after _hash_pw is defined
        "role":     "admin",
        "joined":   "2024-01-01",
        "avatar":   "🛡️",
    }
}


def _hash_pw(pw: str) -> str:
    if BCRYPT_OK:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    # Fallback: SHA-256 (not production-safe but works without bcrypt)
    return hashlib.sha256(pw.encode()).hexdigest()


def _verify_pw(pw: str, hashed: str) -> bool:
    if BCRYPT_OK:
        try:
            return bcrypt.checkpw(pw.encode(), hashed.encode())
        except Exception:
            pass
    return hashlib.sha256(pw.encode()).hexdigest() == hashed


# ── Rebuild default with correct hash ────────────────────────────────────────
_DEFAULT_USERS["admin"]["password"] = _hash_pw("1234")


def _load_users() -> dict:
    if not USERS_FILE.exists():
        _save_users(_DEFAULT_USERS)
        return _DEFAULT_USERS.copy()
    try:
        with open(USERS_FILE, "r") as f:
            data = json.load(f)
        # Ensure admin always exists
        if "admin" not in data:
            data["admin"] = _DEFAULT_USERS["admin"]
            _save_users(data)
        return data
    except Exception:
        return _DEFAULT_USERS.copy()


def _save_users(users: dict):
    USERS_FILE.parent.mkdir(exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def login(username: str, password: str) -> tuple[bool, str, dict]:
    """
    Returns (success, message, user_info_dict)
    """
    users = _load_users()
    un = username.strip().lower()
    if un not in users:
        return False, "User not found.", {}
    user = users[un]
    if not _verify_pw(password, user["password"]):
        return False, "Incorrect password.", {}
    return True, "Login successful!", user


def signup(username: str, password: str, name: str, email: str) -> tuple[bool, str]:
    """
    Returns (success, message)
    """
    users = _load_users()
    un = username.strip().lower()
    if len(un) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
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
    _save_users(users)
    return True, "Account created successfully! You can now log in."


def get_google_auth_url(client_id: str, redirect_uri: str) -> str:
    """
    Build Google OAuth2 consent screen URL.
    Requires: GOOGLE_CLIENT_ID in environment.
    """
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
    return base + "?" + urllib.parse.urlencode(params)


def exchange_google_code(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """
    Exchange auth code for user info.
    Returns user info dict or {} on failure.
    """
    import urllib.request
    import urllib.parse

    token_url = "https://oauth2.googleapis.com/token"
    data = urllib.parse.urlencode({
        "code":          code,
        "client_id":     client_id,
        "client_secret": client_secret,
        "redirect_uri":  redirect_uri,
        "grant_type":    "authorization_code",
    }).encode()

    try:
        req = urllib.request.Request(token_url, data=data)
        with urllib.request.urlopen(req) as resp:
            tokens = json.loads(resp.read())

        id_token = tokens.get("id_token", "")
        # Decode JWT payload (no verification for simplicity – use google-auth in production)
        import base64
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        return {
            "name":   payload.get("name", "Google User"),
            "email":  payload.get("email", ""),
            "avatar": payload.get("picture", ""),
            "sub":    payload.get("sub", ""),
        }
    except Exception:
        return {}