"""Supabase sign-up, sign-in and saved inspections.

Every request carries the inspector's own token, so row-level security in the database
decides what they can read and write. The key below is Supabase's public key, which is
safe to publish. No admin key is used, and passwords are never stored or logged.
"""
import os
import time

import httpx

URL = os.environ.get("SUPABASE_URL", "https://jfypacirbfpbxiopolqn.supabase.co")
KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_UFxoczBu3H0P0wtiyiStTQ_Qiqg29dA")
TIMEOUT = 15
REFRESH_MARGIN = 60
UNREACHABLE = "Could not reach the account service. Check your connection and try again."


class AccountError(Exception):
    """A problem to show the inspector, in plain words."""


def headers(token: str | None = None) -> dict:
    values = {"apikey": KEY}
    if token:
        values["Authorization"] = f"Bearer {token}"
    return values


def problem(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = {}
    return body.get("msg") or body.get("error_description") or body.get("message") or "Something went wrong. Try again."


def auth_post(path: str, payload: dict, token: str | None = None, params: dict | None = None) -> dict:
    try:
        response = httpx.post(f"{URL}/auth/v1/{path}", json=payload, params=params, headers=headers(token), timeout=TIMEOUT)
    except httpx.HTTPError as error:
        raise AccountError(UNREACHABLE) from error
    if not response.is_success:
        raise AccountError(problem(response))
    return response.json() if response.content else {}


def to_session(data: dict, email: str = "") -> dict:
    user = data.get("user") or {}
    return {"access_token": data["access_token"], "refresh_token": data["refresh_token"],
            "expires_at": time.time() + data.get("expires_in", 3600), "email": user.get("email") or email}


def sign_up(email: str, password: str) -> dict | None:
    """Returns a session, or None when the project requires the inspector to confirm their email first."""
    data = auth_post("signup", {"email": email, "password": password})
    return to_session(data, email) if data.get("access_token") else None


def sign_in(email: str, password: str) -> dict:
    return to_session(auth_post("token", {"email": email, "password": password}, params={"grant_type": "password"}), email)


def sign_out(session: dict) -> None:
    try:
        auth_post("logout", {}, token=session["access_token"])
    except AccountError:
        pass  # The token expires by itself, and the browser forgets the session either way.


def fresh(session: dict | None) -> dict | None:
    """Renews a session close to expiry. Returns None if it can no longer be renewed."""
    if not session:
        return None
    if session.get("expires_at", 0) - time.time() > REFRESH_MARGIN:
        return session
    try:
        data = auth_post("token", {"refresh_token": session["refresh_token"]}, params={"grant_type": "refresh_token"})
    except AccountError:
        return None
    return to_session(data, session.get("email", ""))


def fetch_inspections(session: dict) -> list[dict]:
    """The signed-in inspector's saves, oldest first."""
    try:
        response = httpx.get(f"{URL}/rest/v1/inspections", headers=headers(session["access_token"]), timeout=TIMEOUT,
                             params={"select": "business,checklist,notes,created_at", "order": "created_at.asc"})
    except httpx.HTTPError as error:
        raise AccountError(UNREACHABLE) from error
    if not response.is_success:
        raise AccountError(f"Could not load your inspections: {problem(response)}")
    return [{"time": row["created_at"], "business": row["business"], "checklist": row["checklist"], "notes": row["notes"]}
            for row in response.json()]


def add_inspection(session: dict, business: str, checklist: dict, notes: str) -> None:
    try:
        response = httpx.post(f"{URL}/rest/v1/inspections", headers={**headers(session["access_token"]), "Prefer": "return=minimal"},
                              json={"business": business, "checklist": checklist, "notes": notes}, timeout=TIMEOUT)
    except httpx.HTTPError as error:
        raise AccountError(UNREACHABLE) from error
    if not response.is_success:
        raise AccountError(f"Could not save your inspection: {problem(response)}")
