import json
from typing import Any, Dict, Optional

import httpx


SEND_OTP_URL = "https://api.aifiesta.ai/api/auth/v2/send-otp"
VERIFY_OTP_URL = "https://api.aifiesta.ai/api/auth/v2/verify-otp"


class FiestaAuthError(Exception):
    """Raised when a Fiesta auth OTP call fails."""


def send_otp(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Fiesta's send-otp endpoint.

    The exact payload shape depends on Fiesta's API (e.g. email / phone fields).
    This function just forwards whatever you pass in and returns the JSON body.
    """
    headers = {"Content-Type": "application/json"}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(SEND_OTP_URL, json=payload, headers=headers)

    if resp.status_code != 200 and resp.status_code != 201:
        try:
            detail = resp.json()
        except json.JSONDecodeError:
            detail = resp.text
        raise FiestaAuthError(
            f"send-otp failed with status {resp.status_code}: {detail}"
        )

    return resp.json()


def verify_otp(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Fiesta's verify-otp endpoint and return the JSON body.

    `payload` should include the OTP code plus any other required fields
    (e.g. email / phone, requestId, etc.).
    """
    headers = {"Content-Type": "application/json"}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(VERIFY_OTP_URL, json=payload, headers=headers)

    if resp.status_code != 200 and resp.status_code != 201:
        try:
            detail = resp.json()
        except json.JSONDecodeError:
            detail = resp.text
        raise FiestaAuthError(
            f"verify-otp failed with status {resp.status_code}: {detail}"
        )

    return resp.json()


def extract_fiesta_session(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract the FIESTA_SESSION-style token from a verify-otp response.

    Based on the sample response, the access token lives at:
        data["session"]["access_token"]

    If that path is missing, this falls back to a few common locations and
    returns None if nothing is found.
    """
    # Preferred, per your sample:
    session = data.get("session")
    if isinstance(session, dict):
        access_token = session.get("access_token")
        if isinstance(access_token, str):
            return access_token

    # Fallbacks: look for common key names at the top level.
    candidate_keys = [
        "access_token",
        "token",
    ]
    for key in candidate_keys:
        value = data.get(key)
        if isinstance(value, str):
            return value

    # Sometimes the token may be nested under "data" or similar.
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in candidate_keys:
            value = nested.get(key)
            if isinstance(value, str):
                return value

    return None


def run_interactive_otp_flow(
    send_payload: Dict[str, Any], verify_base_payload: Dict[str, Any]
) -> str:
    """
    Convenience function:

    1. Calls send-otp with `send_payload`.
    2. Prompts the user in the console: "Enter OTP: ".
    3. Calls verify-otp with `verify_base_payload` + {"otp": <user input>}.
    4. Tries to extract and return the FIESTA_SESSION token.

    Raises FiestaAuthError if the HTTP calls fail, or ValueError if a token
    cannot be found in the response.
    """
    # Step 1: send OTP
    send_otp(send_payload)

    # Step 2: read OTP from user
    otp = input("Enter OTP: ").strip()
    if not otp:
        raise ValueError("OTP cannot be empty.")

    # Step 3: verify OTP
    verify_payload = {**verify_base_payload, "otp": otp}
    data = verify_otp(verify_payload)

    # Step 4: extract token
    token = extract_fiesta_session(data)
    if not token:
        raise ValueError(
            "Could not find FIESTA_SESSION token in verify-otp response. "
            "Inspect the returned JSON to locate the correct field."
        )
    return token
