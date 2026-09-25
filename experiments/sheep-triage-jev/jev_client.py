"""TypeSafe Jev text decision client.

One POST per track to ``https://api.typesafe.ai/v1/systemone``. The question
is a Noul (calibrated yes/no). ``noul >= threshold`` maps to ``look``;
anything lower maps to ``dont``.

The key is read from ``TYPESAFE_API_KEY`` by the caller and sent only as a
Bearer token. It is never written into request bodies, artifacts, or errors.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
QUESTION_ID = "look"

LOOK_INSTRUCTIONS = (
    "Should a person look at this tracked sheep? "
    "Yes when the pose summary is worth a human glance: "
    "ear-angle spread, an ear missing on many frames, a very short track, "
    "or a centroid that moves a lot. "
    "No when both ears are measured on most frames, spread is low, "
    "and the head stays put."
)
LOOK_TRUE = (
    "The numbers show something to glance at: high ear-angle spread, "
    "an ear often not measured, few frames, or a lot of centroid motion."
)
LOOK_FALSE = (
    "The numbers look steady: both ears measured on most frames, "
    "low ear-angle spread, and little centroid motion. Skip this track."
)


class JevError(RuntimeError):
    """The decision API failed or returned a shape we cannot triage on."""


def decision_from_noul(noul: float, threshold: float) -> str:
    if noul >= threshold:
        return "look"
    return "dont"


def request_body(state: dict, *, model: str = DEFAULT_MODEL) -> dict:
    return {
        "model": model,
        "state": state,
        "questions": {
            QUESTION_ID: {
                "type": "noul",
                "instructions": LOOK_INSTRUCTIONS,
                "criteria": {"true": LOOK_TRUE, "false": LOOK_FALSE},
            }
        },
    }


def _scrub(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text


def triage_track(
    state: dict,
    *,
    api_key: str,
    threshold: float = 0.5,
    model: str = DEFAULT_MODEL,
    endpoint: str = ENDPOINT,
    timeout: float = 60.0,
    opener=urllib.request.urlopen,
) -> dict:
    """Ask Jev whether to look at one track. Returns a JSON-ready record."""
    if not api_key:
        raise JevError("TYPESAFE_API_KEY is empty")
    body = request_body(state, model=model)
    raw_body = json.dumps(body).encode()
    req = urllib.request.Request(
        endpoint,
        data=raw_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SamSeesSheep-sheep-triage-jev/0.1",
        },
        method="POST",
    )
    try:
        with opener(req, timeout=timeout) as resp:
            payload_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise JevError(
            f"Jev HTTP {exc.code}: {_scrub(detail, api_key)[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise JevError(f"Jev request failed: {_scrub(str(exc.reason), api_key)}") from exc

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise JevError("Jev response was not JSON") from exc

    answers = payload.get("answers") or {}
    answer = answers.get(QUESTION_ID)
    if not isinstance(answer, dict) or "noul" not in answer:
        raise JevError(
            f"Jev response missing answers.{QUESTION_ID}.noul "
            f"(keys: {sorted(answers)})"
        )
    try:
        noul = float(answer["noul"])
    except (TypeError, ValueError) as exc:
        raise JevError("Jev noul value was not a number") from exc
    if not math.isfinite(noul):
        raise JevError("Jev noul value was not a finite number")

    return {
        "decision": decision_from_noul(noul, threshold),
        "noul": noul,
        "threshold": threshold,
        "model": payload.get("model", model),
        "usage": payload.get("usage"),
        "answer": {"type": answer.get("type", "noul"), "noul": noul},
    }
