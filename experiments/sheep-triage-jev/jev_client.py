"""TypeSafe Jev client for the SPFES ear check.

One POST per track to ``https://api.typesafe.ai/v1/systemone``. The state is
the track's ``EarPack``; three questions are evaluated in parallel:

- ``triage``: a Choice of look | skip | cannot under the SPFES ear protocol.
  This typed class is the decision.
- ``look``: a Noul for the same judgment, used only as a confidence gate on
  a typed look.
- ``reason``: a Choice over a fixed reason vocabulary. Jev does not write
  text; code turns the reason into the farmer's one-line sentence.

A typed look stays look when ``noul >= threshold`` and becomes cannot
(reason ``not_sure``) when it does not. A typed skip is always skip and a
typed cannot is always cannot; the noul never turns either into a look. The
reason is otherwise Jev's most probable reason among those that fit the
decision.

After the tracks, one pen call turns the day's counts and reasons into
walk_tomorrow | later | fine for the next morning.

The key is read from ``TYPESAFE_API_KEY`` by the caller and sent only as a
Bearer token. It is never written into request bodies, artifacts, or errors.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast, get_args

from spfes_ear import (
    DECISIONS,
    GATE_REASON,
    JEV_REASONS,
    REASONS_FOR,
    Decision,
    Reason,
    Verdict,
)

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
QUESTION_SET = "spfes-ear-v2"  # v2: overnight pen question (walk_tomorrow); track questions as in v1
DECISION_RULE = (
    "Jev's typed class decides. A typed look needs noul >= threshold, otherwise it becomes cannot "
    "(not_sure). A typed skip stays skip and a typed cannot stays cannot, whatever the noul."
)

TRIAGE_ID = "triage"
LOOK_ID = "look"
REASON_ID = "reason"
PEN_ID = "pen"

PenCall = Literal["walk_tomorrow", "later", "fine"]
PEN_CALLS: tuple[PenCall, ...] = get_args(PenCall)

SPFES_EAR_PROTOCOL = (
    "Ear items of McLennan's SPFES sheep facial scale, read from head-pose numbers. "
    "Signs worth a person's look: unusual ear carriage (an ear held far forward, far back "
    "toward the neck, or so flat that only one ear can be measured while the face is toward "
    "the camera), clear asymmetry between the left and right ears, or frequent ear posture "
    "change across frames. Cannot check: the face was mostly not toward the camera, or the "
    "ears could not be measured on most facing frames. Walking, speed and a short time in "
    "view are not signs."
)


def _instructions(question: str) -> dict[str, str]:
    return {"protocol": SPFES_EAR_PROTOCOL, "question": question}


TRACK_QUESTIONS: dict[str, dict[str, Any]] = {
    TRIAGE_ID: {
        "type": "choice",
        "instructions": _instructions(
            "Under this protocol, should a person look at this sheep, skip it, or can it not be checked?"
        ),
        "criteria": {
            "look": "At least one ear sign: unusual carriage, clear left-right asymmetry, or frequent posture change.",
            "skip": "Face toward the camera, ears measured, carriage even and steady. Nothing to look at.",
            "cannot": "The face was mostly turned away, or the ears could not be measured on most facing frames.",
        },
    },
    LOOK_ID: {
        "type": "noul",
        "instructions": _instructions("Does this sheep show an ear sign worth a person's look?"),
        "criteria": {
            "true": "Unusual carriage, clear asymmetry, or frequent posture change in the ear numbers.",
            "false": "Ears even and steady, or too little measured to call anything a sign.",
        },
    },
    REASON_ID: {
        "type": "choice",
        "instructions": _instructions("Which one best describes this sheep's ears?"),
        "criteria": {
            "flutter": "Ear angle changed a lot from frame to frame: frequent posture change.",
            "asymmetry": "Left and right ears held at clearly different angles.",
            "carriage": "An ear held at an unusual angle: far forward, or far back toward the neck.",
            "one_ear_missing": "Face toward the camera; one ear measured, the other not on most facing frames.",
            "steady": "Both ears measured, even and steady.",
            "not_facing": "Face mostly not toward the camera.",
            "ears_unmeasurable": "Face toward the camera, but neither ear measured on most facing frames.",
        },
    },
}

PEN_QUESTIONS: dict[str, dict[str, Any]] = {
    PEN_ID: {
        "type": "choice",
        "instructions": _instructions(
            "These are the overnight ear checks for every sheep tracked in one day's footage of one pen. "
            "The farmer reads this the next morning. What should the farmer do tomorrow?"
        ),
        "criteria": {
            "walk_tomorrow": (
                "Several sheep, or a large share of the pen, show ear signs worth a look. "
                "Walk the pen first thing tomorrow."
            ),
            "later": (
                "One or two sheep show ear signs, or many could not be checked. "
                "Look at them on the next routine walk-through."
            ),
            "fine": "No sheep, or almost none, show ear signs, and most were checked. No special walk needed.",
        },
    },
}


class JevError(RuntimeError):
    """The decision API failed or returned a shape we cannot triage on."""


class _Body(Protocol):
    def read(self) -> bytes: ...


class Opener(Protocol):
    def __call__(self, req: urllib.request.Request, *, timeout: float) -> AbstractContextManager[_Body]: ...


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackTriage:
    decision: Decision
    reason: Reason
    noul: float
    threshold: float
    triage: ChoiceAnswer
    reason_answer: ChoiceAnswer
    model: str
    usage: dict[str, Any] | None

    @property
    def verdict(self) -> Verdict:
        return Verdict(self.decision, self.reason)

    @property
    def typed_class(self) -> Decision:
        return cast(Decision, self.triage.choice)

    @property
    def gated(self) -> bool:
        """True when the noul gate changed Jev's typed class."""
        return self.decision != self.typed_class

    def to_json(self) -> dict[str, Any]:
        return {
            "typed_class": self.typed_class,
            "noul": self.noul,
            "threshold": self.threshold,
            "decision": self.decision,
            "gated": self.gated,
            "reason": self.reason,
            "choice": self.triage.to_json(),
            "reason_choice": self.reason_answer.to_json(),
            "model": self.model,
            "usage": self.usage,
        }


@dataclass(frozen=True)
class PenTriage:
    call: PenCall
    answer: ChoiceAnswer
    model: str
    usage: dict[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        return {
            "call": self.call,
            "confidence": self.answer.confidence,
            "probabilities": self.answer.probabilities,
            "model": self.model,
            "usage": self.usage,
        }


def decide(typed_class: Decision, noul: float, threshold: float) -> Decision:
    """Jev's typed class, with the noul gating a typed look only."""
    if typed_class == "look" and noul < threshold:
        return "cannot"
    return typed_class


def fitting_reason(decision: Decision, answer: ChoiceAnswer) -> Reason:
    """Jev's reason if it fits the decision, else its most probable reason that does."""
    allowed = REASONS_FOR[decision]
    if answer.choice in allowed:
        return answer.choice
    return max(allowed, key=lambda reason: answer.probabilities.get(reason, 0.0))


def _body(state: Mapping[str, Any], questions: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {"model": model, "state": dict(state), "questions": dict(questions)}


def request_body(state: Mapping[str, Any], *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return _body(state, TRACK_QUESTIONS, model)


def pen_request_body(state: Mapping[str, Any], *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return _body(state, PEN_QUESTIONS, model)


def pen_state(verdicts: Sequence[Verdict], *, failed: int = 0) -> dict[str, Any]:
    """Pen-level state: per-sheep outcomes only, no numbers Jev already judged."""
    counts = Counter(v.decision for v in verdicts)
    return {
        "task": (
            "Pen-level ear check from one day's footage, run overnight. No image is attached. "
            "Each tracked sheep was already checked on its own under the protocol."
        ),
        "sheep_tracked": len(verdicts) + failed,
        "counts": {**{d: counts[d] for d in DECISIONS}, "check_failed": failed},
        "look_reasons": dict(Counter(v.reason for v in verdicts if v.decision == "look")),
        "cannot_reasons": dict(Counter(v.reason for v in verdicts if v.decision == "cannot")),
    }


def _scrub(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text


def _post(
    body: Mapping[str, Any],
    *,
    api_key: str,
    endpoint: str,
    timeout: float,
    opener: Opener | None,
) -> Mapping[str, Any]:
    if not api_key:
        raise JevError("TYPESAFE_API_KEY is empty")
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SamSeesSheep-sheep-triage-jev/0.2",
        },
        method="POST",
    )
    send = opener if opener is not None else urllib.request.urlopen
    try:
        with send(req, timeout=timeout) as resp:
            payload_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise JevError(f"Jev HTTP {exc.code}: {_scrub(detail, api_key)[:500]}") from exc
    except urllib.error.URLError as exc:
        raise JevError(f"Jev request failed: {_scrub(str(exc.reason), api_key)}") from exc
    except TimeoutError as exc:
        raise JevError(f"Jev request timed out after {timeout:g} s") from exc
    except OSError as exc:  # resets mid-read, and socket.timeout before Python 3.10
        raise JevError(f"Jev request failed: {_scrub(str(exc), api_key)}") from exc

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JevError("Jev response was not JSON") from exc
    if not isinstance(payload, dict):
        raise JevError("Jev response was not a JSON object")
    return payload


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError(f"{where} was not a number")
    if not math.isfinite(value):
        raise JevError(f"{where} was not a finite number")
    return float(value)


def _answer(payload: Mapping[str, Any], qid: str, kind: str) -> Mapping[str, Any]:
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise JevError("Jev response has no answers")
    answer = answers.get(qid)
    if not isinstance(answer, Mapping):
        raise JevError(f"Jev response missing answers.{qid} (keys: {sorted(answers)})")
    if answer.get("type", kind) != kind:
        raise JevError(f"answers.{qid} is a {answer.get('type')!r}, expected {kind!r}")
    return answer


def _noul(payload: Mapping[str, Any], qid: str) -> float:
    noul = _number(_answer(payload, qid, "noul").get("noul"), f"answers.{qid}.noul")
    if not 0.0 <= noul <= 1.0:
        raise JevError(f"answers.{qid}.noul {noul} is outside 0..1")
    return noul


def _choice(payload: Mapping[str, Any], qid: str, options: Sequence[str]) -> ChoiceAnswer:
    answer = _answer(payload, qid, "choice")
    choice = answer.get("choice")
    if choice not in options:
        raise JevError(f"answers.{qid}.choice {choice!r} is not one of {list(options)}")
    raw = answer.get("probabilities")
    if not isinstance(raw, Mapping):
        raise JevError(f"answers.{qid}.probabilities is missing")
    probabilities = {
        option: _number(raw.get(option, 0.0), f"answers.{qid}.probabilities.{option}") for option in options
    }
    confidence = _number(answer.get("confidence"), f"answers.{qid}.confidence")
    return ChoiceAnswer(cast(str, choice), confidence, probabilities)


def _usage(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else None


def triage_track(
    state: Mapping[str, Any],
    *,
    api_key: str,
    threshold: float = 0.5,
    model: str = DEFAULT_MODEL,
    endpoint: str = ENDPOINT,
    timeout: float = 60.0,
    opener: Opener | None = None,
) -> TrackTriage:
    """Ask Jev for one track's look / skip / cannot and the reason behind it."""
    payload = _post(request_body(state, model=model), api_key=api_key, endpoint=endpoint, timeout=timeout, opener=opener)
    triage = _choice(payload, TRIAGE_ID, DECISIONS)
    noul = _noul(payload, LOOK_ID)
    reason = _choice(payload, REASON_ID, JEV_REASONS)
    typed_class = cast(Decision, triage.choice)
    decision = decide(typed_class, noul, threshold)
    return TrackTriage(
        decision=decision,
        reason=GATE_REASON if decision != typed_class else fitting_reason(decision, reason),
        noul=noul,
        threshold=threshold,
        triage=triage,
        reason_answer=reason,
        model=str(payload.get("model", model)),
        usage=_usage(payload),
    )


def triage_pen(
    state: Mapping[str, Any],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = ENDPOINT,
    timeout: float = 60.0,
    opener: Opener | None = None,
) -> PenTriage:
    """Ask Jev whether the pen needs walking now, later, or not at all."""
    payload = _post(pen_request_body(state, model=model), api_key=api_key, endpoint=endpoint, timeout=timeout, opener=opener)
    answer = _choice(payload, PEN_ID, PEN_CALLS)
    return PenTriage(
        call=cast(PenCall, answer.choice),
        answer=answer,
        model=str(payload.get("model", model)),
        usage=_usage(payload),
    )
