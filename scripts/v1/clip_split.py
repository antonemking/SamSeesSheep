"""Split and masking rules for the v1.0 face line. Standard library only, so the rules can be tested anywhere.

Three rules live here, and the exporter (export_yolo.py) cannot write a dataset without passing through them:

1. Splits are by clip, never by frame (CLAUDE.md decision 6). The caller names the held-out clips; every head
   of a clip goes to one split.
2. Evaluation footage never trains. Judge clips, any forbidden frame id (judge manifest, Blind-20), any source
   whose content hash is on the forbidden list, and any source path under a test-only folder are refused. There is
   no override.
3. A slot can be MASKED on the heads of named clips. Masked is a fourth state beside visible / occluded / absent:
   the labeler did not mark the slot on purpose, so it must carry no position loss and no "absent" signal.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

JUDGE_CLIPS = ("IMG_3877", "IMG_4425")            # Blind-20 is drawn from these two
FORBIDDEN_PATH_MARKERS = ("test-only-NEVER-TRAIN",)
STATES = ("visible", "occluded", "masked", "absent")


class SplitError(Exception):
    """A split or training-set rule is broken. Never caught by the exporter."""


def assign_by_clip(heads: Sequence[Mapping[str, Any]], val_clips: Iterable[str]) -> dict[str, str]:
    """clip -> 'train' | 'val'. Whole clips only."""
    clips = []
    for h in heads:
        c = h.get("clip")
        if not c:
            raise SplitError(f"head {h.get('id')!r} has no clip; a clip-level split needs one on every head")
        clips.append(str(c))
    val = {str(c) for c in val_clips}
    if not val:
        raise SplitError("no held-out clip named; refusing to fall back to a per-frame split")
    present = set(clips)
    if val - present:
        raise SplitError(f"held-out clips with no heads: {sorted(val - present)}")
    if not present - val:
        raise SplitError("every clip is held out; nothing left to train on")
    return {c: ("val" if c in val else "train") for c in sorted(present)}


def check_no_straddle(rows: Iterable[tuple[str, str]]) -> None:
    """rows = (clip, split) per written head. Raises if any clip is in both splits."""
    seen: dict[str, set[str]] = {}
    for clip, split in rows:
        seen.setdefault(clip, set()).add(split)
    bad = sorted(c for c, s in seen.items() if len(s) > 1)
    if bad:
        raise SplitError(f"clips in both train and val: {bad}")


def guard_training_set(heads: Sequence[Mapping[str, Any]], sources: Sequence[Mapping[str, Any]] = (),
                       forbidden_ids: Iterable[str] = (), forbidden_sha: Iterable[str] = (),
                       forbidden_clips: Iterable[str] = JUDGE_CLIPS) -> dict[str, Any]:
    """Raise SplitError if any evaluation footage is present; otherwise return what was checked.

    heads   : label rows with 'id' (frame id) and 'clip'.
    sources : one row per batch with 'source' (clip path) and 'sha256' (content hash of the clip).
    """
    ids, shas = set(forbidden_ids), {s.lower() for s in forbidden_sha}
    clips = {c.lower() for c in forbidden_clips}
    problems = []
    for h in heads:
        clip, fid = str(h.get("clip", "")), str(h.get("id", ""))
        if clip.lower() in clips or any(fid.lower().startswith(c + "_") for c in clips):
            problems.append(f"head {fid} is from judge clip {clip}")
        elif fid in ids:
            problems.append(f"head {fid} is a forbidden (judge / Blind-20) frame")
    for s in sources:
        path, sha = str(s.get("source", "")), str(s.get("sha256", "")).lower()
        stem = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem.lower() in clips:
            problems.append(f"source {path} is a judge clip")
        if sha and sha in shas:
            problems.append(f"source {path} has the content hash of forbidden footage")
        for m in FORBIDDEN_PATH_MARKERS:
            if m.lower() in path.lower():
                problems.append(f"source {path} is under {m}")
    if problems:
        raise SplitError(f"{len(problems)} evaluation-footage violations; nothing was written. First: {problems[0]}")
    return {"judge_clips": sorted(forbidden_clips), "forbidden_frame_ids": len(ids), "forbidden_sha256": len(shas),
            "forbidden_path_markers": list(FORBIDDEN_PATH_MARKERS), "heads_checked": len(heads),
            "sources_checked": len(sources), "violations": 0, "allow_judge": False}


def parse_slots(spec: str) -> list[int]:
    """'27-34' or '27,28,30-32' -> sorted slot list."""
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def check_flip_closed(slots: Iterable[int], flip_idx: Sequence[int]) -> None:
    """A horizontal flip permutes slots by flip_idx. A masked set that is not closed under it would, after a
    flip, mask a marked slot and un-mask an unmarked one."""
    s = set(slots)
    bad = sorted(i for i in s if flip_idx[i] not in s)
    if bad:
        raise SplitError(f"masked slots {bad} have their flip partner outside the masked set")


def masked_slots_for(clip: str, mask_clips: Iterable[str], mask_slots: Sequence[int]) -> list[int]:
    return list(mask_slots) if str(clip) in set(mask_clips) else []


def state(v: int, masked: bool) -> str:
    """The state of one slot of one head. Masked wins: a masked slot is never counted as absent."""
    if masked:
        return "masked"
    return {2: "visible", 1: "occluded"}.get(int(v), "absent")


def count_states(rows: Iterable[tuple[str, Sequence[int], Sequence[int]]]) -> dict[str, dict[str, int]]:
    """rows = (clip, v per slot, masked slots) per head -> {clip: {state: n}}."""
    out: dict[str, Counter] = {}
    for clip, vs, masked in rows:
        m = set(masked)
        c = out.setdefault(clip, Counter({s: 0 for s in STATES}))
        for i, v in enumerate(vs):
            c[state(v, i in m)] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def canonical(doc: Any) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
