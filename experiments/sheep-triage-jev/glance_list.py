#!/usr/bin/env python3
"""Morning brief: one static HTML check list per run.

The day's footage is tracked and ear-checked overnight; this page is what the
farmer opens next morning. It reads ``tracks.json`` and ``triage.json`` from a
run folder and writes ``morning-brief.html`` next to them.

The main view is the date, the pen, "Check these N tomorrow" with the pen
call, then one card per Look sheep, most clearly flagged first: the sheep
number, a thumbnail when the replay made one, when it was seen, one
plain-English reason, and a Watch button that seeks ``annotated.mp4``. Sheep
that looked fine or could not be checked sit in a collapsed section under the
list. Reasons come from Jev's reason choice, worded here from the track's ear
numbers; runs without one fall back to a reason read off the pose summary.

    python experiments/sheep-triage-jev/glance_list.py experiments/sheep-triage-jev/runs/<name>/ \\
        [--pen "North pen"] [--date 2026-09-24]

Stdlib only, no network. The page never shows the raw score, the model name,
JSON or file paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import date
from html import escape
from pathlib import Path
from typing import Any, Literal, TypedDict

from pose_features import EarPack
from spfes_ear import CARRIAGE_HIGH_DEG, CARRIAGE_LOW_DEG, FLUTTER_DEG, REASONS, Reason

PAGE_NAME = "morning-brief.html"
VIDEO_NAME = "annotated.mp4"
THUMBS_DIR = "thumbs"
DEFAULT_PEN = "Your flock"

LOOK = "Look"
SKIP = "Skip"
UNCHECKED = "Not checked"
# "dont" is how runs from before the SPFES ear check spelled skip.
BADGES = {"look": LOOK, "skip": SKIP, "cannot": UNCHECKED, "dont": SKIP}
ORDER = {LOOK: 0, UNCHECKED: 1, SKIP: 2}
FACE_OR_EARS_UNSEEN = ("not_facing", "ears_unmeasurable")

SPREAD_DEG = 15.0  # ear-angle spread worth a glance
SEEN_MOST = 0.5  # ear or head measured on at least half the frames
BRIEF_S = 1.0
MOVING_WIDTHS_PER_S = 0.5  # centroid speed in box widths per second; well above tracker jitter

PEN_LINES = {
    "walk_tomorrow": "Walk the pen first thing tomorrow.",
    "later": "No rush — your next routine walk-through is soon enough.",
    "fine": "The pen looks fine — no special walk needed.",
}
# Runs from before the overnight framing asked for walk_now.
OLD_PEN_CALLS = {"walk_now": "walk_tomorrow"}
FAILED_TEXT = "The ear check failed for this one."

Status = Literal["sorted", "cannot", "failed", "not_run"]


class Card(TypedDict):
    track_id: int
    badge: str
    status: Status
    reason_code: str | None
    reason: str
    rank: int | None
    start_s: float | None
    end_s: float | None


def badge_for(decision: str | None) -> str:
    """look → Look, skip → Skip. cannot, pose-only and API errors → Not checked."""
    return BADGES.get(decision or "", UNCHECKED)


def decisions_ran(triage_doc: Mapping[str, Any]) -> bool:
    """True when at least one track got a look / skip / cannot."""
    return any(row.get("decision") in BADGES for row in triage_doc.get("tracks", []))


def status_for(row: Mapping[str, Any]) -> Status:
    if row.get("error"):
        return "failed"
    decision = row.get("decision")
    if decision == "cannot":
        return "cannot"
    if decision in BADGES:
        return "sorted"
    return "not_run"


def clock(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def thumb_path(track_id: int) -> str:
    """Where the replay renderer saves a sheep's thumbnail, relative to the run folder."""
    return f"{THUMBS_DIR}/sheep-{track_id}.jpg"


def day_text(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B}"


def footage_length(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


def farmer_reason(reason: Reason, pack: EarPack) -> str:
    """Word one of Jev's reasons for a farmer, with the numbers from the ear pack."""
    measured = [
        (side, ear.median_deg, ear.std_deg)
        for side, ear in (("Left", pack.left_ear), ("Right", pack.right_ear))
        if ear.median_deg is not None and ear.std_deg is not None
    ]
    if reason == "flutter":
        if len(measured) == 2 and all(std >= FLUTTER_DEG for _, _, std in measured):
            return f"Both ears kept changing position — their angles varied by about {max(s for _, _, s in measured):.0f}°."
        if measured:
            side, _, std = max(measured, key=lambda m: m[2])
            return f"{side} ear kept changing position — its angle varied by about {std:.0f}°."
        return "Its ears kept changing position."
    if reason == "asymmetry":
        if pack.asymmetry_deg is not None:
            return f"Ears held unevenly — left and right differed by about {pack.asymmetry_deg:.0f}°."
        return "Ears held unevenly — one sat differently from the other."
    if reason == "carriage":
        if measured:
            side, median, _ = max(measured, key=lambda m: max(CARRIAGE_LOW_DEG - m[1], m[1] - CARRIAGE_HIGH_DEG))
            direction = "back" if median > (CARRIAGE_LOW_DEG + CARRIAGE_HIGH_DEG) / 2 else "forward"
            return f"{side} ear held unusually far {direction} (about {median:.0f}°)."
        return "An ear was held at an unusual angle."
    if reason == "one_ear_missing":
        side = "left" if pack.left_ear.measured_fraction <= pack.right_ear.measured_fraction else "right"
        return f"Facing the camera, but its {side} ear was hard to see on most frames."
    if reason == "steady":
        return "Both ears seen, even and steady."
    if reason == "not_facing":
        return "Couldn't see its face well enough — it was mostly turned away."
    if reason == "ears_unmeasurable":
        return "Couldn't see its ears well enough to check."
    if reason == "not_sure":
        return "Not sure enough to call — the ear signs were weak or mixed."
    raise ValueError(f"no farmer wording for reason {reason!r}")


def reason_for(track: Mapping[str, Any], badge: str | None = None) -> str:
    """One sentence from the pose summary, for runs without a Jev reason. Most notable thing first."""
    n = track.get("n_frames") or 0
    ear: Mapping[str, Any] = track.get("ear_angle_deg") or {}
    motion: Mapping[str, Any] = track.get("motion") or {}
    left_std = ear.get("left_std")
    right_std = ear.get("right_std")
    left_wide = left_std is not None and left_std >= SPREAD_DEG
    right_wide = right_std is not None and right_std >= SPREAD_DEG
    if left_wide and right_wide:
        return f"Both ears kept moving — their angles varied by about {max(ear['left_std'], ear['right_std']):.0f}°."
    if left_wide or right_wide:
        side, std = ("Left", left_std) if left_wide else ("Right", right_std)
        return f"{side} ear kept moving — its angle varied by about {std:.0f}°."

    if n and (track.get("facing_fraction") or 0.0) < SEEN_MOST:
        return "Head turned away or hidden on most frames."
    left_seen = n and ear.get("n_left", 0) / n >= SEEN_MOST
    right_seen = n and ear.get("n_right", 0) / n >= SEEN_MOST
    if not left_seen and not right_seen:
        return "Both ears hard to see on most frames."
    if not left_seen or not right_seen:
        return f"{'Left' if not left_seen else 'Right'} ear hard to see on most frames."

    duration = track.get("duration_s") or 0.0
    path = motion.get("centroid_path_px") or 0.0
    width = motion.get("bbox_median_w_px") or 0.0
    if duration > 0 and width > 0 and path / duration / width >= MOVING_WIDTHS_PER_S:
        return "Walked around a lot while in view."
    if duration < BRIEF_S:
        return f"Only in view for a moment ({duration:.1f} s)."
    if badge == LOOK:
        return "Nothing stood out in the ear and head numbers, but it was still flagged for a glance."
    return "Both ears seen and steady; the head stayed put."


def card_reason(track: Mapping[str, Any], row: Mapping[str, Any], badge: str) -> str:
    reason = row.get("reason")
    pack = track.get("ear_pack")
    if reason in REASONS and pack is not None:
        return farmer_reason(reason, EarPack.from_json(pack))
    return reason_for(track, badge)


def _strength(row: Mapping[str, Any]) -> float | None:
    noul = row.get("noul")
    if isinstance(noul, bool) or not isinstance(noul, (int, float)):
        return None
    return float(noul)


def build_cards(tracks_doc: Mapping[str, Any], triage_doc: Mapping[str, Any]) -> list[Card]:
    """Look cards first, most clearly flagged first, then Not checked, then Skip."""
    fps = tracks_doc.get("fps")
    rows = {row["track_id"]: row for row in triage_doc.get("tracks", [])}
    keyed: list[tuple[tuple[Any, ...], Card]] = []
    for track in tracks_doc.get("tracks", []):
        row = rows.get(track["track_id"], {})
        badge = badge_for(row.get("decision"))
        status = status_for(row)
        start_s = end_s = None
        if fps:
            start_s = track["frame_first"] / fps
            end_s = (track["frame_last"] + 1) / fps
        card = Card(
            track_id=track["track_id"],
            badge=badge,
            status=status,
            reason_code=row.get("reason"),
            reason=FAILED_TEXT if status == "failed" else card_reason(track, row, badge),
            rank=None,
            start_s=start_s,
            end_s=end_s,
        )
        strength = _strength(row) if badge == LOOK else None
        key = (ORDER[badge], strength is None, -(strength or 0.0), start_s or 0.0, track["track_id"])
        keyed.append((key, card))
    cards = [card for _, card in sorted(keyed, key=lambda kc: kc[0])]
    for rank, card in enumerate((c for c in cards if c["badge"] == LOOK), start=1):
        card["rank"] = rank
    return cards


def _counts(cards: list[Card]) -> tuple[int, int, int, int]:
    n = len(cards)
    look = sum(c["badge"] == LOOK for c in cards)
    skip = sum(c["badge"] == SKIP for c in cards)
    return n, look, skip, n - look - skip


def headline(cards: list[Card]) -> str:
    """The one line under the date. Counts Look sheep only; the rest never make the headline."""
    n, look, skip, _ = _counts(cards)
    statuses = {c["status"] for c in cards}
    if n == 0:
        return "No sheep were seen in this footage."
    if look == 0 and skip == 0:
        if statuses == {"not_run"}:
            return "Not sorted yet — the ear check didn't run, so there's no list for tomorrow."
        if statuses <= {"failed", "not_run"}:
            return "The ear check failed, so there's no list for tomorrow."
        if statuses == {"cannot"} and all(c["reason_code"] in FACE_OR_EARS_UNSEEN for c in cards):
            if n == 1:
                return "Couldn't check the one sheep — its face or ears weren't clear enough."
            return f"Couldn't check any of the {n} sheep — their faces or ears weren't clear enough."
        if n == 1:
            return "Couldn't check the one sheep, so there's no list for tomorrow."
        return f"Couldn't check any of the {n} sheep, so there's no list for tomorrow."
    if look == 0:
        return "Nothing needs a check tomorrow." if skip == n else "Nothing flagged for tomorrow."
    if look == n:
        return "Check the one sheep tomorrow." if n == 1 else f"Check all {n} tomorrow."
    return "Check this one tomorrow." if look == 1 else f"Check these {look} tomorrow."


def detail(cards: list[Card]) -> str | None:
    """A second line for the states the headline alone would leave unclear."""
    n, look, skip, unchecked = _counts(cards)
    if look == n and n > 1:
        return f"Every sheep seen was flagged ({n} of {n})."
    if look == 0 and skip:
        if skip == n:
            return "The one sheep seen looked fine." if n == 1 else f"All {n} sheep looked fine."
        return f"{skip} looked fine and {unchecked} couldn't be checked — see below."
    return None


def pen_line(triage_doc: Mapping[str, Any]) -> tuple[str, str] | None:
    """(call, sentence) for the pen call, or None when there is none."""
    pen = triage_doc.get("pen")
    call = pen.get("call") if isinstance(pen, Mapping) else None
    if not isinstance(call, str):
        return None
    call = OLD_PEN_CALLS.get(call, call)
    if call not in PEN_LINES:
        return None
    return call, PEN_LINES[call]


def _text(value: str) -> str:
    return escape(value, quote=False)


def _watch(video: str | None, start: float | None, label: str, text: str, extra: str = "") -> str:
    if not video or start is None:
        return ""
    return (
        f'<a class="watch{extra}" href="{escape(video)}#t={start:.2f}" data-t="{start:.2f}" '
        f'data-label="{escape(label)} · from {clock(start)}">{_text(text)}</a>'
    )


def _when(card: Card) -> str:
    start, end = card["start_s"], card["end_s"]
    if start is None or end is None:
        return ""
    return f"Seen {clock(start)} – {clock(end)}"


def _look_card_html(card: Card, video: str | None, thumb: str | None) -> str:
    name = f"Sheep #{card['track_id']}"
    when = _when(card)
    lines = [
        f'<li class="card look{"" if thumb else " no-thumb"}">',
        f'  <span class="rank" aria-hidden="true">{card["rank"]}</span>',
        '  <div class="head">',
        f"    <h2>{name}</h2>",
    ]
    if when:
        lines.append(f'    <p class="when">{when}</p>')
    lines.append("  </div>")
    if thumb:
        lines.append(
            f'  <img class="thumb" src="{escape(thumb)}" alt="{name}" width="320" height="240" loading="lazy">'
        )
    lines.append(f'  <p class="why">{_text(card["reason"])}</p>')
    start = card["start_s"]
    watch = _watch(video, start, name, f"Watch from {clock(start)}" if start is not None else "")
    if watch:
        lines.append(f"  {watch}")
    lines.append("</li>")
    return "\n".join(lines)


def _row_html(card: Card, video: str | None, thumb: str | None) -> str:
    name = f"Sheep #{card['track_id']}"
    when = _when(card)
    lines = [f'<li class="row{"" if thumb else " no-thumb"}">']
    if thumb:
        lines.append(f'  <img class="mini" src="{escape(thumb)}" alt="{name}" width="320" height="240" loading="lazy">')
    head = f"<strong>{name}</strong>" + (f' <span class="when">{when}</span>' if when else "")
    lines.append(f'  <p class="row-head">{head}</p>')
    lines.append(f'  <p class="why">{_text(card["reason"])}</p>')
    watch = _watch(video, card["start_s"], name, "Watch", " small")
    if watch:
        lines.append(f"  {watch}")
    lines.append("</li>")
    return "\n".join(lines)


def _rest_html(cards: list[Card], video: str | None, thumbs: Mapping[int, str]) -> str:
    groups = [
        ("Not checked", [c for c in cards if c["badge"] == UNCHECKED]),
        ("Looked fine", [c for c in cards if c["badge"] == SKIP]),
    ]
    groups = [(title, group) for title, group in groups if group]
    if not groups:
        return ""
    counts = " · ".join(f"{len(group)} {title.lower()}" for title, group in groups)
    body = []
    for title, group in groups:
        rows = "\n".join(_row_html(c, video, thumbs.get(c["track_id"])) for c in group)
        body.append(f'<h3>{title}</h3>\n<ul class="rows">\n{rows}\n</ul>')
    return (
        '<details class="rest">\n'
        '<summary><span class="rest-title">Not on the list</span> '
        f'<span class="rest-counts">{counts}</span></summary>\n'
        f'<div class="rest-body">\n{chr(10).join(body)}\n</div>\n'
        "</details>"
    )


STYLE = """
:root {
  color-scheme: light;
  --bg: #f4f1e8; --card: #fff; --ink: #1d221f; --muted: #56605a; --line: #e0dacb;
  --look: #f2a23a; --button: #1d221f; --focus: #2a64d0;
  --walk-tomorrow: #fbe2bf; --walk-tomorrow-dot: #d97a0c;
  --later: #f7eecb; --later-dot: #b8911a;
  --fine: #dfeee1; --fine-dot: #3b8753;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 17px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.page { max-width: 760px; margin: 0 auto; padding: 24px 16px 120px; }
.eyebrow { margin: 0; font-size: 13px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { margin: 6px 0 4px; font-size: 28px; line-height: 1.15; letter-spacing: -.01em; }
.meta { margin: 0; color: var(--muted); }
.summary { margin: 20px 0 18px; padding: 16px 18px; border-radius: 16px; background: var(--card); box-shadow: 0 0 0 1px var(--line); }
.summary.pen-walk-tomorrow { background: var(--walk-tomorrow); box-shadow: none; }
.summary.pen-later { background: var(--later); box-shadow: none; }
.summary.pen-fine { background: var(--fine); box-shadow: none; }
.headline { margin: 0; font-size: 24px; line-height: 1.2; font-weight: 750; }
.detail { margin: 6px 0 0; color: var(--muted); }
.pen { display: flex; align-items: center; gap: 10px; margin: 10px 0 0; font-weight: 650; }
.dot { flex: none; width: 12px; height: 12px; border-radius: 50%; background: var(--muted); }
.pen-walk-tomorrow .dot { background: var(--walk-tomorrow-dot); }
.pen-later .dot { background: var(--later-dot); }
.pen-fine .dot { background: var(--fine-dot); }
.list-note { margin: 0 0 8px; font-size: 14px; color: var(--muted); }
.checklist { list-style: none; margin: 0; padding: 0; display: grid; gap: 12px; }
.card { display: grid; align-items: start; gap: 4px 12px; padding: 14px; background: var(--card); border-radius: 16px;
  box-shadow: 0 0 0 1px var(--line), 0 2px 6px rgba(40, 35, 20, .05);
  grid-template-columns: 34px 1fr 104px; grid-template-areas: "rank head thumb" "why why why" "watch watch watch"; }
.card.no-thumb { grid-template-columns: 34px 1fr; grid-template-areas: "rank head" "why why" "watch watch"; }
.rank { grid-area: rank; display: grid; place-items: center; width: 34px; height: 34px; border-radius: 50%;
  background: var(--look); font-weight: 800; }
.head { grid-area: head; min-width: 0; }
.card h2 { margin: 3px 0 0; font-size: 20px; line-height: 1.2; }
.when { margin: 2px 0 0; color: var(--muted); font-size: 15px; }
.thumb { grid-area: thumb; display: block; width: 100%; height: auto; aspect-ratio: 4 / 3; object-fit: cover;
  border-radius: 10px; background: #d9d3c4; }
.why { grid-area: why; margin: 6px 0 0; }
.watch { grid-area: watch; justify-self: start; display: inline-flex; align-items: center; gap: 9px; min-height: 44px;
  margin-top: 8px; padding: 0 16px; border-radius: 10px; background: var(--button); color: #fff;
  font-size: 16px; font-weight: 650; text-decoration: none; }
.watch::before { content: ""; border-style: solid; border-width: 6px 0 6px 10px;
  border-color: transparent transparent transparent currentColor; }
.watch:hover { background: #3a423c; }
.watch.small { min-height: 38px; padding: 0 12px; font-size: 15px; }
.rest { margin-top: 20px; background: var(--card); border-radius: 16px; box-shadow: 0 0 0 1px var(--line); }
.rest summary { position: relative; display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px;
  min-height: 52px; padding: 14px 48px 14px 16px; cursor: pointer; list-style: none; }
.rest summary::-webkit-details-marker { display: none; }
.rest summary::after { content: ""; position: absolute; right: 20px; top: 50%; width: 9px; height: 9px;
  border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted); transform: translateY(-70%) rotate(45deg); }
.rest[open] summary::after { transform: translateY(-20%) rotate(-135deg); }
.rest-title { font-weight: 700; }
.rest-counts { color: var(--muted); }
.rest-body { padding: 0 16px 8px; border-top: 1px solid var(--line); }
.rest h3 { margin: 16px 0 2px; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.rows { list-style: none; margin: 0; padding: 0; }
.row { display: grid; align-items: start; gap: 2px 12px; padding: 12px 0; border-top: 1px solid var(--line);
  grid-template-columns: 64px 1fr; grid-template-areas: "mini head" "why why" "watch watch"; }
.row:first-child { border-top: 0; }
.row.no-thumb { grid-template-columns: 1fr; grid-template-areas: "head" "why" "watch"; }
.mini { grid-area: mini; display: block; width: 64px; height: auto; aspect-ratio: 4 / 3; object-fit: cover;
  border-radius: 8px; background: #d9d3c4; }
.row-head { grid-area: head; margin: 0; min-width: 0; }
.row-head .when { display: block; margin: 0; }
.row .why { margin: 4px 0 0; }
.row .watch { margin-top: 8px; }
footer { margin-top: 28px; color: var(--muted); font-size: 15px; }
footer p { margin: 0 0 10px; }
.watch-all { color: var(--ink); font-weight: 650; }
.player { position: fixed; z-index: 10; left: 0; right: 0; bottom: 0; display: flex; flex-direction: column; gap: 8px;
  padding: 10px 10px calc(10px + env(safe-area-inset-bottom)); background: #161a17; color: #fff;
  box-shadow: 0 -8px 28px rgba(0, 0, 0, .28); }
.player[hidden] { display: none; }
.player-bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-weight: 650; }
.player-bar button { min-height: 40px; padding: 0 14px; border: 0; border-radius: 8px; background: #333a35; color: #fff;
  font: inherit; cursor: pointer; }
.player video { display: block; width: 100%; max-height: 45vh; background: #000; border-radius: 8px; }
a:focus-visible, button:focus-visible, summary:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
@media (min-width: 700px) {
  body { font-size: 18px; }
  .page { padding: 44px 24px 140px; }
  h1 { font-size: 42px; }
  .summary { padding: 20px 24px; }
  .headline { font-size: 28px; }
  .card { gap: 4px 18px; padding: 16px; grid-template-columns: 40px 184px 1fr; grid-template-rows: auto auto 1fr;
    grid-template-areas: "rank thumb head" "rank thumb why" "rank thumb watch"; }
  .card.no-thumb { grid-template-columns: 40px 1fr; grid-template-areas: "rank head" "rank why" "rank watch"; }
  .rank { width: 40px; height: 40px; }
  .row { align-items: center; grid-template-columns: 72px 1fr auto; grid-template-areas: "mini head watch" "mini why watch"; }
  .row.no-thumb { grid-template-columns: 1fr auto; grid-template-areas: "head watch" "why watch"; }
  .row-head .when { display: inline; margin-left: 6px; }
  .mini { width: 72px; }
  .row .watch { margin-top: 0; }
}
@media (min-width: 1000px) {
  .player { left: auto; right: 24px; bottom: 24px; width: 520px; padding: 12px; border-radius: 14px; }
}
""".strip()

SCRIPT = """
(function () {
  var player = document.getElementById("player");
  var video = document.getElementById("replay");
  var label = document.getElementById("player-label");
  if (!player || !video) return;
  function seek(t) {
    function go() {
      video.currentTime = t;
      var playing = video.play();
      if (playing && playing.catch) playing.catch(function () {});
    }
    if (video.readyState >= 1) go();
    else video.addEventListener("loadedmetadata", go, { once: true });
  }
  document.querySelectorAll("a[data-t]").forEach(function (link) {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      label.textContent = link.dataset.label;
      player.hidden = false;
      seek(parseFloat(link.dataset.t));
    });
  });
  function close() {
    video.pause();
    player.hidden = true;
  }
  document.getElementById("player-close").addEventListener("click", close);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !player.hidden) close();
  });
})();
""".strip()


def _footage_day(tracks_doc: Mapping[str, Any]) -> date | None:
    value = tracks_doc.get("footage_date")
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def render_html(
    tracks_doc: Mapping[str, Any],
    triage_doc: Mapping[str, Any],
    *,
    video: str | None,
    thumbs: Mapping[int, str] | None = None,
    day: date | None = None,
    pen_label: str | None = None,
) -> str:
    """The brief as one self-contained page. ``day`` and ``pen_label`` default to the run's own."""
    thumbs = thumbs or {}
    cards = build_cards(tracks_doc, triage_doc)
    looks = [c for c in cards if c["badge"] == LOOK]
    day = day or _footage_day(tracks_doc)
    pen_name = (pen_label or tracks_doc.get("pen_label") or DEFAULT_PEN).strip() or DEFAULT_PEN
    when = day_text(day) if day else "Morning brief"
    clip = tracks_doc.get("clip") or "clip"
    clip_name = "synthetic test (no real video)" if clip == "synthetic" else Path(clip).name
    fps = tracks_doc.get("fps")
    n_frames = tracks_doc.get("n_frames")
    length = n_frames / fps if fps and n_frames else None
    meta = f"{footage_length(length)} of footage, sorted overnight" if length else "Sorted overnight"
    source = f"Footage: {clip_name}" + (f" · {clock(length)} long." if length else ".")

    call = pen_line(triage_doc)
    summary_class = f"summary pen-{call[0].replace('_', '-')}" if call else "summary"
    summary = [f'<p class="headline">{_text(headline(cards))}</p>']
    extra = detail(cards)
    if extra:
        summary.append(f'<p class="detail">{_text(extra)}</p>')
    if call:
        summary.append(f'<p class="pen"><span class="dot" aria-hidden="true"></span>{_text(call[1])}</p>')

    checklist = ""
    if looks:
        note = '<p class="list-note">Most clearly flagged first.</p>\n' if len(looks) > 1 else ""
        items = "\n".join(_look_card_html(c, video, thumbs.get(c["track_id"])) for c in looks)
        checklist = f'{note}<ol class="checklist">\n{items}\n</ol>'
    rest = _rest_html(cards, video, thumbs)

    dropped = len(tracks_doc.get("dropped_track_ids") or [])
    ignored = ""
    if dropped:
        ignored = f" {dropped} very brief detection{'s were' if dropped != 1 else ' was'} ignored."
    watch_all = player = script = ""
    if video:
        watch_all = (
            f'<p><a class="watch-all" href="{escape(video)}" data-t="0" data-label="Full replay">'
            "Watch the full replay</a></p>\n"
        )
        player = (
            '<div class="player" id="player" hidden>\n'
            '<div class="player-bar"><span id="player-label">Replay</span>'
            '<button type="button" id="player-close">Close</button></div>\n'
            f'<video id="replay" src="{escape(video)}" controls preload="metadata" playsinline></video>\n'
            "</div>"
        )
        script = f"<script>\n{SCRIPT}\n</script>"
    iso = f' datetime="{day.isoformat()}"' if day else ""
    title = f"{when} — {pen_name} — morning brief" if day else f"{pen_name} — morning brief"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_text(title)}</title>
<style>
{STYLE}
</style>
</head>
<body>
<div class="page">
<header>
<p class="eyebrow">{_text(pen_name)} · Morning brief</p>
<h1><time{iso}>{_text(when)}</time></h1>
<p class="meta">{_text(meta)}</p>
</header>
<main>
<section class="{summary_class}">
{chr(10).join(summary)}
</section>
{checklist}
{rest}
</main>
<footer>
{watch_all}<p>Overnight, each sheep in the footage gets an ear check: how its ears are held, whether one sits
differently from the other, and whether they keep changing position. Sheep worth a look go on the list, most
clearly flagged first. Looked fine means both ears were even and steady. Not checked means its face or ears
weren't clear enough to judge, or the check wasn't sure. This only looks at ears — it can't tell you what's
wrong with a sheep, and it can miss things.</p>
<p>Sheep numbers are labels for this footage only, not ear tags.{ignored}</p>
<p>{_text(source)}</p>
</footer>
</div>
{player}
{script}
</body>
</html>
"""


def write_glance_list(run_dir: Path, *, day: date | None = None, pen_label: str | None = None) -> Path:
    """Write ``morning-brief.html`` into ``run_dir``. Without a recorded date, the brief uses the run's file date."""
    tracks_path = run_dir / "tracks.json"
    tracks_doc = json.loads(tracks_path.read_text())
    triage_doc = json.loads((run_dir / "triage.json").read_text())
    video = VIDEO_NAME if (run_dir / VIDEO_NAME).is_file() else None
    thumbs = {
        t["track_id"]: thumb_path(t["track_id"])
        for t in tracks_doc.get("tracks", [])
        if (run_dir / thumb_path(t["track_id"])).is_file()
    }
    day = day or _footage_day(tracks_doc) or date.fromtimestamp(tracks_path.stat().st_mtime)
    page = run_dir / PAGE_NAME
    page.write_text(render_html(tracks_doc, triage_doc, video=video, thumbs=thumbs, day=day, pen_label=pen_label))
    return page


def iso_day(value: str) -> date:
    """argparse type for --date."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date like 2026-09-24") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="Run folder with tracks.json and triage.json.")
    p.add_argument(
        "--pen", default=None, help=f'Pen or flock name for the top of the brief (default: the run\'s, else "{DEFAULT_PEN}").'
    )
    p.add_argument(
        "--date", type=iso_day, default=None, help="Day the footage was taken, YYYY-MM-DD (default: the run's)."
    )
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    for name in ("tracks.json", "triage.json"):
        if not (args.run_dir / name).is_file():
            raise SystemExit(f"{name} not found in {args.run_dir}. Run run_pipeline.py first.")
    print(f"wrote {write_glance_list(args.run_dir, day=args.date, pen_label=args.pen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
