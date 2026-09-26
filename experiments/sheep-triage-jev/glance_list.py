#!/usr/bin/env python3
"""Farmer glance list: one static HTML page per run.

Reads ``tracks.json`` and ``triage.json`` from a run folder and writes
``glance-list.html`` next to them. Look cards come first. Each card has the
track number, when it is in view, one plain-English reason, and a jump link
into ``annotated.mp4`` when that file exists. The reason comes from Jev's
reason choice, worded here from the track's ear numbers; runs without one
fall back to a reason read off the pose summary. If the run has a pen call,
the page opens with it.

    python experiments/sheep-triage-jev/glance_list.py experiments/sheep-triage-jev/runs/<name>/

Stdlib only. The page says Look / Skip / Not checked. It never shows the raw
score, the model name, or JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any, Literal, TypedDict

from pose_features import EarPack
from spfes_ear import CARRIAGE_HIGH_DEG, CARRIAGE_LOW_DEG, FLUTTER_DEG, REASONS, Reason

PAGE_NAME = "glance-list.html"
VIDEO_NAME = "annotated.mp4"

LOOK = "Look"
SKIP = "Skip"
UNCHECKED = "Not checked"
# "dont" is how runs from before the SPFES ear check spelled skip.
BADGES = {"look": LOOK, "skip": SKIP, "cannot": UNCHECKED, "dont": SKIP}
ORDER = {LOOK: 0, UNCHECKED: 1, SKIP: 2}

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

Status = Literal["sorted", "cannot", "failed", "not_run"]


class Card(TypedDict):
    track_id: int
    badge: str
    status: Status
    reason: str
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


def build_cards(tracks_doc: Mapping[str, Any], triage_doc: Mapping[str, Any]) -> list[Card]:
    fps = tracks_doc.get("fps")
    rows = {row["track_id"]: row for row in triage_doc.get("tracks", [])}
    cards: list[Card] = []
    for track in tracks_doc.get("tracks", []):
        row = rows.get(track["track_id"], {})
        badge = badge_for(row.get("decision"))
        start_s = end_s = None
        if fps:
            start_s = track["frame_first"] / fps
            end_s = (track["frame_last"] + 1) / fps
        cards.append(
            Card(
                track_id=track["track_id"],
                badge=badge,
                status=status_for(row),
                reason=card_reason(track, row, badge),
                start_s=start_s,
                end_s=end_s,
            )
        )
    cards.sort(key=lambda c: (ORDER[c["badge"]], c["start_s"] or 0.0, c["track_id"]))
    return cards


def headline(cards: list[Card]) -> str:
    n = len(cards)
    look = sum(c["badge"] == LOOK for c in cards)
    skip = sum(c["badge"] == SKIP for c in cards)
    unchecked = n - look - skip
    statuses = {c["status"] for c in cards}
    if n == 0:
        return "No sheep were tracked in this clip."
    if statuses == {"not_run"}:
        return "Not sorted yet — the look/skip check did not run, so nothing is marked Look or Skip."
    if "failed" in statuses and statuses <= {"failed", "not_run"}:
        return "The look/skip check failed, so nothing is marked Look or Skip."
    if statuses == {"cannot"}:
        if n == 1:
            return "Couldn't check the one sheep — its face or ears weren't clear enough."
        return f"Couldn't check any of the {n} sheep — their faces or ears weren't clear enough."
    if look == n:
        return f"Every sheep in this clip needs a look ({n} of {n})."
    if skip == n:
        return f"Nothing stood out — all {n} can be skipped."
    parts = [f"{look} of {n} need a look"]
    if skip:
        parts.append(f"{skip} can be skipped")
    if unchecked:
        parts.append(f"{unchecked} could not be checked")
    return ", ".join(parts) + "."


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


def _card_html(card: Card, video: str | None) -> str:
    kind = {LOOK: "look", SKIP: "skip", UNCHECKED: "unchecked"}[card["badge"]]
    lines = [
        f'<li class="card {kind}">',
        f'  <span class="badge {kind}">{escape(card["badge"])}</span>',
        f"  <h2>Sheep #{card['track_id']}</h2>",
    ]
    start, end = card["start_s"], card["end_s"]
    if start is not None and end is not None:
        lines.append(f'  <p class="when">In view {clock(start)} – {clock(end)}</p>')
    lines.append(f'  <p class="why">{escape(card["reason"])}</p>')
    if card["status"] == "failed":
        lines.append('  <p class="note">The look/skip check failed for this one.</p>')
    if start is not None and video:
        lines.append(
            f'  <a class="jump" href="{escape(video)}#t={start:.2f}" data-t="{start:.2f}">'
            f"Jump to {clock(start)}</a>"
        )
    elif start is not None:
        lines.append(f'  <p class="jump-missing">At {clock(start)} in the clip.</p>')
    lines.append("</li>")
    return "\n".join(lines)


STYLE = """
body { font: 18px/1.4 -apple-system, system-ui, sans-serif; margin: 0 auto; max-width: 1280px; padding: 16px; color: #1d1d1b; background: #f6f4ee; }
h1 { font-size: 28px; margin: 0 0 4px; }
.clip { color: #666; margin: 0 0 12px; }
.headline { font-size: 22px; font-weight: 600; margin: 0 0 8px; }
main { display: grid; gap: 16px; margin-top: 8px; }
@media (min-width: 900px) { main.with-video { grid-template-columns: 3fr 2fr; align-items: start; } .player { position: sticky; top: 12px; } }
video { width: 100%; background: #000; border-radius: 8px; }
.cards { list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }
.card { background: #fff; border-radius: 10px; padding: 12px 16px; border-left: 8px solid #bbb; }
.card.look { border-left-color: #ff9f1a; }
.card.skip { border-left-color: #3caf50; opacity: 0.85; }
.card h2 { display: inline; font-size: 20px; margin-left: 8px; }
.card p { margin: 6px 0; }
.when { color: #555; }
.note { color: #a33; }
.badge { display: inline-block; font-weight: 700; padding: 2px 12px; border-radius: 999px; background: #bbb; color: #1d1d1b; }
.badge.look { background: #ff9f1a; }
.badge.skip { background: #3caf50; color: #fff; }
.pen { display: inline-block; font-size: 20px; font-weight: 700; margin: 4px 0 10px; padding: 6px 16px; border-radius: 8px; background: #e4e1d8; }
.pen.walk-now { background: #ff9f1a; }
.pen.later { background: #ffe0a8; }
.pen.fine { background: #cfe9d2; }
.jump { display: inline-block; margin-top: 4px; padding: 6px 14px; border-radius: 6px; background: #1d1d1b; color: #fff; text-decoration: none; font-weight: 600; }
footer { color: #666; font-size: 15px; margin-top: 24px; }
""".strip()

SCRIPT = """
document.querySelectorAll("a.jump").forEach(function (link) {
  link.addEventListener("click", function (event) {
    var video = document.getElementById("replay");
    if (!video) return;
    event.preventDefault();
    video.currentTime = parseFloat(link.dataset.t);
    video.play();
    video.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
});
""".strip()


def render_html(tracks_doc: Mapping[str, Any], triage_doc: Mapping[str, Any], *, video: str | None) -> str:
    cards = build_cards(tracks_doc, triage_doc)
    look = sum(c["badge"] == LOOK for c in cards)
    skip = sum(c["badge"] == SKIP for c in cards)
    unchecked = len(cards) - look - skip
    clip = tracks_doc.get("clip") or "clip"
    clip_name = "synthetic test (no real video)" if clip == "synthetic" else Path(clip).name
    fps = tracks_doc.get("fps")
    n_frames = tracks_doc.get("n_frames")
    clip_line = escape(clip_name)
    if fps and n_frames:
        clip_line += f" · {clock(n_frames / fps)} long"
    pen = ""
    call = pen_line(triage_doc)
    if call is not None:
        pen = f'<p class="pen {call[0].replace("_", "-")}">{escape(call[1])}</p>'
    counts = ""
    if decisions_ran(triage_doc):
        pills = [
            f'<span class="badge look">Look {look}</span>',
            f'<span class="badge skip">Skip {skip}</span>',
        ]
        if unchecked:
            pills.append(f'<span class="badge unchecked">Not checked {unchecked}</span>')
        counts = f'<p class="counts">{" ".join(pills)}</p>'
    player = ""
    if video:
        player = (
            f'<div class="player"><video id="replay" src="{escape(video)}" '
            f'controls preload="metadata" playsinline></video></div>'
        )
    dropped = len(tracks_doc.get("dropped_track_ids") or [])
    ignored = ""
    if dropped:
        ignored = f" {dropped} very brief detection{'s were' if dropped != 1 else ' was'} ignored."
    body = "\n".join(_card_html(card, video) for card in cards)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which sheep need a glance — {escape(clip_name)}</title>
<style>
{STYLE}
</style>
</head>
<body>
<header>
<h1>Which sheep need a glance</h1>
<p class="clip">{clip_line}</p>
<p class="headline">{escape(headline(cards))}</p>
{pen}
{counts}
</header>
<main class="{"with-video" if video else "no-video"}">
{player}
<ol class="cards">
{body}
</ol>
</main>
<footer>
<p>Look means the ears showed something worth a quick glance: an unusual angle, one ear held
differently from the other, or ears that kept changing position. Skip means both ears looked even and steady.
Not checked means its face or ears weren't clear enough to judge. This looks at ear position only;
it is not a health or pain score.</p>
<p>Sheep numbers are tracking labels for this clip only, not ear tags.{ignored}</p>
</footer>
<script>
{SCRIPT}
</script>
</body>
</html>
"""


def write_glance_list(run_dir: Path) -> Path:
    tracks_doc = json.loads((run_dir / "tracks.json").read_text())
    triage_doc = json.loads((run_dir / "triage.json").read_text())
    video = VIDEO_NAME if (run_dir / VIDEO_NAME).is_file() else None
    page = run_dir / PAGE_NAME
    page.write_text(render_html(tracks_doc, triage_doc, video=video))
    return page


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="Run folder with tracks.json and triage.json.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    for name in ("tracks.json", "triage.json"):
        if not (args.run_dir / name).is_file():
            raise SystemExit(f"{name} not found in {args.run_dir}. Run run_pipeline.py first.")
    print(f"wrote {write_glance_list(args.run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
