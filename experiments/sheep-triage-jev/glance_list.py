#!/usr/bin/env python3
"""Farmer glance list: one static HTML page per run.

Reads ``tracks.json`` and ``triage.json`` from a run folder and writes
``glance-list.html`` next to them. Look cards come first. Each card has the
track number, when it is in view, one plain-English reason taken from the
pose summary, and a jump link into ``annotated.mp4`` when that file exists.

    python experiments/sheep-triage-jev/glance_list.py experiments/sheep-triage-jev/runs/<name>/

Stdlib only. The page says Look / Skip. It never shows the raw score, the
model name, or JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

PAGE_NAME = "glance-list.html"
VIDEO_NAME = "annotated.mp4"

LOOK = "Look"
SKIP = "Skip"
UNCHECKED = "Not checked"
BADGES = {"look": LOOK, "dont": SKIP}
ORDER = {LOOK: 0, UNCHECKED: 1, SKIP: 2}

SPREAD_DEG = 15.0  # ear-angle spread worth a glance
SEEN_MOST = 0.5  # ear or head measured on at least half the frames
BRIEF_S = 1.0
MOVING_WIDTHS_PER_S = 0.5  # centroid speed in box widths per second; well above tracker jitter


def badge_for(decision: str | None) -> str:
    """look → Look, dont → Skip. Anything else (pose-only, API error) → Not checked."""
    return BADGES.get(decision or "", UNCHECKED)


def clock(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def reason_for(track: dict, badge: str | None = None) -> str:
    """One sentence from the pose summary. Most notable thing first."""
    n = track.get("n_frames") or 0
    ear = track.get("ear_angle_deg") or {}
    motion = track.get("motion") or {}
    left_std = ear.get("left_std")
    right_std = ear.get("right_std")
    left_wide = left_std is not None and left_std >= SPREAD_DEG
    right_wide = right_std is not None and right_std >= SPREAD_DEG
    if left_wide and right_wide:
        return f"Both ears kept moving — their angles varied by about {max(left_std, right_std):.0f}°."
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


def build_cards(tracks_doc: dict, triage_doc: dict) -> list[dict]:
    fps = tracks_doc.get("fps")
    rows = {row["track_id"]: row for row in triage_doc.get("tracks", [])}
    cards = []
    for track in tracks_doc.get("tracks", []):
        row = rows.get(track["track_id"], {})
        badge = badge_for(row.get("decision"))
        card = {
            "track_id": track["track_id"],
            "badge": badge,
            "reason": reason_for(track, badge),
            "failed": bool(row.get("error")),
            "start_s": None,
            "end_s": None,
        }
        if fps:
            card["start_s"] = track["frame_first"] / fps
            card["end_s"] = (track["frame_last"] + 1) / fps
        cards.append(card)
    cards.sort(key=lambda c: (ORDER[c["badge"]], c["start_s"] or 0.0, c["track_id"]))
    return cards


def headline(cards: list[dict]) -> str:
    n = len(cards)
    look = sum(c["badge"] == LOOK for c in cards)
    skip = sum(c["badge"] == SKIP for c in cards)
    unchecked = n - look - skip
    if n == 0:
        return "No sheep were tracked in this clip."
    if unchecked == n and any(c["failed"] for c in cards):
        return "The look/skip check failed, so nothing is marked Look or Skip."
    if unchecked == n:
        return "Not sorted yet — the look/skip check did not run, so nothing is marked Look or Skip."
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


def _card_html(card: dict, video: str | None) -> str:
    kind = {LOOK: "look", SKIP: "skip", UNCHECKED: "unchecked"}[card["badge"]]
    lines = [
        f'<li class="card {kind}">',
        f'  <span class="badge {kind}">{escape(card["badge"])}</span>',
        f"  <h2>Sheep #{card['track_id']}</h2>",
    ]
    start = card["start_s"]
    if start is not None:
        lines.append(f'  <p class="when">In view {clock(start)} – {clock(card["end_s"])}</p>')
    lines.append(f'  <p class="why">{escape(card["reason"])}</p>')
    if card["failed"]:
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


def render_html(tracks_doc: dict, triage_doc: dict, *, video: str | None) -> str:
    cards = build_cards(tracks_doc, triage_doc)
    look = sum(c["badge"] == LOOK for c in cards)
    skip = sum(c["badge"] == SKIP for c in cards)
    clip = tracks_doc.get("clip") or "clip"
    clip_name = "synthetic test (no real video)" if clip == "synthetic" else Path(clip).name
    fps = tracks_doc.get("fps")
    n_frames = tracks_doc.get("n_frames")
    clip_line = escape(clip_name)
    if fps and n_frames:
        clip_line += f" · {clock(n_frames / fps)} long"
    counts = ""
    if look or skip:
        counts = (
            f'<p class="counts"><span class="badge look">Look {look}</span> '
            f'<span class="badge skip">Skip {skip}</span></p>'
        )
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
{counts}
</header>
<main class="{"with-video" if video else "no-video"}">
{player}
<ol class="cards">
{body}
</ol>
</main>
<footer>
<p>Look means the head and ear positions are worth a quick glance. Skip means nothing stood out.
This is not a health or pain score.</p>
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
