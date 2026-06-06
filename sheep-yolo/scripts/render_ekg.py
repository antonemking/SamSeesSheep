"""Live ear-angle 'EKG' video — tracks the calm ewe across the whole clip
(ByteTrack) so the readout runs as long as she's visible, not just the 5 s
window. Left: her crop (dynamically centered) with v0.7 keypoints. Right: her
ear-angle trace drawing in real time.

No clinical/SPFES bands and no burned-in caption — keep the "not yet validated
against stress events" framing in the post copy (VALIDATION.md), not the pixels.

  python render_ekg.py
"""
from __future__ import annotations
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CLIP = ROOT.parent / "test-clips" / "IMG_3651.MOV"
WEIGHTS = ROOT / "weights" / "sheep-pose-v0.7-yolo26n.pt"

WINDOW = (367, 522)            # used only to identify which track is the calm ewe
ROI = (1220, 80, 1560, 390)
FPS = 30
KPT_TH = 0.4
SKEL = [(0, 1), (0, 2), (1, 3), (2, 4)]
KCOL = (255, 0, 255)
CROP = 560                     # dynamic crop side (px), centered on her box
PANEL = 540


def ang(m, e):
    return float(np.degrees(np.arctan2(m[0] * e[1] - m[1] * e[0], m[0] * e[0] + m[1] * e[1])))


def ears(k):
    nose, lb, rb, lt, rt = k
    if nose[2] <= 0 or lb[2] <= 0 or rb[2] <= 0:
        return np.nan, np.nan
    mid = (lb[:2] + rb[:2]) / 2
    ml = nose[:2] - mid
    return (abs(ang(ml, lt[:2] - lb[:2])) if lt[2] > 0 else np.nan,
            abs(ang(ml, rt[:2] - rb[:2])) if rt[2] > 0 else np.nan)


def draw_angle(img, base, v_ear, v_mid, value, color):
    """QuickPose-style: arc + degree value between the ear vector and the
    muzzle midline, drawn at the ear base. The number equals the chart's."""
    base = np.asarray(base, float)
    b_i = tuple(base.astype(int))
    ue = v_ear / (np.linalg.norm(v_ear) + 1e-6)
    cv2.line(img, b_i, tuple((base + ue * 62).astype(int)), color, 2)   # ear vector
    a_ear = np.degrees(np.arctan2(v_ear[1], v_ear[0]))
    a_mid = np.degrees(np.arctan2(v_mid[1], v_mid[0]))
    diff = (a_mid - a_ear + 180) % 360 - 180                            # signed minor arc
    cv2.ellipse(img, b_i, (32, 32), 0, a_ear, a_ear + diff, color, 2)
    bis = np.radians(a_ear + diff / 2)
    lp = (base + np.array([np.cos(bis), np.sin(bis)]) * 52).astype(int)
    cv2.putText(img, f"{abs(value):.0f}", tuple(lp), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def plot_panel(ts, L, R, Lb, Rb, i, h, w):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ll = f"left  {L[i]:5.0f}°" if not np.isnan(L[i]) else "left   —"
    rl = f"right {R[i]:5.0f}°" if not np.isnan(R[i]) else "right  —"
    ax.axhline(Lb, color="#1f77b4", lw=1.0, ls=":", alpha=0.5)
    ax.axhline(Rb, color="#9467bd", lw=1.0, ls=":", alpha=0.5)
    ax.plot(ts[:i + 1], L[:i + 1], color="#1f77b4", lw=1.8, label=ll)
    ax.plot(ts[:i + 1], R[:i + 1], color="#9467bd", lw=1.8, label=rl)
    ax.axvline(ts[i], color="0.4", lw=1, ls="--")
    ax.text(0.02, 0.04, "···  this ewe's baseline (her median, not a clinical threshold)",
            transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_xlim(0, float(ts[-1]))
    ax.set_ylim(60, 150)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ear angle |deg|")
    ax.set_title("ear angle — live (v0.7)", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def main():
    model = YOLO(str(WEIGHTS))
    per_frame = []   # per video frame: {track_id: (box_xyxy, kpts)}
    for r in model.track(source=str(CLIP), tracker="bytetrack.yaml",
                         stream=True, conf=0.25, verbose=False):
        d = {}
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xy = r.boxes.xyxy.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy()
            for j, tid in enumerate(ids):
                d[int(tid)] = (xy[j], kp[j])
        per_frame.append(d)
    N = len(per_frame)

    # which track is the calm ewe? the one most in-ROI during WINDOW
    c = Counter()
    for i in range(WINDOW[0], min(WINDOW[1], N)):
        for tid, (box, kp) in per_frame[i].items():
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]:
                c[tid] += 1
    if not c:
        raise SystemExit("no track found in ROI during window")
    target = c.most_common(1)[0][0]
    frames = [i for i in range(N) if target in per_frame[i]]
    print(f"target track id={target}: present in {len(frames)} frames "
          f"({len(frames) / FPS:.1f}s of {N / FPS:.1f}s clip)")

    L = np.array([ears(per_frame[f][target][1])[0] for f in frames])
    R = np.array([ears(per_frame[f][target][1])[1] for f in frames])
    ts = np.arange(len(frames)) / FPS
    Lb, Rb = float(np.nanmedian(L)), float(np.nanmedian(R))

    cap = cv2.VideoCapture(str(CLIP))
    out = ART / "v0.7-ear-angle-ekg-IMG_3651-long.mp4"
    vw = None
    f = -1
    fi = 0
    fset = set(frames)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1
        if f >= N or target not in per_frame[f] or f not in fset:
            continue
        box, kp = per_frame[f][target]
        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        x2 = min(frame.shape[1], max(CROP, cx + CROP // 2))
        y2 = min(frame.shape[0], max(CROP, cy + CROP // 2))
        x1, y1 = x2 - CROP, y2 - CROP
        crop = frame[y1:y2, x1:x2].copy()
        kc = kp[:, :2] - np.array([x1, y1])
        cf = kp[:, 2]
        pts = {j: tuple(kc[j].astype(int)) for j in range(5) if cf[j] > KPT_TH}
        for a, b in SKEL:
            if a in pts and b in pts:
                cv2.line(crop, pts[a], pts[b], KCOL, 2)
        for p in pts.values():
            cv2.circle(crop, p, 5, KCOL, -1)
        crop = cv2.resize(crop, (PANEL, PANEL))
        canvas = np.hstack([crop, plot_panel(ts, L, R, Lb, Rb, fi, PANEL, 720)])
        if vw is None:
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                 (canvas.shape[1], canvas.shape[0]))
        vw.write(canvas)
        fi += 1
    cap.release()
    if vw:
        vw.release()
    print(f"wrote {out}  ({fi} frames, {fi / FPS:.1f}s)")


if __name__ == "__main__":
    main()
