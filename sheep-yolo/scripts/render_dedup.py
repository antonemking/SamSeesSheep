"""Keypoint-centroid de-duplication + clean re-render / black-sheep figure.

Dedup rule: two detections whose confident-keypoint centroids are within
DUP_PX are the same sheep -> keep the higher box-confidence one. Robust for a
dense flock (won't merge two real sheep, whose noses sit far apart), unlike
lowering NMS IoU which can fuse genuinely-adjacent animals.

`dedup()` is the reusable bit — port it into the barn-inference pipeline so
per-animal stats aren't double-counted.

  python render_dedup.py figure                                  # frame-457 black-sheep 3-up PNG
  python render_dedup.py video --pose weights/sheep-pose-v0.7-yolo26n.pt   # clean deduped clip -> mp4
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT.parent / "test-clips" / "IMG_3651.MOV"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

DUP_PX = 30          # centroids closer than this = same sheep
KPT_TH = 0.5         # keypoint draw / confidence threshold
CONF = 0.25          # detector confidence (matches the benches)
# skeleton: nose-Lbase, nose-Rbase, Lbase-Ltip, Rbase-Rtip
SKEL = [(0, 1), (0, 2), (1, 3), (2, 4)]

MODELS = {
    "v0.4": (ROOT / "weights" / "sheep-pose-v0.4-yolo26n.pt", (255, 0, 255)),    # magenta
    "v0.5": (ROOT / "weights" / "sheep-pose-v0.5-yolo26n.pt", (255, 165, 0)),    # light blue
    "v0.7": (ROOT / "weights" / "sheep-pose-v0.7-yolo26n.pt", (0, 210, 80)),     # green
}


def dedup(xyxy, kpts, bconf, dup_px=DUP_PX):
    """Return kept detection indices after keypoint-centroid de-duplication."""
    order = np.argsort(-bconf)
    kept, cents = [], []
    for i in order:
        vis = kpts[i][kpts[i][:, 2] > KPT_TH]
        if len(vis):
            c = vis[:, :2].mean(0)
        else:
            c = np.array([(xyxy[i][0] + xyxy[i][2]) / 2, (xyxy[i][1] + xyxy[i][3]) / 2])
        if all(np.hypot(*(c - kc)) >= dup_px for kc in cents):
            kept.append(i)
            cents.append(c)
    return kept


def draw_pose(img, kpts, color, r=4, t=2):
    for a, b in SKEL:
        if kpts[a][2] > KPT_TH and kpts[b][2] > KPT_TH:
            cv2.line(img, tuple(kpts[a][:2].astype(int)), tuple(kpts[b][:2].astype(int)), color, t)
    for k in kpts:
        if k[2] > KPT_TH:
            cv2.circle(img, tuple(k[:2].astype(int)), r, color, -1)


def infer(model, img):
    r = model.predict(img, conf=CONF, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0, 5, 3)), np.zeros(0)
    return (r.boxes.xyxy.cpu().numpy(),
            r.keypoints.data.cpu().numpy(),
            r.boxes.conf.cpu().numpy())


def cmd_video(pose_path):
    model = YOLO(str(pose_path))
    color = (0, 210, 80)   # green, matches v0.7 in the figure
    cap = cv2.VideoCapture(str(CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    stem = Path(pose_path).stem
    out = ART / f"{stem}-dedup-on-{CLIP.stem}.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = removed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        xy, kp, bc = infer(model, frame)
        keep = dedup(xy, kp, bc)
        removed += (len(xy) - len(keep))
        for i in keep:
            draw_pose(frame, kp[i], color)
        cv2.putText(frame, f"{stem}  deduped: {len(keep)} sheep",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        vw.write(frame)
        n += 1
    cap.release()
    vw.release()
    print(f"wrote {out}  ({n} frames, {removed} duplicate detections removed)")


def cmd_figure(frame_idx=457):
    cap = cv2.VideoCapture(str(CLIP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_idx}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # locate the black sheep via v0.7's darkest deduped box
    m7 = YOLO(str(MODELS["v0.7"][0]))
    xy, kp, bc = infer(m7, img)
    keep = dedup(xy, kp, bc)
    cand = []
    for i in keep:
        x1, y1, x2, y2 = [int(v) for v in xy[i]]
        cr = gray[max(0, y1):y2, max(0, x1):x2]
        cand.append((float(cr.mean()) if cr.size else 255, i))
    _, bi = min(cand)
    bx = xy[bi]
    bcx, bcy = int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)
    pad = 190
    cx1, cy1 = max(0, bcx - pad), max(0, bcy - pad)
    cx2, cy2 = min(img.shape[1], bcx + pad), min(img.shape[0], bcy + pad)

    panels = []
    for lab, (w, color) in MODELS.items():
        canvas = img.copy()
        xy, kp, bc = infer(YOLO(str(w)), canvas)
        keep = dedup(xy, kp, bc)
        for i in keep:
            draw_pose(canvas, kp[i], color, r=5, t=3)
        # status on the black sheep: nearest deduped detection to its centre
        status, kc = "MISSED", 0.0
        if keep:
            best = min(keep, key=lambda i: (((xy[i][0]+xy[i][2])/2 - bcx) ** 2 +
                                            ((xy[i][1]+xy[i][3])/2 - bcy) ** 2))
            d = (((xy[best][0]+xy[best][2])/2 - bcx) ** 2 +
                 ((xy[best][1]+xy[best][3])/2 - bcy) ** 2) ** 0.5
            nk = int((kp[best][:, 2] > KPT_TH).sum())
            if d < 60 and nk >= 3:
                status, kc = "DETECTED", float(kp[best][:, 2].mean())
        crop = canvas[cy1:cy2, cx1:cx2].copy()
        crop = cv2.resize(crop, (480, 480))
        cv2.rectangle(crop, (0, 0), (479, 70), (0, 0, 0), -1)
        cv2.putText(crop, lab, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        txt = f"black sheep: {status}" + (f" (kpt {kc:.2f})" if status == "DETECTED" else "")
        cv2.putText(crop, txt, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 230, 0) if status == "DETECTED" else (0, 0, 230), 2)
        panels.append(crop)

    out = ART / f"black-sheep-frame{frame_idx}-3way.png"
    cv2.imwrite(str(out), np.hstack(panels))
    print(f"black sheep @({bcx},{bcy}) brightness={min(c for c, _ in cand):.0f}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("figure")
    v = sub.add_parser("video")
    v.add_argument("--pose", required=True)
    args = ap.parse_args()
    if args.cmd == "figure":
        cmd_figure()
    else:
        cmd_video(args.pose if Path(args.pose).is_absolute() else ROOT / args.pose)


if __name__ == "__main__":
    main()
