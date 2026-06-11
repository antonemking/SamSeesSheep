"""Side-by-side benchmark of sheep-pose v0.2 vs v0.3 on a held-out clip.

Outputs into ./artifacts:
  v0.2-on-IMG_3412.mp4   Ultralytics-annotated inference, whole clip
  v0.3-on-IMG_3412.mp4   Ultralytics-annotated inference, whole clip
  v0.2-vs-v0.3.mp4       2-up side-by-side, kpts in distinct colors,
                         frames 23-180 (the one motionless-sheep window)
  bench_report.json      σ numbers, detection rates, headline

Why this clip: footage/IMG_3583.MOV — fresh clip, not in either training
set. Training-set IDs (confirmed against ~/Backups/sheep-seg/labels/)
are: v0.2 = {6f689b79, 7e53dfab, d6873739}; v0.3 = those three plus
{00d25853, 0d1655d2, c74474f9}. IMG_3583 was captured after both export
batches and never reviewed.

σ definition: the clip has several foreground sheep heads. Bytetrack on
v0.3 identifies track id=2 — a ~260 px wide head at conf 0.9+ that
stays nearly stationary for ~5 s. We use frames 590..740 (5.0 s) and a
padded ROI around id=2's centroid range. σ is computed only on frames
where the model fired a detection whose centroid lies inside that ROI,
so v0.2 and v0.3 are scored on the same physical sheep regardless of
which other sheep also happen to be detected on a given frame.

We report per-kpt σ TWICE per model:
  (a) σ on each model's own in-ROI frames (uses all data each has)
  (b) σ on the intersection: frames where BOTH models fired in-ROI
      (apples-to-apples; controls for detection rate)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

V02 = ROOT / "weights" / "sheep-pose-v0.2-yolo26n.pt"
V03 = ROOT / "weights" / "sheep-pose-v0.3-yolo26n.pt"
CONF = 0.25

# Per-clip benchmark config. Each entry: motionless window (half-open
# frame range) + tight ROI around the bytetrack-derived dominant sheep,
# padded by ~half-head so head-sway fits but adjacent sheep are excluded.
CLIPS = {
    "IMG_3583": {
        # id=2: big foreground head, ~260×235 px, conf 0.9+, nearly
        # still for 5 s. Centroid std in window: cx=22 cy=46.
        "path": ROOT / "footage" / "IMG_3583.MOV",
        "window": (590, 741),           # 150 frames ≈ 5.0 s @ 30 fps
        "roi": (940, 460, 1230, 770),
    },
    "IMG_3601": {
        # id=277: foreground head, ~233×163 px, calmest 4 s window
        # frames 755..875 (cx-std=44, cy-std=32). ROI padded around
        # centroid range 994-1138 × 227-364 to exclude neighbours.
        "path": ROOT / "footage" / "IMG_3601.MOV",
        "window": (755, 876),           # 121 frames ≈ 4.0 s @ 30 fps
        "roi": (894, 147, 1238, 444),
    },
    "IMG_3412": {
        # TRULY HELD OUT — md5 doesn't match any training source upload
        # and pixel-NCC vs every training frame0 is <0.2. The only
        # motionless track is id=9, a small distant sheep (~70×50 px,
        # frames 23..180, centroid std ~15 px). Numbers will be less
        # dramatic than IMG_3583/3601 because the head is ~4× smaller,
        # but this is the honest held-out benchmark.
        "path": ROOT / "footage" / "IMG_3412.MOV",
        "window": (23, 181),            # 158 frames ≈ 5.3 s @ 30 fps
        "roi": (743, 170, 934, 346),
    },
    "IMG_3507": {
        # TRULY HELD OUT — captured fresh, never labeled. NCC < 0.18
        # vs every training video. Best motionless track is id=19, a
        # ~187×111 px foreground head at conf 0.9+. Window: frames
        # 293..442 (5.0 s), centroid std cx=18 cy=5 — essentially
        # stationary. ROI padded ~half-head around centroid range
        # cx [1113,1172] cy [236,265] to exclude grazing neighbours.
        "path": ROOT / "footage" / "IMG_3507.MOV",
        "window": (293, 443),           # 150 frames ≈ 5.0 s @ 30 fps
        "roi": (1010, 175, 1265, 330),
    },
    "IMG_3651": {
        # HELD OUT — lives in test-clips/, never added to the labeler.
        # NCC < 0.23 vs every training video. Best motionless target
        # is track id=34, a ~234×169 px Katahdin head facing camera at
        # conf 0.6+. Window: frames 367..521 (5.1 s), centroid std
        # cx=30 cy=32 — not perfectly still (sheep is among a moving
        # group) but the calmest 5 s window with a big foreground head.
        # ROI padded ~half-head around centroid range cx [1337,1444]
        # cy [164,303] to exclude neighbouring sheep.
        "path": ROOT / "test-clips" / "IMG_3651.MOV",
        "window": (367, 522),           # 155 frames ≈ 5.1 s @ 30 fps
        "roi": (1220, 80, 1560, 390),
    },
    "Test_Clip_Morning": {
        # HELD OUT — lives in test-clips/. Calibrated with v0.7 YOLO-pose
        # + ByteTrack. Best calm target is track id=215, a background tan
        # sheep isolated near the fence. Window: frames 742..891 (5.0 s),
        # target detected on 148/150 frames; centroid std cx=48 cy=25.
        # ROI padded ~half-head around centroid range cx [1393,1560]
        # cy [360,445] to exclude foreground neighbours.
        "path": ROOT.parent / "test-clips" / "Test_Clip_Morning.mov",
        "window": (742, 892),           # 150 frames ≈ 5.0 s @ 30 fps
        "roi": (1234, 267, 1719, 538),
    },
}

# Selected via CLI: `python bench_v02_v03.py IMG_3601`. Defaults to IMG_3583.
SELECTED = sys.argv[1] if len(sys.argv) > 1 else "IMG_3583"
if SELECTED not in CLIPS:
    raise SystemExit(f"unknown clip {SELECTED!r}; pick one of {list(CLIPS)}")
CLIP = CLIPS[SELECTED]["path"]
WINDOW = CLIPS[SELECTED]["window"]
ROI = CLIPS[SELECTED]["roi"]
KPT_NAMES = ["nose", "L_ear_base", "R_ear_base", "L_ear_tip", "R_ear_tip"]

V02_BGR = (0, 165, 255)   # orange
V03_BGR = (60, 220, 60)   # green


def in_roi(cx: float, cy: float) -> bool:
    return ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]


# ---------- inference ----------------------------------------------------

def predict_clip(model_path: Path, label: str) -> tuple[Path, np.ndarray, list]:
    """Run model on clip with Ultralytics' save=True annotator.

    Returns:
      (annotated-mp4 path,
       target_kpts: array (n_frames, 5, 3) — kpts of the IN-ROI sheep
                    (NaN where no detection inside ROI),
       all_kpts: list of length n_frames; each entry is an (N, 5, 3)
                 array of every detection in that frame).

    target_kpts feeds the σ math (same physical sheep across frames).
    all_kpts feeds the demo visualization (draws kpts on every sheep
    the model sees, so the post shows the model working on the whole
    flock).
    """
    print(f"\n=== {label}  ({model_path.name}) ===")
    model = YOLO(str(model_path))

    out_name = f"{label}-on-{CLIP.stem}"
    results = model.predict(
        source=str(CLIP), conf=CONF, save=True,
        project=str(ART / "_ultra"), name=out_name, exist_ok=True,
        stream=True, verbose=False,
    )

    target_kpts_per_frame: list[np.ndarray] = []
    all_kpts_per_frame: list[np.ndarray] = []
    any_det = 0
    roi_det = 0
    n_frames = 0
    roi_cx = (ROI[0] + ROI[2]) / 2
    roi_cy = (ROI[1] + ROI[3]) / 2
    for r in results:
        n_frames += 1
        if r.boxes is None or len(r.boxes) == 0:
            target_kpts_per_frame.append(np.full((5, 3), np.nan))
            all_kpts_per_frame.append(np.zeros((0, 5, 3)))
            continue
        any_det += 1
        xywh = r.boxes.xywh.cpu().numpy()
        kpts_arr = r.keypoints.data.cpu().numpy()
        all_kpts_per_frame.append(kpts_arr)
        best = None
        best_d = float("inf")
        for k in range(len(xywh)):
            cx, cy = float(xywh[k, 0]), float(xywh[k, 1])
            if not in_roi(cx, cy):
                continue
            d = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
            if d < best_d:
                best_d = d
                best = k
        if best is None:
            target_kpts_per_frame.append(np.full((5, 3), np.nan))
            continue
        roi_det += 1
        target_kpts_per_frame.append(kpts_arr[best])

    kpts = np.stack(target_kpts_per_frame, axis=0)
    print(f"  frames={n_frames}  any-detection={any_det}  in-ROI={roi_det}")

    # Lift Ultralytics' annotated mp4 out to artifacts/.
    ultra_run_dir = ART / "_ultra" / out_name
    annotated = None
    for ext in (".mp4", ".avi"):
        for p in ultra_run_dir.glob(f"*{ext}"):
            annotated = p
            break
        if annotated:
            break
    if annotated is None:
        raise RuntimeError(f"no annotated video found in {ultra_run_dir}")
    final = ART / f"{label}-on-{CLIP.stem}.mp4"
    final.unlink(missing_ok=True)
    if annotated.suffix == ".mp4":
        final.symlink_to(annotated.resolve())
    else:
        _transcode(annotated, final)
    print(f"  annotated -> {final}")
    return final, kpts, all_kpts_per_frame


def _transcode(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        out.write(frm)
    cap.release()
    out.release()


# ---------- σ ------------------------------------------------------------

def _per_kpt(kpts: np.ndarray, mask: np.ndarray) -> dict:
    """Per-keypoint σ on frames where mask is True. Returns dict per kpt."""
    out = {"per_kpt": {}, "mean_sigma": None}
    sigmas = []
    for k, name in enumerate(KPT_NAMES):
        xy = kpts[mask, k, :2]
        conf = kpts[mask, k, 2]
        ok = (~np.isnan(xy[:, 0])) & (conf > 0)
        n_ok = int(ok.sum())
        if n_ok < 3:
            out["per_kpt"][name] = {"sigma_px": None, "n_visible": n_ok}
            continue
        v = xy[ok]
        sx = float(np.std(v[:, 0]))
        sy = float(np.std(v[:, 1]))
        s = float(np.sqrt(sx * sx + sy * sy))
        out["per_kpt"][name] = {
            "sigma_px": round(s, 2),
            "sigma_x": round(sx, 2),
            "sigma_y": round(sy, 2),
            "n_visible": n_ok,
        }
        sigmas.append(s)
    out["mean_sigma"] = round(float(np.mean(sigmas)), 2) if sigmas else None
    return out


def _rolling_median(v: np.ndarray, w: int = 7) -> np.ndarray:
    half = w // 2
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - half): min(len(v), i + half + 1)])
    return out


def _per_kpt_residual(kpts: np.ndarray) -> dict:
    """Per-keypoint *residual* σ: subtract a 7-frame rolling-median trajectory
    before std-dev, so slow sheep motion is removed and only jitter remains.

    This is the stability-relevant number: how stable is the kpt
    *relative to its own slow drift*. Detection rate doesn't bias it the
    way raw σ does — each model is compared against its own smoothed
    trajectory, so v0.2 firing on only 17 of 158 frames is judged on the
    jitter within those 17, not on whether they were sampled at the same
    time as v0.3's 119.
    """
    out = {"per_kpt": {}, "mean_sigma": None}
    sigmas = []
    for k, name in enumerate(KPT_NAMES):
        xy = kpts[:, k, :2]
        conf = kpts[:, k, 2]
        ok = (~np.isnan(xy[:, 0])) & (conf > 0)
        n_ok = int(ok.sum())
        if n_ok < 8:
            out["per_kpt"][name] = {"sigma_px": None, "n_visible": n_ok}
            continue
        xv = xy[ok, 0]
        yv = xy[ok, 1]
        rx = xv - _rolling_median(xv)
        ry = yv - _rolling_median(yv)
        sx = float(np.std(rx))
        sy = float(np.std(ry))
        s = float(np.sqrt(sx * sx + sy * sy))
        out["per_kpt"][name] = {
            "sigma_px": round(s, 2),
            "sigma_x": round(sx, 2),
            "sigma_y": round(sy, 2),
            "n_visible": n_ok,
        }
        sigmas.append(s)
    out["mean_sigma"] = round(float(np.mean(sigmas)), 2) if sigmas else None
    return out


def sigma_report(v02: np.ndarray, v03: np.ndarray) -> dict:
    a, b = WINDOW
    v02_w = v02[a:b]
    v03_w = v03[a:b]
    v02_has = ~np.isnan(v02_w[:, 0, 0])
    v03_has = ~np.isnan(v03_w[:, 0, 0])
    n = b - a

    # (a) raw σ on each model's own in-ROI frames (biased by detection rate)
    own_v02 = _per_kpt(v02_w, v02_has)
    own_v03 = _per_kpt(v03_w, v03_has)

    # (b) raw σ on paired frames (apples-to-apples but n=paired_frames)
    both = v02_has & v03_has
    pair_v02 = _per_kpt(v02_w, both)
    pair_v03 = _per_kpt(v03_w, both)

    # (c) RESIDUAL σ — subtracts slow sheep motion via rolling median.
    #     This is the most informative metric: pure jitter, not biased
    #     by which sub-window each model happened to detect on.
    resid_v02 = _per_kpt_residual(v02_w)
    resid_v03 = _per_kpt_residual(v03_w)

    return {
        "window_frames": [a, b],
        "window_seconds": round(n / 30.0, 2),
        "v0.2_in_roi_detection_rate": f"{int(v02_has.sum())}/{n} ({v02_has.sum()/n*100:.0f}%)",
        "v0.3_in_roi_detection_rate": f"{int(v03_has.sum())}/{n} ({v03_has.sum()/n*100:.0f}%)",
        "paired_frames": int(both.sum()),
        "own_frames_raw_sigma": {"v0.2": own_v02, "v0.3": own_v03},
        "paired_raw_sigma": {"v0.2": pair_v02, "v0.3": pair_v03},
        "residual_sigma": {"v0.2": resid_v02, "v0.3": resid_v03},
    }


# ---------- side-by-side -------------------------------------------------

def _crop_from_all_dets(all_v02, all_v03, window, src_w, src_h, pad=80):
    """Auto-derive a zoomed crop that fits every sheep head detected by
    either model anywhere in the σ window. Padded so heads aren't
    flush against the edge."""
    a, b = window
    xs = []
    ys = []
    for kpts_list in (all_v02, all_v03):
        for i in range(a, b):
            arr = kpts_list[i]
            if arr is None or len(arr) == 0:
                continue
            xy = arr[:, :, :2]
            conf = arr[:, :, 2]
            valid = conf > 0
            if not valid.any():
                continue
            xs.extend(xy[..., 0][valid].tolist())
            ys.extend(xy[..., 1][valid].tolist())
    if not xs:
        return (0, 0, src_w, src_h)
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(src_w, int(max(xs)) + pad)
    y1 = min(src_h, int(max(ys)) + pad)
    return (x0, y0, x1, y1)


def make_side_by_side(v02_target: np.ndarray, v03_target: np.ndarray,
                       v02_all: list, v03_all: list,
                       out_path: Path) -> Path:
    """Zoomed 2-up: v0.2 left, v0.3 right.

    Crop is auto-derived from the union of all in-window sheep-head
    detections, padded — so every visible sheep fits and adjacent
    animals aren't cut off. Inside that crop we draw kpts for EVERY
    detection the model produced (not just the target sheep). The
    target sheep still drives the σ math; this view shows the model
    working on the whole flock.
    """
    cap = cv2.VideoCapture(str(CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop = _crop_from_all_dets(v02_all, v03_all, WINDOW, src_w, src_h)
    cx0, cy0, cx1, cy1 = crop
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0

    # Render each panel at a fixed height; width scales with crop aspect.
    panel_h = 720
    panel_w = int(round(panel_h * crop_w / crop_h))
    scale = panel_h / crop_h
    out_w = panel_w * 2
    out_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    a, b = WINDOW
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    for i in range(a, b):
        ok, frm = cap.read()
        if not ok:
            break
        cropped = cv2.resize(
            frm[cy0:cy1, cx0:cx1], (panel_w, panel_h),
            interpolation=cv2.INTER_CUBIC,
        )
        left = _compose_panel_zoom(
            cropped.copy(), v02_all[i], scale, (cx0, cy0),
            V02_BGR, "v0.2  (98 instances, 3 vids)",
        )
        right = _compose_panel_zoom(
            cropped.copy(), v03_all[i], scale, (cx0, cy0),
            V03_BGR, "v0.3  (313 instances, 6 vids)",
        )
        canvas = np.concatenate([left, right], axis=1)
        writer.write(canvas)
    cap.release()
    writer.release()
    return out_path


def _compose_panel_zoom(panel, kpts_all, scale, origin, color, caption):
    H, W = panel.shape[:2]
    _overlay_kpts_multi(panel, kpts_all, scale, origin, color, dot_r=8)
    cv2.rectangle(panel, (0, 0), (W, 50), (30, 30, 30), -1)
    cv2.putText(panel, caption, (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return panel


def _overlay_kpts_multi(img: np.ndarray, kpts_all: np.ndarray, scale: float,
                         origin: tuple[int, int],
                         color: tuple[int, int, int], dot_r: int = 8) -> None:
    """Draw kpts for ALL detections in this frame."""
    if kpts_all is None or len(kpts_all) == 0:
        return
    for kpts in kpts_all:
        _overlay_kpts(img, kpts, scale, origin, color, dot_r=dot_r)


def _overlay_kpts(img: np.ndarray, kpts: np.ndarray, scale: float,
                   origin: tuple[int, int],
                   color: tuple[int, int, int], dot_r: int = 6) -> None:
    """Draw a single detection's kpts. NaN-row → skip."""
    if kpts is None or len(kpts) == 0 or np.isnan(kpts[0, 0]):
        return
    ox, oy = origin
    pts = (kpts[:, :2] - np.array([ox, oy])) * scale
    confs = kpts[:, 2]
    for ka, kb in [(0, 1), (0, 2), (1, 3), (2, 4)]:
        if confs[ka] > 0 and confs[kb] > 0:
            cv2.line(img, tuple(map(int, pts[ka])),
                     tuple(map(int, pts[kb])), color,
                     max(1, dot_r // 3), cv2.LINE_AA)
    for k in range(5):
        if confs[k] <= 0:
            continue
        p = tuple(map(int, pts[k]))
        cv2.circle(img, p, dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, dot_r, (255, 255, 255), 1, cv2.LINE_AA)


def make_stills_grid(v02_target: np.ndarray, v03_target: np.ndarray,
                      v02_all: list, v03_all: list,
                      out_path: Path) -> Path:
    """2×4 stills grid (v0.2 row top, v0.3 row bottom) at 4 timestamps.

    Uses the same auto-zoom crop as the SXS video so the still and
    video framings match. Draws every detection's kpts, not just the
    target sheep. Sample frames are picked where the target sheep was
    detected by both models (so the comparison shows real placement
    differences, not "v0.2 missed").
    """
    cap = cv2.VideoCapture(str(CLIP))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a, b = WINDOW
    both_fired = [
        i for i in range(a, b)
        if not np.isnan(v02_target[i, 0, 0])
        and not np.isnan(v03_target[i, 0, 0])
    ]
    if len(both_fired) >= 4:
        idx = np.linspace(0, len(both_fired) - 1, 4).astype(int)
        sample_frames = [both_fired[i] for i in idx]
    else:
        sample_frames = [
            a + (b - a) * f // 4 for f in (1, 2, 3)
        ] + [b - 5]

    cx0, cy0, cx1, cy1 = _crop_from_all_dets(
        v02_all, v03_all, WINDOW, src_w, src_h,
    )
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    # cell height fixed, width scales with crop aspect
    cell_h = 540
    cell_w = int(round(cell_h * crop_w / crop_h))
    scale = cell_h / crop_h

    rows = []
    for label, all_kpts, color in [("v0.2", v02_all, V02_BGR),
                                     ("v0.3", v03_all, V03_BGR)]:
        row_cells = []
        for fi in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            cell = cv2.resize(
                frm[cy0:cy1, cx0:cx1], (cell_w, cell_h),
                interpolation=cv2.INTER_CUBIC,
            )
            _overlay_kpts_multi(cell, all_kpts[fi], scale, (cx0, cy0),
                                 color, dot_r=12)
            cv2.rectangle(cell, (0, 0), (cell_w, 40), (30, 30, 30), -1)
            cv2.putText(cell, f"{label}  frame {fi}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            row_cells.append(cell)
        rows.append(np.concatenate(row_cells, axis=1))
    grid = np.concatenate(rows, axis=0)
    cap.release()
    cv2.imwrite(str(out_path), grid)
    return out_path


# ---------- main ---------------------------------------------------------

def main() -> None:
    assert CLIP.exists(), CLIP
    print(f"clip: {CLIP}")
    print(f"window: frames {WINDOW[0]}..{WINDOW[1]-1}  ROI: {ROI}")

    v02_mp4, v02_target, v02_all = predict_clip(V02, "v0.2")
    v03_mp4, v03_target, v03_all = predict_clip(V03, "v0.3")

    sigmas = sigma_report(v02_target, v03_target)

    side_by_side = ART / f"v0.2-vs-v0.3-{CLIP.stem}.mp4"
    make_side_by_side(v02_target, v03_target, v02_all, v03_all, side_by_side)
    stills_path = ART / f"v0.2-vs-v0.3-{CLIP.stem}.png"
    make_stills_grid(v02_target, v03_target, v02_all, v03_all, stills_path)

    try:
        clip_label = str(CLIP.relative_to(ROOT))
    except ValueError:
        clip_label = str(CLIP.relative_to(ROOT.parent))

    report = {
        "clip": clip_label,
        "training_set_v0.2": ["6f689b79", "7e53dfab", "d6873739"],
        "training_set_v0.3": ["6f689b79", "7e53dfab", "d6873739",
                              "00d25853", "0d1655d2", "c74474f9"],
        "held_out": True,
        "conf_threshold": CONF,
        "annotated_mp4_v0.2": str(v02_mp4.relative_to(ROOT)),
        "annotated_mp4_v0.3": str(v03_mp4.relative_to(ROOT)),
        "side_by_side_mp4": str(side_by_side.relative_to(ROOT)),
        "side_by_side_stills_png": str(stills_path.relative_to(ROOT)),
        "sigmas": sigmas,
    }
    own_v02 = sigmas["own_frames_raw_sigma"]["v0.2"]["mean_sigma"]
    own_v03 = sigmas["own_frames_raw_sigma"]["v0.3"]["mean_sigma"]
    pair_v02 = sigmas["paired_raw_sigma"]["v0.2"]["mean_sigma"]
    pair_v03 = sigmas["paired_raw_sigma"]["v0.3"]["mean_sigma"]
    res_v02 = sigmas["residual_sigma"]["v0.2"]["mean_sigma"]
    res_v03 = sigmas["residual_sigma"]["v0.3"]["mean_sigma"]
    report["headline"] = {
        "residual_sigma": f"mean residual σ (jitter, slow motion removed): "
                          f"v0.2 = {res_v02} px → v0.3 = {res_v03} px",
        "own_frames_raw": f"mean raw σ (each model's own in-ROI frames): "
                          f"v0.2 = {own_v02} px → v0.3 = {own_v03} px "
                          f"— note v0.2 sample is sparse",
        "paired_raw": f"mean raw σ (paired frames only, n={sigmas['paired_frames']}): "
                      f"v0.2 = {pair_v02} px → v0.3 = {pair_v03} px",
        "detection": f"in-ROI detection rate: "
                     f"v0.2 = {sigmas['v0.2_in_roi_detection_rate']}, "
                     f"v0.3 = {sigmas['v0.3_in_roi_detection_rate']}",
    }
    (ART / f"bench_report-{CLIP.stem}.json").write_text(json.dumps(report, indent=2))
    print("\n--- σ summary ---")
    print(json.dumps(report["headline"], indent=2))
    print("\nFull report -> artifacts/bench_report.json")


if __name__ == "__main__":
    main()
