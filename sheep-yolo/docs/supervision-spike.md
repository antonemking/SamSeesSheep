# Roboflow `supervision` spike — LOR-123

Status: optional prototype, not a production rewrite.

## What was tested

`render_supervision_spike.py` runs the v0.7 YOLO-pose model over a bounded sample of `../test-clips/IMG_3651.MOV` and writes an annotated MP4 under `artifacts/`.

It exercises these Roboflow `supervision` 0.28.0 APIs:

- `sv.Detections.from_ultralytics(result)`
- `sv.KeyPoints.from_ultralytics(result)`
- `sv.VideoInfo.from_video_path(...)`
- `sv.get_video_frames_generator(...)`
- `sv.VideoSink(...)`
- `sv.VertexAnnotator`
- `sv.EdgeAnnotator`
- `sv.BoxAnnotator`
- `sv.PolygonZone`
- `sv.PolygonZoneAnnotator`

The ear-angle math is still SamSeesSheep-owned code reused from `render_ekg.py` via `ears()` and `SKEL`.

## How to run

From `sheep-yolo/`:

```bash
uv run --with supervision scripts/render_supervision_spike.py
uv run --with supervision scripts/render_supervision_spike.py \
  --start-frame 380 --max-frames 90 \
  --output artifacts/supervision-spike-IMG_3651-roi.mp4
```

The first command samples the start of the clip. The ROI count can be zero there because the calm ewe has not entered the ROI yet.

Use `--start-frame 380` to exercise the existing calm-ewe ROI.

## Observed results

Claude Code generated and ran the prototype with these observed checks:

- `py_compile` passed.
- Frame-0 90-frame sample wrote an MP4 and reported detections but zero inside the ROI, expected for that window.
- `--start-frame 380 --max-frames 90` wrote an ROI sample and reported non-zero ROI hits.
- `ffprobe` saw valid MP4 output.
- Generated artifacts live under `sheep-yolo/artifacts/`, which is gitignored.

Hermes independently verified:

- the prototype script compiles;
- README/doc diffs have no whitespace errors;
- generated MP4 artifacts are ignored by git;
- generated MP4 artifacts are valid ISO Media/MP4 files;
- `supervision` is not added to `pyproject.toml`.

## Recommendation

Adopt `supervision` only as an optional artifact/reporting layer for now.

Good fits:

- faster video artifact scripts;
- cleaner video IO with `VideoInfo`/frame generators/`VideoSink`;
- cleaner Ultralytics result normalization through `Detections` and `KeyPoints`;
- ROI/zone logic with `PolygonZone`, especially if SamSeesSheep or Lorewood Vision grows into multi-region paddock/evidence reports.

Not yet a fit:

- core training/export path;
- replacing the ear-angle geometry;
- replacing the live matplotlib EKG panel in `render_ekg.py`;
- tracking migration.

Important caveat: `sv.ByteTrack` is deprecated in supervision 0.28.0 and moving to the separate `trackers` package. This spike deliberately does not use `sv.ByteTrack`; `render_ekg.py` should keep using Ultralytics' `model.track(...)` path unless a separate tracking spike proves an alternative.
