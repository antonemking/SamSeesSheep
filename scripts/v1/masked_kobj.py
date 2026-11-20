"""Make Ultralytics pose training honour MASKED keypoint slots (kpt_mask.json from export_yolo.py).

Why this exists. Ultralytics has three states per keypoint (v = 0, 1, 2). For v = 0 it skips the position loss
but still trains the visibility output towards "not there" (the kobj term: BCE against `v != 0`). A slot the
labeler skipped on purpose would therefore teach the model that the part is absent. This hook removes masked
slots from that BCE, so they carry no position loss and no absent signal.

Use, in the training script, before `model.train(...)`:

    import masked_kobj
    hook = masked_kobj.install("train/data/v1.0-b/kpt_mask.json")
    YOLO("yolo26n-pose.pt").train(data="train/data/v1.0-b/data.yaml", ...)
    hook.require_hits()            # raises if no masked image ever reached the loss

Limits, stated:
- Single process only. Multi-GPU (DDP) training starts fresh interpreters that do not carry the patch.
- Images are matched by file stem, so stems must be unique in the dataset (the exporter's are).
- The masked set must be closed under the dataset's flip_idx (the exporter checks), because a horizontal flip
  permutes slots after the mask was written.
- It relies on three facts about ultralytics.utils.loss (checked against 8.4.180): the pose criterion has
  `loss(preds, batch)` with `batch["im_file"]`, `calculate_keypoints_loss(fg_mask, ...)`, and calls
  `self.bce_pose(pred_visibility, target)` with tensors of shape (foreground anchors, keypoints). Shapes are
  asserted on every call; a version that differs fails loudly instead of training wrong.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class Hook:
    def __init__(self, images: dict[str, list[int]]):
        self.images = {k: sorted(set(v)) for k, v in images.items() if v}
        self.hits = 0                 # masked images seen by the loss
        self.calls = 0                # loss calls
        self._undo = []

    def ignore_for(self, im_files, nkpt, torch, device):
        """(batch, nkpt) bool tensor: True where the slot is masked for that image."""
        ig = torch.zeros((len(im_files), nkpt), dtype=torch.bool, device=device)
        for b, f in enumerate(im_files):
            slots = self.images.get(Path(str(f)).stem)
            if slots:
                if max(slots) >= nkpt:
                    raise RuntimeError(f"kpt_mask slot {max(slots)} is outside kpt_shape[0]={nkpt}")
                ig[b, slots] = True
                self.hits += 1
        return ig

    def require_hits(self):
        if self.images and not self.hits:
            raise RuntimeError("kpt_mask.json lists masked images but none reached the loss: the mask was not applied")

    def uninstall(self):
        for cls, name, orig in reversed(self._undo):
            setattr(cls, name, orig)
        self._undo.clear()


def masked_bce(pred, target, ignore, torch):
    """Mean BCE-with-logits over the entries that are not ignored. All ignored -> a zero that keeps the graph."""
    if pred.shape != target.shape or pred.shape != ignore.shape:
        raise RuntimeError(f"masked kobj: shapes differ, pred {tuple(pred.shape)} target {tuple(target.shape)} "
                           f"ignore {tuple(ignore.shape)}; this Ultralytics version is not supported by the hook")
    keep = ~ignore
    if not keep.any():
        return pred.sum() * 0.0
    per = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return per[keep].mean()


def load_mask(path) -> dict[str, list[int]]:
    doc = json.loads(Path(path).read_text())
    return doc["images"] if "images" in doc else doc


def install(mask_json: str | os.PathLike) -> Hook:
    import torch
    from ultralytics.utils import loss as L

    hook = Hook(load_mask(mask_json))
    classes = [c for c in (getattr(L, "v8PoseLoss", None), getattr(L, "PoseLoss26", None)) if c is not None]
    if not classes:
        raise RuntimeError("no pose loss class found in ultralytics.utils.loss")

    def patch(cls):
        if "loss" not in cls.__dict__ or "calculate_keypoints_loss" not in cls.__dict__:
            raise RuntimeError(f"{cls.__name__} lacks loss/calculate_keypoints_loss; hook not supported here")
        orig_loss, orig_ckl = cls.__dict__["loss"], cls.__dict__["calculate_keypoints_loss"]

        def loss(self, preds, batch, *a, **k):
            if "im_file" not in batch:
                raise RuntimeError("masked kobj: batch has no 'im_file'; cannot tell which images are masked")
            hook.calls += 1
            self._mk_ignore = hook.ignore_for(batch["im_file"], self.kpt_shape[0], torch, self.device)
            if not getattr(self, "_mk_wrapped", False):
                stock = self.bce_pose

                def bce(pred, target, _self=self, _stock=stock):
                    ig = getattr(_self, "_mk_fg_ignore", None)
                    if ig is None or not ig.any():
                        return _stock(pred, target)
                    return masked_bce(pred, target, ig, torch)

                self.bce_pose, self._mk_wrapped = bce, True
            try:
                return orig_loss(self, preds, batch, *a, **k)
            finally:
                self._mk_ignore = self._mk_fg_ignore = None

        def calculate_keypoints_loss(self, masks, *a, **k):
            ig = getattr(self, "_mk_ignore", None)
            self._mk_fg_ignore = None if ig is None else ig[masks.nonzero(as_tuple=True)[0]]
            return orig_ckl(self, masks, *a, **k)

        for name, new, orig in (("loss", loss, orig_loss), ("calculate_keypoints_loss", calculate_keypoints_loss, orig_ckl)):
            setattr(cls, name, new)
            hook._undo.append((cls, name, orig))

    for c in classes:
        patch(c)
    return hook
