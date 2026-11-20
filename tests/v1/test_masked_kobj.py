"""The masked-slot loss hook. Needs torch; the last test also needs ultralytics. No optimizer, no weights:
one loss evaluation on a randomly initialised model and random pixels."""
import json

import pytest

torch = pytest.importorskip("torch")

import masked_kobj as mk  # noqa: E402

EDGE = list(range(27, 35))


def test_masked_bce_ignores_masked_entries_and_gives_them_no_gradient():
    torch.manual_seed(0)
    pred = torch.randn(5, 36, requires_grad=True)
    target = torch.ones(5, 36)
    target[:, EDGE] = 0                                   # what a stock run would call "absent"
    ignore = torch.zeros(5, 36, dtype=torch.bool)
    ignore[:3, EDGE] = True                               # three masked heads, two ordinary ones
    loss = mk.masked_bce(pred, target, ignore, torch)
    loss.backward()
    assert torch.all(pred.grad[:3, EDGE] == 0)            # no absent signal on masked slots
    assert torch.all(pred.grad[3:, EDGE] != 0)            # ordinary heads still learn "absent"
    keep = ~ignore
    ref = torch.nn.functional.binary_cross_entropy_with_logits(pred.detach()[keep], target[keep])
    assert torch.isclose(loss.detach(), ref)


def test_masked_bce_with_everything_masked_is_zero_and_shapes_are_checked():
    pred = torch.randn(2, 4, requires_grad=True)
    out = mk.masked_bce(pred, torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool), torch)
    assert out.item() == 0.0 and out.requires_grad
    with pytest.raises(RuntimeError, match="shapes differ"):
        mk.masked_bce(pred, torch.zeros(2, 4), torch.ones(2, 3, dtype=torch.bool), torch)


def test_hook_maps_image_files_to_slots_and_demands_hits():
    h = mk.Hook({"Sheep_at_night_00030_h0": EDGE})
    ig = h.ignore_for(["/d/images/train/Sheep_at_night_00030_h0.jpg", "/d/images/train/IMG_3656_00030_h0.jpg"],
                      36, torch, "cpu")
    assert ig[0, EDGE].all() and ig[0].sum() == 8 and not ig[1].any() and h.hits == 1
    idle = mk.Hook({"x": [1]})
    with pytest.raises(RuntimeError, match="not applied"):
        idle.require_hits()
    with pytest.raises(RuntimeError, match="outside kpt_shape"):
        mk.Hook({"x": [40]}).ignore_for(["x.jpg"], 36, torch, "cpu")


def test_hook_changes_only_the_visibility_loss_of_masked_images(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import PoseModel

    torch.manual_seed(0)
    model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(36, 3), verbose=False)
    model.args = get_cfg()
    model.train()
    kp = torch.rand(2, 36, 3) * 0.5 + 0.25
    kp[..., 2] = 2
    kp[:, EDGE, :] = 0                                    # both heads: edge slots are 0 0 0 in the row
    batch = {"img": torch.rand(2, 3, 320, 320), "cls": torch.zeros(2, 1), "batch_idx": torch.tensor([0.0, 1.0]),
             "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 2), "keypoints": kp,
             "im_file": ["/d/images/train/night_h0.jpg", "/d/images/train/day_h0.jpg"]}

    def items():
        model.criterion = None
        torch.manual_seed(1)
        _, it = model.loss(dict(batch))
        return it

    stock = items()
    path = tmp_path / "kpt_mask.json"
    path.write_text(json.dumps({"images": {"night_h0": EDGE}}))
    hook = mk.install(path)
    try:
        masked = items()
        assert hook.hits >= 1 and hook.calls >= 1
        hook.require_hits()
    finally:
        hook.uninstall()
    after = items()
    assert "kobj_loss" in stock and len(stock) >= 5
    for n in stock:
        if n == "kobj_loss":
            assert not torch.isclose(masked[n], stock[n])          # the masked head no longer says "absent"
            assert torch.isfinite(masked[n])
        else:
            assert torch.isclose(masked[n], stock[n]), n           # nothing else moves
    assert all(torch.isclose(after[n], stock[n]) for n in stock)   # uninstall restores the stock loss

    path.write_text(json.dumps({"images": {"some_other_image": EDGE}}))
    hook = mk.install(path)
    try:
        same = items()
        assert all(torch.isclose(same[n], stock[n]) for n in stock)   # no masked image in the batch -> stock loss
        with pytest.raises(RuntimeError, match="not applied"):
            hook.require_hits()
    finally:
        hook.uninstall()
