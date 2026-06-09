# test-clips

The actual test clips live in the repo-level `test-clips/` directory.

This directory is intentionally real (not a symlink) so fresh clones do not carry a broken `sheep-yolo/test-clips` symlink. From inside `sheep-yolo`, refer to clips with paths like:

```bash
../test-clips/IMG_3651.MOV
```
