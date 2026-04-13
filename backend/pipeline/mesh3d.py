"""3D mesh reconstruction using TripoSR.

Single-image to 3D mesh: takes a sheep photo, removes background,
runs TripoSR inference, exports GLB with vertex colors for Three.js.

TripoSR installed at: /home/toneking/dev/lorewood-advisors/TripoSR/
"""

from __future__ import annotations

import base64
import logging
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TRIPOSR_DIR = Path("/home/toneking/dev/lorewood-advisors/TripoSR")

_model = None
_rembg_session = None


def _ensure_triposr_path():
    """Add TripoSR to Python path."""
    tsr_path = str(TRIPOSR_DIR)
    if tsr_path not in sys.path:
        sys.path.insert(0, tsr_path)


def _load_model():
    """Load TripoSR model. First run downloads ~1.68GB weights."""
    global _model
    _ensure_triposr_path()

    try:
        import torch
        from tsr.system import TSR

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading TripoSR model on %s...", device)

        _model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        _model.renderer.set_chunk_size(8192)
        _model.to(device)
        logger.info("TripoSR model loaded.")
    except Exception as e:
        logger.error("Failed to load TripoSR: %s", e)
        _model = None


def _get_rembg_session():
    """Lazy-load rembg session for background removal."""
    global _rembg_session
    if _rembg_session is None:
        import rembg
        _rembg_session = rembg.new_session()
    return _rembg_session


def _crop_to_mask(
    image_bgr: np.ndarray, mask: np.ndarray, padding: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to the mask's bounding box with padding.

    Makes the crop square (TripoSR expects square input) and adds padding
    so the subject isn't clipped at the edges.
    """
    h, w = mask.shape[:2]
    mask_bool = mask > 127
    coords = np.where(mask_bool)
    if len(coords[0]) == 0:
        return image_bgr, mask

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    obj_h = y_max - y_min
    obj_w = x_max - x_min
    pad = int(max(obj_h, obj_w) * padding)

    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)

    # Make square
    crop_h = y_max - y_min
    crop_w = x_max - x_min
    size = max(crop_h, crop_w)
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2

    y1 = max(0, cy - size // 2)
    x1 = max(0, cx - size // 2)
    y2 = min(h, y1 + size)
    x2 = min(w, x1 + size)

    if y2 - y1 < size:
        y1 = max(0, y2 - size)
    if x2 - x1 < size:
        x1 = max(0, x2 - size)

    return image_bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def generate_mesh_glb(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    resolution: int = 256,
) -> bytes | None:
    """Generate a 3D mesh from a sheep photo.

    Args:
        image_bgr: BGR image from cv2
        mask: optional SAM head mask. If provided, used to crop and mask the subject.
        resolution: marching cubes resolution (higher = more detail, more VRAM)

    Returns:
        GLB bytes with vertex colors, ready for Three.js, or None on failure.
    """
    global _model
    _ensure_triposr_path()

    if _model is None:
        _load_model()
    if _model is None:
        return None

    import torch
    import trimesh
    from PIL import Image as PILImage
    from tsr.utils import remove_background, resize_foreground

    start = time.time()

    # Crop to mask bounding box — gives TripoSR a focused, clean input
    if mask is not None and mask.sum() > 0:
        image_cropped, mask_cropped = _crop_to_mask(image_bgr, mask)
    else:
        image_cropped = image_bgr
        mask_cropped = mask

    image_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(image_rgb)

    # Create RGBA: use SAM mask as alpha channel
    if mask_cropped is not None and mask_cropped.sum() > 0:
        mask_bool = mask_cropped > 127
        rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = image_rgb
        rgba[:, :, 3] = (mask_bool * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(rgba, "RGBA")
    else:
        session = _get_rembg_session()
        pil_image = remove_background(pil_image, session)

    # Resize foreground to center it
    pil_image = resize_foreground(pil_image, 0.85)

    # Convert RGBA to RGB with neutral grey background
    image_arr = np.array(pil_image).astype(np.float32) / 255.0
    alpha = image_arr[:, :, 3:4]
    image_arr = image_arr[:, :, :3] * alpha + (1 - alpha) * 0.5
    processed = PILImage.fromarray((image_arr * 255.0).astype(np.uint8))

    logger.info("Preprocessing done in %.1fs", time.time() - start)

    # Free any cached VRAM from SAM before running TripoSR
    torch.cuda.empty_cache()

    # Run inference
    t1 = time.time()
    device = next(_model.parameters()).device
    with torch.no_grad():
        scene_codes = _model([processed], device=device)
    logger.info("Inference done in %.1fs", time.time() - t1)

    # Extract mesh — TripoSR returns trimesh.Trimesh directly with vertex colors
    t2 = time.time()
    try:
        meshes = _model.extract_mesh(
            scene_codes, has_vertex_color=True, resolution=resolution
        )
    except torch.cuda.OutOfMemoryError:
        # Free VRAM and retry at lower resolution
        torch.cuda.empty_cache()
        fallback_res = min(resolution, 192)
        logger.warning("OOM at res=%d, retrying at %d", resolution, fallback_res)
        meshes = _model.extract_mesh(
            scene_codes, has_vertex_color=True, resolution=fallback_res
        )
        resolution = fallback_res
    mesh = meshes[0]
    logger.info(
        "Mesh extraction done in %.1fs (res=%d, %d verts, %d faces)",
        time.time() - t2, resolution, len(mesh.vertices), len(mesh.faces),
    )

    # Smooth the mesh to reduce blobby artifacts from low-res marching cubes
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=10)

    # Fix TripoSR orientation: flip upright and face toward camera
    rotation = trimesh.transformations.euler_matrix(-np.pi / 2, np.pi, 0, "sxyz")
    mesh.apply_transform(rotation)

    # Center and normalize to unit scale
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    max_extent = extent.max()
    if max_extent > 0:
        mesh.vertices /= max_extent

    # Export to GLB — vertex colors are preserved from TripoSR
    scene = trimesh.Scene(mesh)
    glb_bytes = scene.export(file_type="glb")

    elapsed = time.time() - start
    logger.info(
        "TripoSR complete: %d verts, %d faces, %.1f KB GLB, %.1fs total",
        len(mesh.vertices), len(mesh.faces), len(glb_bytes) / 1024, elapsed,
    )
    return glb_bytes


def render_mesh_views(glb_bytes: bytes) -> list[str]:
    """Render a GLB mesh from multiple angles as base64 JPEGs.

    Returns 5 views: front, left, right, three-quarter, above.
    """
    import io as _io
    import trimesh

    mesh = trimesh.load(_io.BytesIO(glb_bytes), file_type="glb", force="mesh")

    mesh.visual.vertex_colors = np.full(
        (len(mesh.vertices), 4), [180, 183, 186, 255], dtype=np.uint8
    )

    views = []
    angles = [
        (0, 0, "front"),
        (0, -35, "left"),
        (0, 35, "right"),
        (-30, 15, "three-quarter"),
        (-45, 0, "above"),
    ]

    for pitch_deg, yaw_deg, name in angles:
        img = _render_single_view(mesh, pitch_deg, yaw_deg)
        if img is not None:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            views.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        else:
            views.append("")

    return views


def render_turntable(
    glb_bytes: bytes, n_frames: int = 24, size: int = 512,
    full_rotation: bool = False,
) -> list[str]:
    """Render a turntable animation as JPEG frames.

    For 2.5D depth meshes: rocks between -40 and +40 degrees (no back).
    For full 3D meshes: full 360-degree rotation.

    Returns n_frames base64 JPEGs, auto-cycled in the browser.
    """
    import io as _io
    import math
    import trimesh

    start = time.time()
    mesh = trimesh.load(_io.BytesIO(glb_bytes), file_type="glb", force="mesh")

    frames = []
    for i in range(n_frames):
        t = i / n_frames
        if full_rotation:
            yaw = 360.0 * t
            pitch = -8
        else:
            # Rock back and forth: -40° → +40° → -40° with slight pitch
            yaw = 40.0 * math.sin(2 * math.pi * t)
            pitch = -5 + 8.0 * math.sin(4 * math.pi * t)

        img = _render_single_view(mesh, pitch_deg=pitch, yaw_deg=yaw, size=size)
        if img is not None:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        else:
            frames.append("")

    logger.info(
        "Turntable: %d frames at %dpx in %.1fs",
        len(frames), size, time.time() - start,
    )
    return frames


def _render_single_view(
    mesh, pitch_deg: float, yaw_deg: float, size: int = 512
) -> np.ndarray | None:
    """Render a mesh from a given angle using Open3D offscreen rendering."""
    import math

    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.71, 0.72, 0.74])

    o3d_mesh.translate(-o3d_mesh.get_center())
    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    extent = max(bbox.get_extent())
    if extent > 0:
        o3d_mesh.scale(1.0 / extent, center=[0, 0, 0])

    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
        renderer.scene.set_background([0.051, 0.067, 0.09, 1.0])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [0.71, 0.72, 0.74, 1.0]
        mat.base_roughness = 0.65
        mat.base_metallic = 0.05

        renderer.scene.add_geometry("mesh", o3d_mesh, mat)
        renderer.scene.scene.set_sun_light([1, 1, 1], [1, 1, 1], 60000)
        renderer.scene.scene.enable_sun_light(True)

        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        dist = 1.5
        eye = [
            dist * math.cos(pitch) * math.sin(yaw),
            dist * math.sin(pitch),
            dist * math.cos(pitch) * math.cos(yaw),
        ]
        renderer.setup_camera(35.0, [0, 0, 0], eye, [0, 1, 0])

        img = np.asarray(renderer.render_to_image())
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning("Open3D render failed: %s", e)
        return None
