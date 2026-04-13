"""Depth estimation and 2.5D mesh reconstruction.

Uses Depth Anything V2 for monocular depth, then Open3D Poisson
surface reconstruction for a clean, smooth mesh with RGB vertex colors.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_depth_model = None
_device = None


def _load_depth_model():
    """Load Depth Anything V2 small model (~50MB)."""
    global _depth_model, _device
    try:
        import torch
        from transformers import pipeline

        _device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading Depth Anything V2 (small)...")
        _depth_model = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=_device,
        )
        logger.info("Depth model loaded.")
    except Exception as e:
        logger.error("Failed to load depth model: %s", e)
        _depth_model = None


def estimate_depth(image_rgb: np.ndarray) -> np.ndarray | None:
    """Estimate monocular depth. Returns (H, W) float32 normalized 0-1."""
    global _depth_model
    if _depth_model is None:
        _load_depth_model()
    if _depth_model is None:
        return None

    from PIL import Image
    pil_img = Image.fromarray(image_rgb)
    result = _depth_model(pil_img)
    depth = np.array(result["depth"], dtype=np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


def _crop_to_mask(
    image: np.ndarray, mask: np.ndarray, padding: float = 0.2
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop image and mask to bounding box with padding. Returns crop + bbox."""
    h, w = mask.shape[:2]
    coords = np.where(mask > 127)
    if len(coords[0]) == 0:
        return image, mask, (0, 0, h, w)

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    pad = int(max(y_max - y_min, x_max - x_min) * padding)
    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)

    # Make square
    crop_h, crop_w = y_max - y_min, x_max - x_min
    size = max(crop_h, crop_w)
    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
    y1 = max(0, cy - size // 2)
    x1 = max(0, cx - size // 2)
    y2 = min(h, y1 + size)
    x2 = min(w, x1 + size)

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], (y1, x1, y2, x2)


def build_mesh_from_photo(
    image: np.ndarray,
    head_mask: np.ndarray,
) -> bytes | None:
    """Full pipeline: depth estimation + Poisson reconstruction → GLB.

    Args:
        image: BGR numpy array
        head_mask: binary mask (H, W)

    Returns: GLB bytes with RGB vertex colors, or None
    """
    import tempfile

    import open3d as o3d

    start = time.time()

    # Crop to head region for better depth estimation
    image_cropped, mask_cropped, _ = _crop_to_mask(image, head_mask)
    image_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)

    # Estimate depth on the cropped region
    depth = estimate_depth(image_rgb)
    if depth is None:
        return None

    h, w = depth.shape[:2]
    mask_bool = mask_cropped > 127

    if depth.shape[:2] != mask_cropped.shape[:2]:
        depth = cv2.resize(depth, (mask_cropped.shape[1], mask_cropped.shape[0]))
    if image_rgb.shape[:2] != mask_cropped.shape[:2]:
        image_rgb = cv2.resize(image_rgb, (mask_cropped.shape[1], mask_cropped.shape[0]))

    logger.info("Depth estimated in %.1fs", time.time() - start)

    # Build point cloud from masked depth
    fx = fy = float(w)
    cx, cy_cam = w / 2.0, h / 2.0

    ys, xs = np.where(mask_bool)
    if len(ys) < 100:
        return None

    # Subsample for performance (max ~80K points for good detail)
    step = max(1, len(ys) // 80000)
    ys, xs = ys[::step], xs[::step]

    z = depth[ys, xs].astype(np.float64)
    z = 1.0 - z  # Invert: closer = larger z
    z = z * 0.4 + 0.5  # Scale to reasonable depth range

    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy_cam) * z / fy

    points = np.column_stack([x, -y, -z])
    colors = image_rgb[ys, xs].astype(np.float64) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson surface reconstruction
    try:
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False
        )
    except Exception as e:
        logger.error("Poisson reconstruction failed: %s", e)
        return None

    # Remove low-density vertices (cleans up blobby edges)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    # Remove degenerate triangles before smoothing
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_unreferenced_vertices()

    # Smooth for clean surface
    mesh_o3d = mesh_o3d.filter_smooth_laplacian(number_of_iterations=3)
    mesh_o3d.compute_vertex_normals()

    # Clean up any NaN vertices introduced by smoothing
    vertices_arr = np.asarray(mesh_o3d.vertices)
    nan_mask = np.any(np.isnan(vertices_arr) | np.isinf(vertices_arr), axis=1)
    if nan_mask.any():
        logger.warning("Removing %d NaN vertices", nan_mask.sum())
        mesh_o3d.remove_vertices_by_mask(nan_mask.tolist())
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_unreferenced_vertices()

    n_verts = len(mesh_o3d.vertices)
    n_faces = len(mesh_o3d.triangles)

    if n_verts == 0 or n_faces == 0:
        logger.error("Poisson produced empty mesh")
        return None

    # Center and scale
    mesh_o3d.translate(-mesh_o3d.get_center())
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    extent = max(bbox.get_extent())
    if extent > 0:
        mesh_o3d.scale(1.0 / extent, center=[0, 0, 0])

    # Clay material
    mesh_o3d.paint_uniform_color([0.71, 0.72, 0.74])

    # Export directly from Open3D to GLB
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        o3d.io.write_triangle_mesh(f.name, mesh_o3d, write_vertex_normals=True)
        glb_bytes = Path(f.name).read_bytes()
        Path(f.name).unlink()

    elapsed = time.time() - start
    logger.info(
        "Depth mesh: %d verts, %d faces, %.1f KB GLB, %.1fs total",
        n_verts, n_faces, len(glb_bytes) / 1024, elapsed,
    )
    return glb_bytes
