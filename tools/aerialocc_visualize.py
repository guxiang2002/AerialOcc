import argparse
import json
import os
from pathlib import Path

import numpy as np


def get_mlab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from mayavi import mlab

    mlab.options.offscreen = True
    return mlab


LABEL_COLORS = np.array(
    [
        [0, 0, 0, 255],          # 0  empty / unknown
        [128, 64, 128, 255],     # 1  road
        [244, 35, 232, 255],     # 2  sidewalk
        [70, 70, 70, 255],       # 3  building
        [102, 102, 156, 255],    # 4  wall
        [190, 153, 153, 255],    # 5  fence
        [153, 153, 153, 255],    # 6  pole
        [250, 170, 30, 255],     # 7  traffic light
        [220, 220, 0, 255],      # 8  traffic sign
        [107, 142, 35, 255],     # 9  vegetation
        [152, 251, 152, 255],    # 10 terrain
        [70, 130, 180, 255],     # 11 sky
        [220, 20, 60, 255],      # 12 pedestrian
        [255, 0, 0, 255],        # 13 rider
        [0, 0, 142, 255],        # 14 car
        [0, 0, 70, 255],         # 15 truck
        [0, 60, 100, 255],       # 16 bus
        [0, 80, 100, 255],       # 17 train
        [0, 0, 230, 255],        # 18 motorcycle
        [119, 11, 32, 255],      # 19 bicycle
        [110, 190, 160, 255],    # 20 static
        [170, 120, 50, 255],     # 21 dynamic
        [55, 90, 80, 255],       # 22 other
        [45, 60, 150, 255],      # 23 water
        [157, 234, 50, 255],     # 24 road lines
        [81, 0, 81, 255],        # 25 ground
        [150, 100, 100, 255],    # 26 bridge
        [230, 150, 140, 255],    # 27 rail track
        [180, 165, 180, 255],    # 28 guard rail
    ],
    dtype=np.uint8,
)

PEDESTRIAN_LABEL = 12
GROUND_LABELS = np.array([1, 2, 10, 24, 25], dtype=np.uint32)
MODEL_GRID_SHAPE = (256, 256, 32)
MODEL_MIN_BOUND = np.array([-25.6, -25.6, -32.0], dtype=np.float32)
MODEL_MAX_BOUND = np.array([51.2, 25.6, 8.0], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize AerialOcc-20K occupancy predictions.")
    parser.add_argument(
        "--data-root",
        default="data/AerialOcc-20K",
        help="AerialOcc-20K dataset root.",
    )
    parser.add_argument(
        "--pred-root",
        default="outputs/predictions",
        help="Prediction save root from main.py --save_path, or a direct predictions folder.",
    )
    parser.add_argument(
        "--preprocess-root",
        default="data/AerialOcc-20K/preprocess",
        help="Preprocessed Carla labels root used by gt_model mode.",
    )
    parser.add_argument("--split", default="Val", choices=["Train", "Val"])
    parser.add_argument("--scene", default="scene_08")
    parser.add_argument(
        "--output-root",
        default="outputs/visualizations",
        help="Rendered output directory.",
    )
    parser.add_argument(
        "--source",
        default="pred",
        choices=["pred", "gt", "gt_model", "dataset_pred"],
        help="pred: model outputs, gt: native evaluation labels, gt_model: model-grid GT, dataset_pred: dataset predictions/*.bin",
    )
    parser.add_argument("--frame", default=None, help="Optional single frame id, e.g. 000123.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--video-view", action="store_true")
    parser.add_argument(
        "--view",
        default="default",
        choices=[
            "default",
            "video",
            "topdown",
            "overview_top",
            "lane_oblique",
            "panorama_oblique",
            "panorama_front",
            "panorama_back",
            "side",
            "low_side",
            "front_low",
            "lane_front",
            "lane_back",
        ],
        help="Camera preset for rendering.",
    )
    parser.add_argument(
        "--camera-offset-scale",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional custom camera offset scale. Overrides --view.",
    )
    parser.add_argument("--draw-frustum", action="store_true")
    parser.add_argument("--frustum-depth", type=float, default=1)
    parser.add_argument("--frustum-line-width", type=float, default=0.5)
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        default=(2200, 2200),
        metavar=("WIDTH", "HEIGHT"),
        help="Output render size in pixels.",
    )
    parser.add_argument(
        "--match-image-aspect",
        action="store_true",
        help="Render with the same width/height as the matching BEV image.",
    )
    parser.add_argument(
        "--camera-reference-frame",
        default=None,
        help="Use the selected frame id to set the camera, e.g. 000001.",
    )
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--camera-height", type=float, default=20.0)
    parser.add_argument(
        "--ground-max-layers",
        type=int,
        default=2,
        help="Visualization-only cleanup for predictions: keep at most this many lowest ground voxels per (x, y) column.",
    )
    parser.add_argument(
        "--disable-visibility-mask",
        action="store_true",
        help="Render prediction voxels without applying the scene visibility/supervision mask.",
    )
    parser.add_argument(
        "--disable-ground-clean",
        action="store_true",
        help="Disable visualization-only ground cleanup for predictions.",
    )
    parser.add_argument(
        "--only-label",
        type=int,
        default=None,
        help="Visualization-only filter: render only this semantic label and hide all other voxels.",
    )
    parser.add_argument(
        "--panorama-front-tilt",
        type=float,
        default=0.0,
        help="Vertical rotation adjustment for panorama_front. Positive is more top-down; negative is flatter.",
    )
    return parser.parse_args()


def resolve_camera_offset_scale(args):
    if args.camera_offset_scale is not None:
        return list(args.camera_offset_scale)

    # Keep compatibility with the older flag while making the newer presets explicit.
    if args.video_view or args.view == "video":
        return [1.25, -1.05, 1.55]

    presets = {
        "default": [1.1, -0.95, 1.7],
        "topdown": [0.05, -0.05, 2.8],
        "overview_top": [0.05, -0.05, 2.8],
        "lane_oblique": [0.05, -0.05, 2.8],
        "panorama_oblique": [0.05, -0.05, 2.8],
        "panorama_front": [0.05, -0.05, 2.8],
        "panorama_back": [0.05, -0.05, 2.8],
        "side": [2.2, -0.15, 0.55],
        "low_side": [2.4, -0.2, 0.18],
        "front_low": [0.25, -2.25, 0.22],
        # Carla scene_16 lanes are aligned mostly with the x-axis in the voxel grid.
        # These presets look along the lane direction instead of diagonally across it.
        "lane_front": [-2.35, 0.18, 0.42],
        "lane_back": [2.35, -0.18, 0.42],
    }
    return presets[args.view]


def resolve_source_dir(args):
    scene = args.scene
    if args.source in {"gt", "gt_model"}:
        return os.path.join(args.data_root, args.split, scene, "cartesian", "evaluation")
    if args.source == "dataset_pred":
        return os.path.join(args.data_root, args.split, scene, "cartesian", "predictions")

    candidates = [
        os.path.join(args.pred_root, "sequences", scene, "predictions"),
        os.path.join(args.pred_root, scene, "predictions"),
        args.pred_root,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find prediction directory from {args.pred_root}")


def list_occ_files(source_dir, frame_id=None, frame_stride=1, max_frames=None):
    files_by_stem = {}
    for pattern in ("*.label", "*.bin"):
        for path in sorted(Path(source_dir).glob(pattern)):
            files_by_stem.setdefault(path.stem, path)
    files = [files_by_stem[stem] for stem in sorted(files_by_stem)]
    if frame_id is not None:
        files = [path for path in files if path.stem == frame_id]
    files = files[:: max(1, frame_stride)]
    if max_frames is not None:
        files = files[:max_frames]
    return files


def find_occ_file(source_dir, frame_id):
    files = list_occ_files(source_dir, frame_id=frame_id)
    if not files:
        raise FileNotFoundError(f"No occupancy file found for frame {frame_id} in {source_dir}")
    return files[0]


def load_scene_params(eval_dir):
    params_path = os.path.join(eval_dir, "params.json")
    if not os.path.exists(params_path):
        return None
    with open(params_path, "r") as f:
        params = json.load(f)
    return {
        "grid_shape": tuple(int(v) for v in params["grid_size"]),
        "min_bound": np.asarray(params["min_bound"], dtype=np.float32),
        "max_bound": np.asarray(params["max_bound"], dtype=np.float32),
        "coordinates": params.get("coordinates", "cartesian"),
        "num_channels": int(params.get("num_channels", len(LABEL_COLORS))),
    }


def find_first_image(image_dir, frame_id=None):
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted(Path(image_dir).glob(ext)))

    if not image_paths:
        raise FileNotFoundError(f"No BEV image found in {image_dir}")

    if frame_id is None:
        return str(image_paths[0])

    for path in image_paths:
        if path.stem == frame_id:
            return str(path)

    return str(image_paths[0])


def resolve_render_size(args, image_dir, frame_id):
    if not args.match_image_aspect:
        return tuple(int(v) for v in args.render_size)

    image_path = find_first_image(image_dir, frame_id)
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def build_topdown_calib(image_path, min_bound, max_bound, camera_height):
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size

    center_x = (min_bound[0] + max_bound[0]) / 2.0
    center_y = (min_bound[1] + max_bound[1]) / 2.0
    x_extent = max(abs(min_bound[0] - center_x), abs(max_bound[0] - center_x))
    y_extent = max(abs(min_bound[1] - center_y), abs(max_bound[1] - center_y))

    fx = (width * camera_height) / max(2.0 * y_extent, 1e-6)
    fy = (height * camera_height) / max(2.0 * x_extent, 1e-6)
    cx = width / 2.0
    cy = height / 2.0

    intr = np.eye(4, dtype=np.float32)
    intr[:3, :4] = np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    lidar2cam = np.eye(4, dtype=np.float32)
    lidar2cam[:3, :4] = np.array(
        [
            [0.0, 1.0, 0.0, -center_y],
            [-1.0, 0.0, 0.0, center_x],
            [0.0, 0.0, -1.0, camera_height],
        ],
        dtype=np.float32,
    )
    return intr, lidar2cam


def draw_frustum(mlab, lidar2cam, intr, img_size, depth, color=(0.0, 0.0, 0.0), line_width=6.0):
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    width, height = img_size

    image_corners = np.array(
        [
            [0.0, 0.0, depth],
            [width, 0.0, depth],
            [width, height, depth],
            [0.0, height, depth],
        ],
        dtype=np.float32,
    )

    cam_corners = np.stack(
        [
            (image_corners[:, 0] - cx) * depth / fx,
            (image_corners[:, 1] - cy) * depth / fy,
            image_corners[:, 2],
        ],
        axis=1,
    )
    apex = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    cam_points = np.concatenate([apex, cam_corners], axis=0)
    cam_points_h = np.concatenate(
        [cam_points, np.ones((cam_points.shape[0], 1), dtype=np.float32)], axis=1
    )
    world_points = (np.linalg.inv(lidar2cam) @ cam_points_h.T).T[:, :3]

    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    ]
    for start_idx, end_idx in edges:
        segment = world_points[[start_idx, end_idx]]
        mlab.plot3d(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            tube_radius=max(depth * 0.006, 0.18),
        )


def load_gt_voxels(occ_file, scene_params):
    if scene_params is None:
        raise FileNotFoundError(
            f"Missing params.json for GT visualization near {occ_file}."
        )
    voxels = np.fromfile(str(occ_file), dtype=np.uint32)
    expected_size = int(np.prod(scene_params["grid_shape"]))
    if voxels.size != expected_size:
        raise ValueError(
            f"GT file {occ_file} has {voxels.size} elements, expected {expected_size}."
        )
    return voxels.reshape(scene_params["grid_shape"]).astype(np.uint32), scene_params


def infer_spec(occ_file, scene_params, source):
    candidate_params = []
    if scene_params is not None:
        candidate_params.append(scene_params)
    candidate_params.append(
        {
            "grid_shape": MODEL_GRID_SHAPE,
            "min_bound": MODEL_MIN_BOUND,
            "max_bound": MODEL_MAX_BOUND,
            "coordinates": "cartesian",
            "num_channels": len(LABEL_COLORS),
        }
    )

    # main.py saves model predictions as uint16, while AerialOcc-20K GT labels are uint32.
    candidate_dtypes = [np.uint16, np.uint32] if source == "pred" else [np.uint32, np.uint16]

    for dtype in candidate_dtypes:
        label = np.fromfile(str(occ_file), dtype=dtype)
        for params in candidate_params:
            if label.size == int(np.prod(params["grid_shape"])):
                return label.reshape(params["grid_shape"]), params

    expected_sizes = [int(np.prod(params["grid_shape"])) for params in candidate_params]
    raise ValueError(
        f"Unsupported prediction size for {occ_file}. "
        f"Tried dtypes {[np.dtype(dtype).name for dtype in candidate_dtypes]} "
        f"and expected element counts {expected_sizes}."
    )


def load_counts(source, occ_file, grid_shape):
    count_path = Path(occ_file).with_suffix(".bin")
    if source in {"gt", "dataset_pred"} and count_path.exists():
        counts = load_count_array(count_path, grid_shape)
        if counts is not None:
            return counts
    return np.ones(grid_shape, dtype=np.float32)


def load_count_array(count_path, grid_shape):
    counts = np.fromfile(str(count_path), dtype=np.float32)
    expected_size = int(np.prod(grid_shape))
    if counts.size != expected_size:
        return None
    return counts.reshape(grid_shape)


def load_gt_counts(occ_file, scene_params):
    count_path = Path(occ_file).with_suffix(".bin")
    if not count_path.exists():
        raise FileNotFoundError(f"Missing GT count file: {count_path}")

    counts = load_count_array(count_path, scene_params["grid_shape"])
    if counts is None:
        expected_size = int(np.prod(scene_params["grid_shape"]))
        raise ValueError(
            f"GT count file {count_path} has unexpected size, expected {expected_size}."
        )
    return counts


def load_scene_visibility_counts(frame_id, scene_params, target_params, args):
    if scene_params is None:
        return None

    count_path = (
        Path(args.data_root)
        / args.split
        / args.scene
        / "cartesian"
        / "evaluation"
        / f"{frame_id}.bin"
    )
    if not count_path.exists():
        return None

    native_counts = load_count_array(count_path, scene_params["grid_shape"])
    if native_counts is None:
        return None

    native_counts = (native_counts > 0).astype(np.float32)
    source_occ_size = np.array(scene_params["grid_shape"], dtype=np.int32)
    source_pc_range = np.concatenate(
        [scene_params["min_bound"], scene_params["max_bound"]]
    ).astype(np.float32)
    target_occ_size = np.array(target_params["grid_shape"], dtype=np.int32)
    target_pc_range = np.concatenate(
        [target_params["min_bound"], target_params["max_bound"]]
    ).astype(np.float32)

    return remap_grid_to_model(
        native_counts,
        source_occ_size=source_occ_size,
        source_pc_range=source_pc_range,
        target_occ_size=target_occ_size,
        target_pc_range=target_pc_range,
    ).astype(np.float32)


def remap_grid_to_model(grid, source_occ_size, source_pc_range, target_occ_size, target_pc_range):
    if (
        np.array_equal(source_occ_size, target_occ_size)
        and np.allclose(source_pc_range, target_pc_range)
    ):
        return grid.copy()

    target = np.zeros(tuple(target_occ_size.tolist()), dtype=grid.dtype)

    source_min = source_pc_range[:3]
    source_max = source_pc_range[3:]
    target_min = target_pc_range[:3]
    target_max = target_pc_range[3:]

    source_steps = (source_max - source_min) / source_occ_size
    target_steps = (target_max - target_min) / target_occ_size

    active_coords = np.argwhere(grid > 0)
    if active_coords.size == 0:
        return target

    active_values = grid[tuple(active_coords.T)]
    active_centers = source_min + (active_coords.astype(np.float32) + 0.5) * source_steps
    target_coords = np.floor((active_centers - target_min) / target_steps).astype(np.int32)

    valid_mask = np.all((target_coords >= 0) & (target_coords < target_occ_size), axis=1)
    if not np.any(valid_mask):
        return target

    target_coords = target_coords[valid_mask]
    active_values = active_values[valid_mask]

    target_flat = target.reshape(-1)
    flat_indices = np.ravel_multi_index(target_coords.T, tuple(target_occ_size.tolist()))
    np.maximum.at(target_flat, flat_indices, active_values)
    return target


def load_gt_model_voxels(occ_file, scene_params, args):
    frame_id = Path(occ_file).stem
    preprocess_path = os.path.join(
        args.preprocess_root,
        "labels",
        args.scene,
        f"{frame_id}_1_1.npy",
    )
    if os.path.exists(preprocess_path):
        voxels = np.load(preprocess_path).astype(np.uint32)
        counts = (voxels != args.ignore_label).astype(np.float32)
        model_params = {
            "grid_shape": MODEL_GRID_SHAPE,
            "min_bound": MODEL_MIN_BOUND,
            "max_bound": MODEL_MAX_BOUND,
            "coordinates": "cartesian",
            "num_channels": len(LABEL_COLORS),
        }
        return voxels, counts, model_params

    voxels_native, native_params = load_gt_voxels(occ_file, scene_params)
    counts_native = load_gt_counts(occ_file, scene_params)
    model_occ_size = np.array(MODEL_GRID_SHAPE, dtype=np.int32)
    model_pc_range = np.concatenate([MODEL_MIN_BOUND, MODEL_MAX_BOUND]).astype(np.float32)
    source_occ_size = np.array(native_params["grid_shape"], dtype=np.int32)
    source_pc_range = np.concatenate(
        [native_params["min_bound"], native_params["max_bound"]]
    ).astype(np.float32)

    voxels = remap_grid_to_model(
        voxels_native.astype(np.uint32),
        source_occ_size=source_occ_size,
        source_pc_range=source_pc_range,
        target_occ_size=model_occ_size,
        target_pc_range=model_pc_range,
    )
    counts = remap_grid_to_model(
        (counts_native > 0).astype(np.float32),
        source_occ_size=source_occ_size,
        source_pc_range=source_pc_range,
        target_occ_size=model_occ_size,
        target_pc_range=model_pc_range,
    )
    model_params = {
        "grid_shape": MODEL_GRID_SHAPE,
        "min_bound": MODEL_MIN_BOUND,
        "max_bound": MODEL_MAX_BOUND,
        "coordinates": "cartesian",
        "num_channels": len(LABEL_COLORS),
    }
    return voxels.astype(np.uint32), counts.astype(np.float32), model_params


def get_grid_coords(dims, min_bound, max_bound, coordinates):
    dims = np.asarray(dims, dtype=np.int32)
    min_bound = np.asarray(min_bound, dtype=np.float32)
    max_bound = np.asarray(max_bound, dtype=np.float32)

    intervals = (max_bound - min_bound) / dims
    axes = [
        np.linspace(min_bound[i], max_bound[i], num=dims[i], endpoint=False) + intervals[i] / 2
        for i in range(3)
    ]
    xv, yv, zv = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    coords = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    if coordinates == "cylindrical":
        rho = coords[:, 0]
        theta = coords[:, 1]
        z = coords[:, 2]
        coords = np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1)

    return coords, intervals


def get_occupied_points(voxels, counts, params):
    grid_coords, _ = get_grid_coords(
        voxels.shape,
        params["min_bound"],
        params["max_bound"],
        params["coordinates"],
    )
    flat_labels = voxels.reshape(-1)
    flat_counts = counts.reshape(-1)

    valid_mask = flat_counts > 0
    occupied_mask = flat_labels > 0
    color_mask = flat_labels < len(LABEL_COLORS)
    mask = valid_mask & occupied_mask & color_mask
    if not np.any(mask):
        return None
    return grid_coords[mask]


def suppress_floating_ground(voxels, counts, max_layers):
    if max_layers <= 0:
        return voxels

    ground_mask = np.isin(voxels, GROUND_LABELS) & (counts > 0)
    active_columns = np.argwhere(np.any(ground_mask, axis=2))
    if active_columns.size == 0:
        return voxels

    cleaned = voxels.copy()
    for x_idx, y_idx in active_columns:
        z_indices = np.flatnonzero(ground_mask[x_idx, y_idx])
        if z_indices.size <= max_layers:
            continue
        cleaned[x_idx, y_idx, z_indices[max_layers:]] = 0
    return cleaned


def configure_camera(scene, occupied_points, camera_offset_scale):
    mins = occupied_points.min(axis=0)
    maxs = occupied_points.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = np.maximum(maxs - mins, 1e-3)
    z_extent = extents[2]
    max_extent = max(float(np.max(extents)), 20.0)

    focal_point = center.copy()
    focal_point[2] -= 0.15 * z_extent
    position = focal_point + np.asarray(camera_offset_scale, dtype=np.float32) * max_extent

    scene.camera.position = position.tolist()
    scene.camera.focal_point = focal_point.tolist()
    scene.camera.view_angle = 24.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [1.0, 8.0 * max_extent]
    scene.camera.compute_view_plane_normal()


def render_overview_top(voxels, counts, save_path, render_size, background_color=(245, 245, 245)):
    valid_mask = (counts > 0) & (voxels > 0) & (voxels < len(LABEL_COLORS))
    occupied_columns = np.any(valid_mask, axis=2)

    if not np.any(occupied_columns):
        print("No occupied voxels found for", save_path)
        return

    reverse_top_idx = np.argmax(valid_mask[:, :, ::-1], axis=2)
    top_idx = np.where(
        occupied_columns,
        valid_mask.shape[2] - 1 - reverse_top_idx,
        0,
    ).astype(np.int32)
    top_labels = np.take_along_axis(voxels, top_idx[..., None], axis=2)[..., 0]

    # Match the dataset BEV convention: x decreases downward, y increases rightward.
    top_labels = top_labels[::-1, :]
    top_idx = top_idx[::-1, :]
    occupied_columns = occupied_columns[::-1, :]

    active_rows = np.where(np.any(occupied_columns, axis=1))[0]
    active_cols = np.where(np.any(occupied_columns, axis=0))[0]
    row_pad = 4
    col_pad = 4
    row_start = max(int(active_rows[0]) - row_pad, 0)
    row_end = min(int(active_rows[-1]) + row_pad + 1, occupied_columns.shape[0])
    col_start = max(int(active_cols[0]) - col_pad, 0)
    col_end = min(int(active_cols[-1]) + col_pad + 1, occupied_columns.shape[1])

    top_labels = top_labels[row_start:row_end, col_start:col_end]
    top_idx = top_idx[row_start:row_end, col_start:col_end]
    occupied_columns = occupied_columns[row_start:row_end, col_start:col_end]

    rgb = np.full((*top_labels.shape, 3), background_color, dtype=np.uint8)
    rgb[occupied_columns] = LABEL_COLORS[top_labels[occupied_columns], :3]

    height_map = top_idx.astype(np.float32)
    height_norm = np.zeros_like(height_map, dtype=np.float32)
    if np.any(occupied_columns):
        min_height = float(height_map[occupied_columns].min())
        max_height = float(height_map[occupied_columns].max())
        denom = max(max_height - min_height, 1e-6)
        height_norm[occupied_columns] = (height_map[occupied_columns] - min_height) / denom

    shade = (0.90 + 0.25 * height_norm)[..., None]
    rgb[occupied_columns] = np.clip(
        rgb[occupied_columns].astype(np.float32) * shade[occupied_columns],
        0,
        255,
    ).astype(np.uint8)

    edge_mask = np.zeros_like(occupied_columns, dtype=bool)
    edge_mask[1:, :] |= occupied_columns[1:, :] & (
        (~occupied_columns[:-1, :]) | ((height_map[1:, :] - height_map[:-1, :]) > 1.5)
    )
    edge_mask[:-1, :] |= occupied_columns[:-1, :] & (
        (~occupied_columns[1:, :]) | ((height_map[:-1, :] - height_map[1:, :]) > 1.5)
    )
    edge_mask[:, 1:] |= occupied_columns[:, 1:] & (
        (~occupied_columns[:, :-1]) | ((height_map[:, 1:] - height_map[:, :-1]) > 1.5)
    )
    edge_mask[:, :-1] |= occupied_columns[:, :-1] & (
        (~occupied_columns[:, 1:]) | ((height_map[:, :-1] - height_map[:, 1:]) > 1.5)
    )
    rgb[edge_mask] = np.clip(rgb[edge_mask].astype(np.float32) * 0.72, 0, 255).astype(np.uint8)

    from PIL import Image

    surface = Image.fromarray(rgb, mode="RGB")
    canvas = Image.new("RGB", render_size, background_color)
    scale = min(render_size[0] / max(surface.width, 1), render_size[1] / max(surface.height, 1)) * 0.96
    resized_size = (
        max(1, int(round(surface.width * scale))),
        max(1, int(round(surface.height * scale))),
    )
    surface = surface.resize(resized_size, Image.Resampling.NEAREST)
    paste_xy = (
        (render_size[0] - resized_size[0]) // 2,
        (render_size[1] - resized_size[1]) // 2,
    )
    canvas.paste(surface, paste_xy)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)
    print("saved to", save_path)


def render_lane_oblique(voxels, counts, save_path, render_size, background_color=(245, 245, 245)):
    valid_mask = (counts > 0) & (voxels > 0) & (voxels < len(LABEL_COLORS))
    occupied_columns = np.any(valid_mask, axis=2)

    if not np.any(occupied_columns):
        print("No occupied voxels found for", save_path)
        return

    reverse_top_idx = np.argmax(valid_mask[:, :, ::-1], axis=2)
    top_idx = np.where(
        occupied_columns,
        valid_mask.shape[2] - 1 - reverse_top_idx,
        0,
    ).astype(np.int32)
    top_labels = np.take_along_axis(voxels, top_idx[..., None], axis=2)[..., 0]

    # Match the dataset BEV convention: lanes run from bottom to top.
    top_labels = top_labels[::-1, :]
    top_idx = top_idx[::-1, :]
    occupied_columns = occupied_columns[::-1, :]

    active_rows = np.where(np.any(occupied_columns, axis=1))[0]
    active_cols = np.where(np.any(occupied_columns, axis=0))[0]
    row_pad = 4
    col_pad = 4
    row_start = max(int(active_rows[0]) - row_pad, 0)
    row_end = min(int(active_rows[-1]) + row_pad + 1, occupied_columns.shape[0])
    col_start = max(int(active_cols[0]) - col_pad, 0)
    col_end = min(int(active_cols[-1]) + col_pad + 1, occupied_columns.shape[1])

    top_labels = top_labels[row_start:row_end, col_start:col_end]
    top_idx = top_idx[row_start:row_end, col_start:col_end]
    occupied_columns = occupied_columns[row_start:row_end, col_start:col_end]

    crop_h, crop_w = top_labels.shape
    max_height = int(top_idx[occupied_columns].max()) + 1

    cell_size = 8
    skew_x = 3
    depth_step = 2
    height_step = 7
    margin = 60

    native_width = crop_w * cell_size + crop_h * skew_x + 2 * margin
    native_height = crop_h * depth_step + (max_height + 2) * height_step + crop_h * 2 + 2 * margin

    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (native_width, native_height), background_color)
    draw = ImageDraw.Draw(canvas)

    def project(x_idx, depth_idx, z_idx):
        px = margin + x_idx * cell_size + depth_idx * skew_x
        py = native_height - margin - depth_idx * depth_step - z_idx * height_step
        return (px, py)

    max_height_float = float(max(max_height, 1))
    max_depth_float = float(max(crop_h - 1, 1))

    # Paint far-to-near so closer columns overwrite the background properly.
    for row_idx in range(crop_h):
        depth_idx = crop_h - 1 - row_idx
        depth_ratio = depth_idx / max_depth_float
        for col_idx in range(crop_w):
            if not occupied_columns[row_idx, col_idx]:
                continue

            top_height = int(top_idx[row_idx, col_idx]) + 1
            color = LABEL_COLORS[int(top_labels[row_idx, col_idx]), :3].astype(np.float32)

            shade = 0.92 + 0.18 * (top_height / max_height_float) - 0.10 * depth_ratio
            top_color = tuple(np.clip(color * shade, 0, 255).astype(np.uint8))

            top_face = [
                project(col_idx, depth_idx, top_height),
                project(col_idx + 1, depth_idx, top_height),
                project(col_idx + 1, depth_idx + 1, top_height),
                project(col_idx, depth_idx + 1, top_height),
            ]
            draw.polygon(top_face, fill=top_color)

            near_height = 0
            if row_idx + 1 < crop_h and occupied_columns[row_idx + 1, col_idx]:
                near_height = int(top_idx[row_idx + 1, col_idx]) + 1
            if near_height < top_height:
                front_color = tuple(np.clip(color * 0.62, 0, 255).astype(np.uint8))
                front_face = [
                    project(col_idx, depth_idx, near_height),
                    project(col_idx + 1, depth_idx, near_height),
                    project(col_idx + 1, depth_idx, top_height),
                    project(col_idx, depth_idx, top_height),
                ]
                draw.polygon(front_face, fill=front_color)

            left_height = 0
            if col_idx - 1 >= 0 and occupied_columns[row_idx, col_idx - 1]:
                left_height = int(top_idx[row_idx, col_idx - 1]) + 1
            if left_height < top_height:
                side_color = tuple(np.clip(color * 0.78, 0, 255).astype(np.uint8))
                side_face = [
                    project(col_idx, depth_idx, left_height),
                    project(col_idx, depth_idx + 1, left_height),
                    project(col_idx, depth_idx + 1, top_height),
                    project(col_idx, depth_idx, top_height),
                ]
                draw.polygon(side_face, fill=side_color)

    canvas_array = np.asarray(canvas)
    bg = np.asarray(background_color, dtype=np.uint8)
    content_mask = np.any(canvas_array != bg[None, None, :], axis=2)
    if np.any(content_mask):
        ys, xs = np.where(content_mask)
        pad = 12
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, canvas.width)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, canvas.height)
        canvas = canvas.crop((x0, y0, x1, y1))
        native_width, native_height = canvas.size

    output = Image.new("RGB", render_size, background_color)
    scale = min(render_size[0] / max(native_width, 1), render_size[1] / max(native_height, 1)) * 0.96
    resized_size = (
        max(1, int(round(native_width * scale))),
        max(1, int(round(native_height * scale))),
    )
    canvas = canvas.resize(resized_size, Image.Resampling.BICUBIC)
    paste_xy = (
        (render_size[0] - resized_size[0]) // 2,
        (render_size[1] - resized_size[1]) // 2,
    )
    output.paste(canvas, paste_xy)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output.save(save_path)
    print("saved to", save_path)


def render_panorama_oblique(voxels, counts, save_path, render_size, background_color=(245, 245, 245)):
    valid_mask = (counts > 0) & (voxels > 0) & (voxels < len(LABEL_COLORS))
    occupied_columns = np.any(valid_mask, axis=2)

    if not np.any(occupied_columns):
        print("No occupied voxels found for", save_path)
        return

    reverse_top_idx = np.argmax(valid_mask[:, :, ::-1], axis=2)
    top_idx = np.where(
        occupied_columns,
        valid_mask.shape[2] - 1 - reverse_top_idx,
        0,
    ).astype(np.int32)
    top_labels = np.take_along_axis(voxels, top_idx[..., None], axis=2)[..., 0]

    top_labels = top_labels[::-1, :]
    top_idx = top_idx[::-1, :]
    occupied_columns = occupied_columns[::-1, :]

    active_rows = np.where(np.any(occupied_columns, axis=1))[0]
    active_cols = np.where(np.any(occupied_columns, axis=0))[0]
    row_pad = 4
    col_pad = 4
    row_start = max(int(active_rows[0]) - row_pad, 0)
    row_end = min(int(active_rows[-1]) + row_pad + 1, occupied_columns.shape[0])
    col_start = max(int(active_cols[0]) - col_pad, 0)
    col_end = min(int(active_cols[-1]) + col_pad + 1, occupied_columns.shape[1])

    top_labels = top_labels[row_start:row_end, col_start:col_end]
    top_idx = top_idx[row_start:row_end, col_start:col_end]
    occupied_columns = occupied_columns[row_start:row_end, col_start:col_end]

    crop_h, crop_w = top_labels.shape
    max_height = int(top_idx[occupied_columns].max()) + 1

    # A higher bird's-eye axonometric basis: keeps the whole scene readable
    # while preserving building height and roof shape.
    basis_x = np.array([8.0, 2.8], dtype=np.float32)
    basis_y = np.array([3.4, -8.2], dtype=np.float32)
    basis_z = np.array([0.0, -10.5], dtype=np.float32)
    margin = 70

    corners = []
    for x_idx in (0, crop_w):
        for y_idx in (0, crop_h):
            for z_idx in (0, max_height + 2):
                corners.append(x_idx * basis_x + y_idx * basis_y + z_idx * basis_z)
    corners = np.asarray(corners, dtype=np.float32)
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
    native_width, native_height = np.ceil(max_xy - min_xy + 2 * margin).astype(np.int32)
    origin = np.array([margin - min_xy[0], margin - min_xy[1]], dtype=np.float32)

    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (int(native_width), int(native_height)), background_color)
    draw = ImageDraw.Draw(canvas)

    def project(x_idx, y_idx, z_idx):
        projected = origin + x_idx * basis_x + y_idx * basis_y + z_idx * basis_z
        return (float(projected[0]), float(projected[1]))

    max_height_float = float(max(max_height, 1))
    max_depth_float = float(max(crop_h, 1))

    for row_idx in range(crop_h):
        depth_idx = crop_h - 1 - row_idx
        for col_idx in range(crop_w):
            if not occupied_columns[row_idx, col_idx]:
                continue

            top_height = int(top_idx[row_idx, col_idx]) + 1
            color = LABEL_COLORS[int(top_labels[row_idx, col_idx]), :3].astype(np.float32)

            top_face = [
                project(col_idx, depth_idx, top_height),
                project(col_idx + 1, depth_idx, top_height),
                project(col_idx + 1, depth_idx + 1, top_height),
                project(col_idx, depth_idx + 1, top_height),
            ]
            shade = 0.90 + 0.20 * (top_height / max_height_float) - 0.03 * (depth_idx / max_depth_float)
            top_color = tuple(np.clip(color * shade, 0, 255).astype(np.uint8))
            draw.polygon(top_face, fill=top_color)

            front_height = 0
            if row_idx + 1 < crop_h and occupied_columns[row_idx + 1, col_idx]:
                front_height = int(top_idx[row_idx + 1, col_idx]) + 1
            if front_height < top_height:
                front_face = [
                    project(col_idx, depth_idx, front_height),
                    project(col_idx + 1, depth_idx, front_height),
                    project(col_idx + 1, depth_idx, top_height),
                    project(col_idx, depth_idx, top_height),
                ]
                front_color = tuple(np.clip(color * 0.62, 0, 255).astype(np.uint8))
                draw.polygon(front_face, fill=front_color)

            left_height = 0
            if col_idx - 1 >= 0 and occupied_columns[row_idx, col_idx - 1]:
                left_height = int(top_idx[row_idx, col_idx - 1]) + 1
            if left_height < top_height:
                left_face = [
                    project(col_idx, depth_idx, left_height),
                    project(col_idx, depth_idx + 1, left_height),
                    project(col_idx, depth_idx + 1, top_height),
                    project(col_idx, depth_idx, top_height),
                ]
                left_color = tuple(np.clip(color * 0.78, 0, 255).astype(np.uint8))
                draw.polygon(left_face, fill=left_color)

    canvas_array = np.asarray(canvas)
    bg = np.asarray(background_color, dtype=np.uint8)
    content_mask = np.any(canvas_array != bg[None, None, :], axis=2)
    if np.any(content_mask):
        ys, xs = np.where(content_mask)
        pad = 12
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, canvas.width)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, canvas.height)
        canvas = canvas.crop((x0, y0, x1, y1))

    output = Image.new("RGB", render_size, background_color)
    scale = min(render_size[0] / max(canvas.width, 1), render_size[1] / max(canvas.height, 1)) * 0.96
    resized_size = (
        max(1, int(round(canvas.width * scale))),
        max(1, int(round(canvas.height * scale))),
    )
    canvas = canvas.resize(resized_size, Image.Resampling.BICUBIC)
    paste_xy = (
        (render_size[0] - resized_size[0]) // 2,
        (render_size[1] - resized_size[1]) // 2,
    )
    output.paste(canvas, paste_xy)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output.save(save_path)
    print("saved to", save_path)


def render_panorama_front(
    voxels,
    counts,
    save_path,
    render_size,
    tilt=0.0,
    reverse=False,
    background_color=(245, 245, 245),
):
    valid_mask = (counts > 0) & (voxels > 0) & (voxels < len(LABEL_COLORS))
    occupied_columns = np.any(valid_mask, axis=2)

    if not np.any(occupied_columns):
        print("No occupied voxels found for", save_path)
        return

    reverse_top_idx = np.argmax(valid_mask[:, :, ::-1], axis=2)
    top_idx = np.where(
        occupied_columns,
        valid_mask.shape[2] - 1 - reverse_top_idx,
        0,
    ).astype(np.int32)
    top_labels = np.take_along_axis(voxels, top_idx[..., None], axis=2)[..., 0]

    top_labels = top_labels[::-1, :]
    top_idx = top_idx[::-1, :]
    occupied_columns = occupied_columns[::-1, :]

    active_rows = np.where(np.any(occupied_columns, axis=1))[0]
    active_cols = np.where(np.any(occupied_columns, axis=0))[0]
    row_pad = 4
    col_pad = 4
    row_start = max(int(active_rows[0]) - row_pad, 0)
    row_end = min(int(active_rows[-1]) + row_pad + 1, occupied_columns.shape[0])
    col_start = max(int(active_cols[0]) - col_pad, 0)
    col_end = min(int(active_cols[-1]) + col_pad + 1, occupied_columns.shape[1])

    top_labels = top_labels[row_start:row_end, col_start:col_end]
    top_idx = top_idx[row_start:row_end, col_start:col_end]
    occupied_columns = occupied_columns[row_start:row_end, col_start:col_end]

    crop_h, crop_w = top_labels.shape
    max_height = int(top_idx[occupied_columns].max()) + 1

    from PIL import Image, ImageDraw

    tilt = float(np.clip(tilt, -1.0, 1.0))

    render_width, render_height = render_size
    margin_top = 70
    margin_bottom = 100
    near_y = render_height - margin_bottom
    far_y = margin_top + 150 + 120.0 * tilt
    center_x = render_width / 2.0

    near_scale = 7.2 * (1.0 - 0.18 * tilt)
    far_scale = 2.35 * (1.0 + 0.30 * tilt)
    height_unit = 9.5 * (1.0 - 0.18 * tilt)

    scales = []
    y_lines = [near_y]
    for depth_idx in range(crop_h + 1):
        t = depth_idx / max(crop_h, 1)
        scale = near_scale * (1.0 - t) ** 1.15 + far_scale * (1.0 - (1.0 - t) ** 1.15)
        scales.append(scale)
    scales = np.asarray(scales, dtype=np.float32)

    for depth_idx in range(crop_h):
        step = 2.2 + 0.72 * scales[depth_idx]
        y_lines.append(y_lines[-1] - step)

    current_far = y_lines[-1]
    y_shift = far_y - current_far
    y_lines = [y + y_shift for y in y_lines]

    def x_line(depth_idx, col_idx):
        scale = scales[depth_idx]
        width = crop_w * scale
        left = center_x - width / 2.0
        return left + col_idx * scale

    def project(depth_idx, col_idx, z_idx):
        scale = scales[depth_idx]
        z_scale = height_unit * (0.55 + 0.45 * scale / near_scale)
        return (
            float(x_line(depth_idx, col_idx)),
            float(y_lines[depth_idx] - z_idx * z_scale),
        )

    canvas = Image.new("RGB", render_size, background_color)
    draw = ImageDraw.Draw(canvas)

    max_height_float = float(max(max_height, 1))
    max_depth_float = float(max(crop_h, 1))

    if reverse:
        row_iter = range(crop_h - 1, -1, -1)
    else:
        row_iter = range(crop_h)

    for row_idx in row_iter:
        depth_idx = row_idx if reverse else crop_h - 1 - row_idx
        depth_ratio = depth_idx / max_depth_float
        for col_idx in range(crop_w):
            if not occupied_columns[row_idx, col_idx]:
                continue

            top_height = int(top_idx[row_idx, col_idx]) + 1
            color = LABEL_COLORS[int(top_labels[row_idx, col_idx]), :3].astype(np.float32)

            top_face = [
                project(depth_idx, col_idx, top_height),
                project(depth_idx, col_idx + 1, top_height),
                project(depth_idx + 1, col_idx + 1, top_height),
                project(depth_idx + 1, col_idx, top_height),
            ]
            shade = 0.92 + 0.16 * (top_height / max_height_float) - 0.06 * depth_ratio
            top_color = tuple(np.clip(color * shade, 0, 255).astype(np.uint8))
            draw.polygon(top_face, fill=top_color)

            front_height = 0
            neighbor_row = row_idx - 1 if reverse else row_idx + 1
            if 0 <= neighbor_row < crop_h and occupied_columns[neighbor_row, col_idx]:
                front_height = int(top_idx[neighbor_row, col_idx]) + 1
            if front_height < top_height:
                front_face = [
                    project(depth_idx, col_idx, front_height),
                    project(depth_idx, col_idx + 1, front_height),
                    project(depth_idx, col_idx + 1, top_height),
                    project(depth_idx, col_idx, top_height),
                ]
                front_color = tuple(np.clip(color * 0.58, 0, 255).astype(np.uint8))
                draw.polygon(front_face, fill=front_color)

    canvas_array = np.asarray(canvas)
    bg = np.asarray(background_color, dtype=np.uint8)
    content_mask = np.any(canvas_array != bg[None, None, :], axis=2)
    if np.any(content_mask):
        ys, xs = np.where(content_mask)
        pad = 12
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, canvas.width)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, canvas.height)
        canvas = canvas.crop((x0, y0, x1, y1))

    output = Image.new("RGB", render_size, background_color)
    scale = min(render_width / max(canvas.width, 1), render_height / max(canvas.height, 1)) * 0.96
    resized_size = (
        max(1, int(round(canvas.width * scale))),
        max(1, int(round(canvas.height * scale))),
    )
    canvas = canvas.resize(resized_size, Image.Resampling.BICUBIC)
    paste_xy = (
        (render_width - resized_size[0]) // 2,
        (render_height - resized_size[1]) // 2,
    )
    output.paste(canvas, paste_xy)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output.save(save_path)
    print("saved to", save_path)


def draw(
    voxels,
    counts,
    params,
    save_path,
    camera_offset_scale,
    render_size,
    view_name="default",
    panorama_front_tilt=0.0,
    camera_points=None,
    frustum=None,
):
    if view_name == "overview_top":
        render_overview_top(
            voxels=voxels,
            counts=counts,
            save_path=save_path,
            render_size=render_size,
        )
        return
    if view_name == "lane_oblique":
        render_lane_oblique(
            voxels=voxels,
            counts=counts,
            save_path=save_path,
            render_size=render_size,
        )
        return
    if view_name == "panorama_oblique":
        render_panorama_oblique(
            voxels=voxels,
            counts=counts,
            save_path=save_path,
            render_size=render_size,
        )
        return
    if view_name == "panorama_front":
        render_panorama_front(
            voxels=voxels,
            counts=counts,
            save_path=save_path,
            render_size=render_size,
            tilt=panorama_front_tilt,
        )
        return
    if view_name == "panorama_back":
        render_panorama_front(
            voxels=voxels,
            counts=counts,
            save_path=save_path,
            render_size=render_size,
            tilt=panorama_front_tilt,
            reverse=True,
        )
        return

    mlab = get_mlab()
    grid_coords, intervals = get_grid_coords(
        voxels.shape,
        params["min_bound"],
        params["max_bound"],
        params["coordinates"],
    )
    flat_labels = voxels.reshape(-1)
    flat_counts = counts.reshape(-1)

    valid_mask = flat_counts > 0
    occupied_mask = flat_labels > 0
    color_mask = flat_labels < len(LABEL_COLORS)
    mask = valid_mask & occupied_mask & color_mask

    if not np.any(mask):
        print("No occupied voxels found for", save_path)
        return

    occupied = np.hstack([grid_coords[mask], flat_labels[mask].reshape(-1, 1)])
    scale_factor = max(0.1, float(min(intervals[0], intervals[2]) * 0.9))

    figure = mlab.figure(size=render_size, bgcolor=(1, 1, 1))
    plot = mlab.points3d(
        occupied[:, 0],
        occupied[:, 1],
        occupied[:, 2],
        occupied[:, 3],
        mode="cube",
        scale_factor=scale_factor,
        opacity=1.0,
        vmin=0,
        vmax=len(LABEL_COLORS) - 1,
    )
    plot.glyph.scale_mode = "scale_by_vector"
    plot.module_manager.scalar_lut_manager.lut.table = LABEL_COLORS

    ped_mask = mask & (flat_labels == PEDESTRIAN_LABEL)
    if np.any(ped_mask):
        ped_coords = grid_coords[ped_mask]
        ped_scalars = flat_labels[ped_mask]
        ped_plot = mlab.points3d(
            ped_coords[:, 0],
            ped_coords[:, 1],
            ped_coords[:, 2],
            ped_scalars,
            mode="cube",
            scale_factor=scale_factor * 1.8,
            opacity=1.0,
            vmin=0,
            vmax=len(LABEL_COLORS) - 1,
        )
        ped_plot.glyph.scale_mode = "scale_by_vector"
        ped_plot.module_manager.scalar_lut_manager.lut.table = LABEL_COLORS

    if frustum is not None:
        draw_frustum(
            mlab=mlab,
            lidar2cam=frustum["lidar2cam"],
            intr=frustum["intr"],
            img_size=frustum["img_size"],
            depth=frustum["depth"],
            line_width=frustum["line_width"],
        )

    configure_camera(
        figure.scene,
        camera_points if camera_points is not None else occupied[:, :3],
        camera_offset_scale,
    )
    figure.scene.render()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mlab.savefig(save_path)
    print("saved to", save_path)
    mlab.close(figure)


def load_render_voxels(args, occ_file, scene_params):
    if args.source == "gt":
        voxels, params = load_gt_voxels(occ_file, scene_params)
        counts = load_gt_counts(occ_file, params)
    elif args.source == "gt_model":
        voxels, counts, params = load_gt_model_voxels(occ_file, scene_params, args)
    else:
        voxels, params = infer_spec(occ_file, scene_params, args.source)
        voxels = voxels.astype(np.uint32)
        counts = load_counts(args.source, occ_file, params["grid_shape"])
        if args.source == "pred" and not args.disable_visibility_mask:
            scene_counts = load_scene_visibility_counts(
                frame_id=occ_file.stem,
                scene_params=scene_params,
                target_params=params,
                args=args,
            )
            if scene_counts is not None:
                counts = scene_counts
        if (
            args.source in {"pred", "dataset_pred"}
            and args.ground_max_layers > 0
            and not args.disable_ground_clean
        ):
            voxels = suppress_floating_ground(
                voxels=voxels,
                counts=counts,
                max_layers=args.ground_max_layers,
            )

    voxels[voxels == args.ignore_label] = 0
    if args.only_label is not None:
        voxels = np.where(voxels == args.only_label, voxels, 0).astype(voxels.dtype)
    return voxels, counts, params


def main():
    args = parse_args()
    source_dir = resolve_source_dir(args)
    eval_dir = os.path.join(args.data_root, args.split, args.scene, "cartesian", "evaluation")
    scene_params = load_scene_params(eval_dir)
    bev_dir = os.path.join(args.data_root, args.split, args.scene, "cartesian", "bev")

    occ_files = list_occ_files(source_dir, args.frame, args.frame_stride, args.max_frames)
    if not occ_files:
        raise FileNotFoundError(f"No occupancy files found in {source_dir}")

    output_dir = Path(args.output_root) / args.scene / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_offset_scale = resolve_camera_offset_scale(args)
    camera_points = None
    if args.camera_reference_frame is not None:
        ref_file = find_occ_file(source_dir, args.camera_reference_frame)
        ref_voxels, ref_counts, ref_params = load_render_voxels(args, ref_file, scene_params)
        camera_points = get_occupied_points(ref_voxels, ref_counts, ref_params)
        if camera_points is None:
            raise ValueError(f"No occupied voxels found in camera reference frame {ref_file.stem}")

    for occ_file in occ_files:
        frustum = None
        voxels, counts, params = load_render_voxels(args, occ_file, scene_params)
        if args.draw_frustum:
            image_path = find_first_image(bev_dir, occ_file.stem)
            intr, lidar2cam = build_topdown_calib(
                image_path=image_path,
                min_bound=params["min_bound"],
                max_bound=params["max_bound"],
                camera_height=args.camera_height,
            )
            from PIL import Image
            with Image.open(image_path) as image:
                img_size = image.size
            frustum = {
                "intr": intr,
                "lidar2cam": lidar2cam,
                "img_size": img_size,
                "depth": args.frustum_depth if args.frustum_depth is not None else args.camera_height,
                "line_width": args.frustum_line_width,
            }
        save_path = output_dir / f"{occ_file.stem}.png"
        render_size = resolve_render_size(args, bev_dir, occ_file.stem)
        draw(
            voxels=voxels,
            counts=counts,
            params=params,
            save_path=str(save_path),
            camera_offset_scale=camera_offset_scale,
            render_size=render_size,
            view_name=args.view,
            panorama_front_tilt=args.panorama_front_tilt,
            camera_points=camera_points,
            frustum=frustum,
        )


if __name__ == "__main__":
    main()
