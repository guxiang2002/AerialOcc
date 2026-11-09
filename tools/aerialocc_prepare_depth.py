import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export Carla AerialOcc-20K LiDAR-projected depth maps as .npy files.'
    )
    parser.add_argument(
        '--data-root',
        default='data/AerialOcc-20K',
        help='AerialOcc-20K root containing Train/ and Val/.',
    )
    parser.add_argument(
        '--output-root',
        default='data/AerialOcc-20K/depth',
        help='Output root for offline depth maps.',
    )
    parser.add_argument(
        '--split',
        default='all',
        choices=['all', 'train', 'val'],
        help='Which split to export.',
    )
    parser.add_argument(
        '--scene',
        default=None,
        help='Optional single scene name such as scene_08.',
    )
    return parser.parse_args()


def read_scene_params(params_path):
    with open(params_path, 'r') as f:
        params = json.load(f)

    min_bound = np.asarray(params['min_bound'], dtype=np.float32)
    max_bound = np.asarray(params['max_bound'], dtype=np.float32)
    pc_range = np.concatenate([min_bound, max_bound]).astype(np.float32)
    return {
        'min_bound': min_bound,
        'max_bound': max_bound,
        'pc_range': pc_range,
    }


def build_topdown_calib(image_path, pc_range, camera_height):
    with Image.open(image_path) as image:
        width, height = image.size

    center_x = (pc_range[0] + pc_range[3]) / 2.0
    center_y = (pc_range[1] + pc_range[4]) / 2.0
    x_extent = max(abs(pc_range[0] - center_x), abs(pc_range[3] - center_x))
    y_extent = max(abs(pc_range[1] - center_y), abs(pc_range[4] - center_y))

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


def project_lidar_to_depth(lidar_points, intr, lidar2cam, image_size):
    width, height = image_size

    points_h = np.concatenate(
        [lidar_points.astype(np.float32), np.ones((lidar_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    points_cam = (lidar2cam @ points_h.T).T
    points_proj = (intr @ points_cam.T).T

    depths = points_proj[:, 2]
    valid_mask = depths > 0
    if not np.any(valid_mask):
        return np.zeros((height, width), dtype=np.float32)

    points_proj = points_proj[valid_mask]
    depths = depths[valid_mask]
    u = points_proj[:, 0] / depths
    v = points_proj[:, 1] / depths

    valid_mask = (
        (u >= 0)
        & (v >= 0)
        & (u <= width - 1)
        & (v <= height - 1)
    )
    if not np.any(valid_mask):
        return np.zeros((height, width), dtype=np.float32)

    u = u[valid_mask]
    v = v[valid_mask]
    depths = depths[valid_mask]

    order = np.argsort(depths)[::-1]
    u = np.round(u[order]).astype(np.int32)
    v = np.round(v[order]).astype(np.int32)
    depths = depths[order].astype(np.float32)

    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_map[v, u] = depths
    return depth_map


def find_image_path(image_dir, frame_id):
    for ext in ('.png', '.jpg', '.jpeg'):
        candidate = os.path.join(image_dir, frame_id + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'Could not find image for frame {frame_id} in {image_dir}')


def collect_scene_dirs(data_root, split, scene_name=None):
    split_map = {
        'train': ['Train'],
        'val': ['Val'],
        'all': ['Train', 'Val'],
    }
    roots = [os.path.join(data_root, split_dir) for split_dir in split_map[split]]
    scene_dirs = []
    for root in roots:
        for scene_dir in sorted(glob.glob(os.path.join(root, 'scene_*'))):
            if scene_name is not None and os.path.basename(scene_dir) != scene_name:
                continue
            scene_dirs.append(scene_dir)
    return scene_dirs


def export_scene(scene_dir, output_root, camera_height=60.0):
    scene_name = os.path.basename(scene_dir)
    cartesian_dir = os.path.join(scene_dir, 'cartesian')
    image_dir = os.path.join(cartesian_dir, 'bev')
    lidar_dir = os.path.join(cartesian_dir, 'velodyne')
    eval_dir = os.path.join(cartesian_dir, 'evaluation')

    params = read_scene_params(os.path.join(eval_dir, 'params.json'))
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, '*.png'))
        + glob.glob(os.path.join(image_dir, '*.jpg'))
        + glob.glob(os.path.join(image_dir, '*.jpeg'))
    )
    if not image_paths:
        raise FileNotFoundError(f'No BEV image found under {image_dir}')

    intr, lidar2cam = build_topdown_calib(
        image_paths[0],
        params['pc_range'],
        camera_height,
    )
    with Image.open(image_paths[0]) as image:
        image_size = image.size

    frame_ids = sorted(
        {
            Path(path).stem
            for path in image_paths
        }
        & {
            Path(path).stem
            for path in glob.glob(os.path.join(lidar_dir, '*.bin'))
        }
    )

    depth_dir = os.path.join(output_root, 'sequences', scene_name)
    os.makedirs(depth_dir, exist_ok=True)

    meta = {
        'scene': scene_name,
        'image_size': list(image_size),
        'camera_height': float(camera_height),
        'pc_range': params['pc_range'].tolist(),
    }
    with open(os.path.join(output_root, f'{scene_name}_depth_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    for frame_id in tqdm(frame_ids, desc=f'exporting {scene_name}'):
        image_path = find_image_path(image_dir, frame_id)
        lidar_path = os.path.join(lidar_dir, frame_id + '.bin')

        with Image.open(image_path) as image:
            frame_image_size = image.size
        if frame_image_size != image_size:
            intr, lidar2cam = build_topdown_calib(
                image_path,
                params['pc_range'],
                camera_height,
            )
            image_size = frame_image_size

        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        depth_map = project_lidar_to_depth(
            lidar_points=lidar_points,
            intr=intr,
            lidar2cam=lidar2cam,
            image_size=image_size,
        )
        np.save(os.path.join(depth_dir, frame_id + '.npy'), depth_map.astype(np.float32))


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    scene_dirs = collect_scene_dirs(args.data_root, args.split, args.scene)
    if not scene_dirs:
        raise FileNotFoundError('No Carla scenes found for the requested split/scene.')

    for scene_dir in scene_dirs:
        export_scene(scene_dir, args.output_root)


if __name__ == '__main__':
    main()
