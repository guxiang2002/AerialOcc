"""
Preprocess AerialOcc-20K occupancy labels into a training-friendly format.

Prepare the original AerialOcc-20K occupancy labels:
1. reads evaluation/*.label and evaluation/*.bin
2. treats counts <= valid_threshold as invalid / unknown voxels
3. remaps each scene's native grid into a common target grid
4. merges source labels on the target grid with the original vote / mode rule
5. writes downsampled labels as .npy files
5. optionally exports packed .invalid masks
"""

import argparse
import glob
import json
import os

import numpy as np
from tqdm import tqdm


def pack_bits(array):
    array = array.reshape(-1).astype(np.uint8)
    pad = (-len(array)) % 8
    if pad:
        array = np.pad(array, (0, pad), mode="constant")
    return np.array(
        array[::8] << 7
        | array[1::8] << 6
        | array[2::8] << 5
        | array[3::8] << 4
        | array[4::8] << 3
        | array[5::8] << 2
        | array[6::8] << 1
        | array[7::8],
        dtype=np.uint8,
    )


def downsample_label(label, voxel_size, downscale=2):
    if downscale == 1:
        return label

    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds,
            y * ds : (y + 1) * ds,
            z * ds : (z + 1) * ds,
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size
        zero_count = zero_count_0 + zero_count_255

        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))

    return label_downscale


def read_scene_params(eval_dir):
    params_path = os.path.join(eval_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    occ_size = np.array([int(v) for v in params["grid_size"]], dtype=np.int32)
    min_bound = np.array(params["min_bound"], dtype=np.float32)
    max_bound = np.array(params["max_bound"], dtype=np.float32)
    pc_range = np.concatenate([min_bound, max_bound]).astype(np.float32)
    return {
        "occ_size": occ_size,
        "pc_range": pc_range,
        "num_channels": int(params.get("num_channels", 0)),
        "coordinates": params.get("coordinates", "cartesian"),
    }


def build_axis_map(src_size, src_min, src_max, tgt_size, tgt_min, tgt_max):
    src_step = (src_max - src_min) / src_size
    tgt_step = (tgt_max - tgt_min) / tgt_size
    centers = src_min + (np.arange(src_size, dtype=np.float32) + 0.5) * src_step
    tgt_idx = np.floor((centers - tgt_min) / tgt_step).astype(np.int32)
    valid = (tgt_idx >= 0) & (tgt_idx < tgt_size)
    return tgt_idx, valid


def build_scene_mapper(source_occ_size, source_pc_range, target_occ_size, target_pc_range):
    x_idx, x_valid = build_axis_map(
        source_occ_size[0], source_pc_range[0], source_pc_range[3],
        target_occ_size[0], target_pc_range[0], target_pc_range[3],
    )
    y_idx, y_valid = build_axis_map(
        source_occ_size[1], source_pc_range[1], source_pc_range[4],
        target_occ_size[1], target_pc_range[1], target_pc_range[4],
    )
    z_idx, z_valid = build_axis_map(
        source_occ_size[2], source_pc_range[2], source_pc_range[5],
        target_occ_size[2], target_pc_range[2], target_pc_range[5],
    )

    flat_x = np.repeat(x_idx, source_occ_size[1] * source_occ_size[2])
    flat_y = np.tile(np.repeat(y_idx, source_occ_size[2]), source_occ_size[0])
    flat_z = np.tile(z_idx, source_occ_size[0] * source_occ_size[1])

    valid_flat = (
        np.repeat(x_valid, source_occ_size[1] * source_occ_size[2])
        & np.tile(np.repeat(y_valid, source_occ_size[2]), source_occ_size[0])
        & np.tile(z_valid, source_occ_size[0] * source_occ_size[1])
    )

    flat_target = np.full(int(np.prod(source_occ_size)), -1, dtype=np.int64)
    flat_target[valid_flat] = np.ravel_multi_index(
        (flat_x[valid_flat], flat_y[valid_flat], flat_z[valid_flat]),
        tuple(target_occ_size.tolist()),
    )
    return flat_target, valid_flat


def merge_labels_by_mode(target_label, flat_indices, labels):
    if flat_indices.size == 0:
        return target_label

    order = np.argsort(flat_indices, kind="mergesort")
    sorted_indices = flat_indices[order]
    sorted_labels = labels[order].astype(np.int64)

    unique_indices, start_positions = np.unique(sorted_indices, return_index=True)
    end_positions = np.append(start_positions[1:], sorted_indices.size)

    for target_idx, start, end in zip(unique_indices, start_positions, end_positions):
        votes = np.bincount(sorted_labels[start:end])
        target_label[target_idx] = np.argmax(votes)

    return target_label


def remap_frame(
    label,
    counts,
    flat_target,
    valid_flat,
    target_occ_size,
    valid_threshold,
    label_merge_mode,
):
    target_num_voxels = int(np.prod(target_occ_size))

    label = label.reshape(-1)
    counts = counts.reshape(-1)
    valid_obs = valid_flat & (counts > valid_threshold)

    target_seen = np.zeros(target_num_voxels, dtype=np.uint8)
    np.maximum.at(target_seen, flat_target[valid_obs], 1)

    target_label = np.full(target_num_voxels, 255, dtype=np.uint16)
    target_label[target_seen > 0] = 0

    valid_sem = valid_obs & (label > 0) & (label < 255)
    if np.any(valid_sem):
        valid_indices = flat_target[valid_sem]
        valid_labels = label[valid_sem].astype(np.uint16)
        if label_merge_mode == "max":
            np.maximum.at(target_label, valid_indices, valid_labels)
        elif label_merge_mode in {"mode", "vote"}:
            merge_labels_by_mode(target_label, valid_indices, valid_labels)
        elif label_merge_mode == "first":
            empty_mask = target_label[valid_indices] == 0
            target_label[valid_indices[empty_mask]] = valid_labels[empty_mask]
        else:
            raise ValueError(f"Unsupported label_merge_mode: {label_merge_mode}")

    target_invalid = (target_seen == 0).astype(np.uint8)
    return (
        target_label.reshape(tuple(target_occ_size.tolist())),
        target_invalid.reshape(tuple(target_occ_size.tolist())),
    )


def process_scene(
    scene_dir,
    output_root,
    target_occ_size,
    target_pc_range,
    valid_threshold,
    write_invalid,
    label_merge_mode,
):
    eval_dir = os.path.join(scene_dir, "cartesian", "evaluation")
    scene_name = os.path.basename(scene_dir)
    scene_params = read_scene_params(eval_dir)

    flat_target, valid_flat = build_scene_mapper(
        source_occ_size=scene_params["occ_size"],
        source_pc_range=scene_params["pc_range"],
        target_occ_size=target_occ_size,
        target_pc_range=target_pc_range,
    )

    label_paths = sorted(glob.glob(os.path.join(eval_dir, "*.label")))
    label_out_dir = os.path.join(output_root, "labels", scene_name)
    invalid_out_dir = os.path.join(output_root, "invalid", scene_name)
    os.makedirs(label_out_dir, exist_ok=True)
    if write_invalid:
        os.makedirs(invalid_out_dir, exist_ok=True)

    meta = {
        "source_occ_size": scene_params["occ_size"].tolist(),
        "source_pc_range": scene_params["pc_range"].tolist(),
        "target_occ_size": target_occ_size.tolist(),
        "target_pc_range": target_pc_range.tolist(),
        "valid_threshold": float(valid_threshold),
        "label_merge_mode": label_merge_mode,
    }
    with open(os.path.join(output_root, f"{scene_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for label_path in tqdm(label_paths, desc=f"processing {scene_name}"):
        frame_id = os.path.splitext(os.path.basename(label_path))[0]
        count_path = os.path.join(eval_dir, frame_id + ".bin")
        if not os.path.exists(count_path):
            raise FileNotFoundError(f"Missing count file for {label_path}")

        label = np.fromfile(label_path, dtype=np.uint32).reshape(tuple(scene_params["occ_size"].tolist()))
        counts = np.fromfile(count_path, dtype=np.float32).reshape(tuple(scene_params["occ_size"].tolist()))

        label_1_1, invalid_1_1 = remap_frame(
            label=label,
            counts=counts,
            flat_target=flat_target,
            valid_flat=valid_flat,
            target_occ_size=target_occ_size,
            valid_threshold=valid_threshold,
            label_merge_mode=label_merge_mode,
        )

        np.save(os.path.join(label_out_dir, frame_id + "_1_1.npy"), label_1_1.astype(np.uint8))
        np.save(
            os.path.join(label_out_dir, frame_id + "_1_2.npy"),
            downsample_label(label_1_1.astype(np.uint8), tuple(target_occ_size.tolist()), downscale=2),
        )

        if write_invalid:
            packed_invalid = pack_bits(invalid_1_1)
            packed_invalid.tofile(os.path.join(invalid_out_dir, frame_id + ".invalid"))


def main():
    parser = argparse.ArgumentParser("./aerialocc_prepare_labels.py")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/AerialOcc-20K",
        help="AerialOcc-20K root containing Train/ and Val/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/AerialOcc-20K/preprocess",
        help="Where to write preprocessed labels and invalid masks.",
    )
    parser.add_argument(
        "--valid-threshold",
        type=float,
        default=0.0,
        help="Voxels with counts <= threshold are treated as invalid.",
    )
    parser.add_argument(
        "--write-invalid",
        action="store_true",
        help="Also export packed .invalid masks on the target 256x256x32 grid.",
    )
    parser.add_argument(
        "--label-merge-mode",
        type=str,
        default="mode",
        choices=["first", "max", "mode", "vote"],
        help="How to merge multiple source voxels that land in the same target voxel.",
    )
    parser.add_argument(
        "--target-occ-size",
        type=int,
        nargs=3,
        default=[256, 256, 32],
        help="Target occupancy grid size used by the current model.",
    )
    parser.add_argument(
        "--target-pc-range",
        type=float,
        nargs=6,
        default=[-25.6, -25.6, -32.0, 51.2, 25.6, 8.0],
        help="Target point cloud range used by the current model.",
    )
    args = parser.parse_args()

    target_occ_size = np.array(args.target_occ_size, dtype=np.int32)
    target_pc_range = np.array(args.target_pc_range, dtype=np.float32)

    scene_dirs = sorted(
        glob.glob(os.path.join(args.data_root, "Train", "scene_*"))
        + glob.glob(os.path.join(args.data_root, "Val", "scene_*"))
    )
    for scene_dir in scene_dirs:
        process_scene(
            scene_dir=scene_dir,
            output_root=args.output_root,
            target_occ_size=target_occ_size,
            target_pc_range=target_pc_range,
            valid_threshold=args.valid_threshold,
            write_invalid=args.write_invalid,
            label_merge_mode=args.label_merge_mode,
        )


if __name__ == "__main__":
    main()
