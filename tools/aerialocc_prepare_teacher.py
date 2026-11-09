import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export VGGT teacher depth for AerialOcc-20K using temporal windows."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/AerialOcc-20K",
        help="AerialOcc-20K dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="data/AerialOcc-20K/vggt_depth",
        help="Output root for normalized teacher depth.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["all", "train", "val"],
        help="Which AerialOcc-20K split to export.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Optional single scene like scene_08.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Temporal window size. Use 3 to match t-1,t,t+1 training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported depth files.",
    )
    parser.add_argument(
        "--vggt-repo",
        default=None,
        help="Optional local path to the official VGGT repository if it is not installed as a package.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/VGGT-1B",
        help="Pretrained VGGT checkpoint name.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional local VGGT checkpoint path (.pt). If set, skip Hugging Face download.",
    )
    return parser.parse_args()


def maybe_add_vggt_repo(vggt_repo):
    if vggt_repo is None:
        return
    vggt_repo = os.path.abspath(vggt_repo)
    package_like_root = (
        os.path.basename(vggt_repo) == "vggt"
        and os.path.isdir(os.path.join(vggt_repo, "models"))
        and os.path.isdir(os.path.join(vggt_repo, "utils"))
    )
    import_root = os.path.dirname(vggt_repo) if package_like_root else vggt_repo
    if import_root not in sys.path:
        sys.path.insert(0, import_root)


def import_vggt_modules():
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    return VGGT, load_and_preprocess_images


def load_vggt_model(VGGT, torch, model_name, device, checkpoint_path=None):
    if checkpoint_path is not None:
        print(f"Loading VGGT checkpoint from local file: {checkpoint_path}")
        model = VGGT().to(device)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    try:
        model = VGGT.from_pretrained(model_name).to(device)
        return model
    except Exception as exc:
        print(f"from_pretrained({model_name}) failed: {exc}")
        fallback_url = f"https://huggingface.co/{model_name}/resolve/main/model.pt"
        print(f"Falling back to direct checkpoint URL: {fallback_url}")
        model = VGGT().to(device)
        cache_name = f"{model_name.replace('/', '_')}_model.pt"
        checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        cached_file = checkpoints_dir / cache_name
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                fallback_url,
                map_location="cpu",
                progress=True,
                file_name=cache_name,
            )
        except RuntimeError as runtime_exc:
            error_text = str(runtime_exc)
            if "failed finding central directory" not in error_text:
                raise
            if cached_file.exists():
                print(f"Removing corrupted checkpoint cache: {cached_file}")
                cached_file.unlink()
            print("Retrying checkpoint download from scratch...")
            state_dict = torch.hub.load_state_dict_from_url(
                fallback_url,
                map_location="cpu",
                progress=True,
                file_name=cache_name,
            )
        model.load_state_dict(state_dict)
        return model


def resolve_split_names(split):
    if split == "all":
        return ["Train", "Val"]
    if split == "train":
        return ["Train"]
    return ["Val"]


def list_scene_dirs(dataset_root, split_names, scene_filter=None):
    scene_dirs = []
    for split_name in split_names:
        split_dir = Path(dataset_root) / split_name
        if not split_dir.exists():
            continue
        for scene_dir in sorted(split_dir.glob("scene_*")):
            if scene_filter is not None and scene_dir.name != scene_filter:
                continue
            scene_dirs.append((split_name, scene_dir))
    return scene_dirs


def list_frame_paths(bev_dir):
    image_paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted(bev_dir.glob(pattern)))
    return image_paths


def build_window(frame_paths, center_idx, window_size):
    half = window_size // 2
    indices = []
    for offset in range(-half, half + 1):
        idx = min(max(center_idx + offset, 0), len(frame_paths) - 1)
        indices.append(idx)
    return indices


def get_autocast_dtype(torch, device):
    if device.startswith("cuda") and torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return None


def run_vggt_depth(model, load_and_preprocess_images, torch, image_paths, device, autocast_dtype):
    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
    with torch.no_grad():
        if autocast_dtype is not None:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    if isinstance(predictions, dict):
        depth = predictions.get("depth_map")
        if depth is None:
            depth = predictions.get("depth")
    else:
        depth = getattr(predictions, "depth_map", None)
        if depth is None:
            depth = getattr(predictions, "depth", None)

    if depth is None:
        raise KeyError("Could not find depth prediction in VGGT output.")

    confidence = (predictions.get('depth_conf') if isinstance(predictions, dict)
                  else getattr(predictions, 'depth_conf', None))
    if confidence is None:
        raise KeyError('VGGT must return depth_conf for confidence filtering.')
    confidence = confidence.detach().float().cpu().numpy()
    depth = depth.detach().float().cpu().numpy()

    # Common VGGT depth output layouts we have seen:
    # - [B, V, H, W, 1]
    # - [V, H, W, 1]
    # - [B, V, H, W]
    # - [V, H, W]
    # - [H, W]
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 2:
        depth = depth[None, ...]
    if depth.ndim != 3:
        raise ValueError(f"Unexpected VGGT depth shape after normalization: {depth.shape}")
    if confidence.ndim == 5 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim == 4 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if confidence.ndim == 2:
        confidence = confidence[None]
    if confidence.shape != depth.shape:
        raise ValueError('VGGT confidence and depth shapes differ.')
    return depth, confidence


def normalize_teacher(depth, confidence):
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not np.any(valid):
        raise ValueError('No finite positive teacher depth.')
    valid &= confidence >= np.quantile(confidence[valid], 0.2)
    result = np.zeros_like(depth, dtype=np.float32)
    result[valid] = depth[valid] / np.median(depth[valid])
    return result


def main():
    args = parse_args()
    if args.window_size % 2 != 1:
        raise ValueError("--window-size must be odd.")

    maybe_add_vggt_repo(args.vggt_repo)

    import torch

    VGGT, load_and_preprocess_images = import_vggt_modules()
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    autocast_dtype = get_autocast_dtype(torch, device)

    model = load_vggt_model(
        VGGT=VGGT,
        torch=torch,
        model_name=args.model_name,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )
    model.eval()

    scene_dirs = list_scene_dirs(
        dataset_root=args.dataset_root,
        split_names=resolve_split_names(args.split),
        scene_filter=args.scene,
    )
    if not scene_dirs:
        raise FileNotFoundError("No matching AerialOcc-20K scenes found.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exported = 0
    for _, scene_dir in scene_dirs:
        scene_name = scene_dir.name
        frame_paths = list_frame_paths(scene_dir / "cartesian" / "bev")
        if not frame_paths:
            continue

        output_scene_dir = output_root / "sequences" / scene_name
        output_scene_dir.mkdir(parents=True, exist_ok=True)

        for center_idx, center_path in enumerate(frame_paths):
            frame_id = center_path.stem
            output_path = output_scene_dir / f"{frame_id}.npy"
            if output_path.exists() and not args.overwrite:
                continue

            window_indices = build_window(frame_paths, center_idx, args.window_size)
            window_paths = [frame_paths[idx] for idx in window_indices]
            depth_stack, confidence_stack = run_vggt_depth(
                model=model,
                load_and_preprocess_images=load_and_preprocess_images,
                torch=torch,
                image_paths=window_paths,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            center_index = len(window_paths) // 2
            center_depth = normalize_teacher(
                depth_stack[center_index], confidence_stack[center_index])
            np.save(output_path, center_depth.astype(np.float32))
            exported += 1
            if exported % 20 == 0:
                print(f"exported {exported} depth maps")

    print(f"Finished exporting {exported} VGGT teacher depth maps to {output_root}")


if __name__ == "__main__":
    main()
