import glob
import json
import os

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset


@DATASETS.register_module()
class AerialOcc20KDataset(Dataset):
    def __init__(
        self,
        data_root,
        ann_file=None,
        pipeline=None,
        split='train',
        camera_used=None,
        occ_size=None,
        pc_range=None,
        stereo_depth_root=None,
        teacher_depth_root=None,
        test_mode=False,
        load_continuous=False,
        image_dir='bev',
        lidar_dir='velodyne',
        point_label_dir='labels',
        occ_dir='evaluation',
        occ_label_suffix='.label',
        occ_count_suffix='.bin',
        label_merge_mode='max',
        camera_height=60.0,
        num_classes=None,
        ignore_label=255,
        preprocess_label_subdir='labels',
        preprocess_scale='1_1',
        prefer_preprocessed_labels=False,
        use_occ_count_mask=False,
        valid_threshold=0.0,
        label_mapping=None,
        scene_names=None,
        temporal_offsets=(0,),
    ):
        super().__init__()

        self.data_root = data_root
        self.ann_file = ann_file if ann_file is not None else data_root
        self.test_mode = test_mode
        self.load_continuous = load_continuous
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.stereo_depth_root = stereo_depth_root
        self.teacher_depth_root = teacher_depth_root

        self.image_dir = image_dir
        self.lidar_dir = lidar_dir
        self.point_label_dir = point_label_dir
        self.occ_dir = occ_dir
        self.occ_label_suffix = occ_label_suffix
        self.occ_count_suffix = occ_count_suffix
        self.label_merge_mode = label_merge_mode
        self.camera_height = camera_height
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.preprocess_label_subdir = preprocess_label_subdir
        self.preprocess_scale = preprocess_scale
        self.prefer_preprocessed_labels = prefer_preprocessed_labels
        self.use_occ_count_mask = use_occ_count_mask
        self.valid_threshold = valid_threshold
        self.label_mapping = self.build_label_mapping(label_mapping)
        self.scene_names = None if scene_names is None else set(scene_names)
        self.temporal_offsets = tuple(int(x) for x in temporal_offsets)
        # The supervised frame is the frame at temporal offset 0.  It is not
        # necessarily the middle input (for example, [-2, -1, 0]).
        self.temporal_center_index = (
            self.temporal_offsets.index(0)
            if 0 in self.temporal_offsets
            else len(self.temporal_offsets) // 2
        )
        self.target_occ_size = np.array(self.occ_size, dtype=np.int32)
        self.target_pc_range = np.array(self.pc_range, dtype=np.float32)

        self.camera_map = {'bev': '0'}
        camera_used = ['bev'] if camera_used is None else camera_used
        self.camera_used = [self.camera_map[camera] for camera in camera_used]

        split_map = {
            'train': 'Train',
            'val': 'Val',
            'test': 'Val',
        }
        if split not in split_map:
            raise ValueError(f'Unsupported split: {split}')
        self.scene_root = os.path.join(self.data_root, split_map[split])

        self.data_infos = self.load_annotations()
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        return self.pipeline(input_dict)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in test data')
            return None
        return self.pipeline(input_dict)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            sequence=info['sequence'],
            frame_id=info['frame_id'],
            lidar_path=info['lidar_path'],
            lidar_label_path=info['lidar_label_path'],
        )

        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        stereo_depth_paths = []
        teacher_depth_paths = []
        temporal_frames = info.get('temporal_frames', [info])
        center_pose = info['pose']
        base_lidar2cam = info['T_velo_2_cam_base']

        for temporal_info in temporal_frames:
            relative_lidar = self.compute_relative_lidar_transform(
                source_pose=center_pose,
                target_pose=temporal_info['pose'],
            )
            lidar2cam = base_lidar2cam @ relative_lidar
            proj_matrix = info['P0'] @ lidar2cam
            for cam_type in self.camera_used:
                image_paths.append(temporal_info[f'img_{int(cam_type)}_path'])
                lidar2img_rts.append(proj_matrix)
                cam_intrinsics.append(info[f'P{int(cam_type)}'])
                lidar2cam_rts.append(lidar2cam)
                stereo_depth_paths.append(temporal_info.get('stereo_depth_path'))
                teacher_depth_paths.append(temporal_info.get('teacher_depth_path'))

        focal_length = info['P0'][0, 0]
        baseline = 0.0

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline,
                stereo_depth_path=stereo_depth_paths,
                teacher_depth_path=teacher_depth_paths,
                temporal_center_index=self.temporal_center_index,
                temporal_offsets=np.array(self.temporal_offsets, dtype=np.int64),
            )
        )
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')
        return input_dict

    def load_annotations(self):
        scans = []
        scene_dirs = sorted(glob.glob(os.path.join(self.scene_root, 'scene_*')))
        for scene_dir in scene_dirs:
            sequence = os.path.basename(scene_dir)
            if self.scene_names is not None and sequence not in self.scene_names:
                continue
            cartesian_dir = os.path.join(scene_dir, 'cartesian')
            image_base_path = os.path.join(cartesian_dir, self.image_dir)
            lidar_base_path = os.path.join(cartesian_dir, self.lidar_dir)
            point_label_base_path = os.path.join(cartesian_dir, self.point_label_dir)
            voxel_base_path = os.path.join(cartesian_dir, self.occ_dir)
            scene_params = self.read_scene_params(os.path.join(voxel_base_path, 'params.json'))

            frame_ids = self.collect_frame_ids(image_base_path, lidar_base_path)
            if not frame_ids:
                continue

            sample_image_path = self.find_first_image(image_base_path, frame_ids[0])
            calib = self.build_topdown_calib(sample_image_path, scene_params['pc_range'], self.camera_height)
            P0 = calib['P0']
            T_velo_2_cam_base = calib['Tr']
            scene_poses = self.read_scene_poses(os.path.join(cartesian_dir, 'poses.txt'))
            scene_times = self.read_scene_times(os.path.join(cartesian_dir, 'times.txt'))

            scene_records = []
            for scene_order_idx, frame_id in enumerate(frame_ids):
                img_path = self.find_first_image(image_base_path, frame_id)
                lidar_path = os.path.join(lidar_base_path, frame_id + '.bin')
                lidar_label_path = os.path.join(point_label_base_path, frame_id + '.label')
                voxel_path = os.path.join(voxel_base_path, frame_id + self.occ_label_suffix)
                voxel_count_path = os.path.join(voxel_base_path, frame_id + self.occ_count_suffix)

                if not os.path.exists(img_path) or not os.path.exists(lidar_path):
                    continue
                if not os.path.exists(lidar_label_path):
                    lidar_label_path = None
                if not os.path.exists(voxel_path):
                    voxel_path = None
                if not os.path.exists(voxel_count_path):
                    voxel_count_path = None
                stereo_depth_path = None
                if self.stereo_depth_root is not None:
                    candidate_depth_path = os.path.join(
                        self.stereo_depth_root,
                        'sequences',
                        sequence,
                        frame_id + '.npy',
                    )
                    if os.path.exists(candidate_depth_path):
                        stereo_depth_path = candidate_depth_path
                teacher_depth_path = None
                if self.teacher_depth_root is not None:
                    candidate_teacher_path = os.path.join(
                        self.teacher_depth_root,
                        'sequences',
                        sequence,
                        frame_id + '.npy',
                    )
                    if os.path.exists(candidate_teacher_path):
                        teacher_depth_path = candidate_teacher_path

                frame_pose = self.resolve_scene_pose(scene_poses, frame_id, scene_order_idx)
                frame_time = self.resolve_scene_time(scene_times, frame_id, scene_order_idx)

                scene_records.append(
                    {
                        'img_0_path': img_path,
                        'sequence': sequence,
                        'frame_id': frame_id,
                        'frame_order_idx': scene_order_idx,
                        'frame_time': frame_time,
                        'pose': frame_pose,
                        'P0': P0,
                        'T_velo_2_cam_base': T_velo_2_cam_base,
                        'voxel_path': voxel_path,
                        'voxel_count_path': voxel_count_path,
                        'stereo_depth_path': stereo_depth_path,
                        'teacher_depth_path': teacher_depth_path,
                        'preprocessed_voxel_path': self.resolve_preprocessed_voxel_path(
                            sequence=sequence,
                            frame_id=frame_id,
                        ),
                        'lidar_path': lidar_path,
                        'lidar_label_path': lidar_label_path,
                        'source_occ_size': scene_params['occ_size'],
                        'source_pc_range': scene_params['pc_range'],
                        'scene_num_channels': scene_params['num_channels'],
                    }
                )
            for scene_order_idx, record in enumerate(scene_records):
                temporal_frames = []
                for offset in self.temporal_offsets:
                    temporal_idx = int(np.clip(scene_order_idx + offset, 0, len(scene_records) - 1))
                    temporal_frames.append(scene_records[temporal_idx])
                record['temporal_frames'] = temporal_frames
                center_relative = self.compute_relative_lidar_transform(
                    source_pose=record['pose'],
                    target_pose=record['pose'],
                )
                record['proj_matrix_0'] = record['P0'] @ (record['T_velo_2_cam_base'] @ center_relative)
                record['T_velo_2_cam'] = record['T_velo_2_cam_base'] @ center_relative
                scans.append(record)
        return scans

    def get_ann_info(self, index, key='voxel_path'):
        info_dict = self.data_infos[index]
        preprocessed_info = info_dict.get('preprocessed_voxel_path')
        if self.prefer_preprocessed_labels and preprocessed_info is not None and os.path.exists(preprocessed_info):
            label = np.load(preprocessed_info)
            return self.remap_training_labels(label)

        info = info_dict[key]
        if info is None:
            return None

        source_occ_size = np.array(info_dict.get('source_occ_size', self.occ_size), dtype=np.int32)
        source_pc_range = np.array(info_dict.get('source_pc_range', self.pc_range), dtype=np.float32)

        label = np.fromfile(info, dtype=np.uint32)
        source_num_voxels = int(np.prod(source_occ_size))
        if label.size != source_num_voxels:
            raise ValueError(
                f'Unexpected occupancy label length {label.size} for source occ_size {source_occ_size.tolist()}'
            )

        label = label.reshape(tuple(source_occ_size.tolist()))
        count = None
        if self.use_occ_count_mask:
            count_info = info_dict.get('voxel_count_path')
            if count_info is not None and os.path.exists(count_info):
                count = np.fromfile(count_info, dtype=np.float32)
                if count.size != source_num_voxels:
                    raise ValueError(
                        f'Unexpected occupancy count length {count.size} for source occ_size {source_occ_size.tolist()}'
                    )
                count = count.reshape(tuple(source_occ_size.tolist()))

        label = self.remap_occ_to_target(
            label,
            count=count,
            source_occ_size=source_occ_size,
            source_pc_range=source_pc_range,
            target_occ_size=self.target_occ_size,
            target_pc_range=self.target_pc_range,
        )
        return self.remap_training_labels(label)

    def resolve_preprocessed_voxel_path(self, sequence, frame_id):
        if self.ann_file is None or self.ann_file == self.data_root:
            return None

        candidate = os.path.join(
            self.ann_file,
            self.preprocess_label_subdir,
            sequence,
            f'{frame_id}_{self.preprocess_scale}.npy',
        )
        return candidate if os.path.exists(candidate) else None

    def build_label_mapping(self, label_mapping):
        if label_mapping is None:
            return None

        if isinstance(label_mapping, dict):
            max_label = max(max(label_mapping.keys()), self.ignore_label)
            lut = np.full(max_label + 1, self.ignore_label, dtype=np.uint16)
            for raw_label, compact_label in label_mapping.items():
                lut[int(raw_label)] = int(compact_label)
            return lut

        lut = np.asarray(label_mapping, dtype=np.uint16)
        if lut.ndim != 1:
            raise ValueError('label_mapping must be a 1D LUT or a dict')
        return lut

    def remap_training_labels(self, label):
        label = label.astype(np.uint16, copy=True)

        if self.label_mapping is not None:
            invalid_mask = label >= self.label_mapping.shape[0]
            if np.any(invalid_mask):
                label[invalid_mask] = self.ignore_label
            valid_mask = ~invalid_mask
            if np.any(valid_mask):
                label[valid_mask] = self.label_mapping[label[valid_mask]]

        if self.num_classes is not None:
            label[label >= self.num_classes] = self.ignore_label

        return label.astype(np.uint8)

    @staticmethod
    def read_scene_params(params_path):
        if not os.path.exists(params_path):
            raise FileNotFoundError(f'Missing scene params file: {params_path}')

        with open(params_path, 'r') as f:
            params = json.load(f)

        occ_size = np.array([int(v) for v in params['grid_size']], dtype=np.int32)
        min_bound = np.array(params['min_bound'], dtype=np.float32)
        max_bound = np.array(params['max_bound'], dtype=np.float32)
        pc_range = np.concatenate([min_bound, max_bound]).astype(np.float32)

        return {
            'occ_size': occ_size,
            'pc_range': pc_range,
            'num_channels': int(params.get('num_channels', 0)),
            'coordinates': params.get('coordinates', 'cartesian'),
        }

    def remap_occ_to_target(
        self,
        label,
        count,
        source_occ_size,
        source_pc_range,
        target_occ_size,
        target_pc_range,
    ):
        if (
            np.array_equal(source_occ_size, target_occ_size)
            and np.allclose(source_pc_range, target_pc_range)
        ):
            if count is None:
                return label
            target = label.copy()
            target[count <= self.valid_threshold] = self.ignore_label
            return target

        if count is not None:
            target = np.full(tuple(target_occ_size.tolist()), self.ignore_label, dtype=np.uint32)
        else:
            target = np.zeros(tuple(target_occ_size.tolist()), dtype=np.uint32)

        source_min = source_pc_range[:3]
        source_max = source_pc_range[3:]
        target_min = target_pc_range[:3]
        target_max = target_pc_range[3:]

        source_steps = (source_max - source_min) / source_occ_size
        target_steps = (target_max - target_min) / target_occ_size

        if count is not None:
            valid_seen_coords = np.argwhere(count > self.valid_threshold)
            if valid_seen_coords.size > 0:
                valid_seen_centers = source_min + (valid_seen_coords.astype(np.float32) + 0.5) * source_steps
                valid_seen_target = np.floor((valid_seen_centers - target_min) / target_steps).astype(np.int32)
                valid_seen_mask = np.all(
                    (valid_seen_target >= 0) & (valid_seen_target < target_occ_size),
                    axis=1,
                )
                if np.any(valid_seen_mask):
                    valid_seen_target = valid_seen_target[valid_seen_mask]
                    valid_seen_indices = np.ravel_multi_index(
                        valid_seen_target.T,
                        tuple(target_occ_size.tolist()),
                    )
                    target.reshape(-1)[valid_seen_indices] = 0

        occupied_mask = (label > 0) & (label != self.ignore_label)
        if count is not None:
            occupied_mask &= count > self.valid_threshold

        occupied_coords = np.argwhere(occupied_mask)
        if occupied_coords.size == 0:
            return target

        occupied_labels = label[tuple(occupied_coords.T)]
        occupied_centers = source_min + (occupied_coords.astype(np.float32) + 0.5) * source_steps
        target_coords = np.floor((occupied_centers - target_min) / target_steps).astype(np.int32)

        valid_mask = np.all((target_coords >= 0) & (target_coords < target_occ_size), axis=1)
        if not np.any(valid_mask):
            return target

        target_coords = target_coords[valid_mask]
        occupied_labels = occupied_labels[valid_mask]

        target_flat = target.reshape(-1)
        flat_indices = np.ravel_multi_index(target_coords.T, tuple(target_occ_size.tolist()))

        if self.label_merge_mode == 'first':
            empty_mask = target_flat[flat_indices] == 0
            target_flat[flat_indices[empty_mask]] = occupied_labels[empty_mask]
        elif self.label_merge_mode == 'max':
            np.maximum.at(target_flat, flat_indices, occupied_labels)
        elif self.label_merge_mode in {'mode', 'vote'}:
            self.merge_labels_by_mode(target_flat, flat_indices, occupied_labels)
        else:
            raise ValueError(f'Unsupported label_merge_mode: {self.label_merge_mode}')

        return target

    @staticmethod
    def merge_labels_by_mode(target_label, flat_indices, labels):
        if flat_indices.size == 0:
            return

        order = np.argsort(flat_indices, kind='mergesort')
        sorted_indices = flat_indices[order]
        sorted_labels = labels[order].astype(np.int64)

        unique_indices, start_positions = np.unique(sorted_indices, return_index=True)
        end_positions = np.append(start_positions[1:], sorted_indices.size)

        for target_idx, start, end in zip(unique_indices, start_positions, end_positions):
            votes = np.bincount(sorted_labels[start:end])
            target_label[target_idx] = np.argmax(votes)

    @staticmethod
    def read_scene_poses(poses_path):
        if not os.path.exists(poses_path):
            return []
        poses = []
        with open(poses_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = np.fromstring(line, sep=' ', dtype=np.float32)
                if values.size == 12:
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :4] = values.reshape(3, 4)
                elif values.size == 16:
                    pose = values.reshape(4, 4).astype(np.float32)
                else:
                    raise ValueError(f'Unsupported pose format in {poses_path}: {values.size} values')
                poses.append(pose)
        return poses

    @staticmethod
    def read_scene_times(times_path):
        if not os.path.exists(times_path):
            return []
        times = []
        with open(times_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                times.append(float(line))
        return times

    @staticmethod
    def resolve_scene_pose(scene_poses, frame_id, scene_order_idx):
        frame_numeric_idx = int(frame_id)
        if frame_numeric_idx < len(scene_poses):
            return scene_poses[frame_numeric_idx]
        if scene_order_idx < len(scene_poses):
            return scene_poses[scene_order_idx]
        return np.eye(4, dtype=np.float32)

    @staticmethod
    def resolve_scene_time(scene_times, frame_id, scene_order_idx):
        frame_numeric_idx = int(frame_id)
        if frame_numeric_idx < len(scene_times):
            return scene_times[frame_numeric_idx]
        if scene_order_idx < len(scene_times):
            return scene_times[scene_order_idx]
        return float(scene_order_idx)

    @staticmethod
    def compute_relative_lidar_transform(source_pose, target_pose):
        return np.linalg.inv(target_pose).astype(np.float32) @ source_pose.astype(np.float32)

    @staticmethod
    def collect_frame_ids(image_base_path, lidar_base_path):
        image_exts = ('*.png', '*.jpg', '*.jpeg')
        frame_ids = set()
        for pattern in image_exts:
            for image_path in glob.glob(os.path.join(image_base_path, pattern)):
                frame_ids.add(os.path.splitext(os.path.basename(image_path))[0])

        lidar_ids = {
            os.path.splitext(os.path.basename(path))[0]
            for path in glob.glob(os.path.join(lidar_base_path, '*.bin'))
        }
        return sorted(frame_ids & lidar_ids)

    @staticmethod
    def find_first_image(image_base_path, frame_id):
        for ext in ('.png', '.jpg', '.jpeg'):
            image_path = os.path.join(image_base_path, frame_id + ext)
            if os.path.exists(image_path):
                return image_path
        raise FileNotFoundError(f'Image for frame {frame_id} not found in {image_base_path}')

    @staticmethod
    def build_topdown_calib(image_path, pc_range, camera_height):
        from PIL import Image

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

        P0 = np.eye(4, dtype=np.float32)
        P0[:3, :4] = np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        T_velo_2_cam = np.eye(4, dtype=np.float32)
        T_velo_2_cam[:3, :4] = np.array(
            [
                [0.0, 1.0, 0.0, -center_y],
                [-1.0, 0.0, 0.0, center_x],
                [0.0, 0.0, -1.0, camera_height],
            ],
            dtype=np.float32,
        )
        return {'P0': P0, 'Tr': T_velo_2_cam}

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
