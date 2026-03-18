import os
import mmcv
import numpy as np
from pyquaternion import Quaternion
from ..core.bbox import get_box_type
from .builder import DATASETS
from .pipelines import Compose



@DATASETS.register_module(force=True)
class FishDataset():
    psd_shape = {'parallel': 0, 'vertival': 1, 'slanted': 2}
    psd_occupy = {'non_occ': 0, 
                  'self_occ': 1, 
                  'vehicle_occ': 2, 
                  'closed_lock_occ': 3, 
                  'open_lock_occ': 4, 
                  'cone_occ': 5, 
                  'other_occ': 6}
    CLASSES = None
    PALETTE = None


    def __init__(self,
                 ann_dir = 'jsons',
                 pipeline=None,
                 data_root=None,
                 load_interval=1,
                 with_velocity=True,
                 modality = dict(
                    use_lidar=True,
                    use_camera=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False),
                 filter_empty_gt=True,
                 box_type_3d='LiDAR',
                 test_mode=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 stereo=False,
                 cam_names=['CAM_FISHEYE_FORWARD', 'CAM_FISHEYE_LEFT', 'CAM_FISHEYE_BACKWARD', 'CAM_FISHEYE_RIGHT']):
        
        self.modality = modality
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.stereo = stereo
        

        self.test_mode = test_mode
        self.cam_names = cam_names
        self.data_root = data_root
        self.data_infos = self.load_annotations(os.path.join(self.data_root, ann_dir))
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.flag = np.zeros(len(self), dtype=np.uint8)

        # temp proc
        self.data_dir = '/vePFS/data/zhixing_data'

    def load_annotations(self, json_dir):
        files = [os.path.join(json_dir, f) for f in os.listdir(json_dir)]
        data = [mmcv.load(f, file_format='json') for f in files]
        data_infos = list(sorted(data, key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        return data_infos

    def read_calib(self, calib_path):
        fish = mmcv.load(calib_path, file_format='json')['fisheye']
        rotation_M = Quaternion(fish['rotation']).rotation_matrix
        E = np.eye(4)
        E[:3, :3] = np.array(rotation_M, dtype=float).reshape(3, 3)
        E[:3, 3] = np.array(fish['translation'], dtype=float).reshape(3,)
        instr_M = np.array(fish['camera_intrinsic'], dtype=float).reshape(3, 3)
        dist = fish['distortion']

        return E, instr_M, dist
    
    def read_lidar_calib(self, calib_path):
        body = mmcv.load(calib_path, file_format='json')['body']
        rotation_M = Quaternion(body['rotation']).rotation_matrix
        E = np.eye(4)
        E[:3, :3] = np.array(rotation_M, dtype=float).reshape(3, 3)
        E[:3, 3] = np.array(body['translation'], dtype=float).reshape(3,)
        return E

    def get_data_info(self, index):
        info = self.data_infos[index]

        scene_id = info['scene_id']
        data_dir = os.path.join(self.data_dir, scene_id)

        input_dict = dict(
            timestamp = int(info['timestamp']),
            annotations = info['annotations'],
            cam_names = self.cam_names
        )

        if self.modality['use_camera']:
            image_paths = []
            calibs = []
            if self.img_info_prototype == 'mmcv':
                for cam_name in self.cam_names:
                    image_paths.append(os.path.join(data_dir, info['image_path'][cam_name]))
                    calib_path = os.path.join(data_dir, info['calib'][cam_name])
                    calibs.append(self.read_calib(calib_path))

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        calibs=calibs,
                    ))
    
                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos

            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))

        if self.modality['use_lidar']:
            lidar_path = os.path.join(data_dir, info['lidar']['lidar_path'])
            calib_path = info['lidar']['calib_path']
            E = self.read_lidar_calib(os.path.join(data_dir, calib_path))
            input_dict.update(
                dict(
                    lidar_path = lidar_path,
                    lidar_calib = E
                )
            )

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        parking_slot_detection = info['annotations']['parking_slot_detection']
        psd_list = []
        spe_list = []
        opy_list = []
        for psd in parking_slot_detection:
            if psd['type'] == 'parking_slot':
                points = np.array([[corner['x'], corner['y'], corner['z']] for corner in psd['points_3d']])
                psd_list.append(points)
                spe_list.append(psd['attributes']['shape'])
                opy_list.append(psd['attributes']['occupy'])

        ann_results = dict(
            corner_points = psd_list,
            spe = spe_list,
            opy = opy_list 
        )

        return ann_results

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        if self.pipeline is not None:
            example = self.pipeline(input_dict)
        else:
            return input_dict
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    
@DATASETS.register_module(force=True)
class FishDatasetStop(FishDataset):
    """FishDatasetV2 with parking slot and wheel stop detection.
    This dataset includes both parking slot detection and wheel stop detection functionality.
    Wheel stops are annotated with two endpoints.
    """

    def get_ann_info(self, index):
        info = self.data_infos[index]
        parking_slot_detection = info['annotations']['parking_slot_detection']
        psd_list = []
        spe_list = []
        opy_list = []
        wheel_stop_list = []
        for psd in parking_slot_detection:
            if psd['type'] == 'parking_slot':
                points = np.array([[corner['x'], corner['y'], corner['z']] for corner in psd['points_3d']])
                psd_list.append(points)
                spe_list.append(psd['attributes']['shape'])
                opy_list.append(psd['attributes']['occupy'])
            elif psd['type'] == 'wheel_stop':
                points = np.array([[corner['x'], corner['y'], corner['z']] for corner in psd['points_3d']])
                wheel_stop_list.append(points)

        ann_results = dict(
            corner_points = psd_list,
            spe = spe_list,
            opy = opy_list,
            wheel_stop_points = wheel_stop_list
        )

        return ann_results

if __name__ == '__main__':
    data_root = '/Workspace/BEV_PLD/data/zhixing_data_0108/P_SuZhou_20250506-223928_E0Y-4297_0_scene175'
    fish_set = FishDatasetStop(data_root=data_root)
    sample = fish_set[0]
    print(sample)
    import pdb;pdb.set_trace()