# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16

import copy


@DETECTORS.register_module()
class FastBEV(BaseDetector):
    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        n_voxels,
        voxel_size,
        bbox_head=None,
        seg_head=None,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
        style='v4',
        n_cams=6,
        fisheye=False,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        self.backproject = backproject
        # checkpoint
        self.with_cp = with_cp
        self.n_cams = n_cams
        self.fisheye = fisheye

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    @staticmethod
    def _fisheye_project_torch(X, Y, Z, K, D, eps=1e-6):
        valid_z = Z > eps
        Zc = torch.clamp(Z, min=eps)
        x = X / Zc
        y = Y / Zc
        r = torch.sqrt(x * x + y * y)
        theta = torch.atan(r)

        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        k1 = D[:, 0:1]
        k2 = D[:, 1:2]
        k3 = D[:, 2:3]
        k4 = D[:, 3:4]
        theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)

        scale = theta_d / torch.clamp(r, min=eps)
        xd = x * scale
        yd = y * scale

        fx = K[:, 0, 0:1]
        fy = K[:, 1, 1:2]
        cx = K[:, 0, 2:3]
        cy = K[:, 1, 2:3]

        u = fx * xd + cx
        v = fy * yd + cy
        return u, v, valid_z

    def _backproject_inplace_fisheye(
        self,
        features,
        points,
        cam2ego,
        K,
        D,
        post_rot,
        post_tran,
        stride,
        eps=1e-6,
    ):
        ncam, c, h, w = features.shape
        dx, dy, dz = points.shape[-3:]
        p = dx * dy * dz
        device = features.device
        dtype = features.dtype
        geom_dtype = torch.float32

        cam2ego = cam2ego.to(device=device, dtype=geom_dtype)
        K = K.to(device=device, dtype=geom_dtype)
        D = D.to(device=device, dtype=geom_dtype)
        post_rot = post_rot.to(device=device, dtype=geom_dtype)
        post_tran = post_tran.to(device=device, dtype=geom_dtype)
        points = points.to(device=device, dtype=geom_dtype)

        pts = points.view(3, -1)
        pts = torch.cat([pts, torch.ones((1, p), device=device, dtype=geom_dtype)], dim=0)
        pts = pts.unsqueeze(0).expand(ncam, 4, p)
        ego2cam = torch.linalg.inv(cam2ego)
        cam_pts = torch.bmm(ego2cam, pts)[:, :3, :]
        X, Y, Z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]

        u, v, valid_z = self._fisheye_project_torch(X, Y, Z, K, D, eps=eps)
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        uv1 = torch.matmul(post_rot.unsqueeze(1), uv1.unsqueeze(-1)).squeeze(-1)
        uv1 = uv1 + post_tran.unsqueeze(1)
        u = uv1[..., 0] / stride
        v = uv1[..., 1] / stride

        xi = torch.round(u).long()
        yi = torch.round(v).long()
        valid = valid_z & (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)

        vol = torch.zeros((c, p), device=device, dtype=dtype)
        for i in range(ncam):
            m = valid[i]
            if m.any():
                vol[:, m] = features[i, :, yi[i, m], xi[i, m]]
        return vol.view(c, dx, dy, dz)

    def _build_fisheye_inputs(self, img_meta, seq_id, n_cams, device, dtype):
        lidar2img_aug = img_meta.get("lidar2img", {}).get("lidar2img_aug", None)
        if not lidar2img_aug:
            return None
        cam_infos = lidar2img_aug[seq_id * n_cams:(seq_id + 1) * n_cams]
        if len(cam_infos) != n_cams:
            return None

        cam2ego_list, intrin_list, dist_list, post_rot_list, post_tran_list = [], [], [], [], []
        for cam in cam_infos:
            dist = cam.get("dist", None)
            lidar2cam = cam.get("lidar2cam_rt", None)
            intrin = cam.get("intrin", None)
            if dist is None or lidar2cam is None or intrin is None:
                return None
            cam2ego = np.linalg.inv(np.array(lidar2cam, dtype=np.float32))
            post_rot = cam.get("post_rot", np.eye(3, dtype=np.float32))
            post_tran = cam.get("post_tran", np.zeros(3, dtype=np.float32))

            cam2ego_list.append(cam2ego)
            intrin_list.append(np.array(intrin, dtype=np.float32))
            dist_list.append(np.array(dist, dtype=np.float32)[:4])
            post_rot_list.append(np.array(post_rot, dtype=np.float32))
            post_tran_list.append(np.array(post_tran, dtype=np.float32))

        return dict(
            cam2ego=torch.tensor(np.stack(cam2ego_list), device=device, dtype=dtype),
            intrin=torch.tensor(np.stack(intrin_list), device=device, dtype=dtype),
            dist=torch.tensor(np.stack(dist_list), device=device, dtype=dtype),
            post_rot=torch.tensor(np.stack(post_rot_list), device=device, dtype=dtype),
            post_tran=torch.tensor(np.stack(post_tran_list), device=device, dtype=dtype),
        )

    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)

        mlvl_feats = self.neck(x)
        mlvl_feats = list(mlvl_feats)

        features_2d = None

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, self.n_cams, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][
                        seq_id * self.n_cams:(seq_id + 1) * self.n_cams]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][
                            seq_id * self.n_cams:(seq_id + 1) * self.n_cams]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)
                    
                    if self.style in ['v1', 'v2']:
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    
                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                    ).to(feat_i.device)

                    if self.backproject == 'inplace':
                        fisheye_inputs = None
                        if self.fisheye:
                            fisheye_inputs = self._build_fisheye_inputs(
                                seq_img_meta, seq_id, self.n_cams, feat_i.device, feat_i.dtype)
                        if fisheye_inputs is not None:
                            volume = self._backproject_inplace_fisheye(
                                feat_i[:, :, :height, :width],
                                points,
                                fisheye_inputs["cam2ego"],
                                fisheye_inputs["intrin"],
                                fisheye_inputs["dist"],
                                fisheye_inputs["post_rot"],
                                fisheye_inputs["post_tran"],
                                stride_i,
                            )
                        else:
                            projection = self._compute_projection(
                                img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                            volume = backproject_inplace(
                                feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                    

                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
        
        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]

        x = mlvl_volumes
        x = self.neck_3d(x)

        return x, None, features_2d

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if kwargs["export_2d"]:
                return self.onnx_export_2d(img, img_metas)
            elif kwargs["export_3d"]:
                return self.onnx_export_3d(img, img_metas)
            else:
                raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        feature_bev, valids, features_2d = self.extract_feat(img, img_metas, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None:
            gt_bboxes = kwargs["gt_bboxes"][0]
            gt_labels = kwargs["gt_labels"][0]
            assert len(kwargs["gt_bboxes"]) == 1 and len(kwargs["gt_labels"]) == 1
            # hack a img_metas_2d
            img_metas_2d = []
            img_info = img_metas[0]["img_info"]
            for idx, info in enumerate(img_info):
                tmp_dict = dict(
                    filename=info["filename"],
                    ori_filename=info["filename"].split("/")[-1],
                    ori_shape=img_metas[0]["ori_shape"],
                    img_shape=img_metas[0]["img_shape"],
                    pad_shape=img_metas[0]["pad_shape"],
                    scale_factor=img_metas[0]["scale_factor"],
                    flip=False,
                    flip_direction=None,
                )
                img_metas_2d.append(tmp_dict)

            rank, world_size = get_dist_info()
            loss_2d = self.bbox_head_2d.forward_train(
                features_2d, img_metas_2d, gt_bboxes, gt_labels
            )
            losses.update(loss_2d)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        if not self.test_cfg.get('use_tta', False):
            return self.simple_test(img, img_metas)
        return self.aug_test(img, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = resize(
            c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c3 = resize(
            c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c4 = resize(
            c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.neck_fuse(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        # x: [6, 200, 100, 3, 256]
        # if bool(os.getenv("DEPLOY_DEBUG", False)):
        #     x = x.sum(dim=0, keepdim=True)
        #     return [x]
        if self.style == "v1":
            x = x.sum(dim=0, keepdim=True)  # [1, 200, 100, 3, 256]
            x = self.neck_3d(x)  # [[1, 256, 100, 50], ]
        elif self.style == "v2":
            x = self.neck_3d(x)  # [6, 256, 100, 50]
            x = [x[0].sum(dim=0, keepdim=True)]  # [1, 256, 100, 50]
        elif self.style == "v3":
            x = self.neck_3d(x)  # [1, 256, 100, 50]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            cls_score = [item.sigmoid() for item in cls_score]

        if os.getenv("DEPLOY", False):
            if dir_cls_preds is None:
                x = [cls_score, bbox_pred]
            else:
                x = [cls_score, bbox_pred, dir_cls_preds]
            return x

        return x

    def simple_test(self, img, img_metas):
        bbox_results = []
        feature_bev, _, features_2d = self.extract_feat(img, img_metas, "test")
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]

        else:
            bbox_results = [dict()]

        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev

        return bbox_results

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])
        per_tta = len(extrinsic_copy) // 2

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[per_tta * tta_id:per_tta * (tta_id + 1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[per_tta * tta_id:per_tta * (tta_id + 1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, per_tta * tta_id:per_tta * (tta_id + 1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


@DETECTORS.register_module()
class FastBEVFish(BaseDetector):
    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        n_voxels,
        voxel_size,
        psd_head,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        multi_scale_id=None,
        backproject='inplace',
        style='v1',
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        if psd_head is not None:
            psd_head.update(train_cfg=train_cfg.pts)
            psd_head.update(test_cfg=test_cfg)
            self.psd_head = build_head(psd_head)

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.backproject = backproject
        self.style = style
        self.multi_scale_id = multi_scale_id

    def fisheye_project_torch(
        self,
        X, Y, Z,
        K, D,
        eps=1e-6,
    ):
        """
        X,Y,Z: (Ncam,P) cam coords
        K: (Ncam,3,3)
        D: (Ncam,4)  k1..k4
        return:
        u,v: (Ncam,P) float
        valid_z: (Ncam,P) bool
        """
        valid_z = Z > eps
        Zc = torch.clamp(Z, min=eps)

        x = X / Zc
        y = Y / Zc
        r = torch.sqrt(x * x + y * y)                 # (Ncam,P)
        theta = torch.atan(r)                         # (Ncam,P)

        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        k1 = D[:, 0:1]
        k2 = D[:, 1:2]
        k3 = D[:, 2:3]
        k4 = D[:, 3:4]

        theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)

        scale = theta_d / torch.clamp(r, min=eps)
        xd = x * scale
        yd = y * scale

        fx = K[:, 0, 0:1]
        fy = K[:, 1, 1:2]
        cx = K[:, 0, 2:3]
        cy = K[:, 1, 2:3]

        u = fx * xd + cx
        v = fy * yd + cy
        return u, v, valid_z

    def backproject_inplace_fisheye(
        self,
        features,   # (Ncam,C,H,W)
        points,     # (3,Dx,Dy,Dz)
        cam2ego,    # (Ncam,4,4)
        K,          # (Ncam,3,3)
        D,          # (Ncam,4)
        post_rot,
        post_tran,
        stride,
        eps=1e-6,
    ):
        """
        output:
        volume: (C,Dx,Dy,Dz)   # overwrite by camera order (0..Ncam-1)
        """
        assert features.dim() == 4
        Ncam, C, H, W = features.shape
        assert points.shape[0] == 3

        Dx, Dy, Dz = points.shape[-3:]
        P = Dx * Dy * Dz
        device = features.device
        dtype = features.dtype
        geom_dtype = torch.float32

        cam2ego = cam2ego.to(device=device, dtype=geom_dtype)
        K = K.to(device=device, dtype=geom_dtype)
        D = D.to(device=device, dtype=geom_dtype)
        post_rot = post_rot.to(device=device, dtype=geom_dtype)
        post_tran = post_tran.to(device=device, dtype=geom_dtype)
        points = points.to(device=device, dtype=geom_dtype)

        # points -> (Ncam,4,P)
        pts = points.view(3, -1)
        pts = torch.cat([pts, torch.ones((1, P), device=device, dtype=pts.dtype)], dim=0)  # (4,P)
        pts = pts.unsqueeze(0).expand(Ncam, 4, P)

        ego2cam = torch.linalg.inv(cam2ego)
        cam_pts = torch.bmm(ego2cam, pts)[:, :3, :]  # (Ncam,3,P)
        X = cam_pts[:, 0, :]
        Y = cam_pts[:, 1, :]
        Z = cam_pts[:, 2, :]

        u, v, valid_z = self.fisheye_project_torch(X, Y, Z, K, D, eps=eps)
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)   # (Ncam, P, 3)

        uv1 = torch.matmul(
            post_rot.unsqueeze(1),   # (Ncam, 1, 3, 3)
            uv1.unsqueeze(-1)        # (Ncam, P, 3, 1)
        ).squeeze(-1)                # (Ncam, P, 3)

        uv1 = uv1 + post_tran.unsqueeze(1)   # (Ncam, P, 3)

        u = uv1[..., 0]
        v = uv1[..., 1]
        u = u / stride
        v = v / stride

        xi = torch.round(u).long()
        yi = torch.round(v).long()

        valid = valid_z & (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)

        vol = torch.zeros((C, P), device=device, dtype=dtype)

        for i in range(Ncam):
            m = valid[i]
            if m.any():
                vol[:, m] = features[i, :, yi[i, m], xi[i, m]]

        vol = vol.view(C, Dx, Dy, Dz)
        return vol

    def get_points(self, n_voxels, voxel_size, origin):
        points = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(n_voxels[0]),
                    torch.arange(n_voxels[1]),
                    torch.arange(n_voxels[2]),
                ]
            )
        )
        new_origin = origin - n_voxels / 2.0 * voxel_size
        points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
        return points

    def forward(self, return_loss=True, **kwargs):

        if return_loss:
            return self.forward_train(kwargs['img_metas'], 
                                      kwargs['img_inputs'], 
                                      kwargs['corner_points'], 
                                      kwargs['spe'], 
                                      kwargs['opy'],
                                      kwargs['wheel_stop_points'], 
                                      kwargs['canvas'], 
                                      )
        else:
            return self.forward_test(kwargs['img_metas'], kwargs['img_inputs'])

    def forward_train(
        self, img_metas, img_inputs, corner_points, spe, opy, wheel_stop_points, canvas
    ):
        
        img_tr, cam2ego_tr,  intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda_rot = self.prepare_inputs(img_inputs)
        feature_bev, valids, features_2d = self.extract_feat(img_tr, cam2ego_tr,  intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda_rot, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        

        losses = dict()
        if self.psd_head is not None:
            x = self.psd_head(feature_bev)
            loss_psd_dict = self.psd_head.loss(x, spe, opy, corner_points, wheel_stop_points)
            losses.update(loss_psd_dict)

        return losses

    def forward_test(self, img_metas, img_inputs):
        img_tr, cam2ego_tr,  intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda_rot = self.prepare_inputs(img_inputs)
        feature_bev, valids, features_2d = self.extract_feat(img_tr, cam2ego_tr,  intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda_rot, "test")
        parkinglot_preds = []
        if self.psd_head is not None:
            x = self.psd_head(feature_bev)
            hm_logit = x[0]['heatmap']
            # print('heatmap shape:', hm_logit.shape)
            # print('slot hm logit max:', hm_logit[:, 0].max().item())
            # print('slot hm logit mean:', hm_logit[:, 0].mean().item())
            # print('ws   hm logit max:', hm_logit[:, 1].max().item())
            # print('ws   hm logit mean:', hm_logit[:, 1].mean().item())

            # hm_prob = torch.sigmoid(hm_logit)
            # print('slot hm prob max:', hm_prob[:, 0].max().item())
            # print('slot hm prob mean:', hm_prob[:, 0].mean().item())
            # print('ws   hm prob max:', hm_prob[:, 1].max().item())
            # print('ws   hm prob mean:', hm_prob[:, 1].mean().item())
            parkinglot_preds = self.psd_head.get_bboxes(x, img_metas) 
        return parkinglot_preds



    def aug_test(self):
        pass

    def simple_test(self):
        pass

    def prepare_inputs(self, inputs):
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape

        img_tr, cam2ego_tr, intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda_rot = inputs


        cam2ego_tr = cam2ego_tr.view(B, N, 4, 4)
        

        return [img_tr, cam2ego_tr,  intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda_rot]

    def extract_feat(self, img_tr, cam2ego_tr,  intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda_rot, mode):
        batch_size = img_tr.shape[0]   # img_tr.shape [24, 4, 3, 768, 960]
        cam_num = img_tr.shape[1]
        img = img_tr.reshape([-1] + list(img_tr.shape)[2:]) # [24 * 4, 3, 768, 960]
        x = self.backbone(img)  # x[0] = [24 * 4, 256, 192, 240], x[1] = [24 * 4, 512, 96, 120], x[2] = [24 * 4, 1024, 48, 60], x[3] = [24 * 4, 2048, 24, 30]

        mlvl_feats = self.neck(x) # x[0] = [24 * 4, 64, 192, 240], x[1] = [24 * 4, 64, 96, 120], x[2] = [24 * 4, 64, 48, 60], x[3] = [24 * 4, 64, 24, 30]
        mlvl_feats = list(mlvl_feats)

        features_2d = None

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            mlvl_feat_split = torch.split(mlvl_feat, cam_num, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []

                for batch_id in range(len(mlvl_feat_split[seq_id])):
                    feat_i = mlvl_feat_split[seq_id][batch_id]
                    cam2ego_i = cam2ego_tr[batch_id][seq_id*cam_num:(seq_id+1)*cam_num]
                    intrin_i = intrin_tr[batch_id][seq_id*cam_num:(seq_id+1)*cam_num]
                    dist_i = dist_tr[batch_id][seq_id*cam_num:(seq_id+1)*cam_num]
                    post_rot_i = post_rot_tr[batch_id][seq_id*cam_num:(seq_id+1)*cam_num]
                    post_tran_i = post_tran_tr[batch_id][seq_id*cam_num:(seq_id+1)*cam_num]
                    bda_rot_i = bda_rot[batch_id][seq_id*cam_num:(seq_id+1)*cam_num]

                    
                    if self.style in ['v1', 'v2']:
                        n_voxels, voxel_size = self.n_voxels, self.voxel_size
                    
                    points = self.get_points(
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor([0, 0, 0]),
                    ).to(feat_i.device)

                    if self.backproject == 'inplace':
                        volume = self.backproject_inplace_fisheye(feat_i, points, cam2ego_i, intrin_i, dist_i, post_rot_i, post_tran_i, stride_i)
                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))
        
        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)

        x = mlvl_volumes
        x = self.neck_3d(x)

        return x, None, features_2d

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss, log_vars=log_vars_, num_samples=len(data['img_metas']))

        return outputs
