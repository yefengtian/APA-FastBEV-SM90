# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
import torch.nn.functional as F

@BBOX_CODERS.register_module()
class CenterPointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor): Mask of the feats. Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor): Regression value of the boxes in 2D with
                the shape of [B, 2, W, H]. Default: None.
            task_id (int): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        xs = xs.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts

@BBOX_CODERS.register_module(force=True)
class CenterPointParkingspotBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: 0.001.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=0.001,
                 code_size=9,
                 nms_kernel_size=3):

        self.pc_range = pc_range    # [x_min, y_min, ...]
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range  # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size
        self.nms_kernel_size = nms_kernel_size

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor, optional): Mask of the feats.
                Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of (B, N_cls, H, W).
            K (int, optional): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()

        # 先是针对每个类别的预测都取topK.
        # (B, N_cls, K), (B, N_cls, K)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)    # (B, N_cls, K), topK对应的像素索引(0, H*W-1).
        topk_ys = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float()    # (B, N_cls, K), y坐标.
        topk_xs = (topk_inds % width).int().float()     # (B, N_cls, K), x坐标.

        # 然后对将所有类别得到的topK数据再次进行topK.
        # (B, K), (B, K)
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()      # (B, K)  对应的类别.
        # (B, N_cls*K, 1) --gather--> (B, K, 1) --> (B, K)  topK对应的像素坐标索引(0, H*W-1).
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K)
        # (B, N_cls*K, 1) --gather--> (B, K, 1) --> (B, K)  topK对应的y坐标.
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K)
        # (B, N_cls*K, 1) --gather--> (B, K, 1) --> (B, K)  topK对应的x坐标.
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of (B, N_c, H, W).
            ind (torch.Tensor): Indexes with the shape of [B, K].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        # (B, N_c, H, W) --> (B, H, W, N_c) --> (B, H*W, N_c)
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)     # (B, K, N_c)
        return feat

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def encode(self):
        pass

    def decode(self,
                heatmap,    # heatmap: (B, 3, H, W) 3:['perpendicular', 'parallel', 'other']
                status,     # availability: (B, 3, H, W)
                reg,        # ctr_offset: (B, 2, H, W)
                kp0,        # (B, 2, H, W)
                kp1,        # (B, 2, H, W)
                kp2,        # (B, 2, H, W)
                kp3,        # (B, 2, H, W)
                task_id=-1):

        status_cls = status.shape[1]
        # NMS + Topk, x&y are in canvas coordinate, x points right, y points down, located at top-left corner
        heat = self._nms(heatmap, kernel=self.nms_kernel_size)
        batch, cat, _, _ = heat.size()
        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        # scores
        scores = scores.view(batch, self.max_num)   # (B, K)
        # print('scores:', scores)
        # class label (type: ['perpendicular', 'parallel', 'other'])
        clses = clses.view(batch, self.max_num).float()     # (B, K)

        # status (availability) by argmax of status at inds locations obtained from heatmap topk
        status = self._transpose_and_gather_feat(status, inds)      # (B, K, 3)
        status = status.view(batch, self.max_num, status_cls)                # (B, K, 3)
        status = status.argmax(dim=2)                               # (B, K)
        status = status.view(batch, self.max_num).float()           # (B, K)

        # Centerpoint regression
        reg = self._transpose_and_gather_feat(reg, inds)    # (B, K, 2)
        reg = reg.view(batch, self.max_num, 2)
        xs_post = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]    # (B, K, 1) + (B, K, 1) --> (B, K, 1)
        ys_post = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]    # (B, K, 1) + (B, K, 1) --> (B, K, 1)

        vcs_y = xs_post.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        vcs_x = ys_post.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]

        # keypoints by adding the kp offset to the topk locations
        center_kp = torch.stack([xs.squeeze(0), ys.squeeze(0)], dim=1).view(batch, self.max_num, 2)
        kps_l = []
        for kp in [kp0, kp1, kp2, kp3]:
            kp = self._transpose_and_gather_feat(kp, inds)    # (B, K, 2)
            kp = kp.view(batch, self.max_num, 2)

            kp += center_kp

            kp_vcs = torch.zeros_like(kp)
            kp_vcs[..., 1] = kp[..., 0] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
            kp_vcs[..., 0] = kp[..., 1] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
            kps_l.append(kp_vcs)

        # mask for filter out the points outside the range, 1 for valid, 0 for invalid
        final_center_preds = torch.cat([vcs_x, vcs_y], dim=2)
        self.post_center_range = torch.tensor(self.post_center_range, device=heat.device)
        mask = (final_center_preds[..., :2] >= self.post_center_range[:2]).all(2)      # (B, K, 2) --> (B, K)
        mask &= (final_center_preds[..., :2] <= self.post_center_range[3:5]).all(2)     # (B, K, 2) --> (B, K)

        thresh_mask = scores > self.score_threshold   # (B, K)

        predictions_dicts = []
        for i in range(batch):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            centers = final_center_preds[i, cmask]      # (K', 2)
            scores = scores[i, cmask]                   # (K', )
            labels = clses[i, cmask]                    # (K', )
            status = status[i, cmask]                   # (K', )
            kps = [kp[i, cmask] for kp in kps_l]        # [(K', 2), (K', 2), (K', 2), (K', 2)]
            prediction_dict = {
                'centers': centers,     # (K', 2)
                'scores': scores,       # (K', )
                'labels': labels,       # (K', )
                'status': status,       # (K', )
                'kps': kps,             # [(K', 2), (K', 2), (K', 2), (K', 2)]
            }
            predictions_dicts.append(prediction_dict)
        return predictions_dicts
    

    @torch.no_grad()
    def decode_slot_ws(self,
                    heatmap_2ch,      # (B,2,H,W) [slot_center, ws_center] after sigmoid OR raw (we'll sigmoid if needed)
                    slot_ctr,         # (B,2,H,W)
                    slot_kp0, slot_kp1, slot_kp2, slot_kp3,  # each (B,2,H,W)
                    ws_ctr,           # (B,2,H,W)
                    ws_kp0, ws_kp1,   # each (B,2,H,W)
                    slot_type_logits, # (B,3,H,W) raw logits
                    slot_occ_logits,  # (B,7,H,W) raw logits
                    ):
        """
        Returns:
        list length B:
            {
            "slots": {"centers":(Ns,2), "scores":(Ns,), "type":(Ns,), "occ":(Ns,), "kps":[(Ns,2)x4] or (Ns,4,2)},
            "wheelstops": {"centers":(Nw,2), "scores":(Nw,), "kps":[(Nw,2)x2] or (Nw,2,2)}
            }
        """

        # 1) ensure prob heatmap
        if heatmap_2ch.dtype.is_floating_point:
            # if already sigmoid-ed, values in [0,1]; if not, still ok-ish but best to sigmoid.
            # you can guard by checking max>1, but simple sigmoid is fine.
            hm = heatmap_2ch
            if hm.max() > 1.0 or hm.min() < 0.0:
                hm = hm.sigmoid()
        else:
            hm = heatmap_2ch.float().sigmoid()

        hm_slot = hm[:, 0:1]  # (B,1,H,W)
        hm_ws   = hm[:, 1:2]

        # 2) NMS per channel
        hm_slot_nms = self._nms(hm_slot, kernel=self.nms_kernel_size)
        hm_ws_nms   = self._nms(hm_ws,   kernel=self.nms_kernel_size)

        B, _, H, W = hm_slot_nms.shape
        K = self.max_num

        def topk_single_channel(hm1, K):
            # hm1: (B,1,H,W)
            scores, inds, clses, ys, xs = self._topk(hm1, K=K)
            # clses will always be 0 because cat=1, keep for compatibility
            return scores, inds, ys, xs

        # 3) topK for slot and ws
        slot_scores, slot_inds, slot_ys, slot_xs = topk_single_channel(hm_slot_nms, K)
        ws_scores,   ws_inds,   ws_ys,   ws_xs   = topk_single_channel(hm_ws_nms,   K)

        # 4) gather center offsets
        slot_reg = self._transpose_and_gather_feat(slot_ctr, slot_inds).view(B, K, 2)
        ws_reg   = self._transpose_and_gather_feat(ws_ctr,   ws_inds).view(B, K, 2)

        # center float in feat coords
        slot_xs_post = slot_xs.view(B, K, 1) + slot_reg[:, :, 0:1]
        slot_ys_post = slot_ys.view(B, K, 1) + slot_reg[:, :, 1:2]
        ws_xs_post   = ws_xs.view(B, K, 1)   + ws_reg[:, :, 0:1]
        ws_ys_post   = ws_ys.view(B, K, 1)   + ws_reg[:, :, 1:2]

        # 5) feat -> metric (keep your convention: feat_x maps to metric_y)
        slot_vcs_y = slot_xs_post * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        slot_vcs_x = slot_ys_post * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ws_vcs_y   = ws_xs_post   * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        ws_vcs_x   = ws_ys_post   * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]

        slot_centers = torch.cat([slot_vcs_x, slot_vcs_y], dim=2)  # (B,K,2)
        ws_centers   = torch.cat([ws_vcs_x,   ws_vcs_y],   dim=2)

        # 6) gather keypoints offsets, add center_int (NOT center_post!)
        # your training target kp_off is relative to center_int (xs,ys), so decode must add center_int.
        slot_center_int = torch.stack([slot_xs, slot_ys], dim=2).view(B, K, 2)  # (B,K,2) [x_int,y_int]
        ws_center_int   = torch.stack([ws_xs,   ws_ys],   dim=2).view(B, K, 2)

        slot_kps_l = []
        for kp in [slot_kp0, slot_kp1, slot_kp2, slot_kp3]:
            kp_off = self._transpose_and_gather_feat(kp, slot_inds).view(B, K, 2)
            kp_feat = kp_off + slot_center_int  # (B,K,2) feat coords
            kp_vcs = torch.zeros_like(kp_feat)
            kp_vcs[..., 1] = kp_feat[..., 0] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
            kp_vcs[..., 0] = kp_feat[..., 1] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
            slot_kps_l.append(kp_vcs)  # list of (B,K,2)

        ws_kps_l = []
        for kp in [ws_kp0, ws_kp1]:
            kp_off = self._transpose_and_gather_feat(kp, ws_inds).view(B, K, 2)
            kp_feat = kp_off + ws_center_int
            kp_vcs = torch.zeros_like(kp_feat)
            kp_vcs[..., 1] = kp_feat[..., 0] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
            kp_vcs[..., 0] = kp_feat[..., 1] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
            ws_kps_l.append(kp_vcs)  # list of (B,K,2)

        # 7) gather slot type/occ logits at slot inds
        # 注意：type/occ 用 softmax CE 训练，推理用 argmax 即可
        type_l = self._transpose_and_gather_feat(slot_type_logits, slot_inds).view(B, K, -1)  # (B,K,3)
        occ_l  = self._transpose_and_gather_feat(slot_occ_logits,  slot_inds).view(B, K, -1)  # (B,K,7)
        slot_type = type_l.argmax(dim=2)  # (B,K)
        slot_occ  = occ_l.argmax(dim=2)   # (B,K)

        # 8) post_center_range & score_threshold filtering (same style as old)
        post = torch.tensor(self.post_center_range, device=hm.device) if self.post_center_range is not None else None

        def make_mask(centers, scores):
            # centers: (B,K,2), scores:(B,K)
            if post is not None:
                m = (centers[..., :2] >= post[:2]).all(2)
                m &= (centers[..., :2] <= post[3:5]).all(2)
            else:
                m = torch.ones_like(scores, dtype=torch.bool)
            m &= (scores > self.score_threshold)
            return m

        slot_mask = make_mask(slot_centers, slot_scores)
        ws_mask   = make_mask(ws_centers,   ws_scores)

        # 9) pack per-sample outputs
        outs = []
        for i in range(B):
            sm = slot_mask[i]
            wm = ws_mask[i]

            out = {
                "slots": {
                    "centers": slot_centers[i, sm],            # (Ns,2)
                    "scores":  slot_scores[i, sm],             # (Ns,)
                    "type":    slot_type[i, sm].to(torch.long),
                    "occ":     slot_occ[i, sm].to(torch.long),
                    "kps":     [kp[i, sm] for kp in slot_kps_l],  # list of 4*(Ns,2)
                },
                "wheelstops": {
                    "centers": ws_centers[i, wm],              # (Nw,2)
                    "scores":  ws_scores[i, wm],               # (Nw,)
                    "kps":     [kp[i, wm] for kp in ws_kps_l], # list of 2*(Nw,2)
                }
            }
            outs.append(out)
        return outs