# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import torch
# import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer, Scale, bias_init_with_prob, normal_init
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)

from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet3d.models.builder import HEADS, build_loss
import torch.nn.functional as F



@HEADS.register_module()
class ParkingSlotWheelstopHead2D(BaseModule):
    """
    No-task head:
      forward(feats) -> List[dict] (by level), usually len=1
    """

    def __init__(self,
                 in_channels=256,
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 common_heads=dict(),
                 bbox_coder=None,
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_reg=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                 train_cfg=None,
                 test_cfg=None,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None):
        
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        # -------- heads --------
        # heatmap: 2ch [slot_center, ws_center]
        self.heatmap_head = self._make_head(share_conv_channel, 2, num_heatmap_convs)

        # slot reg (ctr + 4kps)
        self.slot_ctr_offset = self._make_head(share_conv_channel, common_heads.get('slot_ctr_offset', (2,2))[0], common_heads.get('slot_ctr_offset', (2,2))[1])
        self.slot_kp0 = self._make_head(share_conv_channel, common_heads.get('slot_kp0', (2,2))[0], common_heads.get('slot_kp0', (2,2))[1])
        self.slot_kp1 = self._make_head(share_conv_channel, common_heads.get('slot_kp1', (2,2))[0], common_heads.get('slot_kp1', (2,2))[1])
        self.slot_kp2 = self._make_head(share_conv_channel, common_heads.get('slot_kp2', (2,2))[0], common_heads.get('slot_kp2', (2,2))[1])
        self.slot_kp3 = self._make_head(share_conv_channel, common_heads.get('slot_kp3', (2,2))[0], common_heads.get('slot_kp3', (2,2))[1])

        # ws reg (ctr + 2kps)
        self.ws_ctr_offset = self._make_head(share_conv_channel, common_heads.get('ws_ctr_offset', (2,2))[0], common_heads.get('ws_ctr_offset', (2,2))[1])
        self.ws_kp0 = self._make_head(share_conv_channel, common_heads.get('ws_kp0', (2,2))[0], common_heads.get('ws_kp0', (2,2))[1])
        self.ws_kp1 = self._make_head(share_conv_channel, common_heads.get('ws_kp1', (2,2))[0], common_heads.get('ws_kp1', (2,2))[1])

        # slot cls/occ (separate heads)
        self.slot_type = self._make_head(share_conv_channel, common_heads.get('slot_type', (3,2))[0], common_heads.get('slot_type', (3,2))[1])
        self.slot_occ  = self._make_head(share_conv_channel, common_heads.get('slot_occ', (7,2))[0], common_heads.get('slot_occ', (7,2))[1])

        # init biases like centernet
        self._init_last_bias(self.heatmap_head, -2.19)
        self._init_last_bias(self.slot_type, -2.19)
        self._init_last_bias(self.slot_occ, -2.19)

    def _make_head(self, in_ch, out_ch, num_convs):
        layers = []
        for _ in range(num_convs - 1):
            layers += [nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)]
        return nn.Sequential(*layers)

    @staticmethod
    def _init_last_bias(head, bias):
        for m in reversed(list(head.modules())):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, bias)
                break

    # -------- forward --------
    def forward_single(self, x):
        x = self.shared_conv(x)
        return dict(
            heatmap=self.heatmap_head(x),
            slot_ctr_offset=self.slot_ctr_offset(x),
            slot_kp0=self.slot_kp0(x), slot_kp1=self.slot_kp1(x),
            slot_kp2=self.slot_kp2(x), slot_kp3=self.slot_kp3(x),
            ws_ctr_offset=self.ws_ctr_offset(x),
            ws_kp0=self.ws_kp0(x), ws_kp1=self.ws_kp1(x),
            slot_type=self.slot_type(x),
            slot_occ=self.slot_occ(x),
        )

    # def forward(self, feats):
    #     # feats: list[Tensor], multi level
    #     # return: list[dict] length=num_levels
    #     return multi_apply(self.forward_single, feats)[0]
    
    def forward(self, feats):
        # feats: list[Tensor], multi level
        # return: list[dict] length=num_levels
        return [self.forward_single(x) for x in feats]


    def get_targets(self,
                    slot_type_list,
                    slot_occ_list,
                    slot_geom_list,
                    ws_geom_list):
        """
        Args (建议用 list 输入，和 mmdet3d 一致，长度=B):
            slot_type_list: list[Tensor] each (Ns,) long, 0..2
            slot_occ_list : list[Tensor] each (Ns,) long, 0..6
            slot_geom_list: list[Tensor] each (Ns,4,2) float, metric xy (ctx=x, cty=y)
            ws_geom_list  : list[Tensor] each (Nw,2,2) float, metric xy (two corners)

        Returns:
            heatmap_t:      (B, 2, H, W)
            slot_anno_t:    (B, max_slots, 10)  [ctr_off(2) + kp_off(8)]
            ws_anno_t:      (B, max_ws, 6)      [ctr_off(2) + kp_off(4)]
            slot_inds:      (B, max_slots) long
            slot_masks:     (B, max_slots) uint8/bool
            ws_inds:        (B, max_ws) long
            ws_masks:       (B, max_ws) uint8/bool
            slot_type_ids:  (B, max_slots) long   # only valid when slot_masks=1
            slot_occ_ids:   (B, max_slots) long
        """
        slot_geom_list = [g[..., :2] if g is not None else None for g in slot_geom_list]
        ws_geom_list   = [g[..., :2] if g is not None else None for g in ws_geom_list]

        targets = multi_apply(
            self.get_targets_single,
            slot_type_list, slot_occ_list, slot_geom_list, ws_geom_list
        )

        (heatmap_t,
         slot_anno_t, ws_anno_t,
         slot_inds, slot_masks,
         ws_inds, ws_masks,
         slot_type_ids, slot_occ_ids) = targets

        heatmap_t     = torch.stack(heatmap_t, dim=0)
        slot_anno_t   = torch.stack(slot_anno_t, dim=0)
        ws_anno_t     = torch.stack(ws_anno_t, dim=0)
        slot_inds     = torch.stack(slot_inds, dim=0)
        slot_masks    = torch.stack(slot_masks, dim=0)
        ws_inds       = torch.stack(ws_inds, dim=0)
        ws_masks      = torch.stack(ws_masks, dim=0)
        slot_type_ids = torch.stack(slot_type_ids, dim=0)
        slot_occ_ids  = torch.stack(slot_occ_ids, dim=0)

        return (heatmap_t,
                slot_anno_t, ws_anno_t,
                slot_inds, slot_masks,
                ws_inds, ws_masks,
                slot_type_ids, slot_occ_ids)

    def get_targets_single(self,
                           slot_type: torch.Tensor,
                           slot_occ: torch.Tensor,
                           slot_geom: torch.Tensor,
                           ws_geom: torch.Tensor):
        """
        Build targets for ONE sample.

        slot_geom: (Ns,4,2) metric xy (x=forward? y=left? 按你原 BEV 坐标)
        ws_geom  : (Nw,2,2) metric xy

        Notes about coord transform:
          你原代码里使用：
            coor_x = (cty - pc_range[1]) / voxel_y / out
            coor_y = (ctx - pc_range[0]) / voxel_x / out
          这里保持一致（注意 x/y 交换的那个约定）。
        """
        device = slot_geom.device if slot_geom is not None else ws_geom.device
        dtype  = slot_geom.dtype if slot_geom is not None else ws_geom.dtype

        # ---- train cfg ----
        max_objs      = int(self.train_cfg.get('max_objs', 50))
        dense_reg     = int(self.train_cfg.get('dense_reg', 1))
        max_slots     = max_objs * dense_reg
        max_ws        = max_objs * dense_reg

        grid_size     = torch.tensor(self.train_cfg['grid_size'], device=device, dtype=torch.float32)
        pc_range      = torch.tensor(self.train_cfg['point_cloud_range'], device=device, dtype=torch.float32)
        voxel_size    = torch.tensor(self.train_cfg['voxel_size'], device=device, dtype=torch.float32)
        out_factor    = int(self.train_cfg['out_size_factor'])

        # feature map size (W,H)
        feat_size = (grid_size[:2] / out_factor).long()  # [Wf, Hf] in your comment style
        Wf, Hf = int(feat_size[0].item()), int(feat_size[1].item())

        gaussian_overlap = float(self.train_cfg.get('gaussian_overlap', 0.1))
        min_radius       = int(self.train_cfg.get('min_radius', 2))

        # ---- outputs ----
        heatmap_t   = torch.zeros((2, Hf, Wf), device=device, dtype=dtype)

        slot_anno_t = torch.zeros((max_slots, 10), device=device, dtype=torch.float32)
        ws_anno_t   = torch.zeros((max_ws, 6), device=device, dtype=torch.float32)

        slot_inds   = torch.zeros((max_slots,), device=device, dtype=torch.int64)
        slot_masks  = torch.zeros((max_slots,), device=device, dtype=torch.bool)
        ws_inds     = torch.zeros((max_ws,), device=device, dtype=torch.int64)
        ws_masks    = torch.zeros((max_ws,), device=device, dtype=torch.bool)

        slot_type_ids = torch.zeros((max_slots,), device=device, dtype=torch.int64)
        slot_occ_ids  = torch.zeros((max_slots,), device=device, dtype=torch.int64)

       
        def metric_xy_to_feat_coor(ctx: torch.Tensor, cty: torch.Tensor):
            coor_x = (cty - pc_range[1]) / voxel_size[1] / out_factor
            coor_y = (ctx - pc_range[0]) / voxel_size[0] / out_factor
            return coor_x, coor_y

        draw_gaussian = draw_heatmap_gaussian

        # =========================
        # 1) Slots targets
        # =========================
        Ns = 0 if slot_geom is None else int(slot_geom.shape[0])
        num_slots = min(Ns, max_slots)

        for k in range(num_slots):
            # slot kps metric: (4,2)
            kps = slot_geom[k]  # float
            xs = kps[:, 0]      # metric x
            ys = kps[:, 1]      # metric y

            # compute bbox size for radius (in feature units)
            x_min, x_max = xs.min().item(), xs.max()
            y_min, y_max = ys.min().item(), ys.max()

            slot_w = (x_max - x_min) / float(voxel_size[0]) / out_factor
            slot_h = (y_max - y_min) / float(voxel_size[1]) / out_factor
            if (slot_w <= 0).item() or (slot_h <= 0).item():
                continue

            radius = gaussian_radius((slot_h, slot_w), min_overlap=gaussian_overlap)
            radius = max(min_radius, int(radius))

            # center metric
            ctx = xs.mean()
            cty = ys.mean()

            # center feat coord
            cx, cy = metric_xy_to_feat_coor(ctx, cty)  # (coor_x, coor_y)
            center = torch.stack([cx, cy]).to(torch.float32)
            center_int = center.to(torch.int32)

            # range check
            if not (0 <= center_int[0] < Wf and 0 <= center_int[1] < Hf):
                continue

            # (a) heatmap channel 0: slot center
            draw_gaussian(heatmap_t[0], center_int, radius)

            # (b) inds/mask
            x_i, y_i = int(center_int[0].item()), int(center_int[1].item())
            slot_inds[k] = y_i * Wf + x_i
            slot_masks[k] = 1

            # (c) type/occ id (只在 mask=1 的地方有效)
            # slot_type: 0..2, slot_occ: 0..6
            slot_type_ids[k] = int(slot_type[k].item()) if slot_type is not None and k < slot_type.numel() else 0
            slot_occ_ids[k]  = int(slot_occ[k].item())  if slot_occ is not None and k < slot_occ.numel() else 0

            # (d) ctr offset target (float - int)
            slot_anno_t[k, 0:2] = center - center_int.to(torch.float32)

            # (e) keypoints offsets relative to center_int (same as your old code)
            # first map each kp metric to feat coor (coor_x, coor_y) but note:
            # your old code for kps used:
            #   kpsx = (kp_y - pc_range[1]) / voxel_y / out
            #   kpsy = (kp_x - pc_range[0]) / voxel_x / out
            # i.e. kpsx uses y, kpsy uses x
            kpsx = (kps[:, 1] - pc_range[1]) / voxel_size[1] / out_factor
            kpsy = (kps[:, 0] - pc_range[0]) / voxel_size[0] / out_factor

            kpsx_off = kpsx - center_int[0].to(torch.float32)
            kpsy_off = kpsy - center_int[1].to(torch.float32)

            # flatten as [kp0x,kp0y,kp1x,kp1y,kp2x,kp2y,kp3x,kp3y]
            slot_anno_t[k, 2:10] = torch.stack([kpsx_off, kpsy_off], dim=-1).reshape(-1)

        # =========================
        # 2) Wheelstop targets
        # =========================
        Nw = 0 if ws_geom is None else int(ws_geom.shape[0])
        num_ws = min(Nw, max_ws)

        for k in range(num_ws):
            wkps = ws_geom[k]  # (2,2)
            xs = wkps[:, 0]
            ys = wkps[:, 1]

            x_min, x_max = xs.min().item(), xs.max()
            y_min, y_max = ys.min().item(), ys.max()

            ws_w = (x_max - x_min) / float(voxel_size[0]) / out_factor
            ws_h = (y_max - y_min) / float(voxel_size[1]) / out_factor
            # ws 可能很短，避免 0
            # ws_w = max(ws_w, 1.0)
            # ws_h = max(ws_h, 1.0)

            if (ws_w <= 0).item() or (ws_h <= 0).item():
                continue

            radius = gaussian_radius((ws_h, ws_w), min_overlap=gaussian_overlap)
            radius = max(min_radius, int(radius))

            ctx = xs.mean()
            cty = ys.mean()

            cx, cy = metric_xy_to_feat_coor(ctx, cty)
            center = torch.stack([cx, cy]).to(torch.float32)
            center_int = center.to(torch.int32)

            if not (0 <= center_int[0] < Wf and 0 <= center_int[1] < Hf):
                continue

            # heatmap channel 1: ws center
            draw_gaussian(heatmap_t[1], center_int, radius)

            x_i, y_i = int(center_int[0].item()), int(center_int[1].item())
            ws_inds[k] = y_i * Wf + x_i
            ws_masks[k] = 1

            # ctr offset
            ws_anno_t[k, 0:2] = center - center_int.to(torch.float32)

            # ws keypoints offsets (2 points)
            kpsx = (wkps[:, 1] - pc_range[1]) / voxel_size[1] / out_factor
            kpsy = (wkps[:, 0] - pc_range[0]) / voxel_size[0] / out_factor

            kpsx_off = kpsx - center_int[0].to(torch.float32)
            kpsy_off = kpsy - center_int[1].to(torch.float32)

            ws_anno_t[k, 2:6] = torch.stack([kpsx_off, kpsy_off], dim=-1).reshape(-1)

        return (heatmap_t,
                slot_anno_t, ws_anno_t,
                slot_inds, slot_masks,
                ws_inds, ws_masks,
                slot_type_ids, slot_occ_ids)


    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
    
    def _transpose_and_gather_feat(self, feat_map, inds):
        """
        feat_map: (B, C, H, W)
        inds:     (B, K)
        return:   (B, K, C)
        """
        B, C, H, W = feat_map.shape
        feat = feat_map.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, H*W, C)
        feat = self._gather_feat(feat, inds)  # (B, K, C)
        return feat
    

    def loss(self,
         preds,
         slot_type_list,
         slot_occ_list,
         slot_geom_list,
         ws_geom_list):
        """
        preds: list[dict] from forward(), usually len=1 (single level)
        slot_type_list/slot_occ_list/slot_geom_list/ws_geom_list: list length=B
        """
        pred = preds[0]  # single feature level

        # ---- build targets ----
        (heatmap_t,
        slot_anno_t, ws_anno_t,
        slot_inds, slot_masks,
        ws_inds, ws_masks,
        slot_type_ids, slot_occ_ids) = self.get_targets(
            slot_type_list, slot_occ_list, slot_geom_list, ws_geom_list
        )

        loss_dict = {}

        # =========================================================
        # 1) heatmap loss (dense focal)
        # =========================================================
        # pred heatmap: (B,2,H,W)
        heatmap_pred = clip_sigmoid(pred['heatmap'])

        num_pos = heatmap_t.eq(1).float().sum().item()
        avg_factor = torch.clamp(reduce_mean(heatmap_t.new_tensor(num_pos)), min=1).item()

        loss_hm = self.loss_cls(heatmap_pred, heatmap_t, avg_factor=avg_factor)
        loss_dict['loss_heatmap'] = loss_hm

        # =========================================================
        # 2) slot regression loss (sparse gather @ slot centers)
        #    target: slot_anno_t (B, max_slots, 10)
        # =========================================================
        slot_reg_map = torch.cat(
            [pred['slot_ctr_offset'],
            pred['slot_kp0'], pred['slot_kp1'], pred['slot_kp2'], pred['slot_kp3']],
            dim=1
        )  # (B, 10, H, W)

        slot_reg_pred = self._transpose_and_gather_feat(slot_reg_map, slot_inds)  # (B, max_slots, 10)

        # weights/masks
        slot_mask = slot_masks.unsqueeze(-1).float()  # (B, max_slots, 1)
        
        slot_code_w = self.train_cfg.get('slot_code_weights', None)
        if slot_code_w is None:
            slot_code_w = [1.0] * 10

        slot_code_w = slot_anno_t.new_tensor(slot_code_w).view(1, 1, 10)

        slot_weights = slot_mask * slot_code_w  # (B, max_slots, 10)

        # avg factor = 正样本个数（分布式 reduce）
        slot_num = slot_masks.float().sum()
        slot_num = torch.clamp(reduce_mean(slot_anno_t.new_tensor(slot_num)), min=1.0).item()

        # 防 nan
        isnotnan = (~torch.isnan(slot_anno_t)).float()
        slot_weights = slot_weights * isnotnan

        loss_slot_reg = self.loss_reg(
            slot_reg_pred, slot_anno_t, slot_weights, avg_factor=slot_num
        )
        loss_dict['loss_slot_reg'] = loss_slot_reg

        # =========================================================
        # 3) wheelstop regression loss (sparse gather @ ws centers)
        #    target: ws_anno_t (B, max_ws, 6)
        # =========================================================
        ws_reg_map = torch.cat(
            [pred['ws_ctr_offset'],
            pred['ws_kp0'], pred['ws_kp1']],
            dim=1
        )  # (B, 6, H, W)

        ws_reg_pred = self._transpose_and_gather_feat(ws_reg_map, ws_inds)  # (B, max_ws, 6)

        ws_mask = ws_masks.unsqueeze(-1).float()
        ws_code_w = self.train_cfg.get('ws_code_weights', None)
        if ws_code_w is None:
            ws_code_w = [1.0] * 6
        ws_code_w = ws_anno_t.new_tensor(ws_code_w).view(1, 1, 6)

        ws_weights = ws_mask * ws_code_w
        ws_num = ws_masks.float().sum()
        ws_num = torch.clamp(reduce_mean(ws_anno_t.new_tensor(ws_num)), min=1.0).item()

        isnotnan_ws = (~torch.isnan(ws_anno_t)).float()
        ws_weights = ws_weights * isnotnan_ws

        loss_ws_reg = self.loss_reg(
            ws_reg_pred, ws_anno_t, ws_weights, avg_factor=ws_num
        )
        loss_dict['loss_ws_reg'] = loss_ws_reg

        # =========================================================
        # 4) slot_type / slot_occ classification loss (center-only)
        #    只在 slot_masks == 1 的位置计算
        # =========================================================
        # gather logits at slot centers
        slot_type_logits = self._transpose_and_gather_feat(pred['slot_type'], slot_inds)  # (B, max_slots, 3)
        slot_occ_logits  = self._transpose_and_gather_feat(pred['slot_occ'],  slot_inds)  # (B, max_slots, 7)

        pos_mask = slot_masks.bool()  # (B, max_slots)

        # 如果一个 batch 没有任何 slot（极少但可能），避免 CE 报错
        if pos_mask.any():
            type_pred_pos = slot_type_logits[pos_mask]  # (Npos, 3)
            occ_pred_pos  = slot_occ_logits[pos_mask]   # (Npos, 7)

            type_t_pos = slot_type_ids[pos_mask].long()  # (Npos,)
            occ_t_pos  = slot_occ_ids[pos_mask].long()   # (Npos,)

            loss_type = F.cross_entropy(type_pred_pos, type_t_pos)
            loss_occ  = F.cross_entropy(occ_pred_pos,  occ_t_pos)
        else:
            loss_type = slot_type_logits.sum() * 0.0
            loss_occ  = slot_occ_logits.sum() * 0.0

        # 可选权重
        loss_type_w = float(self.train_cfg.get('loss_type_weight', 1.0))
        loss_occ_w  = float(self.train_cfg.get('loss_occ_weight', 1.0))

        loss_dict['loss_slot_type'] = loss_type * loss_type_w
        loss_dict['loss_slot_occ']  = loss_occ  * loss_occ_w

        return loss_dict
    

    @torch.no_grad()
    def get_bboxes(self, preds, img_metas=None, img=None, rescale=False):
        pred = preds[0]
        return self.bbox_coder.decode_slot_ws(
            heatmap_2ch=pred["heatmap"],   # raw logits ok
            slot_ctr=pred["slot_ctr_offset"],
            slot_kp0=pred["slot_kp0"], slot_kp1=pred["slot_kp1"],
            slot_kp2=pred["slot_kp2"], slot_kp3=pred["slot_kp3"],
            ws_ctr=pred["ws_ctr_offset"],
            ws_kp0=pred["ws_kp0"], ws_kp1=pred["ws_kp1"],
            slot_type_logits=pred["slot_type"],
            slot_occ_logits=pred["slot_occ"],
        )



@HEADS.register_module()
class ParkingSlotWheelstopHead2DCenter(BaseModule):
    """
    单中心版本:
      - heatmap 只预测车位中心
      - 在车位中心处回归:
          1) slot center offset (2)
          2) slot 4 keypoints (8)
          3) wheelstop 2 keypoints (4)
      - 分类:
          1) slot_type
          2) slot_occ
          3) slot_ws_exist  (该车位是否有轮挡)

    GT 约定:
      slot_geom_list[b]: (Ns, 4, 2)
      ws_geom_list[b]:   (Ns, 2, 2), 与 slot 一一对应
                         如果某个 slot 没有轮挡，则该行填 nan
    """

    def __init__(self,
                 in_channels=256,
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 common_heads=dict(),
                 bbox_coder=None,
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_reg=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                 train_cfg=None,
                 test_cfg=None,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        # =========================================================
        # heads
        # =========================================================
        # heatmap: only slot center
        self.heatmap_head = self._make_head(
            share_conv_channel, 1, num_heatmap_convs)

        # slot reg: ctr(2) + 4kps(8)
        self.slot_ctr_offset = self._make_head(
            share_conv_channel,
            common_heads.get('slot_ctr_offset', (2, 2))[0],
            common_heads.get('slot_ctr_offset', (2, 2))[1])

        self.slot_kp0 = self._make_head(
            share_conv_channel,
            common_heads.get('slot_kp0', (2, 2))[0],
            common_heads.get('slot_kp0', (2, 2))[1])

        self.slot_kp1 = self._make_head(
            share_conv_channel,
            common_heads.get('slot_kp1', (2, 2))[0],
            common_heads.get('slot_kp1', (2, 2))[1])

        self.slot_kp2 = self._make_head(
            share_conv_channel,
            common_heads.get('slot_kp2', (2, 2))[0],
            common_heads.get('slot_kp2', (2, 2))[1])

        self.slot_kp3 = self._make_head(
            share_conv_channel,
            common_heads.get('slot_kp3', (2, 2))[0],
            common_heads.get('slot_kp3', (2, 2))[1])

        # ws reg: attached to slot center, only 2 kps = 4 dims
        self.ws_kp0 = self._make_head(
            share_conv_channel,
            common_heads.get('ws_kp0', (2, 2))[0],
            common_heads.get('ws_kp0', (2, 2))[1])

        self.ws_kp1 = self._make_head(
            share_conv_channel,
            common_heads.get('ws_kp1', (2, 2))[0],
            common_heads.get('ws_kp1', (2, 2))[1])

        # slot cls / occ / ws_exist
        self.slot_type = self._make_head(
            share_conv_channel,
            common_heads.get('slot_type', (3, 2))[0],
            common_heads.get('slot_type', (3, 2))[1])

        self.slot_occ = self._make_head(
            share_conv_channel,
            common_heads.get('slot_occ', (7, 2))[0],
            common_heads.get('slot_occ', (7, 2))[1])

        # 2-class logits: [no_ws, has_ws]
        self.slot_ws_exist = self._make_head(
            share_conv_channel,
            common_heads.get('slot_ws_exist', (2, 2))[0],
            common_heads.get('slot_ws_exist', (2, 2))[1])

        # init bias
        self._init_last_bias(self.heatmap_head, -2.19)
        self._init_last_bias(self.slot_type, -2.19)
        self._init_last_bias(self.slot_occ, -2.19)
        self._init_last_bias(self.slot_ws_exist, 0.0)

    def _make_head(self, in_ch, out_ch, num_convs):
        layers = []
        for _ in range(num_convs - 1):
            layers += [
                nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ]
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)]
        return nn.Sequential(*layers)

    @staticmethod
    def _init_last_bias(head, bias):
        for m in reversed(list(head.modules())):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, bias)
                break

    # =========================================================
    # forward
    # =========================================================
    def forward_single(self, x):
        x = self.shared_conv(x)
        return dict(
            heatmap=self.heatmap_head(x),              # (B,1,H,W)
            slot_ctr_offset=self.slot_ctr_offset(x),   # (B,2,H,W)
            slot_kp0=self.slot_kp0(x),                 # (B,2,H,W)
            slot_kp1=self.slot_kp1(x),
            slot_kp2=self.slot_kp2(x),
            slot_kp3=self.slot_kp3(x),
            ws_kp0=self.ws_kp0(x),                     # (B,2,H,W)
            ws_kp1=self.ws_kp1(x),
            slot_type=self.slot_type(x),               # (B,3,H,W)
            slot_occ=self.slot_occ(x),                 # (B,7,H,W)
            slot_ws_exist=self.slot_ws_exist(x),       # (B,2,H,W)
        )

    def forward(self, feats):
        return [self.forward_single(x) for x in feats]

    # =========================================================
    # target helpers
    # =========================================================
    def _has_valid_ws(self, ws_kps: torch.Tensor):
        """ws_kps: (2,2), row is NaN means no wheelstop."""
        if ws_kps is None:
            return False
        if ws_kps.numel() != 4:
            return False
        return torch.isfinite(ws_kps).all().item()

    def get_targets(self,
                    slot_type_list,
                    slot_occ_list,
                    slot_geom_list,
                    ws_geom_list):
        """
        Args:
            slot_type_list: list[Tensor], each (Ns,), 0..2
            slot_occ_list : list[Tensor], each (Ns,), 0..6
            slot_geom_list: list[Tensor], each (Ns,4,2)
            ws_geom_list  : list[Tensor], each (Ns,2,2), 与 slot 一一对应
                            无轮挡时对应行为 NaN

        Returns:
            heatmap_t:         (B, 1, H, W)
            slot_anno_t:       (B, max_slots, 14)
                               [ctr_off(2) + slot_4kps(8) + ws_2kps(4)]
            slot_reg_mask_t:   (B, max_slots, 14)
                               前10维只要有 slot 就有效
                               后4维只有 GT 有轮挡才有效
            slot_inds:         (B, max_slots)
            slot_masks:        (B, max_slots)
            slot_type_ids:     (B, max_slots)
            slot_occ_ids:      (B, max_slots)
            slot_ws_exist_ids: (B, max_slots)  0/1
        """
        slot_geom_list = [g[..., :2] if g is not None else None for g in slot_geom_list]
        ws_geom_list = [g[..., :2] if g is not None else None for g in ws_geom_list]

        targets = multi_apply(
            self.get_targets_single,
            slot_type_list, slot_occ_list, slot_geom_list, ws_geom_list
        )

        (heatmap_t,
         slot_anno_t, slot_reg_mask_t,
         slot_inds, slot_masks,
         slot_type_ids, slot_occ_ids,
         slot_ws_exist_ids) = targets

        heatmap_t = torch.stack(heatmap_t, dim=0)
        slot_anno_t = torch.stack(slot_anno_t, dim=0)
        slot_reg_mask_t = torch.stack(slot_reg_mask_t, dim=0)
        slot_inds = torch.stack(slot_inds, dim=0)
        slot_masks = torch.stack(slot_masks, dim=0)
        slot_type_ids = torch.stack(slot_type_ids, dim=0)
        slot_occ_ids = torch.stack(slot_occ_ids, dim=0)
        slot_ws_exist_ids = torch.stack(slot_ws_exist_ids, dim=0)

        return (
            heatmap_t,
            slot_anno_t, slot_reg_mask_t,
            slot_inds, slot_masks,
            slot_type_ids, slot_occ_ids,
            slot_ws_exist_ids
        )

    def get_targets_single(self,
                           slot_type: torch.Tensor,
                           slot_occ: torch.Tensor,
                           slot_geom: torch.Tensor,
                           ws_geom: torch.Tensor):
        """
        slot_geom: (Ns,4,2)
        ws_geom  : (Ns,2,2), 与 slot 一一对应，无轮挡时填 NaN
        """
        assert slot_geom is not None, "slot_geom should not be None"

        device = slot_geom.device
        dtype = slot_geom.dtype

        # ---- train cfg ----
        max_objs = int(self.train_cfg.get('max_objs', 50))
        dense_reg = int(self.train_cfg.get('dense_reg', 1))
        max_slots = max_objs * dense_reg

        grid_size = torch.tensor(self.train_cfg['grid_size'], device=device, dtype=torch.float32)
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'], device=device, dtype=torch.float32)
        voxel_size = torch.tensor(self.train_cfg['voxel_size'], device=device, dtype=torch.float32)
        out_factor = int(self.train_cfg['out_size_factor'])

        feat_size = (grid_size[:2] / out_factor).long()
        Wf, Hf = int(feat_size[0].item()), int(feat_size[1].item())

        gaussian_overlap = float(self.train_cfg.get('gaussian_overlap', 0.1))
        min_radius = int(self.train_cfg.get('min_radius', 2))

        # ---- outputs ----
        heatmap_t = torch.zeros((1, Hf, Wf), device=device, dtype=dtype)

        # [ctr_off(2) + slot_4kps(8) + ws_2kps(4)] = 14
        slot_anno_t = torch.zeros((max_slots, 14), device=device, dtype=torch.float32)
        slot_reg_mask_t = torch.zeros((max_slots, 14), device=device, dtype=torch.float32)

        slot_inds = torch.zeros((max_slots,), device=device, dtype=torch.int64)
        slot_masks = torch.zeros((max_slots,), device=device, dtype=torch.bool)

        slot_type_ids = torch.zeros((max_slots,), device=device, dtype=torch.int64)
        slot_occ_ids = torch.zeros((max_slots,), device=device, dtype=torch.int64)
        slot_ws_exist_ids = torch.zeros((max_slots,), device=device, dtype=torch.int64)

        def metric_xy_to_feat_coor(ctx: torch.Tensor, cty: torch.Tensor):
            coor_x = (cty - pc_range[1]) / voxel_size[1] / out_factor
            coor_y = (ctx - pc_range[0]) / voxel_size[0] / out_factor
            return coor_x, coor_y

        draw_gaussian = draw_heatmap_gaussian

        Ns = int(slot_geom.shape[0])
        if ws_geom is not None:
            assert ws_geom.shape[0] == Ns, \
                f'ws_geom.shape[0]={ws_geom.shape[0]} must equal slot_geom.shape[0]={Ns}'

        num_slots = min(Ns, max_slots)

        for k in range(num_slots):
            # -------------------------
            # slot target
            # -------------------------
            kps = slot_geom[k]       # (4,2)
            xs = kps[:, 0]
            ys = kps[:, 1]

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            slot_w = (x_max - x_min) / voxel_size[0] / out_factor
            slot_h = (y_max - y_min) / voxel_size[1] / out_factor

            if slot_w.item() <= 0 or slot_h.item() <= 0:
                continue

            radius = gaussian_radius((slot_h, slot_w), min_overlap=gaussian_overlap)
            radius = max(min_radius, int(radius))

            # slot center in metric
            ctx = xs.mean()
            cty = ys.mean()

            # slot center in feature map coord
            cx, cy = metric_xy_to_feat_coor(ctx, cty)
            center = torch.stack([cx, cy]).to(torch.float32)
            center_int = center.to(torch.int32)

            if not (0 <= center_int[0] < Wf and 0 <= center_int[1] < Hf):
                continue

            # heatmap only draw slot center
            draw_gaussian(heatmap_t[0], center_int, radius)

            x_i, y_i = int(center_int[0].item()), int(center_int[1].item())
            slot_inds[k] = y_i * Wf + x_i
            slot_masks[k] = 1

            slot_type_ids[k] = int(slot_type[k].item()) if slot_type is not None and k < slot_type.numel() else 0
            slot_occ_ids[k] = int(slot_occ[k].item()) if slot_occ is not None and k < slot_occ.numel() else 0

            # ctr offset
            slot_anno_t[k, 0:2] = center - center_int.to(torch.float32)
            slot_reg_mask_t[k, 0:2] = 1.0

            # slot 4 keypoints offsets, relative to center_int
            kpsx = (kps[:, 1] - pc_range[1]) / voxel_size[1] / out_factor
            kpsy = (kps[:, 0] - pc_range[0]) / voxel_size[0] / out_factor

            kpsx_off = kpsx - center_int[0].to(torch.float32)
            kpsy_off = kpsy - center_int[1].to(torch.float32)

            slot_anno_t[k, 2:10] = torch.stack([kpsx_off, kpsy_off], dim=-1).reshape(-1)
            slot_reg_mask_t[k, 2:10] = 1.0

            # -------------------------
            # optional wheelstop target
            # -------------------------
            has_ws = False
            if ws_geom is not None:
                wkps = ws_geom[k]  # (2,2) or NaN
                has_ws = self._has_valid_ws(wkps)
            else:
                wkps = None

            slot_ws_exist_ids[k] = 1 if has_ws else 0

            if has_ws:
                wsx = (wkps[:, 1] - pc_range[1]) / voxel_size[1] / out_factor
                wsy = (wkps[:, 0] - pc_range[0]) / voxel_size[0] / out_factor

                wsx_off = wsx - center_int[0].to(torch.float32)
                wsy_off = wsy - center_int[1].to(torch.float32)

                slot_anno_t[k, 10:14] = torch.stack([wsx_off, wsy_off], dim=-1).reshape(-1)
                slot_reg_mask_t[k, 10:14] = 1.0

        return (
            heatmap_t,
            slot_anno_t, slot_reg_mask_t,
            slot_inds, slot_masks,
            slot_type_ids, slot_occ_ids,
            slot_ws_exist_ids
        )

    # =========================================================
    # gather helpers
    # =========================================================
    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat_map, inds):
        """
        feat_map: (B, C, H, W)
        inds:     (B, K)
        return:   (B, K, C)
        """
        B, C, H, W = feat_map.shape
        feat = feat_map.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        feat = self._gather_feat(feat, inds)
        return feat

    # =========================================================
    # loss
    # =========================================================
    def loss(self,
             preds,
             slot_type_list,
             slot_occ_list,
             slot_geom_list,
             ws_geom_list):
        """
        preds: list[dict], usually len=1
        ws_geom_list: 与 slot_geom_list 一一对齐，无轮挡时填 NaN
        """
        pred = preds[0]

        (heatmap_t,
         slot_anno_t, slot_reg_mask_t,
         slot_inds, slot_masks,
         slot_type_ids, slot_occ_ids,
         slot_ws_exist_ids) = self.get_targets(
            slot_type_list, slot_occ_list, slot_geom_list, ws_geom_list
        )

        loss_dict = {}

        # =========================================================
        # 1) heatmap loss
        # =========================================================
        heatmap_pred = clip_sigmoid(pred['heatmap'])   # (B,1,H,W)

        num_pos = heatmap_t.eq(1).float().sum().item()
        avg_factor = torch.clamp(
            reduce_mean(heatmap_t.new_tensor(num_pos)), min=1
        ).item()

        loss_hm = self.loss_cls(heatmap_pred, heatmap_t, avg_factor=avg_factor)
        loss_dict['loss_heatmap'] = loss_hm

        # =========================================================
        # 2) regression loss
        #    target dim = 14
        #    [ctr(2) + slot4kps(8) + ws2kps(4)]
        # =========================================================
        slot_reg_map = torch.cat(
            [
                pred['slot_ctr_offset'],   # 2
                pred['slot_kp0'],          # 2
                pred['slot_kp1'],          # 2
                pred['slot_kp2'],          # 2
                pred['slot_kp3'],          # 2
                pred['ws_kp0'],            # 2
                pred['ws_kp1'],            # 2
            ],
            dim=1
        )  # (B,14,H,W)

        slot_reg_pred = self._transpose_and_gather_feat(slot_reg_map, slot_inds)  # (B,max_slots,14)

        slot_mask = slot_masks.unsqueeze(-1).float()  # (B,max_slots,1)

        slot_code_w = self.train_cfg.get('slot_code_weights', None)
        if slot_code_w is None:
            slot_code_w = [1.0] * 14
        assert len(slot_code_w) == 14, f'slot_code_weights len must be 14, got {len(slot_code_w)}'
        slot_code_w = slot_anno_t.new_tensor(slot_code_w).view(1, 1, 14)

        # 核心：
        # - slot_mask 控制有没有 slot
        # - slot_reg_mask_t 控制某一维是否有效
        #   前10维：有slot就有效
        #   后4维：只有 GT 有轮挡才有效
        slot_weights = slot_mask * slot_reg_mask_t * slot_code_w

        slot_num = slot_masks.float().sum()
        slot_num = torch.clamp(
            reduce_mean(slot_anno_t.new_tensor(slot_num)), min=1.0
        ).item()

        isnotnan = (~torch.isnan(slot_anno_t)).float()
        slot_weights = slot_weights * isnotnan

        loss_slot_reg = self.loss_reg(
            slot_reg_pred, slot_anno_t, slot_weights, avg_factor=slot_num
        )
        loss_dict['loss_slot_reg'] = loss_slot_reg

        # =========================================================
        # 3) slot_type / slot_occ / slot_ws_exist classification
        #    只在 slot center 的正样本位置计算
        # =========================================================
        slot_type_logits = self._transpose_and_gather_feat(pred['slot_type'], slot_inds)              # (B,K,3)
        slot_occ_logits = self._transpose_and_gather_feat(pred['slot_occ'], slot_inds)                # (B,K,7)
        slot_ws_exist_logits = self._transpose_and_gather_feat(pred['slot_ws_exist'], slot_inds)      # (B,K,2)

        pos_mask = slot_masks.bool()

        if pos_mask.any():
            type_pred_pos = slot_type_logits[pos_mask]
            occ_pred_pos = slot_occ_logits[pos_mask]
            ws_exist_pred_pos = slot_ws_exist_logits[pos_mask]

            type_t_pos = slot_type_ids[pos_mask].long()
            occ_t_pos = slot_occ_ids[pos_mask].long()
            ws_exist_t_pos = slot_ws_exist_ids[pos_mask].long()

            loss_type = F.cross_entropy(type_pred_pos, type_t_pos)
            loss_occ = F.cross_entropy(occ_pred_pos, occ_t_pos)
            loss_ws_exist = F.cross_entropy(ws_exist_pred_pos, ws_exist_t_pos)
        else:
            loss_type = slot_type_logits.sum() * 0.0
            loss_occ = slot_occ_logits.sum() * 0.0
            loss_ws_exist = slot_ws_exist_logits.sum() * 0.0

        loss_type_w = float(self.train_cfg.get('loss_type_weight', 1.0))
        loss_occ_w = float(self.train_cfg.get('loss_occ_weight', 1.0))
        loss_ws_exist_w = float(self.train_cfg.get('loss_ws_exist_weight', 1.0))

        loss_dict['loss_slot_type'] = loss_type * loss_type_w
        loss_dict['loss_slot_occ'] = loss_occ * loss_occ_w
        loss_dict['loss_slot_ws_exist'] = loss_ws_exist * loss_ws_exist_w

        return loss_dict

    # =========================================================
    # decode
    # =========================================================
    @torch.no_grad()
    def get_bboxes(self, preds, img_metas=None, img=None, rescale=False):
        """
        这里需要你的 bbox_coder 同步改成新的 decode 接口。
        """
        pred = preds[0]
        return self.bbox_coder.decode_from_slot_center(
            heatmap=pred["heatmap"],              # (B,1,H,W)
            slot_ctr=pred["slot_ctr_offset"],
            slot_kp0=pred["slot_kp0"],
            slot_kp1=pred["slot_kp1"],
            slot_kp2=pred["slot_kp2"],
            slot_kp3=pred["slot_kp3"],
            ws_kp0=pred["ws_kp0"],
            ws_kp1=pred["ws_kp1"],
            slot_type_logits=pred["slot_type"],
            slot_occ_logits=pred["slot_occ"],
            slot_ws_exist_logits=pred["slot_ws_exist"],
        )