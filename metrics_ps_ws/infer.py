import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import mmcv
from pyquaternion import Quaternion
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ========================= Infer =========================
class Infer:
    def __init__(self, config, checkpoint):
        cfg = config
        self.cfg = cfg
        self.model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(self.model, checkpoint, map_location='cpu')
        self.model.to(device).eval()
        


    def infer(self, data):
        with torch.no_grad():
            # 你的 get_bboxes 里是 decode_slot_ws
            result = self.model(return_loss=False, rescale=True, **data)
        return result

    # -------- robust parse decode output --------
    def _parse_decode_slot_ws(self, result):
        """
        支持几种常见返回：
        1) dict: {"slots": {...}, "wheelstops": {...}}
        2) tuple/list: (slots, wheelstops)
        3) batch 外层 list: [ (slots, wheelstops) ] 或 [ {"slots":..} ]
        """
        out = result
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]

        if isinstance(out, dict):
            if "slots" in out or "wheelstops" in out:
                return out.get("slots", None), out.get("wheelstops", None)

        if isinstance(out, (list, tuple)) and len(out) >= 2:
            return out[0], out[1]

        # fallback：只有一个东西，认为是 slots
        return out, None

    def _slots_obj_to_lists(self, slots_obj):
        """
        slots_obj:
          - dict: {centers:(K,2), kps:(K,4,2) or [kp0..kp3], scores:(K,), ...}
          - list/tuple old-style: [centers, scores, labels, occ/status, kps]
        return:
          K, corners_list(list of (4,2)), type_list, occ_list, scores_list
        """
        if slots_obj is None:
            return 0, [], [], [], []

        # dict style
        if isinstance(slots_obj, dict):
            centers = slots_obj.get("centers", slots_obj.get("center", None))
            kps = slots_obj.get("kps", slots_obj.get("corners", None))
            scores = slots_obj.get("scores", slots_obj.get("score", None))
            slot_type = slots_obj.get("slot_type", slots_obj.get("type", None))
            slot_occ = slots_obj.get("slot_occ", slots_obj.get("occ", slots_obj.get("status", None)))

            if centers is None or kps is None:
                return 0, [], [], [], []

            centers = centers.detach().cpu().numpy() if torch.is_tensor(centers) else np.asarray(centers)
            K = int(centers.shape[0])

            # kps can be (K,4,2) or list of 4 (K,2)
            if isinstance(kps, (list, tuple)) and len(kps) == 4:
                kp0 = kps[0].detach().cpu().numpy() if torch.is_tensor(kps[0]) else np.asarray(kps[0])
                kp1 = kps[1].detach().cpu().numpy() if torch.is_tensor(kps[1]) else np.asarray(kps[1])
                kp2 = kps[2].detach().cpu().numpy() if torch.is_tensor(kps[2]) else np.asarray(kps[2])
                kp3 = kps[3].detach().cpu().numpy() if torch.is_tensor(kps[3]) else np.asarray(kps[3])
                corners_list = [np.stack([kp0[i], kp1[i], kp2[i], kp3[i]], axis=0).astype(np.float32) for i in range(K)]
            else:
                kps = kps.detach().cpu().numpy() if torch.is_tensor(kps) else np.asarray(kps)
                corners_list = [kps[i].astype(np.float32) for i in range(K)]  # (4,2)

            if scores is None:
                scores_list = [1.0] * K
            else:
                scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
                scores_list = [float(scores[i]) for i in range(K)]

            if slot_type is None:
                type_list = [0] * K
            else:
                slot_type = slot_type.detach().cpu().numpy() if torch.is_tensor(slot_type) else np.asarray(slot_type)
                type_list = [int(slot_type[i]) for i in range(K)]

            if slot_occ is None:
                occ_list = [0] * K
            else:
                slot_occ = slot_occ.detach().cpu().numpy() if torch.is_tensor(slot_occ) else np.asarray(slot_occ)
                occ_list = [int(slot_occ[i]) for i in range(K)]

            return K, corners_list, occ_list, type_list, scores_list

        # list/tuple old-style
        if isinstance(slots_obj, (list, tuple)) and len(slots_obj) >= 5:
            centers = slots_obj[0]
            scores  = slots_obj[1]
            labels  = slots_obj[2]
            occ     = slots_obj[3]
            kps     = slots_obj[4]

            centers = centers.detach().cpu().numpy() if torch.is_tensor(centers) else np.asarray(centers)
            K = int(centers.shape[0])

            if isinstance(kps, (list, tuple)) and len(kps) == 4:
                kps_np = [kp.detach().cpu().numpy() if torch.is_tensor(kp) else np.asarray(kp) for kp in kps]
                corners_list = [np.stack([kps_np[0][i], kps_np[1][i], kps_np[2][i], kps_np[3][i]], axis=0).astype(np.float32)
                                for i in range(K)]
            else:
                kps = kps.detach().cpu().numpy() if torch.is_tensor(kps) else np.asarray(kps)
                corners_list = [kps[i].astype(np.float32) for i in range(K)]

            scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
            scores_list = [float(scores[i]) for i in range(K)]

            labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
            type_list = [int(labels[i]) for i in range(K)]

            occ = occ.detach().cpu().numpy() if torch.is_tensor(occ) else np.asarray(occ)
            occ_list = [int(occ[i]) for i in range(K)]

            return K, corners_list, occ_list, type_list, scores_list

        return 0, [], [], [], []

    def _ws_obj_to_lists(self, ws_obj):
        """
        wheelstop obj:
          - dict: {centers:(K,2), kps:(K,2,2) or [kp0,kp1], scores:(K,)}
          - list/tuple: [centers, scores, kps]
        return:
          K, ws_kps2_list(list of (2,2)), scores_list
        """
        if ws_obj is None:
            return 0, [], []

        if isinstance(ws_obj, dict):
            centers = ws_obj.get("centers", ws_obj.get("center", None))
            kps = ws_obj.get("kps", None)
            scores = ws_obj.get("scores", ws_obj.get("score", None))

            if centers is None or kps is None:
                return 0, [], []

            centers = centers.detach().cpu().numpy() if torch.is_tensor(centers) else np.asarray(centers)
            K = int(centers.shape[0])

            if isinstance(kps, (list, tuple)) and len(kps) == 2:
                kp0 = kps[0].detach().cpu().numpy() if torch.is_tensor(kps[0]) else np.asarray(kps[0])
                kp1 = kps[1].detach().cpu().numpy() if torch.is_tensor(kps[1]) else np.asarray(kps[1])
                ws_list = [np.stack([kp0[i], kp1[i]], axis=0).astype(np.float32) for i in range(K)]
            else:
                kps = kps.detach().cpu().numpy() if torch.is_tensor(kps) else np.asarray(kps)
                ws_list = [kps[i].astype(np.float32) for i in range(K)]  # (2,2)

            if scores is None:
                scores_list = [1.0] * K
            else:
                scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
                scores_list = [float(scores[i]) for i in range(K)]

            return K, ws_list, scores_list

        if isinstance(ws_obj, (list, tuple)) and len(ws_obj) >= 3:
            centers = ws_obj[0]
            scores = ws_obj[1]
            kps = ws_obj[2] if len(ws_obj) == 3 else ws_obj[-1]

            centers = centers.detach().cpu().numpy() if torch.is_tensor(centers) else np.asarray(centers)
            K = int(centers.shape[0])

            kps = kps.detach().cpu().numpy() if torch.is_tensor(kps) else np.asarray(kps)
            ws_list = [kps[i].astype(np.float32) for i in range(K)]

            scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
            scores_list = [float(scores[i]) for i in range(K)]
            return K, ws_list, scores_list

        return 0, [], []

    def post_process(self, preds):
        """
        统一返回：
          slot: K, corners_list(4,2), slot_occ_list, slot_type_list, slot_scores_list
          ws:   W, ws_kps2_list(2,2), ws_scores_list
        """
        slots_obj, ws_obj = self._parse_decode_slot_ws(preds)

        slot_K, slot_corners_list, slot_occ_list, slot_type_list, slot_scores_list = self._slots_obj_to_lists(slots_obj)
        ws_K, ws_kps2_list, ws_scores_list = self._ws_obj_to_lists(ws_obj)

        return (slot_K, slot_corners_list, slot_occ_list, slot_type_list, slot_scores_list,
                ws_K, ws_kps2_list, ws_scores_list)

    # -------- visualization: draw slot + ws in ONE image --------
    def vis_mix(
        self,
        slot_corners_list,
        slot_scores_list,
        ws_kps2_list=None,
        ws_scores_list=None,
        score_text=True,
        color_by_score=False,
        out_path="mix.png",
        img_size=3000,
        meters_per_pixel=0.01,
        point_radius=4,
        line_thickness=2
    ):
        """
        slot_corners_list: list of (4,2) [x_fwd, y_left]
        ws_kps2_list:      list of (2,2) [x_fwd, y_left]
        """
        if ws_kps2_list is None: ws_kps2_list = []
        if ws_scores_list is None: ws_scores_list = [1.0] * len(ws_kps2_list)

        # slots 允许 scores_list=None
        if slot_scores_list is None:
            slot_scores_list = [1.0] * len(slot_corners_list)

        # ------------ helper ------------
        def score_to_bgr(score01: float):
            s = float(np.clip(score01, 0.0, 1.0))
            r = int(round(255 * (1.0 - s)))
            g = int(round(255 * s))
            b = 0
            return (b, g, r)

        def ego_to_img(x_m, y_m, cx, cy):
            # 你的约定：u = cx - y/pp, v = cy - x/pp
            u = int(round(cx - y_m / meters_per_pixel))
            v = int(round(cy - x_m / meters_per_pixel))
            return u, v

        def draw_axes_with_ticks(img, cx, cy, tick_every_m=1.0):
            H, W = img.shape[:2]
            px_per_m = int(round(1.0 / meters_per_pixel))

            axis_color = (200, 200, 200)
            cv2.line(img, (cx, 0), (cx, H - 1), axis_color, 1, cv2.LINE_AA)
            cv2.line(img, (0, cy), (W - 1, cy), axis_color, 1, cv2.LINE_AA)

            # x+ 向上，y+ 向左
            L = int(round(2.0 / meters_per_pixel))
            cv2.arrowedLine(img, (cx, cy), (cx, max(0, cy - L)), (0, 0, 255), 2, tipLength=0.03)
            cv2.arrowedLine(img, (cx, cy), (max(0, cx - L), cy), (255, 0, 0), 2, tipLength=0.03)
            cv2.putText(img, "x+", (cx + 8, max(15, cy - L + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "y+", (max(5, cx - L + 5), cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(img, "0", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            tick_len_px = 10
            max_steps_x = max(cy // px_per_m, (H - 1 - cy) // px_per_m)
            max_steps_y = max(cx // px_per_m, (W - 1 - cx) // px_per_m)
            max_steps = int(max(max_steps_x, max_steps_y))

            for k in range(1, max_steps + 1):
                d_px = int(round(k * tick_every_m * px_per_m))
                # x 轴刻度（竖轴上）
                v_up = cy - d_px
                if 0 <= v_up < H:
                    cv2.line(img, (cx - tick_len_px, v_up), (cx + tick_len_px, v_up), axis_color, 1, cv2.LINE_AA)
                v_dn = cy + d_px
                if 0 <= v_dn < H:
                    cv2.line(img, (cx - tick_len_px, v_dn), (cx + tick_len_px, v_dn), axis_color, 1, cv2.LINE_AA)

                # y 轴刻度（横轴上，左正右负）
                u_left = cx - d_px
                if 0 <= u_left < W:
                    cv2.line(img, (u_left, cy - tick_len_px), (u_left, cy + tick_len_px), axis_color, 1, cv2.LINE_AA)
                u_right = cx + d_px
                if 0 <= u_right < W:
                    cv2.line(img, (u_right, cy - tick_len_px), (u_right, cy + tick_len_px), axis_color, 1, cv2.LINE_AA)

        # ------------ canvas ------------
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        cx = cy = img_size // 2
        draw_axes_with_ticks(img, cx, cy, tick_every_m=1.0)

        # ------------ score normalize for color ------------
        # slots & ws 分别做归一化，避免两类数值范围不同导致颜色全挤到一边
        slot_scores = np.asarray(slot_scores_list, dtype=np.float32) if len(slot_scores_list) else np.zeros((0,), np.float32)
        ws_scores = np.asarray(ws_scores_list, dtype=np.float32) if len(ws_scores_list) else np.zeros((0,), np.float32)

        def norm01(arr):
            if arr.size == 0:
                return arr
            smin, smax = float(arr.min()), float(arr.max())
            if smax - smin < 1e-6:
                return np.ones_like(arr) * 0.5
            return (arr - smin) / (smax - smin)

        slot_scores01 = norm01(slot_scores)
        ws_scores01 = norm01(ws_scores)

        # ------------ draw slots (polylines) ------------
        for si, (slot, sc, sc01) in enumerate(zip(slot_corners_list, slot_scores, slot_scores01)):
            slot = np.asarray(slot, dtype=np.float32)
            if slot.shape != (4, 2):
                continue

            pts = []
            for i in range(4):
                u, v = ego_to_img(float(slot[i, 0]), float(slot[i, 1]), cx, cy)
                pts.append([u, v])
            pts = np.array(pts, dtype=np.int32)

            color = score_to_bgr(sc01) if color_by_score else (255, 255, 255)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, line_thickness, cv2.LINE_AA)

            # 角点
            for pi, (u, v) in enumerate(pts):
                if 0 <= u < img_size and 0 <= v < img_size:
                    cv2.circle(img, (int(u), int(v)), point_radius, (0, 255, 0), -1)
                    cv2.putText(img, str(pi + 1), (int(u) + 6, int(v) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            if score_text:
                center_xy = slot.mean(axis=0)
                cu, cv_ = ego_to_img(float(center_xy[0]), float(center_xy[1]), cx, cy)
                if 0 <= cu < img_size and 0 <= cv_ < img_size:
                    txt = f"S:{float(sc):.3f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    pad = 6
                    (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
                    x1 = cu - tw // 2 - pad
                    y1 = cv_ - th // 2 - pad
                    x2 = cu + tw // 2 + pad
                    y2 = cv_ + th // 2 + pad
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    org = (cu - tw // 2, cv_ + th // 2 - baseline)
                    cv2.putText(img, txt, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # ------------ draw wheelstops (segment) ------------
        for wi, (seg, sc, sc01) in enumerate(zip(ws_kps2_list, ws_scores, ws_scores01)):
            seg = np.asarray(seg, dtype=np.float32)
            if seg.shape != (2, 2):
                continue

            p0 = ego_to_img(float(seg[0, 0]), float(seg[0, 1]), cx, cy)
            p1 = ego_to_img(float(seg[1, 0]), float(seg[1, 1]), cx, cy)

            # ws 用黄色系更醒目（如果要按score上色就仍用score_to_bgr）
            color = score_to_bgr(sc01) if color_by_score else (0, 255, 255)

            cv2.line(img, p0, p1, color, max(2, line_thickness + 1), cv2.LINE_AA)
            cv2.circle(img, p0, point_radius + 1, (0, 255, 255), -1)
            cv2.circle(img, p1, point_radius + 1, (0, 255, 255), -1)

            if score_text:
                center_xy = seg.mean(axis=0)
                cu, cv_ = ego_to_img(float(center_xy[0]), float(center_xy[1]), cx, cy)
                if 0 <= cu < img_size and 0 <= cv_ < img_size:
                    txt = f"W:{float(sc):.3f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    pad = 6
                    (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
                    x1 = cu - tw // 2 - pad
                    y1 = cv_ - th // 2 - pad
                    x2 = cu + tw // 2 + pad
                    y2 = cv_ + th // 2 + pad
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    org = (cu - tw // 2, cv_ + th // 2 - baseline)
                    cv2.putText(img, txt, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imwrite(out_path, img)
        print(f"[OK] Saved: {out_path}")

    def vis_mix_com(
        self,
        slot_corners_list,
        slot_scores_list,
        slot_occ_list=None,
        slot_type_list=None,
        ws_kps2_list=None,
        ws_scores_list=None,
        score_text=True,
        color_by_score=False,
        out_path="mix.png",
        img_size=3000,
        meters_per_pixel=0.01,
        point_radius=4,
        line_thickness=2
    ):
        """
        slot_corners_list: list of (4,2) [x_fwd, y_left]
        slot_occ_list:     list of occ cls/id or score
        slot_type_list:    list of type cls/id or score
        ws_kps2_list:      list of (2,2) [x_fwd, y_left]
        """
        if ws_kps2_list is None:
            ws_kps2_list = []
        if ws_scores_list is None:
            ws_scores_list = [1.0] * len(ws_kps2_list)

        # slots 允许 scores_list=None
        if slot_scores_list is None:
            slot_scores_list = [1.0] * len(slot_corners_list)

        if slot_occ_list is None:
            slot_occ_list = [None] * len(slot_corners_list)

        if slot_type_list is None:
            slot_type_list = [None] * len(slot_corners_list)

        # ------------ helper ------------
        def score_to_bgr(score01: float):
            s = float(np.clip(score01, 0.0, 1.0))
            r = int(round(255 * (1.0 - s)))
            g = int(round(255 * s))
            b = 0
            return (b, g, r)

        def ego_to_img(x_m, y_m, cx, cy):
            # 你的约定：u = cx - y/pp, v = cy - x/pp
            u = int(round(cx - y_m / meters_per_pixel))
            v = int(round(cy - x_m / meters_per_pixel))
            return u, v

        def format_cls_or_score(x, name=""):
            """兼容 int / float / np标量 / tensor(标量)"""
            if x is None:
                return f"{name}:None" if name else "None"

            # torch tensor scalar
            if hasattr(x, "detach"):
                try:
                    x = x.detach().cpu().item()
                except Exception:
                    pass

            # numpy scalar
            if isinstance(x, np.generic):
                x = x.item()

            if isinstance(x, (int, np.integer)):
                return f"{name}:{int(x)}" if name else f"{int(x)}"

            if isinstance(x, (float, np.floating)):
                # 如果本质是整数，比如 2.0，也按整数显示
                if abs(float(x) - round(float(x))) < 1e-6:
                    return f"{name}:{int(round(float(x)))}" if name else f"{int(round(float(x)))}"
                return f"{name}:{float(x):.3f}" if name else f"{float(x):.3f}"

            return f"{name}:{str(x)}" if name else str(x)

        def draw_axes_with_ticks(img, cx, cy, tick_every_m=1.0):
            H, W = img.shape[:2]
            px_per_m = int(round(1.0 / meters_per_pixel))

            axis_color = (200, 200, 200)
            cv2.line(img, (cx, 0), (cx, H - 1), axis_color, 1, cv2.LINE_AA)
            cv2.line(img, (0, cy), (W - 1, cy), axis_color, 1, cv2.LINE_AA)

            # x+ 向上，y+ 向左
            L = int(round(2.0 / meters_per_pixel))
            cv2.arrowedLine(img, (cx, cy), (cx, max(0, cy - L)), (0, 0, 255), 2, tipLength=0.03)
            cv2.arrowedLine(img, (cx, cy), (max(0, cx - L), cy), (255, 0, 0), 2, tipLength=0.03)
            cv2.putText(img, "x+", (cx + 8, max(15, cy - L + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "y+", (max(5, cx - L + 5), cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(img, "0", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            tick_len_px = 10
            max_steps_x = max(cy // px_per_m, (H - 1 - cy) // px_per_m)
            max_steps_y = max(cx // px_per_m, (W - 1 - cx) // px_per_m)
            max_steps = int(max(max_steps_x, max_steps_y))

            for k in range(1, max_steps + 1):
                d_px = int(round(k * tick_every_m * px_per_m))
                # x 轴刻度（竖轴上）
                v_up = cy - d_px
                if 0 <= v_up < H:
                    cv2.line(img, (cx - tick_len_px, v_up), (cx + tick_len_px, v_up), axis_color, 1, cv2.LINE_AA)
                v_dn = cy + d_px
                if 0 <= v_dn < H:
                    cv2.line(img, (cx - tick_len_px, v_dn), (cx + tick_len_px, v_dn), axis_color, 1, cv2.LINE_AA)

                # y 轴刻度（横轴上，左正右负）
                u_left = cx - d_px
                if 0 <= u_left < W:
                    cv2.line(img, (u_left, cy - tick_len_px), (u_left, cy + tick_len_px), axis_color, 1, cv2.LINE_AA)
                u_right = cx + d_px
                if 0 <= u_right < W:
                    cv2.line(img, (u_right, cy - tick_len_px), (u_right, cy + tick_len_px), axis_color, 1, cv2.LINE_AA)

        def draw_multiline_label(img, lines, center_u, center_v,
                                font=cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale=0.85,
                                thickness=2,
                                text_color=(255, 255, 255),
                                bg_color=(0, 0, 0),
                                pad=6,
                                line_gap=6):
            """在中心位置画多行文本"""
            if lines is None or len(lines) == 0:
                return

            sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]
            widths = [s[0][0] for s in sizes]
            heights = [s[0][1] for s in sizes]
            baselines = [s[1] for s in sizes]

            box_w = max(widths) + 2 * pad
            content_h = sum(heights) + (len(lines) - 1) * line_gap
            box_h = content_h + 2 * pad

            x1 = int(center_u - box_w / 2)
            y1 = int(center_v - box_h / 2)
            x2 = x1 + box_w
            y2 = y1 + box_h

            cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)

            y_cursor = y1 + pad
            for t, (sz, baseline) in zip(lines, sizes):
                tw, th = sz
                org_x = int(center_u - tw / 2)
                org_y = int(y_cursor + th)
                cv2.putText(img, t, (org_x, org_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
                y_cursor += th + line_gap

        # ------------ canvas ------------
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        cx = cy = img_size // 2
        draw_axes_with_ticks(img, cx, cy, tick_every_m=1.0)

        # ------------ score normalize for color ------------
        # slots & ws 分别做归一化，避免两类数值范围不同导致颜色全挤到一边
        slot_scores = np.asarray(slot_scores_list, dtype=np.float32) if len(slot_scores_list) else np.zeros((0,), np.float32)
        ws_scores = np.asarray(ws_scores_list, dtype=np.float32) if len(ws_scores_list) else np.zeros((0,), np.float32)

        def norm01(arr):
            if arr.size == 0:
                return arr
            smin, smax = float(arr.min()), float(arr.max())
            if smax - smin < 1e-6:
                return np.ones_like(arr) * 0.5
            return (arr - smin) / (smax - smin)

        slot_scores01 = norm01(slot_scores)
        ws_scores01 = norm01(ws_scores)

        # ------------ draw slots (polylines) ------------
        for si, (slot, sc, sc01) in enumerate(zip(slot_corners_list, slot_scores, slot_scores01)):
            slot = np.asarray(slot, dtype=np.float32)
            if slot.shape != (4, 2):
                continue

            occ = slot_occ_list[si] if si < len(slot_occ_list) else None
            slot_type = slot_type_list[si] if si < len(slot_type_list) else None

            pts = []
            for i in range(4):
                u, v = ego_to_img(float(slot[i, 0]), float(slot[i, 1]), cx, cy)
                pts.append([u, v])
            pts = np.array(pts, dtype=np.int32)

            color = score_to_bgr(sc01) if color_by_score else (255, 255, 255)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, line_thickness, cv2.LINE_AA)

            # 角点
            for pi, (u, v) in enumerate(pts):
                if 0 <= u < img_size and 0 <= v < img_size:
                    cv2.circle(img, (int(u), int(v)), point_radius, (0, 255, 0), -1)
                    cv2.putText(img, str(pi + 1), (int(u) + 6, int(v) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            if score_text:
                center_xy = slot.mean(axis=0)
                cu, cv_ = ego_to_img(float(center_xy[0]), float(center_xy[1]), cx, cy)
                if 0 <= cu < img_size and 0 <= cv_ < img_size:
                    lines = [
                        f"S:{float(sc):.3f}",
                        format_cls_or_score(occ, "Occ"),
                        format_cls_or_score(slot_type, "Type"),
                    ]
                    draw_multiline_label(img, lines, cu, cv_)

        # ------------ draw wheelstops (segment) ------------
        for wi, (seg, sc, sc01) in enumerate(zip(ws_kps2_list, ws_scores, ws_scores01)):
            seg = np.asarray(seg, dtype=np.float32)
            if seg.shape != (2, 2):
                continue

            p0 = ego_to_img(float(seg[0, 0]), float(seg[0, 1]), cx, cy)
            p1 = ego_to_img(float(seg[1, 0]), float(seg[1, 1]), cx, cy)

            # ws 用黄色系更醒目（如果要按score上色就仍用score_to_bgr）
            color = score_to_bgr(sc01) if color_by_score else (0, 255, 255)

            cv2.line(img, p0, p1, color, max(2, line_thickness + 1), cv2.LINE_AA)
            cv2.circle(img, p0, point_radius + 1, (0, 255, 255), -1)
            cv2.circle(img, p1, point_radius + 1, (0, 255, 255), -1)

            if score_text:
                center_xy = seg.mean(axis=0)
                cu, cv_ = ego_to_img(float(center_xy[0]), float(center_xy[1]), cx, cy)
                if 0 <= cu < img_size and 0 <= cv_ < img_size:
                    txt = f"W:{float(sc):.3f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    pad = 6
                    (tw, th), baseline = cv2.getTextSize(txt, font, font_scale, thickness)
                    x1 = cu - tw // 2 - pad
                    y1 = cv_ - th // 2 - pad
                    x2 = cu + tw // 2 + pad
                    y2 = cv_ + th // 2 + pad
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    org = (cu - tw // 2, cv_ + th // 2 - baseline)
                    cv2.putText(img, txt, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imwrite(out_path, img)
        print(f"[OK] Saved: {out_path}")


# ========================= Data =========================
class Data:
    def __init__(self, config, data_dir):
        self.cfg = config
        self.data_dir = data_dir
        self.spe_cls = getattr(config.model.psd_head.bbox_coder, 'spe_cls', 4)
        self.occ_cls = getattr(config.model.psd_head.bbox_coder, 'occ_cls', 4)

    def mmlabNormalize(self, img):
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        img = img.astype(np.float32, copy=True)
        img = (img - mean) / std
        return torch.from_numpy(img).float().permute(2, 0, 1).contiguous()

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate, post_rot, post_tran):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        rw, rh = resize
        post_rot = post_rot.clone()
        post_tran = post_tran.clone()

        post_rot[0, :] *= rw
        post_rot[1, :] *= rh
        post_tran = post_tran - post_tran.new_tensor(crop[:2])

        return img, post_rot, post_tran

    def sample_augmentation(self, data_config, H, W):
        fH, fW = data_config['input_size']
        resize_w = float(fW) / float(W)
        resize_h = float(fH) / float(H)

        resize_dims = (fW, fH)
        crop = (0, 0, fW, fH)
        flip = False
        rotate = 0.0
        return (resize_w, resize_h), resize_dims, crop, flip, rotate

    def read_calib(self, calib_path):
        fish = mmcv.load(calib_path, file_format='json')['fisheye']
        rotation_M = Quaternion(fish['rotation']).rotation_matrix
        E = np.eye(4)
        E[:3, :3] = np.array(rotation_M, dtype=float).reshape(3, 3)
        E[:3, 3] = np.array(fish['translation'], dtype=float).reshape(3,)
        K = np.array(fish['camera_intrinsic'], dtype=float).reshape(3, 3)
        dist = fish['distortion']
        return E, K, dist

    def get_data(self, json_path):
        cfg = self.cfg
        with open(json_path, 'r') as f:
            data = json.load(f)
        scene_id = data['scene_id']

        img_path_list = [
            self.data_dir + scene_id + '/' + data['image_path']['CAM_FISHEYE_FORWARD'],
            self.data_dir + scene_id + '/' + data['image_path']['CAM_FISHEYE_LEFT'],
            self.data_dir + scene_id + '/' + data['image_path']['CAM_FISHEYE_BACKWARD'],
            self.data_dir + scene_id + '/' + data['image_path']['CAM_FISHEYE_RIGHT'],
        ]
        calib_path_list = [
            self.data_dir + scene_id + '/' + data['calib']['CAM_FISHEYE_FORWARD'],
            self.data_dir + scene_id + '/' + data['calib']['CAM_FISHEYE_LEFT'],
            self.data_dir + scene_id + '/' + data['calib']['CAM_FISHEYE_BACKWARD'],
            self.data_dir + scene_id + '/' + data['calib']['CAM_FISHEYE_RIGHT'],
        ]

        imgs = []
        cam2ego_list, intrin_list, dist_list = [], [], []
        post_rot_list, post_tran_list = [], []

        for i, img_path in enumerate(img_path_list):
            img = Image.open(img_path).convert("RGB")
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            scale, resize_dims, crop, flip, rotate = self.sample_augmentation(cfg.data_config, H=img.height, W=img.width)
            img, post_rot2, post_tran2 = self.img_transform(img, scale, resize_dims, crop, flip, rotate, post_rot, post_tran)

            imgs.append(self.mmlabNormalize(img))

            E, K, dist = self.read_calib(calib_path_list[i])
            cam2ego_list.append(torch.tensor(E, dtype=torch.float32))
            intrin_list.append(torch.tensor(K, dtype=torch.float32))
            dist_list.append(torch.tensor(dist, dtype=torch.float32))

            post_tran3 = torch.zeros(3)
            post_rot3 = torch.eye(3)
            post_tran3[:2] = post_tran2
            post_rot3[:2, :2] = post_rot2
            post_rot_list.append(post_rot3)
            post_tran_list.append(post_tran3)

        img_tr = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
        cam2ego_tr = torch.stack(cam2ego_list, dim=0).unsqueeze(0).to(device)
        intrin_tr = torch.stack(intrin_list, dim=0).unsqueeze(0).to(device)
        dist_tr = torch.stack(dist_list, dim=0).unsqueeze(0).to(device)
        post_rot_tr = torch.stack(post_rot_list, dim=0).unsqueeze(0).to(device)
        post_tran_tr = torch.stack(post_tran_list, dim=0).unsqueeze(0).to(device)

        B = img_tr.shape[0]
        bda = torch.eye(3, dtype=torch.float32, device=device).view(1, 3, 3).repeat(B, 1, 1)

        img_inputs = (img_tr, cam2ego_tr, intrin_tr, post_rot_tr, post_tran_tr, dist_tr, bda)
        return dict(img_inputs=img_inputs, img_metas=None)


    def _map_spe_label(self, x):
        x = int(x)
        x = 3 if x == -1 else x
        x = min(x, self.spe_cls - 1)
        return x

    def _map_occ_label(self, x):
        x = int(x)
        x = min(x, self.occ_cls - 1)
        return x

    def get_gt_slot_ws(self, json_path):
        """
        return:
            slot_kps_list: list of (4,2)
            slot_occ_list
            slot_type_list
            slot_scores_list
            ws_kps2_list: list of (2,2)
            ws_scores_list
        """
        slot_kps_list = []
        slot_occ_list = []
        slot_type_list = []
        slot_scores_list = []

        ws_kps2_list = []
        ws_scores_list = []

        with open(json_path, 'r') as f:
            data = json.load(f)

        items = data.get("annotations", {}).get("parking_slot_detection", [])

        for it in items:
            tp = it.get("type", "")

            # ---------------- parking slot ----------------
            if tp == "parking_slot":
                pts = it.get("points_3d", [])
                if len(pts) != 4:
                    continue

                kps = [[p['x'], p['y']] for p in pts]
                occ = it.get("attributes", {}).get("occupy", -1)
                shape = it.get("attributes", {}).get("shape", -1)

                # slot_kps_list.append(np.asarray(kps, np.float32))
                # slot_occ_list.append(int(occ))
                # slot_type_list.append(int(shape))
                # slot_scores_list.append(1.0)
                occ = self._map_occ_label(occ)
                shape = self._map_spe_label(shape)

                slot_kps_list.append(np.asarray(kps, np.float32))
                slot_occ_list.append(int(occ))
                slot_type_list.append(int(shape))
                slot_scores_list.append(1.0)

            # ---------------- wheel stop ----------------
            elif tp == "wheel_stop":
                pts = it.get("points_3d", [])
                if len(pts) != 2:
                    continue

                kps2 = [[pts[0]['x'], pts[0]['y']],
                        [pts[1]['x'], pts[1]['y']]]

                ws_kps2_list.append(np.asarray(kps2, np.float32))
                ws_scores_list.append(1.0)

        return (slot_kps_list, slot_occ_list, slot_type_list, slot_scores_list,
                ws_kps2_list, ws_scores_list)