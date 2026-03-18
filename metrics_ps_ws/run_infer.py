import os
import json
from metrics_ps_ws.infer import Infer, Data
from tqdm import tqdm
from mmcv import Config

if __name__ == "__main__":
    json_path = "/vePFS/val_data/2026_2_1_full_size/jsons/04b9366f-09c3-41ee-b36e-4203a1d699f1.json"

    config = "/vePFS/Daniel/Fast-BEV-dev/configs/fastbev/exp/paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_new.py"
    checkpoint = "/vePFS/Daniel/Fast-BEV-dev/work_dir400_hook/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_new/best_loss_val.pth"
    data_dir = "/vePFS/data/zhixing_data/"

    save_pred = "/vePFS/Daniel/Fast-BEV-dev/metrics_ps_ws/mix_pred.jpg"
    save_gt   = "/vePFS/Daniel/Fast-BEV-dev/metrics_ps_ws/mix_gt.jpg"

    cfg = Config.fromfile(config)

    cls_infer = Infer(cfg, checkpoint)
    cls_data = Data(cfg, data_dir)

    # -------- infer --------
    data = cls_data.get_data(json_path)
    preds = cls_infer.infer(data)


    (pred_slot_K, pred_slot_corners_list, pred_slot_occ_list, pred_slot_type_list, pred_slot_scores_list,
     pred_ws_K, pred_ws_kps2_list, pred_ws_scores_list) = cls_infer.post_process(preds)

    cls_infer.vis_mix(
        pred_slot_corners_list,
        pred_slot_scores_list,
        ws_kps2_list=pred_ws_kps2_list,
        ws_scores_list=pred_ws_scores_list,
        score_text=True,
        color_by_score=False,
        out_path=save_pred,
        img_size=3000,
        meters_per_pixel=0.01,
        point_radius=4,
        line_thickness=2
    )

    # -------- GT --------
    (gt_slot_kps_list, gt_slot_occ_list, gt_slot_type_list, gt_slot_scores_list,
     gt_ws_kps2_list, gt_ws_scores_list) = cls_data.get_gt_slot_ws(json_path)

    cls_infer.vis_mix(
        gt_slot_kps_list,
        gt_slot_scores_list,
        ws_kps2_list=gt_ws_kps2_list,
        ws_scores_list=gt_ws_scores_list,
        score_text=True,
        color_by_score=False,
        out_path=save_gt,
        img_size=3000,
        meters_per_pixel=0.01,
        point_radius=4,
        line_thickness=2
    )

    print(f"[INFO] pred slots={pred_slot_K} pred ws={pred_ws_K} | gt slots={len(gt_slot_kps_list)} gt ws={len(gt_ws_kps2_list)}")