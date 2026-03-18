import os
import json
from metrics_ps_ws.infer import Infer, Data
from metrics_ps_ws.metrics_v4 import MetricsV4
from metrics_ps_ws.metrics_v5 import MetricsV5
from tqdm import tqdm
from mmcv import Config

config = '/vePFS/Daniel/Fast-BEV-dev/configs/fastbev/exp/paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_new.py'
checkpoint = '/vePFS/Daniel/Fast-BEV-dev/work_dir400_hook/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_psd_new/best_loss_val.pth'
data_dir = '/vePFS/data/zhixing_data/'
save_path = '/vePFS/Daniel/Fast-BEV-dev/metrics_ps_ws/output/render.jpg'
cfg = Config.fromfile(config)
cfg.model.psd_head.bbox_coder.score_threshold = 0.2
cfg.model.psd_head.bbox_coder.nms_kernel_size = 15


cls_infer = Infer(cfg, checkpoint)
cls_data = Data(cfg, data_dir)
cls_metricsV4 = MetricsV4()
cls_metricsV5 = MetricsV5()

if __name__ == "__main__":
    json_dir = '/vePFS/val_data/2026_2_1_full_size/jsons/'
    json_list = os.listdir(json_dir)

    for json_file in tqdm(json_list, "proc"):
        json_full_path = os.path.join(json_dir, json_file)
        data = cls_data.get_data(json_full_path)
        preds = cls_infer.infer(data)
        (pred_slot_K, pred_slot_corners_list, pred_slot_occ_list, pred_slot_type_list, pred_slot_scores_list,
        pred_ws_K, pred_ws_kps2_list, pred_ws_scores_list) = cls_infer.post_process(preds)


        (gt_slot_kps_list, gt_slot_occ_list, gt_slot_type_list, gt_slot_scores_list,
        gt_ws_kps2_list, gt_ws_scores_list) = cls_data.get_gt_slot_ws(json_full_path)


        # v4
        # matches, pred_unmatched, gt_unmatched = cls_metricsV4.update_one_frame(
        #     pred_corners_list = pred_slot_corners_list,
        #     pred_scores_list  = pred_slot_scores_list,
        #     gt_kps_list       = gt_slot_kps_list,
        #     gt_scores_list    = gt_slot_scores_list,
        #     iou_thresh        = 0.8,
        #     kp_thresh         = 0.3,
        #     use_best_kp_match = False,
        #     max_r=5.0,
        # )

        # v5
        matches, pred_unmatched, gt_unmatched = cls_metricsV5.update_one_frame(
            pred_corners_list=pred_slot_corners_list,
            pred_scores_list=pred_slot_scores_list,
            gt_kps_list=gt_slot_kps_list,
            gt_scores_list=gt_slot_scores_list,

            iou_thresh=0.8,
            kp_thresh=0.1,
            use_best_kp_match=False,
            max_r=5.0,

            pred_occtype_list=pred_slot_occ_list,
            gt_occtype_list=gt_slot_occ_list,
            pred_spetype_list=pred_slot_type_list,
            gt_spetype_list=gt_slot_type_list,
        )

report = cls_metricsV5.summarize()
print("\n==== Final Evaluation ====")
for k, v in report.items():
    print(k, ":", v)

save_path = "/Prod/metrics_ps_ws/report_full_size_5.0m_kp30.json"  
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print(f"\nMetrics saved to: {os.path.abspath(save_path)}")