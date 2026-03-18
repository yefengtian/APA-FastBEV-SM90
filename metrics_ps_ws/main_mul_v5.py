import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


TAG = 'V5'   # 'V4' or 'V5'


def _worker_run(json_files, json_dir, config, checkpoint, data_dir,
                score_threshold, nms_kernel_size,
                iou_thresh, kp_thresh, use_best_kp_match, max_r,
                gpu_id=None):
    """
    每个进程处理一批 json 文件，返回可 merge 的中间结果。
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from tqdm import tqdm
    from mmcv import Config
    from metrics_ps_ws.infer import Infer, Data

    if TAG == 'V4':
        from metrics_ps_ws.metrics_v4 import MetricsV4
    else:
        from metrics_ps_ws.metrics_v5 import MetricsV5

    cfg = Config.fromfile(config)
    cfg.model.psd_head.bbox_coder.score_threshold = score_threshold
    cfg.model.psd_head.bbox_coder.nms_kernel_size = nms_kernel_size

    cls_infer = Infer(cfg, checkpoint)
    cls_data = Data(cfg, data_dir)

    if TAG == 'V4':
        cls_metrics = MetricsV4()
    else:
        cls_metrics = MetricsV5()

    for jf in tqdm(json_files, desc=f"worker gpu={gpu_id}", leave=False):
        json_full_path = os.path.join(json_dir, jf)
        try:
            data = cls_data.get_data(json_full_path)
            preds = cls_infer.infer(data)

            (
                pred_slot_K,
                pred_slot_corners_list,
                pred_slot_occ_list,
                pred_slot_type_list,
                pred_slot_scores_list,
                pred_ws_K,
                pred_ws_kps2_list,
                pred_ws_scores_list
            ) = cls_infer.post_process(preds)

            (
                gt_slot_kps_list,
                gt_slot_occ_list,
                gt_slot_type_list,
                gt_slot_scores_list,
                gt_ws_kps2_list,
                gt_ws_scores_list
            ) = cls_data.get_gt_slot_ws(json_full_path)

            if TAG == 'V4':
                cls_metrics.update_one_frame(
                    pred_corners_list=pred_slot_corners_list,
                    pred_scores_list=pred_slot_scores_list,
                    gt_kps_list=gt_slot_kps_list,
                    gt_scores_list=gt_slot_scores_list,
                    iou_thresh=iou_thresh,
                    kp_thresh=kp_thresh,
                    use_best_kp_match=use_best_kp_match,
                    max_r=max_r,
                )
            else:
                cls_metrics.update_one_frame(
                    pred_corners_list=pred_slot_corners_list,
                    pred_scores_list=pred_slot_scores_list,
                    gt_kps_list=gt_slot_kps_list,
                    gt_scores_list=gt_slot_scores_list,
                    iou_thresh=iou_thresh,
                    kp_thresh=kp_thresh,
                    use_best_kp_match=use_best_kp_match,
                    max_r=max_r,
                    pred_occtype_list=pred_slot_occ_list,
                    gt_occtype_list=gt_slot_occ_list,
                    pred_spetype_list=pred_slot_type_list,
                    gt_spetype_list=gt_slot_type_list,
                )

        except Exception as e:
            if not hasattr(cls_metrics, "_bad"):
                cls_metrics._bad = 0
            cls_metrics._bad += 1

            if not hasattr(cls_metrics, "_bad_files"):
                cls_metrics._bad_files = []
            cls_metrics._bad_files.append({
                "file": json_full_path,
                "error": str(e)
            })

    out = {
        "frames": cls_metrics.frames,
        "total_pred": cls_metrics.total_pred,
        "total_gt": cls_metrics.total_gt,
        "tp": cls_metrics.tp,
        "fp": cls_metrics.fp,
        "fn": cls_metrics.fn,
        "iou_list": getattr(cls_metrics, "iou_list", []),
        "kp_sum_list": getattr(cls_metrics, "kp_sum_list", []),
        "kp_mean_list": getattr(cls_metrics, "kp_mean_list", []),
        "kp_max_list": getattr(cls_metrics, "kp_max_list", []),

        # V5 classification metrics
        "occ_tp_total": getattr(cls_metrics, "occ_tp_total", 0),
        "occ_tp_correct": getattr(cls_metrics, "occ_tp_correct", 0),
        "spe_tp_total": getattr(cls_metrics, "spe_tp_total", 0),
        "spe_tp_correct": getattr(cls_metrics, "spe_tp_correct", 0),

        "bad": getattr(cls_metrics, "_bad", 0),
        "bad_files": getattr(cls_metrics, "_bad_files", []),
    }
    return out


def _merge_partials(partials):
    """
    合并多个 worker 的统计。
    """
    merged = {
        "frames": 0,
        "total_pred": 0,
        "total_gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "iou_list": [],
        "kp_sum_list": [],
        "kp_mean_list": [],
        "kp_max_list": [],

        # V5 classification metrics
        "occ_tp_total": 0,
        "occ_tp_correct": 0,
        "spe_tp_total": 0,
        "spe_tp_correct": 0,

        "bad": 0,
        "bad_files": [],
    }

    for p in partials:
        merged["frames"] += p["frames"]
        merged["total_pred"] += p["total_pred"]
        merged["total_gt"] += p["total_gt"]
        merged["tp"] += p["tp"]
        merged["fp"] += p["fp"]
        merged["fn"] += p["fn"]

        merged["iou_list"].extend(p.get("iou_list", []))
        merged["kp_sum_list"].extend(p.get("kp_sum_list", []))
        merged["kp_mean_list"].extend(p.get("kp_mean_list", []))
        merged["kp_max_list"].extend(p.get("kp_max_list", []))

        merged["occ_tp_total"] += p.get("occ_tp_total", 0)
        merged["occ_tp_correct"] += p.get("occ_tp_correct", 0)
        merged["spe_tp_total"] += p.get("spe_tp_total", 0)
        merged["spe_tp_correct"] += p.get("spe_tp_correct", 0)

        merged["bad"] += p.get("bad", 0)
        merged["bad_files"].extend(p.get("bad_files", []))

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--score_threshold", type=float, default=0.2)
    ap.add_argument("--nms_kernel_size", type=int, default=15)
    ap.add_argument("--iou_thresh", type=float, default=0.8)
    ap.add_argument("--kp_thresh", type=float, default=0.1)
    ap.add_argument("--use_best_kp_match", action="store_true")
    ap.add_argument("--max_r", type=float, default=5.0)

    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--gpus", type=str, default="")   # e.g. "0,1,2,3"

    args = ap.parse_args()

    json_list = sorted([f for f in os.listdir(args.json_dir) if f.endswith(".json")])

    gpu_ids = []
    if args.gpus.strip():
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    w = max(1, args.num_workers)
    chunks = [json_list[i::w] for i in range(w)]

    partials = []

    with ProcessPoolExecutor(max_workers=w) as ex:
        futures = []
        for wi, files in enumerate(chunks):
            if not files:
                continue
            gpu_id = None
            if gpu_ids:
                gpu_id = gpu_ids[wi % len(gpu_ids)]

            futures.append(
                ex.submit(
                    _worker_run,
                    files,
                    args.json_dir,
                    args.config,
                    args.checkpoint,
                    args.data_dir,
                    args.score_threshold,
                    args.nms_kernel_size,
                    args.iou_thresh,
                    args.kp_thresh,
                    args.use_best_kp_match,
                    args.max_r,
                    gpu_id
                )
            )

        for fu in as_completed(futures):
            partials.append(fu.result())

    merged = _merge_partials(partials)

    if TAG == 'V4':
        from metrics_ps_ws.metrics_v4 import MetricsV4
        m = MetricsV4()
    else:
        from metrics_ps_ws.metrics_v5 import MetricsV5
        m = MetricsV5()

    m.frames = merged["frames"]
    m.total_pred = merged["total_pred"]
    m.total_gt = merged["total_gt"]
    m.tp = merged["tp"]
    m.fp = merged["fp"]
    m.fn = merged["fn"]
    m.iou_list = merged["iou_list"]
    m.kp_sum_list = merged["kp_sum_list"]
    m.kp_mean_list = merged["kp_mean_list"]
    m.kp_max_list = merged["kp_max_list"]

    if hasattr(m, "occ_tp_total"):
        m.occ_tp_total = merged["occ_tp_total"]
    if hasattr(m, "occ_tp_correct"):
        m.occ_tp_correct = merged["occ_tp_correct"]
    if hasattr(m, "spe_tp_total"):
        m.spe_tp_total = merged["spe_tp_total"]
    if hasattr(m, "spe_tp_correct"):
        m.spe_tp_correct = merged["spe_tp_correct"]

    report = m.summarize()
    report["bad_frames"] = merged["bad"]
    report["bad_files"] = merged["bad_files"]
    report["tag"] = TAG

    print("\n==== Final Evaluation (MP) ====")
    for k, v in report.items():
        print(k, ":", v)

    out_dir = os.path.dirname(args.out)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\nMetrics saved to:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()