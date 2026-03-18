import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------- worker side imports (inside process) --------
def _worker_run(json_files, json_dir, config, checkpoint, data_dir,
                score_threshold, nms_kernel_size,
                iou_thresh, kp_thresh, use_best_kp_match, max_r,
                gpu_id=None):
    """
    每个进程处理一批 json 文件，返回可 merge 的中间结果。
    """
    # 进程内设置 GPU（如果传了）
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from tqdm import tqdm
    from mmcv import Config
    from metrics_ps_ws.infer import Infer, Data
    from metrics_ps_ws.metrics_v4 import Metrics

    cfg = Config.fromfile(config)
    cfg.model.psd_head.bbox_coder.score_threshold = score_threshold
    cfg.model.psd_head.bbox_coder.nms_kernel_size = nms_kernel_size

    cls_infer = Infer(cfg, checkpoint)
    cls_data = Data(cfg, data_dir)
    cls_metrics = Metrics()

    for jf in tqdm(json_files, desc=f"worker gpu={gpu_id}", leave=False):
        json_full_path = os.path.join(json_dir, jf)
        try:
            data = cls_data.get_data(json_full_path)
            preds = cls_infer.infer(data)
            (pred_slot_K, pred_slot_corners_list, pred_slot_occ_list, pred_slot_type_list, pred_slot_scores_list,
            pred_ws_K, pred_ws_kps2_list, pred_ws_scores_list) = cls_infer.post_process(preds)


            (gt_slot_kps_list, gt_slot_occ_list, gt_slot_type_list, gt_slot_scores_list,
            gt_ws_kps2_list, gt_ws_scores_list) = cls_data.get_gt_slot_ws(json_full_path)

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
        except Exception as e:
            # 出错就跳过该帧（你也可以选择 raise）
            # 这里返回错误计数，方便你排查
            if not hasattr(cls_metrics, "_bad"):
                cls_metrics._bad = 0
            cls_metrics._bad += 1

    # 返回可合并数据（不要直接返回 cls_metrics 对象，可能不可 pickle）
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
        "bad": getattr(cls_metrics, "_bad", 0),
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
        "bad": 0,
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
        merged["bad"] += p.get("bad", 0)
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
    ap.add_argument("--gpus", type=str, default="")  # e.g. "0,1,2,3"
    args = ap.parse_args()

    json_list = sorted([f for f in os.listdir(args.json_dir) if f.endswith(".json")])

    # 解析 GPU 列表
    gpu_ids = []
    if args.gpus.strip():
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    # 分 chunk
    n = len(json_list)
    w = max(1, args.num_workers)
    chunks = []
    for i in range(w):
        chunks.append(json_list[i::w])

    partials = []

    # 多进程执行
    with ProcessPoolExecutor(max_workers=w) as ex:
        futures = []
        for wi, files in enumerate(chunks):
            if not files:
                continue
            gpu_id = None
            if gpu_ids:
                gpu_id = gpu_ids[wi % len(gpu_ids)]
            futures.append(ex.submit(
                _worker_run,
                files, args.json_dir, args.config, args.checkpoint, args.data_dir,
                args.score_threshold, args.nms_kernel_size,
                args.iou_thresh, args.kp_thresh, args.use_best_kp_match, args.max_r,
                gpu_id
            ))

        for fu in as_completed(futures):
            partials.append(fu.result())

    merged = _merge_partials(partials)

    # 用 Metrics 自己的 summarize 规则输出（复用你已有逻辑）
    from metrics_ps_ws.metrics_v4 import Metrics
    m = Metrics()
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

    report = m.summarize()
    report["bad_frames"] = merged["bad"]

    print("\n==== Final Evaluation (MP) ====")
    for k, v in report.items():
        print(k, ":", v)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print("\nMetrics saved to:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
