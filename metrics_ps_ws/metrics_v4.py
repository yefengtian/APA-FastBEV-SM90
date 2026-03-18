import numpy as np
import cv2

EPS = 1e-9

class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = 0
        self.total_pred = 0
        self.total_gt = 0

        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.iou_list = []
        self.kp_sum_list = []
        self.kp_mean_list = []
        self.kp_max_list = []

    # ---------- geometry ----------
    def polygon_iou(self, A, B):
        A = np.asarray(A, np.float32)
        B = np.asarray(B, np.float32)
        areaA = abs(cv2.contourArea(A))
        areaB = abs(cv2.contourArea(B))
        if areaA < EPS or areaB < EPS:
            return 0.0
        inter, _ = cv2.intersectConvexConvex(A, B)
        union = areaA + areaB - inter
        return float(inter / (union + EPS))

    def kp_error(self, P, G, use_best_match):
        P = np.asarray(P, np.float32)
        G = np.asarray(G, np.float32)

        if not use_best_match:
            d = np.linalg.norm(P - G, axis=1)
            return d.sum(), d.mean(), d.max(), d

        best = None
        for base in (G, G[::-1].copy()):
            for s in range(4):
                Ga = np.roll(base, s, axis=0)
                d = np.linalg.norm(P - Ga, axis=1)
                ssum = d.sum()
                if best is None or ssum < best[0]:
                    best = (ssum, d)
        ssum, d = best
        return float(ssum), float(d.mean()), float(d.max()), d



    def pass_range_by_kp14_origin(self, kps_4x2, max_r=5.0):
        """
        kps_4x2: (4,2)
        r = max(||p1||, ||p4||)  (p1=kps[0], p4=kps[3])
        """
        k = np.asarray(kps_4x2, np.float32)
        r1 = float(np.linalg.norm(k[0]))
        r4 = float(np.linalg.norm(k[3]))
        return max(r1, r4) < max_r

    # ---------- per frame ----------
    def update_one_frame(
        self,
        pred_corners_list,
        pred_scores_list,
        gt_kps_list,
        gt_scores_list=None,
        iou_thresh=0.8,
        kp_thresh=None,
        use_best_kp_match=False,
        max_r=None, 
    ):
        if max_r is not None:
            # filter preds (keep indices for scores)
            keep_p = [i for i, p in enumerate(pred_corners_list)
                    if self.pass_range_by_kp14_origin(p, max_r=max_r)]
            pred_corners_list = [pred_corners_list[i] for i in keep_p]
            pred_scores_list  = [pred_scores_list[i]  for i in keep_p]

            # filter gts (keep indices for scores if provided)
            keep_g = [i for i, g in enumerate(gt_kps_list)
                    if self.pass_range_by_kp14_origin(g, max_r=max_r)]
            gt_kps_list = [gt_kps_list[i] for i in keep_g]
            if gt_scores_list is not None:
                gt_scores_list = [gt_scores_list[i] for i in keep_g]
            
            
        self.frames += 1
        self.total_pred += len(pred_corners_list)
        self.total_gt += len(gt_kps_list)

        pairs = []

        # 1️⃣ 枚举 IoU 合格的 pair
        for pi, p in enumerate(pred_corners_list):
            for gi, g in enumerate(gt_kps_list):
                iou = self.polygon_iou(p, g)
                if iou < iou_thresh:
                    continue

                kp_sum, kp_mean, kp_max, d = self.kp_error(p, g, use_best_kp_match)

                if kp_thresh is not None:
                    if kp_max > kp_thresh:
                        continue

                pairs.append((pi, gi, iou, kp_sum, kp_mean, kp_max))

        # 2️⃣ IoU 贪心匹配
        pairs.sort(key=lambda x: -x[2])

        matched_p = set()
        matched_g = set()
        matches = []

        for pi, gi, iou, kp_sum, kp_mean, kp_max in pairs:
            if pi in matched_p or gi in matched_g:
                continue
            matched_p.add(pi)
            matched_g.add(gi)
            matches.append((pi, gi, iou, kp_sum, kp_mean, kp_max))

        pred_unmatched = set(range(len(pred_corners_list))) - matched_p
        gt_unmatched = set(range(len(gt_kps_list))) - matched_g

        # 3️⃣ 累加统计
        self.tp += len(matches)
        self.fp += len(pred_unmatched)
        self.fn += len(gt_unmatched)

        for _, _, iou, kp_sum, kp_mean, kp_max in matches:
            self.iou_list.append(iou)
            self.kp_sum_list.append(kp_sum)
            self.kp_mean_list.append(kp_mean)
            self.kp_max_list.append(kp_max)

        return matches, pred_unmatched, gt_unmatched

    # ---------- final report ----------
    def summarize(self):
        precision = self.tp / (self.tp + self.fp + EPS)
        recall = self.tp / (self.tp + self.fn + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)

        def stats(arr):
            if len(arr) == 0:
                return None
            a = np.array(arr)
            return dict(
                mean=float(a.mean()),
                p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)),
                max=float(a.max())
            )

        return {
            "frames": self.frames,
            "total_pred": self.total_pred,
            "total_gt": self.total_gt,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "IoU": stats(self.iou_list),
            "kp_sum": stats(self.kp_sum_list),
            "kp_mean": stats(self.kp_mean_list),
            "kp_max": stats(self.kp_max_list),
        }
