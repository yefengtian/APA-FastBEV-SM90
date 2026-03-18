#!/usr/bin/env python3
import argparse
import glob
import json
import os
import pickle
from typing import Dict, List, Optional

def _find_image_relpath(scene_dir: str, data_root: str, timestamp: str, cam_name: str) -> Optional[str]:
    cam_dir = os.path.join(scene_dir, "img", timestamp, cam_name)
    files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
    if not files:
        files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    if not files:
        return None
    return os.path.relpath(files[0], data_root)


def _extract_camera_calib(cam_record: Dict) -> Dict:
    calib = cam_record.get("calibration_param", {})
    model_name = cam_record.get("camera_model", "fisheye")
    model_key = "fisheye" if "fish" in model_name.lower() or model_name.lower() == "mei" else "pinhole"
    model_calib = calib.get(model_key, None)
    if not model_calib or not model_calib.get("valid", False):
        model_calib = calib.get("fisheye", None) or calib.get("pinhole", None) or {}

    intrin = model_calib.get("intrinsic_K", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    extrinsic = model_calib.get(
        "extrinsic_vcs2ccs",
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )
    distort = model_calib.get("intrinsic_distort", [0, 0, 0, 0])
    if len(distort) >= 9:
        # Many records use [1, 0, k1, 0, k2, 0, k3, 0, k4, 0]
        dist4 = [float(distort[2]), float(distort[4]), float(distort[6]), float(distort[8])]
    elif len(distort) >= 4:
        dist4 = [float(distort[0]), float(distort[1]), float(distort[2]), float(distort[3])]
    else:
        dist4 = [0.0, 0.0, 0.0, 0.0]

    return {
        "cam_intrinsic": intrin,
        "extrinsic": extrinsic,
        "dist": dist4,
        "camera_model": model_name,
    }


def convert_dataset(data_root: str, out_path: str, cam_names: List[str], category_count: int) -> None:
    clip_dirs = sorted([p for p in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(p)])
    infos = []

    for clip_dir in clip_dirs:
        scene_jsons = sorted(glob.glob(os.path.join(clip_dir, "scene*.json")))
        for scene_json in scene_jsons:
            with open(scene_json, "r", encoding="utf-8") as f:
                frames = json.load(f)
            for frame in frames:
                timestamp = str(frame["timestamp"])
                cam_info = frame.get("cam_info", {})
                gt_info = frame.get("gt_info", {})
                gt_od = gt_info.get("gt_od", [])

                cams = {}
                missing_cam = False
                for cam_name in cam_names:
                    cam_record = cam_info.get(cam_name, None)
                    if cam_record is None:
                        missing_cam = True
                        break
                    rel_img_path = _find_image_relpath(clip_dir, data_root, timestamp, cam_name)
                    if rel_img_path is None:
                        missing_cam = True
                        break
                    calib_dict = _extract_camera_calib(cam_record)
                    cams[cam_name] = {
                        "data_path": rel_img_path,
                        "cam_intrinsic": calib_dict["cam_intrinsic"],
                        "extrinsic": calib_dict["extrinsic"],
                        "dist": calib_dict["dist"],
                        "camera_model": calib_dict["camera_model"],
                    }
                if missing_cam:
                    continue

                gt_boxes = []
                gt_names = []
                for obj in gt_od:
                    if obj.get("ignore", False):
                        continue
                    center = obj.get("vcs_3d_ct", None)
                    dim = obj.get("dim", None)
                    yaw = obj.get("vcs_3d_yaw", None)
                    category = obj.get("category", None)
                    if center is None or dim is None or yaw is None or category is None:
                        continue
                    if not (0 <= int(category) < category_count):
                        continue
                    x, y, z = [float(v) for v in center]
                    l, w, h = [float(v) for v in dim]
                    gt_boxes.append([x, y, z, w, l, h, float(yaw)])
                    gt_names.append(f"CAT_{int(category)}")

                info = {
                    "timestamp": int(timestamp),
                    "token": frame.get("token", ""),
                    "center2lidar": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "cams": cams,
                    "gt_boxes": gt_boxes,
                    "gt_names": gt_names,
                }
                infos.append(info)

    infos = sorted(infos, key=lambda x: x["timestamp"])
    with open(out_path, "wb") as f:
        pickle.dump({"infos": infos}, f)
    print(f"[od_scene_converter] saved {len(infos)} frames to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert OD scene json to InternalDataset infos pkl.")
    parser.add_argument("--data-root", required=True, help="Root dir containing clip folders.")
    parser.add_argument("--out-path", default=None, help="Output pkl path.")
    parser.add_argument(
        "--cams",
        default="CAM_FISHEYE_FORWARD,CAM_FISHEYE_LEFT,CAM_FISHEYE_BACKWARD,CAM_FISHEYE_RIGHT",
        help="Comma separated camera names.")
    parser.add_argument("--category-count", type=int, default=10, help="Category id upper bound (exclusive).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path = args.out_path or os.path.join(args.data_root, "od_fisheye_infos.pkl")
    cam_names = [c.strip() for c in args.cams.split(",") if c.strip()]
    convert_dataset(
        data_root=args.data_root,
        out_path=out_path,
        cam_names=cam_names,
        category_count=args.category_count,
    )
