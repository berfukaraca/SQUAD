import os
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import cv2

from pathlib import Path

# =========================
#  FRAME / CAM HELPERS
# =========================
def get_frame_size(frames_path):
    d=os.listdir(frames_path)
    first_image = None
    for file in os.listdir(os.path.join(frames_path,d[0])):
        if file.endswith('.jpg'):
            first_image = os.path.join(frames_path, d[0], file)
            break
    if first_image is None:
        raise FileNotFoundError(f"No .jpg found in {frames_path}")
    img = cv2.imread(first_image)
    h, w = img.shape[:2]
    return w, h


def load_cam_params(filepath):
    camparams = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        view_id_line = lines[i].strip()
        if not view_id_line.isdigit():
            i += 1
            continue
        view_id = view_id_line.strip()

        K = []
        for j in range(1, 4):
            K.append([float(x) for x in lines[i + j].strip().split()])
        K = np.array(K)

        # skip "0 0" line
        i += 5

        extrinsic = []
        for j in range(3):
            extrinsic.append([float(x) for x in lines[i + j].strip().split()])
        extrinsic = np.array(extrinsic)

        R = extrinsic[:, :3]
        t = extrinsic[:, 3].reshape(3, 1)

        camparams[view_id] = (K, R, t)
        i += 3
    return camparams


def parse_dmmr_log(dmmr_log_path):
    return pd.read_csv(dmmr_log_path, sep="\t")


# =========================
#  2D DETECTION INTERPOLATION
# =========================
def interpolate_detections_for_view(view_json_path: str, limit: int = 30):
    """
    Read one view's detections.json and write detections_updated.json
    with linearly interpolated x, y, w, h, conf per obj_id over frames.
    """
    if not os.path.exists(view_json_path):
        print(f" No detections.json at {view_json_path}, skipping.")
        return

    with open(view_json_path, "r") as f:
        data = json.load(f)

    tracks = defaultdict(dict)

    for frame_key, objs in data.items():
        if not frame_key.endswith("_processed"):
            continue
        frame_id = int(frame_key.split("_")[0])

        for det in objs:
            obj_id, x, y, w, h, conf = det
            tracks[int(obj_id)][frame_id] = (
                float(x), float(y), float(w), float(h), float(conf)
            )

    if not tracks:
        print(f"No objects in {view_json_path}, skipping interpolation.")
        return

    updated = defaultdict(list)

    for obj_id, frame_dict in tracks.items():
        frames = sorted(frame_dict.keys())
        start_f, end_f = frames[0], frames[-1]
        full_range = list(range(start_f, end_f + 1))

        def build_axis(idx):
            vals = []
            for f in full_range:
                if f in frame_dict:
                    vals.append(frame_dict[f][idx])
                else:
                    vals.append(np.nan)

            s = pd.Series(vals, index=full_range, dtype=float)
            s_interp = s.interpolate(
                method="linear",
                limit=limit,
                limit_direction="both",
                limit_area="inside"
            )
            return s_interp

        sx = build_axis(0)
        sy = build_axis(1)
        sw = build_axis(2)
        sh = build_axis(3)
        sconf = build_axis(4)

        for f in full_range:
            x = sx.loc[f]
            y = sy.loc[f]
            w = sw.loc[f]
            h = sh.loc[f]
            conf = sconf.loc[f]

            if np.isnan(x) or np.isnan(y):
                continue

            frame_key = f"{f}_processed"
            updated[frame_key].append([
                int(obj_id),
                float(x),
                float(y),
                float(w) if not np.isnan(w) else 0.0,
                float(h) if not np.isnan(h) else 0.0,
                float(conf) if not np.isnan(conf) else 0.0,
            ])

    out_path = os.path.join(os.path.dirname(view_json_path), "detections_updated.json")
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"Saved interpolated 2D detections to {out_path}")


def interpolate_all_views_2d(participant_folder: str, limit: int = 30):
    for folder in os.listdir(participant_folder):
        view_folder = os.path.join(participant_folder, folder)
        if not os.path.isdir(view_folder):
            continue

        detections_path = os.path.join(view_folder, "detections.json")
        interpolate_detections_for_view(detections_path, limit=limit)


def load_all_view_detections(base_path, filename="detections_updated.json", conf_thresh=0.5):
    """
    returns:
        detections[frame_id][obj_id] = [(x, y, view_id, conf), ...]
    """
    detections = {}
    for folder in sorted(os.listdir(base_path)):
        view_folder = os.path.join(base_path, folder)
        if not os.path.isdir(view_folder):
            continue

        json_path = os.path.join(view_folder, filename)
        if not os.path.exists(json_path):
            continue

        folder_view_id = folder.split("_")[-1]
        if not folder_view_id.isdigit():
            continue

        with open(json_path, "r") as f:
            view_dets = json.load(f)

        for frame_view_key, objs in view_dets.items():
            if not frame_view_key.endswith("_processed"):
                continue

            frame_id = frame_view_key.split("_")[0]
            frame_dict = detections.setdefault(frame_id, {})

            for det in objs:
                obj_id, x, y, w, h, conf = det
                if conf < conf_thresh:
                    continue
                frame_dict.setdefault(int(obj_id), []).append(
                    (float(x), float(y), folder_view_id, float(conf))
                )
    return detections


# =========================
#  MULTI-VIEW TRIANGULATION
# =========================
def triangulate_from_multiple_views(points_norm, Ks, Rs, ts, image_width, image_height, confs=None):
    projection_matrices = []
    for K, R, t in zip(Ks, Rs, ts):
        t = t.reshape(3, 1)
        P = K @ np.hstack((R, t))
        projection_matrices.append(P)

    A_rows = []
    for p_norm, P in zip(points_norm, projection_matrices):
        x = p_norm[0] * image_width
        y = p_norm[1] * image_height

        A_rows.append(x * P[2, :] - P[0, :])
        A_rows.append(y * P[2, :] - P[1, :])

    A = np.asarray(A_rows, dtype=float)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]

    if np.isclose(X_h[3], 0):
        return None

    X_h = X_h / X_h[3]
    return X_h[:3]


def triangulate_objects_multi_view(detections, camparams, image_width, image_height, min_views=2):
    triangulated = {}

    for frame_id_str, obj_dict in detections.items():
        triangulated[frame_id_str] = {}

        for obj_id, dets in obj_dict.items():
            points_norm = []
            Ks, Rs, ts, confs = [], [], [], []

            for x, y, view_id, conf in dets:
                view_id_str = str(view_id)
                if view_id_str not in camparams:
                    continue

                K, R, t = camparams[view_id_str]
                norm_x = x / image_width
                norm_y = y / image_height

                points_norm.append(np.array([norm_x, norm_y], dtype=float))
                Ks.append(K)
                Rs.append(R)
                ts.append(t)
                confs.append(conf)

            if len(points_norm) < min_views:
                continue

            pt3d = triangulate_from_multiple_views(
                points_norm, Ks, Rs, ts, image_width, image_height, confs=confs
            )

            if pt3d is None:
                continue

            triangulated[frame_id_str][obj_id] = pt3d

    return triangulated


# =========================
#  3D SMOOTHING
# =========================
def smooth_with_gaussian_preserving_nans(series, sigma=3):
    array = series.to_numpy(dtype=np.float64)
    isnan = np.isnan(array)
    filled = np.where(isnan, 0, array)
    valid_mask = (~isnan).astype(float)
    smoothed_values = gaussian_filter1d(filled, sigma=sigma)
    smoothed_weights = gaussian_filter1d(valid_mask, sigma=sigma)
    result = smoothed_values / np.maximum(smoothed_weights, 1e-5)
    result[isnan] = np.nan
    return pd.Series(result, index=series.index)


def smooth_3d_objects(objects_3d, sigma=3.0):
    tracks = defaultdict(dict)
    for frame_id_str, objs in objects_3d.items():
        frame_id = int(frame_id_str)
        for obj_id, pt in objs.items():
            tracks[obj_id][frame_id] = pt

    smoothed_objects = defaultdict(dict)

    for obj_id, frames_dict in tracks.items():
        if not frames_dict:
            continue

        frames_sorted = sorted(frames_dict.keys())
        xs = [frames_dict[f][0] for f in frames_sorted]
        ys = [frames_dict[f][1] for f in frames_sorted]
        zs = [frames_dict[f][2] for f in frames_sorted]

        sx = smooth_with_gaussian_preserving_nans(pd.Series(xs, index=frames_sorted), sigma=sigma)
        sy = smooth_with_gaussian_preserving_nans(pd.Series(ys, index=frames_sorted), sigma=sigma)
        sz = smooth_with_gaussian_preserving_nans(pd.Series(zs, index=frames_sorted), sigma=sigma)

        for f in frames_sorted:
            x, y, z = sx.loc[f], sy.loc[f], sz.loc[f]
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            smoothed_objects[str(f)][obj_id] = [float(x), float(y), float(z)]

    return smoothed_objects


def save_smoothed_detections_per_view(base_path, smoothed_objects):
    for folder in os.listdir(base_path):
        view_folder = os.path.join(base_path, folder)
        if not os.path.isdir(view_folder):
            continue

        out_path = os.path.join(view_folder, "detections_smoothed.json")
        detections_json = {}

        for frame_id in sorted(smoothed_objects.keys(), key=lambda x: int(x)):
            frame_objects = smoothed_objects[frame_id]
            detections_json[frame_id] = []
            for obj_id, pt3d in frame_objects.items():
                detections_json[frame_id].append([
                    int(obj_id),
                    float(pt3d[0]),
                    float(pt3d[1]),
                    float(pt3d[2])
                ])

        with open(out_path, "w") as f:
            json.dump(detections_json, f, indent=2)

        print(f"Saved smoothed 3D detections to {out_path}")

def load_joints_for_batch_all_joints(joints_file_path, joint_type):

    """
    Loads joints.pkl according to the joint file type:

        P_1: child only  (person00 = child)
        P_2: parent only (person00 = parent)
        P_3: both         (person00 = parent, person01 = child)

        Missing roles are filled with NaNs.
    Returns:
      parent[frame_id] -> np.ndarray (J,3)
      child[frame_id]  -> np.ndarray (J,3)
    """
    parent_frame_joint_map = {}
    child_frame_joint_map = {}

    if not os.path.exists(joints_file_path):
        return parent_frame_joint_map, child_frame_joint_map

    with open(joints_file_path, "rb") as file:
        joints_data = pickle.load(file)

    frames = defaultdict(list)
    for entry in joints_data:
        frame_id, person_id, joint_tensor = entry[0], entry[1], entry[2]
        frames[frame_id].append((person_id, joint_tensor))

    for frame_id, items in frames.items():
        # determine shape from whichever is available
        shape = None
        for _, jt in items:
            arr = jt.detach().numpy()
            shape = arr.shape
            break
        if shape is None:
            continue

        parent_arr = np.full(shape, np.nan, dtype=float)
        child_arr  = np.full(shape, np.nan, dtype=float)
        for person_id, jt in items:
            arr = jt.detach().numpy()

            if joint_type == "P_1":
                # Child only
                if person_id == "person00":
                    child_arr = arr

            elif joint_type == "P_2":
                # Parent only
                if person_id == "person00":
                    parent_arr = arr

            elif joint_type == "P_3":
                # Parent + child
                if person_id == "person00":
                    parent_arr = arr
                elif person_id == "person01":
                    child_arr = arr

        parent_frame_joint_map[frame_id] = parent_arr
        child_frame_joint_map[frame_id]  = child_arr

    return parent_frame_joint_map, child_frame_joint_map


# =========================
#  OBJECT CLASS NAMES
# =========================
YOUTH_CLASSES = [
    'doll', 'car', 'jumpbox', 'shapebox', 'flower', 'book', 'bottle',
    'green_star', 'yellow_cylinder', 'blue_cube', 'red_triangle',
    'shapebox_lid', 'head_parent', 'head_child'
]

if __name__ == "__main__":
    
    # basedir = "/path/to/datafolders"
    # Suggested structure:
    #
    # processed_data/
    # ├── participant_1/
    # │   └── data/
    # │       ├── alphapose_results/
    # │       ├── assigned_ids/
    # │       ├── camera_movement/
    # │       ├── detection/
    # │       ├── dmmr_output/
    # │       ├── extracted_frames/
    # │       ├── frame_offsets/
    # │       ├── input_videos/
    # │       ├── joints/
    # │       └── pose_tracks/
    # └── participant_2/
    #     └── data/
    #         └── ...

    # participants = [
    #     p for p in os.listdir(basedir)
    #     if os.path.isdir(os.path.join(basedir, p, "data"))
    # ]

    #To continue with the SQUAD folder:
    basedir = Path(__file__).resolve().parent

    participants = [
        p for p in os.listdir(os.path.join(basedir, "data", "dmmr_output"))
    ]    

    for participant in participants:
        print(f"Full-video build for: {participant}")

        #if you follow the suggested structure:
        # participant_dir = os.path.join(basedir, participant)
        # data_dir = os.path.join(participant_dir, "data")

        # frames_path = os.path.join(data_dir, "extracted_frames")
        # joints_base_dir = os.path.join(data_dir, "joints")
        # detections_base_path = os.path.join(data_dir, "detection")
        # dmmr_output_base = os.path.join(data_dir, "dmmr_output")


        data_dir = os.path.join(basedir, "data")

        frames_path = os.path.join(data_dir, "extracted_frames",participant)
        joints_base_dir = os.path.join(data_dir, "joints",participant)
        detections_base_path = os.path.join(data_dir, "detection",participant)
        dmmr_output_base = os.path.join(data_dir, "dmmr_output",participant)

        image_width, image_height = get_frame_size(frames_path)

        dmmr_log_file = os.path.join(
            dmmr_output_base,
            # participant,#suggested structure
            f"{participant}_dmmr_log.txt"
        )

        interpolate_all_views_2d(detections_base_path, limit=30)

        detections = load_all_view_detections(
            detections_base_path,
            filename="detections_updated.json",
            conf_thresh=0.5
        )

        dmmr_log_df = parse_dmmr_log(dmmr_log_file)

        # =========================
        # JOINTS
        # =========================
        participant_joints_p = {}
        participant_joints_c = {}

        for fname in sorted(os.listdir(joints_base_dir)):

            if "_joints_1" in fname:
                joint_type = "P_1"
            elif "_joints_2" in fname:
                joint_type = "P_2"
            elif "_joints_3" in fname:
                joint_type = "P_3"
            else:
                continue

            joints_file_path = os.path.join(joints_base_dir, fname)

            pj, cj = load_joints_for_batch_all_joints(
                joints_file_path,
                joint_type=joint_type
            )

            for frame, arr in pj.items():
                if frame not in participant_joints_p:
                    participant_joints_p[frame] = arr
                else:
                    # merge without overwriting valid data
                    existing = participant_joints_p[frame]
                    merged = np.where(np.isnan(existing), arr, existing)
                    participant_joints_p[frame] = merged

            for frame, arr in cj.items():
                if frame not in participant_joints_c:
                    participant_joints_c[frame] = arr
                else:
                    existing = participant_joints_c[frame]
                    merged = np.where(np.isnan(existing), arr, existing)
                    participant_joints_c[frame] = merged

        participant_joints_p = {
            k: participant_joints_p[k]
            for k in sorted(
                participant_joints_p,
                key=lambda s: int(str(s).split(".")[0]) if isinstance(s, str) else int(s)
            )
        }
        participant_joints_c = {
            k: participant_joints_c[k]
            for k in sorted(
                participant_joints_c,
                key=lambda s: int(str(s).split(".")[0]) if isinstance(s, str) else int(s)
            )
        }
        print(f"Merged joints: parent={len(participant_joints_p)} frames, child={len(participant_joints_c)} frames")

        # =========================
        # OBJECT COORDINATES 
        # =========================
        all_triangulated_objects = {}

        for _, row in dmmr_log_df.iterrows():
            batch = row['seq_fld']
            opt_cam = row['opt_cam']
            camparams_fld = row['camparams_fld']

            start_frame = int(batch.split("_")[4])
            num_frames = int(batch.split("_")[-1])
            expected_end_frame = start_frame + num_frames - 1

            if opt_cam:
                camparams_path = os.path.join(
                    dmmr_output_base,
                    # participant, #suggested structure
                    batch,
                    "camparams",
                    participant
                )
            else:
                camparams_path = os.path.join(
                    dmmr_output_base,
                    # participant, #suggested structure
                    camparams_fld,
                    "camparams",
                    participant
                )
            camparam_files = [
                os.path.join(camparams_path, f)
                for f in os.listdir(camparams_path)
                if f.endswith('.txt')
            ]

            camparams = {}
            for camparam_file in camparam_files:
                camparams.update(load_cam_params(camparam_file))

            batch_frame_ids = [str(fid) for fid in range(start_frame, expected_end_frame + 1)]
            batch_detections = {fid: detections[fid] for fid in batch_frame_ids if fid in detections}

            triangulated_objects = triangulate_objects_multi_view(
                batch_detections,
                camparams,
                image_width,
                image_height,
                min_views=2
            )

            all_triangulated_objects.update(triangulated_objects)

        all_smoothed_objects = smooth_3d_objects(all_triangulated_objects, sigma=3.0)
        save_smoothed_detections_per_view(detections_base_path, all_smoothed_objects)

        final_data = {}

        if not all_smoothed_objects:
            print(f"No smoothed detections for {participant}. Skipping JSON.")
            continue

        all_frame_ids_sorted = sorted([int(fid) for fid in all_smoothed_objects.keys()])

        for frame_id_int in all_frame_ids_sorted:
            fid_str = str(frame_id_int)
            frame_detections = all_smoothed_objects[fid_str]

            cand_keys = [
                f"{fid_str}.jpg",
                f"{frame_id_int:02d}.jpg",
                f"{frame_id_int:03d}.jpg",
                f"{frame_id_int:04d}.jpg",
                f"{frame_id_int:05d}.jpg",
                f"{frame_id_int:06d}.jpg",
                f"{frame_id_int:07d}.jpg",
                frame_id_int,
            ]

            jp = None
            for ck in cand_keys:
                if ck in participant_joints_p:
                    jp = participant_joints_p[ck]
                    break

            jc = None
            for ck in cand_keys:
                if ck in participant_joints_c:
                    jc = participant_joints_c[ck]
                    break

            parent_joints_raw = jp.tolist() if isinstance(jp, np.ndarray) else []
            child_joints_raw  = jc.tolist() if isinstance(jc, np.ndarray) else []

            objects_list = []
            for obj_id, coords in frame_detections.items():
                obj_name = YOUTH_CLASSES[int(obj_id)]

                objects_list.append({
                    "obj_id": int(obj_id),
                    "object_name": obj_name,
                    "coords": [float(coords[0]), float(coords[1]), float(coords[2])]
                })

            final_data[fid_str] = {
                "objects": objects_list,
                "parent_joints": parent_joints_raw,
                "child_joints": child_joints_raw
            }

        output_dir = os.path.join(data_dir, "fullvideo")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"{participant}_fullvideo_alljoints.json"
        )
        with open(output_file, "w") as f:
            json.dump(final_data, f, indent=2)
        print(f"Saved FULL-VIDEO data for {participant} to {output_file}")