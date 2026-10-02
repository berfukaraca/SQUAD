import os
import json
import numpy as np
import re

from pathlib import Path

""" Suggested structure
processed_data/
├── annotations/
│   ├── attention/
│   │   ├── participant_1.json
│   │   ├── participant_2.json
│   │   └── ...
│   │
│   ├── touch/
│   │   ├── participant_1.json
│   │   ├── participant_2.json
│   │   └── ...
│   │
│   └── frames/
│       ├── participant_1/
│       │   ├── ...
│       │   └── ...
│       └── participant_2/
│           └── ...
│
├── participant_1/
│   └── data/
│       ├── ...
│       ├── fullvideo/
│       │   └── participant_1_fullvideo_alljoints.json
│       └── inputs/
│           └── participant_1_annotated_inputs.json
│
└── participant_2/
    └── data/
        └── ...
"""

# Script is provided to run one processed participant in SQUAD Path
# BASEDIR = "/path/to/processed_data"

# ANNOTATIONS_DIR = os.path.join(BASEDIR, "annotations")
# ATTN_DIR = os.path.join(ANNOTATIONS_DIR, "attention")
# TOUCH_DIR = os.path.join(ANNOTATIONS_DIR, "touch")
# ANNOT_FRAMES_DIR = os.path.join(ANNOTATIONS_DIR, "frames")

# To run in the SQUAD folder:
BASEDIR = Path(__file__).resolve().parent

DATA_DIR = os.path.join(BASEDIR, "data")

ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations") # Add annotations to the data folder
ANNOT_FRAMES_DIR = os.path.join(ANNOTATIONS_DIR, "frames")

# --- joint index convention AlphaPose ---
NOSE, LEYE, REYE, LWR, RWR, LSH, RSH, LHIP, RHIP, HEAD, NECK = 0, 1, 2, 9, 10, 5, 6, 11, 12, 17, 18
SELECTED_9 = [LEYE, REYE, LSH, RSH, LHIP, RHIP, HEAD, NECK, NOSE]
SELECTED_11 = [LEYE, REYE, LSH, RSH, LHIP, RHIP, HEAD, NECK, NOSE, LWR, RWR]

def to_np(a):
    return np.asarray(a, dtype=np.float32)


# =========================
# PITCH HELPERS
# =========================
def _build_head_axes_from_selected_subset(sel9):
    LEye, REye, Head, Neck = sel9[0], sel9[1], sel9[6], sel9[7]

    eye_center = (LEye + REye) / 2.0
    head_center = (Head + Neck) / 2.0

    x_axis = LEye - REye
    x_axis = x_axis / np.linalg.norm(x_axis)

    v = eye_center - head_center
    v_proj = v - np.dot(v, x_axis) * x_axis 
    z_axis = v_proj / np.linalg.norm(v_proj)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R, eye_center

def angle_gaze_with_worldY_from_selected(sel11):
    R, _ = _build_head_axes_from_selected_subset(sel11)
    gaze = R[:, 2] / np.linalg.norm(R[:, 2])
    world_y = np.array([0., 1., 0.], dtype=np.float32)

    dot = np.dot(gaze, world_y)
    dot = max(min(dot, 1.0), -1.0)

    return float(np.degrees(np.arccos(dot)))


# =========================
# BODY NORMALIZATION
# =========================
def get_body_rotation_matrix_hip_fixedY_projected(joints, left_hip_idx=11, right_hip_idx=12):
    lhip = joints[left_hip_idx]
    rhip = joints[right_hip_idx]
    hip_center = (lhip + rhip) / 2.0

    x_axis = lhip - rhip
    x_axis = x_axis / np.linalg.norm(x_axis)

    target = hip_center - np.array([0, 1, 0], dtype=np.float32)
    y_vec = target - hip_center
    y_proj = y_vec - np.dot(y_vec, x_axis) * x_axis
    y_axis = y_proj / np.linalg.norm(y_proj)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R, hip_center


def normalize_pose_hip_fixedY_projected(joints, left_hip_idx=11, right_hip_idx=12):
    R, hip = get_body_rotation_matrix_hip_fixedY_projected(joints, left_hip_idx, right_hip_idx)
    centered = joints - hip
    return centered @ R


def extract_normalized_subset(norm_full):
    return {
        "nose": norm_full[NOSE].tolist(),
        "LEye": norm_full[LEYE].tolist(),
        "REye": norm_full[REYE].tolist(),
        "LShoulder": norm_full[LSH].tolist(),
        "RShoulder": norm_full[RSH].tolist(),
    }

def has_valid_selected_joints(j):
    return (
        isinstance(j, np.ndarray)
        and j.ndim == 2
        and j.shape[1] == 3
        and j.shape[0] > max(SELECTED_11)
        and np.isfinite(j[SELECTED_11]).all()
    )

def infer_fps_from_darklabel_json(json_path, max_checks=10): #25 or 30fps recordings

    with open(json_path, "r") as f:
        data = json.load(f)

    frame_ids = sorted(int(k) for k in data.keys())
    if len(frame_ids) < 2:
        print(f"Not enough frame ids in {json_path} to infer fps.")
        return None

    diffs = []
    limit = min(len(frame_ids) - 1, max_checks)

    for i in range(limit):
        d = frame_ids[i + 1] - frame_ids[i]
        if d > 0:
            diffs.append(d)

    if not diffs:
        print(f" Could not compute positive frame differences in {json_path}")
        return None

    min_diff = min(diffs)
    fps = min_diff // 10
    return fps


def load_dl_json_as_global_object_ids(json_path, id_key, offset_sec=10):

    fps = infer_fps_from_darklabel_json(json_path)
    if fps is None:
        return {}, None

    with open(json_path, "r") as f:
        data = json.load(f)

    offset_frames = fps * offset_sec
    frame_to_object_ids = {}

    for fr_id, fr_data in data.items():
        local_fid = int(fr_id)
        global_fid = local_fid + offset_frames

        parent_objs = set()
        child_objs = set()

        for obj in fr_data.get("objects", []):
            obj_name = obj.get("object_name")
            tag_id = int(obj.get(id_key, 0))

            if obj_name is None:
                continue

            if tag_id in (1, 3):
                parent_objs.add(obj_name)
            if tag_id in (2, 3):
                child_objs.add(obj_name)

        frame_to_object_ids[global_fid] = {
            "parent": parent_objs,
            "child": child_objs,
        }

    return frame_to_object_ids, fps

def load_annotated_frame_ids_from_images(folder, fps, offset_sec=10):
    offset_frames = fps * offset_sec
    frame_ids = []

    for fname in os.listdir(folder):
        low = fname.lower()
        if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png")):
            continue

        stem = os.path.splitext(fname)[0]
        nums = re.findall(r"\d+", stem)
        if not nums:
            continue

        local_fid = int(nums[-1])
        global_fid = local_fid + offset_frames
        frame_ids.append(global_fid)

    return sorted(set(frame_ids))


def attach_attention_and_touch_ids_to_objects(objects, frame_attention, frame_touch):
    attn_parent = frame_attention.get("parent", set()) if frame_attention else set()
    attn_child = frame_attention.get("child", set()) if frame_attention else set()

    touch_parent = frame_touch.get("parent", set()) if frame_touch else set()
    touch_child = frame_touch.get("child", set()) if frame_touch else set()

    new_objects = []

    for o in objects:
        obj = dict(o)
        obj_name = obj.get("object_name")

        in_attn_parent = obj_name in attn_parent
        in_attn_child = obj_name in attn_child

        if in_attn_parent and in_attn_child:
            obj["attention_id"] = 3
        elif in_attn_parent:
            obj["attention_id"] = 1
        elif in_attn_child:
            obj["attention_id"] = 2
        else:
            obj["attention_id"] = 0

        in_touch_parent = obj_name in touch_parent
        in_touch_child = obj_name in touch_child

        if in_touch_parent and in_touch_child:
            obj["touch_id"] = 3
        elif in_touch_parent:
            obj["touch_id"] = 1
        elif in_touch_child:
            obj["touch_id"] = 2
        else:
            obj["touch_id"] = 0

        new_objects.append(obj)

    return new_objects

# =========================
# ATTENDED CENTERS
# =========================
def compute_attended_centers_from_attention(objects, frame_attention):
    def ok3(v):
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float32).reshape(3)
        if not np.isfinite(a).all():
            return None
        return a

    parent_names = frame_attention.get("parent", set()) if frame_attention else set()
    child_names = frame_attention.get("child", set()) if frame_attention else set()

    p_pts, c_pts = [], []

    for o in objects:
        obj_name = o.get("object_name")
        coords = ok3(o.get("coords"))
        if obj_name is None or coords is None:
            continue

        if obj_name in parent_names:
            p_pts.append(coords)
        if obj_name in child_names:
            c_pts.append(coords)

    def mean_or_none(lst):
        return np.mean(np.stack(lst), axis=0).tolist() if lst else None

    return mean_or_none(p_pts), mean_or_none(c_pts)

def build_fullvideo_inputs_for_participant(participant):

    pid = re.sub(r'_\d{2}_\d{2}_\d{4}$', '', participant)

    joints_path = os.path.join(
        DATA_DIR,
        "fullvideo",
        f"{participant}_fullvideo_alljoints.json"
    )

    attn_path = os.path.join(
        ANNOTATIONS_DIR,
        f"{participant}_attention.json"
    )

    touch_path = os.path.join(
        ANNOTATIONS_DIR,
        f"{participant}_touch.json"
    )

    annot_frames_dir = os.path.join(
        ANNOT_FRAMES_DIR,
        f"{participant}",
        
    )
    if not os.path.exists(joints_path):
        print(f"Missing full-video joints JSON: {joints_path}")
        return 0, 0

    if not os.path.exists(attn_path):
        print(f"Missing attention JSON: {attn_path}")
        return 0, 0

    if not os.path.exists(touch_path):
        print(f"Missing touch JSON: {touch_path}")
        return 0, 0
    
    if not os.path.isdir(annot_frames_dir):
        print(f"Missing annotation frames directory: {annot_frames_dir}")
        return 0, 0
 
    with open(joints_path, "r") as f:
        full = json.load(f)

    frame_attention, fps_attn = load_dl_json_as_global_object_ids(
        attn_path, id_key="attention_id", offset_sec=10
    )
    frame_touch, fps_touch = load_dl_json_as_global_object_ids(
        touch_path, id_key="touch_id", offset_sec=10
    )

    if fps_attn is None and fps_touch is None:
        print(f"Could not infer fps for {participant}")
        return 0, 0

    fps = fps_attn if fps_attn is not None else fps_touch

    # include ALL annotated frames from image folder
    used_global_frames = load_annotated_frame_ids_from_images(
        annot_frames_dir,
        fps=fps,
        offset_sec=10
    )
    print(f"\n{participant}")
    print(f" attention frames used: {len(used_global_frames)}")
    print(f" using fps={fps}, offset={fps*10} frames")

    out = {}
    nan3 = [float("nan")] * 3
    NAN_SUB = {
        "nose": nan3,
        "LEye": nan3,
        "REye": nan3,
        "LShoulder": nan3,
        "RShoulder": nan3,
    }

    parent_gt_and_valid = 0
    child_gt_and_valid = 0
    missing_in_alljoints = 0

    for fid in used_global_frames:
        fid_str = str(fid)

        if fid_str not in full:
            entry = {}
            missing_in_alljoints += 1
        else:
            entry = full[fid_str]

        objs = entry.get("objects", [])
        p_full = to_np(entry.get("parent_joints", [])) if entry.get("parent_joints") else None
        c_full = to_np(entry.get("child_joints", [])) if entry.get("child_joints") else None

        p_sel = None
        if has_valid_selected_joints(p_full):
            p_sel = p_full[SELECTED_11]

        c_sel = None
        if has_valid_selected_joints(c_full):
            c_sel = c_full[SELECTED_11]

        pitch_parent = (
            angle_gaze_with_worldY_from_selected(p_sel[:9])
            if p_sel is not None else None
        )
        pitch_child = (
            angle_gaze_with_worldY_from_selected(c_sel[:9])
            if c_sel is not None else None
        )
        parent_norm = NAN_SUB.copy()
        if (
            isinstance(p_full, np.ndarray)
            and p_full.shape[0] > max(RHIP, LHIP)
            and np.isfinite(p_full).all()
        ):
            p_norm = normalize_pose_hip_fixedY_projected(p_full)
            if p_norm is not None:
                parent_norm = extract_normalized_subset(p_norm)

        child_norm = NAN_SUB.copy()
        if (
            isinstance(c_full, np.ndarray)
            and c_full.shape[0] > max(RHIP, LHIP)
            and np.isfinite(c_full).all()
        ):
            c_norm = normalize_pose_hip_fixedY_projected(c_full)
            if c_norm is not None:
                child_norm = extract_normalized_subset(c_norm)

        this_frame_attention = frame_attention.get(fid, {"parent": set(), "child": set()})
        this_frame_touch = frame_touch.get(fid, {"parent": set(), "child": set()})

        parent_has_gt = len(this_frame_attention["parent"]) > 0
        child_has_gt = len(this_frame_attention["child"]) > 0

        if parent_has_gt and p_sel is not None:
            parent_gt_and_valid += 1
        if child_has_gt and c_sel is not None:
            child_gt_and_valid += 1

        objs_with_tags = attach_attention_and_touch_ids_to_objects(
            objs,
            this_frame_attention,
            this_frame_touch
        )

        p_ctr, c_ctr = compute_attended_centers_from_attention(
            objs_with_tags,
            this_frame_attention
        )

        out[fid_str] = {
            "objects": objs_with_tags,
            "parent_attention_objects": sorted(list(this_frame_attention["parent"])),
            "child_attention_objects": sorted(list(this_frame_attention["child"])),
            "parent_touch_objects": sorted(list(this_frame_touch["parent"])),
            "child_touch_objects": sorted(list(this_frame_touch["child"])),
            "parent_joints": p_sel.tolist() if p_sel is not None else [],
            "child_joints": c_sel.tolist() if c_sel is not None else [],
            "pitch_parent": pitch_parent,
            "pitch_child": pitch_child,
            "parent_norm": parent_norm,
            "child_norm": child_norm,
            "parent_attended_center": p_ctr,
            "child_attended_center": c_ctr,
        }

    # output_dir = os.path.join(data_dir, "inputs")
    output_dir = os.path.join(DATA_DIR, "inputs")
    os.makedirs(output_dir, exist_ok=True)

    dst = os.path.join(
        output_dir,
        f"{participant}_annotated_inputs.json"
    )
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    # print(f"   frames missing in alljoints: {missing_in_alljoints}")
    # print(f"   frames written to output: {len(out)}")
    # print(f"   parent frames with >=1 attention GT and valid joints: {parent_gt_and_valid}")
    # print(f"   child frames with >=1 attention GT and valid joints: {child_gt_and_valid}")
    return parent_gt_and_valid, child_gt_and_valid

def main():

    participant_dir = f"{DATA_DIR}/extracted_frames"
    participant = next(
        name for name in os.listdir(participant_dir)
        if os.path.isdir(os.path.join(participant_dir, name))
    )
    parent_count, child_count = build_fullvideo_inputs_for_participant(participant)

    print("parent frames with joints and at least one attention GT:", parent_count)
    print("child frames with joints and at least one attention GT :", child_count)
    
if __name__ == "__main__":
    main()


