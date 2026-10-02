
import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name

# ============ CONFIG ============
OBJ_CLASSES = [
    'doll', 'car', 'jump box', 'shape box', 'flower', 'book', 'bottle',
    'green star', 'yellow cylinder', 'blue cube', 'red triangle',
    'shape box lid'
]


SELECTED11_IDX = {
    'leye': 0,
    'reye': 1,
    'lshoulder': 2,
    'rshoulder': 3,
    'lhip': 4,
    'rhip': 5,
    'head': 6,
    'neck': 7,
    'nose': 8,
    'lwrist': 9,
    'rwrist': 10,
}

ROLES = ("parent", "child")

DEFAULT_DEGREE_THRESHOLD = 15.0
DEFAULT_DISTANCE_MARGIN = 0.10

DEGREE_GRID = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
MARGIN_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def _split_object_names(txt):
    if txt is None or (isinstance(txt, float) and pd.isna(txt)):
        return set()
    s = str(txt).strip().lower()
    if not s:
        return set()
    s = s.replace(", ", ";").replace(",", ";")

    return {
        _normalize_obj_name_for_eval(x)
        for x in s.split(";")
        if x.strip() and _normalize_obj_name_for_eval(x) in OBJ_CLASSES
    }
def compute_predictions_for_row(row, degree_threshold, distance_margin):
    role = str(row.get("role", "")).strip().lower()

    candidate_objs = []
    candidate_dists = []

    for obj in OBJ_CLASSES:
        if role == "parent" and obj == "head parent":
            continue
        if role == "child" and obj == "head child":
            continue

        d = row.get(f"{obj}_dist", np.nan)
        a = row.get(f"{obj}_angle_deg", np.nan)

        if pd.notna(d) and pd.notna(a) and float(a) <= float(degree_threshold):
            candidate_objs.append(obj)
            candidate_dists.append(float(d))

    if not candidate_objs:
        return set(), np.nan, np.nan

    closest_distance = min(candidate_dists)
    buffer_val = closest_distance + float(distance_margin)

    preds = set()
    for obj in candidate_objs:
        d = row.get(f"{obj}_dist", np.nan)
        a = row.get(f"{obj}_angle_deg", np.nan)

        if (
            pd.notna(d)
            and pd.notna(a)
            and float(a) <= float(degree_threshold)
            and float(d) <= buffer_val
        ):
            preds.add(obj)

    return preds, closest_distance, buffer_val

def evaluate_grid_for_fold(df, degree_grid=DEGREE_GRID, margin_grid=MARGIN_GRID):
    """
    Builds a flat summary table for one fold.

    For each:
      - role: parent / child
      - mode: all_rows / detected_only
      - degree_threshold
      - distance_margin

    Computes:
      - micro metrics: pool TP/FP/FN/TN over all object decisions
      - macro F1: compute F1 per object, replace NaN with 0, then average
    """
    if df is None or df.empty:
        return pd.DataFrame()

    records = []

    for degree_threshold in degree_grid:
        for distance_margin in margin_grid:
            for role in ROLES:
                role_df = df[df["role"] == role]
                if role_df.empty:
                    continue

                for mode in ("all_rows", "detected_only"):
                    # micro totals
                    tp = fp = fn = tn = 0

                    # per-object totals for macro F1
                    per_obj_counts = {
                        obj: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                        for obj in OBJ_CLASSES
                    }

                    for _, row in role_df.iterrows():
                        preds, closest_distance, buffer_val = compute_predictions_for_row(
                            row, degree_threshold, distance_margin
                        )
                        gt = _split_object_names(row["gt_attended_objects"])

                        for obj in OBJ_CLASSES:
                            # detected-only: only evaluate rows where this object is detectable
                            if mode == "detected_only" and pd.isna(row.get(f"{obj}_dist", np.nan)):
                                continue

                            pred_hit = obj in preds
                            gt_hit = obj in gt

                            if pred_hit and gt_hit:
                                tp += 1
                                per_obj_counts[obj]["TP"] += 1
                            elif pred_hit and not gt_hit:
                                fp += 1
                                per_obj_counts[obj]["FP"] += 1
                            elif (not pred_hit) and gt_hit:
                                fn += 1
                                per_obj_counts[obj]["FN"] += 1
                            else:
                                tn += 1
                                per_obj_counts[obj]["TN"] += 1

                    # micro
                    precision_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1_micro = (
                        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
                        if (precision_micro + recall_micro) > 0 else 0.0
                    )

                    # macro F1 = average of object-level F1s, NaN -> 0
                    obj_precs = []
                    obj_recs = []
                    obj_f1s = []

                    for obj in OBJ_CLASSES:
                        obj_tp = per_obj_counts[obj]["TP"]
                        obj_fp = per_obj_counts[obj]["FP"]
                        obj_fn = per_obj_counts[obj]["FN"]

                        obj_prec = obj_tp / (obj_tp + obj_fp) if (obj_tp + obj_fp) > 0 else 0.0
                        obj_rec  = obj_tp / (obj_tp + obj_fn) if (obj_tp + obj_fn) > 0 else 0.0

                        if (obj_prec + obj_rec) > 0:
                            obj_f1 = 2 * obj_prec * obj_rec / (obj_prec + obj_rec)
                        else:
                            obj_f1 = 0.0

                        obj_precs.append(obj_prec)
                        obj_recs.append(obj_rec)
                        obj_f1s.append(obj_f1)

                    precision_macro = float(np.mean(obj_precs)) if len(obj_precs) > 0 else 0.0
                    recall_macro = float(np.mean(obj_recs)) if len(obj_recs) > 0 else 0.0
                    f1_macro = float(np.mean(obj_f1s)) if len(obj_f1s) > 0 else 0.0
                    records.append({
                        "role": role,
                        "mode": mode,
                        "degree_threshold": degree_threshold,
                        "distance_margin": distance_margin,

                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "TN": tn,

                        "Precision_micro": precision_micro,
                        "Recall_micro": recall_micro,
                        "Precision_macro": precision_macro,
                        "Recall_macro": recall_macro,
                        "F1_micro": f1_micro,
                        "F1_macro": f1_macro,
                    })

    out = pd.DataFrame(records)
    if not out.empty:
        out.sort_values(
            by=["role", "mode", "degree_threshold", "distance_margin"],
            inplace=True,
            ignore_index=True
        )
    return out

def evaluate_single_combination(df, degree_threshold, distance_margin, role, mode):
    """
    Evaluate one (degree, distance_margin) pair on a dataframe
    for one specific role and mode.

    mode: "all_rows" or "detected_only"
    """
    role_df = df[df["role"] == role]
    if role_df.empty:
        return None

    tp = fp = fn = tn = 0

    per_obj_counts = {
        obj: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for obj in OBJ_CLASSES
    }

    for _, row in role_df.iterrows():
        preds, _, _ = compute_predictions_for_row(row, degree_threshold, distance_margin)
        gt = _split_object_names(row["gt_attended_objects"])

        for obj in OBJ_CLASSES:
            if mode == "detected_only" and pd.isna(row.get(f"{obj}_dist", np.nan)):
                continue

            pred_hit = obj in preds
            gt_hit = obj in gt

            if pred_hit and gt_hit:
                tp += 1
                per_obj_counts[obj]["TP"] += 1
            elif pred_hit and not gt_hit:
                fp += 1
                per_obj_counts[obj]["FP"] += 1
            elif (not pred_hit) and gt_hit:
                fn += 1
                per_obj_counts[obj]["FN"] += 1
            else:
                tn += 1
                per_obj_counts[obj]["TN"] += 1

    precision_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0 else 0.0
    )

    obj_precs = []
    obj_recs = []
    obj_f1s = []

    for obj in OBJ_CLASSES:
        obj_tp = per_obj_counts[obj]["TP"]
        obj_fp = per_obj_counts[obj]["FP"]
        obj_fn = per_obj_counts[obj]["FN"]

        obj_prec = obj_tp / (obj_tp + obj_fp) if (obj_tp + obj_fp) > 0 else 0.0
        obj_rec  = obj_tp / (obj_tp + obj_fn) if (obj_tp + obj_fn) > 0 else 0.0
        obj_f1 = 2 * obj_prec * obj_rec / (obj_prec + obj_rec) if (obj_prec + obj_rec) > 0 else 0.0

        obj_precs.append(obj_prec)
        obj_recs.append(obj_rec)
        obj_f1s.append(obj_f1)

    precision_macro = float(np.mean(obj_precs)) if len(obj_precs) > 0 else 0.0
    recall_macro = float(np.mean(obj_recs)) if len(obj_recs) > 0 else 0.0
    f1_macro = float(np.mean(obj_f1s)) if len(obj_f1s) > 0 else 0.0
    return {
        "role": role,
        "mode": mode,
        "degree_threshold": degree_threshold,
        "distance_margin": distance_margin,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision_micro": precision_micro,
        "Recall_micro": recall_micro,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_micro": f1_micro,
        "F1_macro": f1_macro,
    }
def select_best_train_and_eval_test_from_grid(train_grid, df_test):
    if train_grid is None or train_grid.empty or df_test is None or df_test.empty:
        return pd.DataFrame()

    rows = []

    for role in ROLES:
        for mode in ("all_rows", "detected_only"):
            sub_train = train_grid[
                (train_grid["role"] == role) &
                (train_grid["mode"] == mode)
            ].copy()

            if sub_train.empty:
                continue

            sub_train_micro = sub_train.sort_values(
                by=["F1_micro", "F1_macro", "degree_threshold", "distance_margin"],
                ascending=[False, False, True, True]
            ).reset_index(drop=True)
            best_micro = sub_train_micro.iloc[0]

            test_micro = evaluate_single_combination(
                df_test,
                degree_threshold=float(best_micro["degree_threshold"]),
                distance_margin=float(best_micro["distance_margin"]),
                role=role,
                mode=mode
            )

            rows.append({
                "role": role,
                "mode": mode,
                "selection_metric": "micro",
                "best_degree_threshold": float(best_micro["degree_threshold"]),
                "best_distance_margin": float(best_micro["distance_margin"]),
                "train_F1_micro": float(best_micro["F1_micro"]),
                "train_F1_macro": float(best_micro["F1_macro"]),
                "train_Precision_macro": float(best_micro["Precision_macro"]),
                "train_Recall_macro": float(best_micro["Recall_macro"]),
                "test_Precision_micro": float(test_micro["Precision_micro"]) if test_micro else 0.0,
                "test_Recall_micro": float(test_micro["Recall_micro"]) if test_micro else 0.0,
                "test_Precision_macro": float(test_micro["Precision_macro"]) if test_micro else 0.0,
                "test_Recall_macro": float(test_micro["Recall_macro"]) if test_micro else 0.0,
                "test_F1_micro": float(test_micro["F1_micro"]) if test_micro else 0.0,
                "test_F1_macro": float(test_micro["F1_macro"]) if test_micro else 0.0,
                "test_TP": int(test_micro["TP"]) if test_micro else 0,
                "test_FP": int(test_micro["FP"]) if test_micro else 0,
                "test_FN": int(test_micro["FN"]) if test_micro else 0,
                "test_TN": int(test_micro["TN"]) if test_micro else 0,
            })

            sub_train_macro = sub_train.sort_values(
                by=["F1_macro", "F1_micro", "degree_threshold", "distance_margin"],
                ascending=[False, False, True, True]
            ).reset_index(drop=True)
            best_macro = sub_train_macro.iloc[0]

            test_macro = evaluate_single_combination(
                df_test,
                degree_threshold=float(best_macro["degree_threshold"]),
                distance_margin=float(best_macro["distance_margin"]),
                role=role,
                mode=mode
            )

            rows.append({
                "role": role,
                "mode": mode,
                "selection_metric": "macro",
                "best_degree_threshold": float(best_macro["degree_threshold"]),
                "best_distance_margin": float(best_macro["distance_margin"]),
                "train_F1_micro": float(best_macro["F1_micro"]),
                "train_F1_macro": float(best_macro["F1_macro"]),
                "test_Precision_micro": float(test_macro["Precision_micro"]) if test_macro else 0.0,
                "test_Recall_micro": float(test_macro["Recall_micro"]) if test_macro else 0.0,
                "test_F1_micro": float(test_macro["F1_micro"]) if test_macro else 0.0,
                "test_F1_macro": float(test_macro["F1_macro"]) if test_macro else 0.0,
                "train_Precision_macro": float(best_macro["Precision_macro"]),
                "train_Recall_macro": float(best_macro["Recall_macro"]),
                "test_Precision_macro": float(test_macro["Precision_macro"]) if test_macro else 0.0,
                "test_Recall_macro": float(test_macro["Recall_macro"]) if test_macro else 0.0,
                "test_TP": int(test_macro["TP"]) if test_macro else 0,
                "test_FP": int(test_macro["FP"]) if test_macro else 0,
                "test_FN": int(test_macro["FN"]) if test_macro else 0,
                "test_TN": int(test_macro["TN"]) if test_macro else 0,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(by=["role", "mode", "selection_metric"], inplace=True, ignore_index=True)
    return out

def write_best_pairs_test_eval_sheet(writer, workbook, df_all, sheet_name="best_pairs_test_eval"):
    if df_all is None or df_all.empty:
        return

    df_all.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    ws = writer.sheets[sheet_name]

    int_fmt = workbook.add_format({'num_format': '0', 'border': 1})
    float_fmt = workbook.add_format({'num_format': '0.000', 'border': 1})
    title_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9EAF7', 'border': 1})

    nrows, ncols = df_all.shape
    ws.autofilter(0, 0, nrows, ncols - 1)
    ws.freeze_panes(1, 0)

    ws.set_column(0, 2, 14)
    ws.set_column(3, 4, 16, float_fmt)
    ws.set_column(5, 10, 14, float_fmt)
    ws.set_column(11, 14, 10, int_fmt)

    # average rows by selection metric / mode / role
    start_row = nrows + 3
    ws.write(start_row, 0, "Average over folds", title_fmt)

    avg_df = (
        df_all.groupby(["role", "mode", "selection_metric"], as_index=False)[
           [
            "train_Precision_macro", "train_Recall_macro",
            "train_F1_micro", "train_F1_macro",
            "test_Precision_micro", "test_Recall_micro",
            "test_Precision_macro", "test_Recall_macro",
            "test_F1_micro", "test_F1_macro"
        ]
        ]
        .mean()
    )

    avg_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row + 1, startcol=0)

def write_grid_sheet_from_grid(writer, workbook, grid_df, sheet_name,
                               degree_grid=DEGREE_GRID, margin_grid=MARGIN_GRID):
    if grid_df is None or grid_df.empty:
        return

    grid_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    ws = writer.sheets[sheet_name]

    title_fmt = workbook.add_format({'bold': True, 'bg_color': '#E8F5E9', 'border': 1})
    int_fmt = workbook.add_format({'num_format': '0', 'border': 1})
    float_fmt = workbook.add_format({'num_format': '0.000', 'border': 1})

    nrows, ncols = grid_df.shape
    ws.autofilter(0, 0, nrows, ncols - 1)
    ws.freeze_panes(1, 0)

    ws.set_column(0, 1, 14)
    ws.set_column(2, 3, 16)
    ws.set_column(4, 7, 10, int_fmt)
    ws.set_column(8, 11, 14, float_fmt)

    start_row = nrows + 4

    pivot_specs = [
        ("parent", "all_rows", "F1_micro", "PARENT - F1 micro (all rows)"),
        ("child", "all_rows", "F1_micro", "CHILD - F1 micro (all rows)"),
        ("parent", "detected_only", "F1_micro", "PARENT - F1 micro (detected only)"),
        ("child", "detected_only", "F1_micro", "CHILD - F1 micro (detected only)"),
        ("parent", "all_rows", "F1_macro", "PARENT - F1 macro (all rows)"),
        ("child", "all_rows", "F1_macro", "CHILD - F1 macro (all rows)"),
        ("parent", "detected_only", "F1_macro", "PARENT - F1 macro (detected only)"),
        ("child", "detected_only", "F1_macro", "CHILD - F1 macro (detected only)"),
    ]

    block_height = len(degree_grid) + 4

    for i, (role, mode, metric_col, title) in enumerate(pivot_specs):
        block_row = start_row + i * block_height
        sub = grid_df[(grid_df["role"] == role) & (grid_df["mode"] == mode)].copy()
        pv = sub.pivot(index="degree_threshold", columns="distance_margin", values=metric_col)
        pv = pv.reindex(index=degree_grid, columns=margin_grid)

        ws.write(block_row, 0, title, title_fmt)
        pv.to_excel(writer, sheet_name=sheet_name, startrow=block_row + 1, startcol=0)

    ws.set_column(0, 0, 18)
    ws.set_column(1, len(margin_grid), 12, float_fmt)

def write_grid_sheet(writer, workbook, df, sheet_name,
                     degree_grid=DEGREE_GRID, margin_grid=MARGIN_GRID):
    """
    Writes:
      1) raw flat grid-results table
      2) compact pivot tables for F1_micro and F1_macro
    """
    grid_df = evaluate_grid_for_fold(df, degree_grid, margin_grid)
    if grid_df.empty:
        return

    grid_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    ws = writer.sheets[sheet_name]

    title_fmt = workbook.add_format({'bold': True, 'bg_color': '#E8F5E9', 'border': 1})
    int_fmt = workbook.add_format({'num_format': '0', 'border': 1})
    float_fmt = workbook.add_format({'num_format': '0.000', 'border': 1})

    nrows, ncols = grid_df.shape
    ws.autofilter(0, 0, nrows, ncols - 1)
    ws.freeze_panes(1, 0)

    ws.set_column(0, 1, 14)
    ws.set_column(2, 3, 16)
    ws.set_column(4, 7, 10, int_fmt)
    ws.set_column(8, 11, 14, float_fmt)

    start_row = nrows + 4

    pivot_specs = [
        ("parent", "all_rows", "F1_micro", "PARENT - F1 micro (all rows)"),
        ("child", "all_rows", "F1_micro", "CHILD - F1 micro (all rows)"),
        ("parent", "detected_only", "F1_micro", "PARENT - F1 micro (detected only)"),
        ("child", "detected_only", "F1_micro", "CHILD - F1 micro (detected only)"),

        ("parent", "all_rows", "F1_macro", "PARENT - F1 macro (all rows)"),
        ("child", "all_rows", "F1_macro", "CHILD - F1 macro (all rows)"),
        ("parent", "detected_only", "F1_macro", "PARENT - F1 macro (detected only)"),
        ("child", "detected_only", "F1_macro", "CHILD - F1 macro (detected only)"),
    ]

    block_height = len(degree_grid) + 4

    for i, (role, mode, metric_col, title) in enumerate(pivot_specs):
        block_row = start_row + i * block_height

        sub = grid_df[(grid_df["role"] == role) & (grid_df["mode"] == mode)].copy()
        pv = sub.pivot(index="degree_threshold", columns="distance_margin", values=metric_col)
        pv = pv.reindex(index=degree_grid, columns=margin_grid)

        ws.write(block_row, 0, title, title_fmt)
        pv.to_excel(writer, sheet_name=sheet_name, startrow=block_row + 1, startcol=0)

    ws.set_column(0, 0, 18)
    ws.set_column(1, len(margin_grid), 12, float_fmt)

EXTRA_COLS = [
    "predictions_distance",
    "predictionsDegree",
    "Predictions",
    "TP",
    "FN",
    "FP",
    "degree_threshold",
    "Distance_threshold",
    "TN",
    "closest_distance",
    "buffer",
    "gt_attended_objects_check",
    "Detected",
    "Count GT",
    "Count Detected",
]

# ============ helpers ============
def _unit_or_none(v):
    v = np.asarray(v, dtype=float)
    if v.shape[-1] != 3 or not np.all(np.isfinite(v)):
        return None
    n = np.linalg.norm(v)
    if n == 0:
        return None
    return v / n

def _finite3(p):
    p = np.asarray(p, dtype=float)
    return p.shape == (3,) and np.all(np.isfinite(p))

def get_head_pose_matrix_from_selected11(joints):
    arr = np.asarray(joints, dtype=float)
    if arr.ndim != 2 or arr.shape != (11, 3):
        return None, None

    LEye = arr[SELECTED11_IDX['leye']]
    REye = arr[SELECTED11_IDX['reye']]
    Head = arr[SELECTED11_IDX['head']]
    Neck = arr[SELECTED11_IDX['neck']]

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

def angles_to_fpv_dir_from_tan(vertical_deg, horizontal_deg):
    v_rad = np.deg2rad(vertical_deg)
    h_rad = np.deg2rad(horizontal_deg)
    z_sign = np.sign(np.cos(h_rad))
    z = float(z_sign if z_sign != 0 else 1.0)
    x = np.tan(h_rad) * z
    y = np.tan(v_rad) * z
    vec = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(vec)
    if n == 0:
        return None
    return vec / n

def _joints_valid_for_role(joints) -> bool:
    try:
        arr = np.asarray(joints, dtype=float)
    except Exception:
        return False

    if arr.ndim != 2 or arr.shape != (11, 3):
        return False

    required = [
        SELECTED11_IDX['leye'],
        SELECTED11_IDX['reye'],
        SELECTED11_IDX['head'],
        SELECTED11_IDX['neck'],
        SELECTED11_IDX['nose'],
    ]
    pts = [arr[i] for i in required]
    return all(np.all(np.isfinite(p)) and p.shape == (3,) for p in pts)

def load_angles_npz(npz_path):
    if not (npz_path and os.path.exists(npz_path)):
        return {}

    d = np.load(npz_path, allow_pickle=True)
    if not {"frames", "pred_h", "pred_v"} <= set(d.files):
        return {}

    out = {}
    for f, h, v in zip(d["frames"], d["pred_h"], d["pred_v"]):
        f_int = int(str(f).split(".")[0])
        out[str(f_int)] = (float(h), float(v))
    return out

def _get_obj_id(obj, fallback_idx=None):
    for k in ("id", "obj_id", "object_id", "class_id", "track_id"):
        if k in obj:
            return int(obj[k])
    return fallback_idx

def build_df_for_participant(json_path: str, opt_npz_paths: dict) -> pd.DataFrame:
    child_opt = load_angles_npz(opt_npz_paths.get("child", ""))
    parent_opt = load_angles_npz(opt_npz_paths.get("parent", ""))

    with open(json_path, "r") as f:
        data = json.load(f)

    frame_ids = sorted([k for k in data.keys() if str(k).isdigit()], key=lambda k: int(k))

    columns = ["frame_id", "role", "gt_attended_objects"]
    for name in OBJ_CLASSES:
        columns += [f"{name}_dist", f"{name}_angle_deg"]

    rows = []

    for frame_id in frame_ids:
        frame = data[frame_id]

        obj_map = {}
        parent_gt_names, child_gt_names = [], []

        for idx, o in enumerate(frame.get('objects', [])):
            oid = _get_obj_id(o, fallback_idx=idx)
            att = int(o.get('attention_id', 0))

            if 0 <= oid < len(OBJ_CLASSES):
                if att in (1, 3):
                    parent_gt_names.append(OBJ_CLASSES[oid])
                if att in (2, 3):
                    child_gt_names.append(OBJ_CLASSES[oid])

            pt = np.asarray(o.get('coords', [np.nan, np.nan, np.nan]), dtype=float)
            obj_map[oid] = pt if _finite3(pt) else None

        for role in ROLES:
            gt_names = parent_gt_names if role == "parent" else child_gt_names


            joints = frame.get(f"{role}_joints")
            if joints is None or not _joints_valid_for_role(joints):
                continue

            R, eye_center = get_head_pose_matrix_from_selected11(joints)
            if R is None or eye_center is None:
                continue

            opt = parent_opt if role == "parent" else child_opt
            dir_opt_world = None
            if frame_id in opt:
                h_opt, v_opt = opt[frame_id]
                dir_opt_fpv = angles_to_fpv_dir_from_tan(v_opt, h_opt)
                if dir_opt_fpv is not None:
                    dir_opt_world = _unit_or_none(R @ dir_opt_fpv)

            row = {
                "frame_id": int(frame_id),
                "role": role,
                "gt_attended_objects": ",".join(sorted(set(gt_names))),
            }

            for obj_name in OBJ_CLASSES:
                row[f"{obj_name}_dist"] = float('nan')
                row[f"{obj_name}_angle_deg"] = float('nan')

            for obj_id, obj_name in enumerate(OBJ_CLASSES):
                pt = obj_map.get(obj_id, None)

                row[f"{obj_name}_dist"] = float(np.linalg.norm(pt - eye_center)) if (
                    eye_center is not None and pt is not None and _finite3(pt)
                ) else float('nan')

                if eye_center is not None and pt is not None and dir_opt_world is not None and _finite3(pt):
                    v = pt - eye_center
                    v_u = _unit_or_none(v)
                    if v_u is not None:
                        cos_t = max(-1.0, min(1.0, float(np.dot(v_u, dir_opt_world))))
                        ang = math.degrees(math.acos(cos_t))
                    else:
                        ang = float('nan')
                else:
                    ang = float('nan')

                row[f"{obj_name}_angle_deg"] = ang

            rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.sort_values(by=["frame_id", "role"], inplace=True, ignore_index=True)
    return df

def load_records_json(path):
    with open(path, "r") as f:
        return json.load(f)

# ============ excel formula builders ============
def col_letter(idx_zero_based: int) -> str:
    return xl_col_to_name(idx_zero_based)

def build_df_from_exact_records(records, processed_dir, fold_dir):
    """
    Build dataframe only from the exact samples saved by the training pipeline.

    GT attended objects are taken from:
      - parent_attention_objects / child_attention_objects
    NOT reconstructed from frame["objects"].

    Distances/angles are still computed from frame["objects"] coords.
    """
    if not records:
        return None

    json_cache = {}
    angle_cache = {}

    rows = []

    for rec in records:
        participant = rec["participant"]
        role = rec["role"]
        frame_id_raw = rec["frame_id"]
        frame_id_str = str(frame_id_raw)

        json_path = os.path.join(
            processed_dir, participant, "data", "inputs",
            f"{participant}_annotated_inputs.json"
        )
        if json_path not in json_cache:
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r") as f:
                json_cache[json_path] = json.load(f)

        data = json_cache[json_path]

        if frame_id_str not in data:
            alt = str(int(frame_id_raw)) if str(frame_id_raw).isdigit() else frame_id_str
            if alt not in data:
                continue
            frame_id_str = alt

        frame = data[frame_id_str]

        joints = frame.get(f"{role}_joints")
        if joints is None or not _joints_valid_for_role(joints):
            continue

        R, eye_center = get_head_pose_matrix_from_selected11(joints)
        if R is None or eye_center is None:
            continue

        cache_key = (participant, role)
        if cache_key not in angle_cache:
            npz_path = os.path.join(
                fold_dir,
                f"{role}_{participant}_fullvideo_angles_pred_gt_angles.npz"
            )
            angle_cache[cache_key] = load_angles_npz(npz_path)

        pred_angle_map = angle_cache[cache_key]

        dir_opt_world = None
        if frame_id_str in pred_angle_map:
            h_opt, v_opt = pred_angle_map[frame_id_str]
            dir_opt_fpv = angles_to_fpv_dir_from_tan(v_opt, h_opt)
            if dir_opt_fpv is not None:
                dir_opt_world = _unit_or_none(R @ dir_opt_fpv)

        # ----- object coordinates for distance/angle -----
        obj_map = {}
        for idx, o in enumerate(frame.get("objects", [])):
            oid = _get_obj_id(o, fallback_idx=idx)
            pt = np.asarray(o.get("coords", [np.nan, np.nan, np.nan]), dtype=float)
            obj_map[oid] = pt if _finite3(pt) else None

        # ----- GT attended objects: use stored attention-object lists -----
        if role == "parent":
            gt_names_raw = frame.get("parent_attention_objects", [])
            attended_center = frame.get("parent_attended_center", None)
        else:
            gt_names_raw = frame.get("child_attention_objects", [])
            attended_center = frame.get("child_attended_center", None)

        gt_names = []
        for x in gt_names_raw:
            x_norm = _normalize_obj_name_for_eval(x)
            if x_norm in OBJ_CLASSES_NORM:
                gt_names.append(x_norm)
        row = {
            "participant": participant,
            "frame_id": int(frame_id_str) if str(frame_id_str).isdigit() else frame_id_str,
            "role": role,
            "gt_attended_objects": ",".join(sorted(set(gt_names))),
            "has_gt_center": attended_center is not None,
        }

        for obj_name in OBJ_CLASSES:
            row[f"{obj_name}_dist"] = float("nan")
            row[f"{obj_name}_angle_deg"] = float("nan")

        for obj_id, obj_name in enumerate(OBJ_CLASSES):
            pt = obj_map.get(obj_id, None)

            if eye_center is not None and pt is not None and _finite3(pt):
                row[f"{obj_name}_dist"] = float(np.linalg.norm(pt - eye_center))

            if eye_center is not None and pt is not None and dir_opt_world is not None and _finite3(pt):
                v = pt - eye_center
                v_u = _unit_or_none(v)
                if v_u is not None:
                    cos_t = max(-1.0, min(1.0, float(np.dot(v_u, dir_opt_world))))
                    ang = math.degrees(math.acos(cos_t))
                else:
                    ang = float("nan")
            else:
                ang = float("nan")

            row[f"{obj_name}_angle_deg"] = ang

        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    ordered_cols = ["participant", "frame_id", "role", "gt_attended_objects", "has_gt_center"] + [
        f"{name}_{suffix}" for name in OBJ_CLASSES for suffix in ("dist", "angle_deg")
    ]
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    df = df[ordered_cols]
    df.sort_values(by=["participant", "frame_id", "role"], inplace=True, ignore_index=True)
    return df


def _normalize_obj_name_for_eval(name):
    s = str(name).strip().lower().replace("_", " ")
    # There were annotation differences in the data, extra steps are included to deal with these annotation problems
    alias_map = {
        "jumpbox": "jump box",
        "shapebox": "shape box",
        "shapebox lid": "shape box lid",
        "shapebox_lid": "shape box lid",
        "greenstar": "green star",
        "green_star": "green star",
        "yellowcylinder": "yellow cylinder",
        "yellow_cylinder":"yellow cylinder",
        "bluecube": "blue cube",
        "blue_cube": "blue cube",
        "red_triangle": "red triangle",
        "redtriangle": "red triangle",
    }

    return alias_map.get(s, s)

OBJ_CLASSES_NORM = {_normalize_obj_name_for_eval(o) for o in OBJ_CLASSES}

def cell(row_zero_based: int, col_zero_based: int, row_abs=False, col_abs=False) -> str:
    return xl_rowcol_to_cell(row_zero_based, col_zero_based, row_abs=row_abs, col_abs=col_abs)

def write_controls(ws, workbook, degree_threshold, distance_margin):
    label_fmt = workbook.add_format({'bold': True, 'bg_color': '#EDE7F6', 'border': 1})
    value_fmt = workbook.add_format({'bg_color': '#FFF59D', 'border': 1, 'num_format': '0.00'})
    note_fmt = workbook.add_format({'italic': True, 'font_color': '#666666'})

    ws.write('A1', 'degree_threshold', label_fmt)
    ws.write_number('B1', float(degree_threshold), value_fmt)
    ws.write('A2', 'distance_margin', label_fmt)
    ws.write_number('B2', float(distance_margin), value_fmt)
    ws.write('D1', 'Edit B1 and B2 only. All formula columns update automatically.', note_fmt)
    ws.freeze_panes(4, 0)

def build_sheet_formulas(sheet_name, df, startrow=3):
    headers = list(df.columns) + EXTRA_COLS
    idx = {h: i for i, h in enumerate(headers)}

    dist_cols = [(obj, idx[f"{obj}_dist"]) for obj in OBJ_CLASSES]
    angle_cols = [(obj, idx[f"{obj}_angle_deg"]) for obj in OBJ_CLASSES]

    formulas = {}

    for r in range(len(df)):
        row = startrow + 1 + r  # excel row index in zero-based for data rows

        gt_col = idx["gt_attended_objects"]
        role_col = idx["role"]
        pred_dist_col = idx["predictions_distance"]
        pred_deg_col = idx["predictionsDegree"]
        pred_col = idx["Predictions"]
        tp_col = idx["TP"]
        fn_col = idx["FN"]
        fp_col = idx["FP"]
        deg_thr_col = idx["degree_threshold"]
        dist_thr_col = idx["Distance_threshold"]
        tn_col = idx["TN"]
        closest_col = idx["closest_distance"]
        buffer_col = idx["buffer"]
        gt_check_col = idx["gt_attended_objects_check"]
        detected_col = idx["Detected"]
        count_gt_col = idx["Count GT"]
        count_det_col = idx["Count Detected"]

        b1 = "$B$1"
        b2 = "$B$2"
        gt_ref = cell(row, gt_col)
        role_ref = cell(row, role_col)
        pred_dist_ref = cell(row, pred_dist_col)
        pred_deg_ref = cell(row, pred_deg_col)
        pred_ref = cell(row, pred_col)
        buffer_ref = cell(row, buffer_col)
        closest_ref = cell(row, closest_col)
        gt_check_ref = cell(row, gt_check_col)

        # predictions_distance
        pieces = []
        for obj, c in dist_cols:
            ref = cell(row, c)

            extra_role_cond = ""
            if obj == "head parent":
                extra_role_cond = f',{role_ref}<>"parent"'
            elif obj == "head child":
                extra_role_cond = f',{role_ref}<>"child"'

            pieces.append(
                f'IF(AND(ISNUMBER({ref}),{ref}<={buffer_ref}{extra_role_cond}),"{obj}","")'
            )

        formulas[(row, pred_dist_col)] = f'=TEXTJOIN(", ",TRUE,{",".join(pieces)})'
        
        pieces = []
        for obj, c in angle_cols:
            ref = cell(row, c)

            extra_role_cond = ""
            if obj == "head parent":
                extra_role_cond = f',{role_ref}<>"parent"'
            elif obj == "head child":
                extra_role_cond = f',{role_ref}<>"child"'

            pieces.append(
                f'IF(AND(ISNUMBER({ref}),{ref}<={b1}{extra_role_cond}),"{obj}","")'
            )

        formulas[(row, pred_deg_col)] = f'=TEXTJOIN(", ",TRUE,{",".join(pieces)})'

        pieces = []
        for obj in OBJ_CLASSES:
            dref = cell(row, idx[f"{obj}_dist"])
            aref = cell(row, idx[f"{obj}_angle_deg"])

            extra_role_cond = ""
            if obj == "head parent":
                extra_role_cond = f',{role_ref}<>"parent"'
            elif obj == "head child":
                extra_role_cond = f',{role_ref}<>"child"'

            pieces.append(
                f'IF(AND(ISNUMBER({dref}),ISNUMBER({aref}),{dref}<={buffer_ref},{aref}<={b1}{extra_role_cond}),"{obj}","")'
            )

        formulas[(row, pred_col)] = f'=TEXTJOIN(", ",TRUE,{",".join(pieces)})'

        formulas[(row, tp_col)] = (
            f'=LET('
            f'P,LOWER(SUBSTITUTE(SUBSTITUTE(SUBSTITUTE({pred_ref},CHAR(160)," "),", ",";"),",",";")),'
            f'G,LOWER(SUBSTITUTE(SUBSTITUTE(SUBSTITUTE({gt_ref},CHAR(160)," "),", ",";"),",",";")),'
            f'PL,IFERROR(FILTER(TRIM(TEXTSPLIT(P,";")),TRIM(TEXTSPLIT(P,";"))<>""),""),'
            f'GL,IFERROR(FILTER(TRIM(TEXTSPLIT(G,";")),TRIM(TEXTSPLIT(G,";"))<>""),""),'
            f'IF(OR(PL="",GL=""),0,SUM(--ISNUMBER(XMATCH(PL,GL)))))'
        )


        formulas[(row, fn_col)] = (
            f'=LET('
            f'gt,LOWER(SUBSTITUTE(SUBSTITUTE({gt_ref},", ",";"),",",";")),'
            f'pr,LOWER(SUBSTITUTE(SUBSTITUTE({pred_ref},", ",";"),",",";")),'
            f'G0,TRIM(TEXTSPLIT(gt,";")),'
            f'P0,TRIM(TEXTSPLIT(pr,";")),'
            f'G,IFERROR(FILTER(G0,G0<>""),""),'
            f'P,IFERROR(FILTER(P0,P0<>""),""),'
            f'IF(OR(G="",ISBLANK(G)),0,SUMPRODUCT(--(LEN(G)>0),--(ISNA(MATCH(G,P,0))))))'
        )


        formulas[(row, fp_col)] = (
            f'=LET('
            f'Ptxt,LOWER(SUBSTITUTE(SUBSTITUTE({pred_ref},", ",";"),",",";")),'
            f'Gtxt,LOWER(SUBSTITUTE(SUBSTITUTE({gt_ref},", ",";"),",",";")),'
            f'P,IFERROR(FILTER(TRIM(TEXTSPLIT(Ptxt,";")),TRIM(TEXTSPLIT(Ptxt,";"))<>""),""),'
            f'G,IFERROR(FILTER(TRIM(TEXTSPLIT(Gtxt,";")),TRIM(TEXTSPLIT(Gtxt,";"))<>""),""),'
            f'IF(OR(P="",G=""),IF(P="",0,ROWS(P)),SUM(--ISNA(XMATCH(P,G)))))'
        )

        formulas[(row, deg_thr_col)] = f'={b1}'
        formulas[(row, dist_thr_col)] = f'={b2}'
        formulas[(row, tn_col)] = f'={len(OBJ_CLASSES)}-{cell(row,tp_col)}-{cell(row,fn_col)}-{cell(row,fp_col)}'

        # closest_distance using predictionsDegree
        gclean = f'LOWER(SUBSTITUTE(SUBSTITUTE(SUBSTITUTE({pred_deg_ref},CHAR(160)," "),", ",";"),",",";"))'
        parts = []
        for obj, c in dist_cols:
            dref = cell(row, c)
            obj_esc = obj.replace('"', '""')
            parts.append(f'IF(ISNUMBER(SEARCH(";{obj_esc};",";"&{gclean}&";")),{dref},1E+99)')
        formulas[(row, closest_col)] = f'=LET(result,MIN({",".join(parts)}),IF(result=1E+99,"",result))'

        formulas[(row, buffer_col)] = f'=IF({closest_ref}="","",{closest_ref}+{b2})'
        formulas[(row, gt_check_col)] = (
            f'=LOWER(SUBSTITUTE(SUBSTITUTE(SUBSTITUTE({gt_ref},CHAR(160)," "),", ",";"),",",";"))'
        )

        det_parts = []
        for obj, c in dist_cols:
            dref = cell(row, c)
            obj_esc = obj.replace('"', '""')
            det_parts.append(f'IF(AND(ISNUMBER({dref}),ISNUMBER(SEARCH(";{obj_esc};",";"&{gt_check_ref}&";"))),"{obj_esc}","")')
        formulas[(row, detected_col)] = f'=TEXTJOIN("; ",TRUE,{",".join(det_parts)})'

        formulas[(row, count_gt_col)] = (
            f'=LET('
            f'T,SUBSTITUTE({gt_check_ref},", ",";"),'
            f'X,IFERROR(FILTER(TRIM(TEXTSPLIT(T,";")),TRIM(TEXTSPLIT(T,";"))<>""),""),'
            f'IF(X="",0,ROWS(X)))'
        )
        formulas[(row, count_det_col)] = (
            f'=LET('
            f'T,SUBSTITUTE({cell(row, detected_col)},", ",";"),'
            f'X,IFERROR(FILTER(TRIM(TEXTSPLIT(T,";")),TRIM(TEXTSPLIT(T,";"))<>""),""),'
            f'IF(X="",0,ROWS(X)))'
        )

    return headers, formulas

def write_sheet(writer, workbook, df, sheet_name, degree_threshold, distance_margin):
    if df is None or df.empty:
        return

    ws = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = ws

    write_controls(ws, workbook, degree_threshold, distance_margin)

    startrow = 3
    base_headers = list(df.columns)
    all_headers, formulas = build_sheet_formulas(sheet_name, df, startrow=startrow)
    full_df = df.copy()
    for c in EXTRA_COLS:
        full_df[c] = ""

    full_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=startrow)
    # overwrite formula columns with formulas
    formula_fmt = workbook.add_format({'font_color': '#000000'})
    input_fmt = workbook.add_format({'font_color': '#0000FF'})
    yellow_fmt = workbook.add_format({'bg_color': '#FFF59D'})
    num_fmt = workbook.add_format({'num_format': '0.000'})
    int_fmt = workbook.add_format({'num_format': '0'})
    wrap_fmt = workbook.add_format({'text_wrap': True})

    # Highlight GT dist/angle cells
    header = list(full_df.columns)
    col_idx = {name: i for i, name in enumerate(header)}
    for r in range(len(df)):
        gt_str = str(df.iloc[r]["gt_attended_objects"]).strip()
        if not gt_str:
            continue
        attended_set = {x.strip() for x in gt_str.split(',') if x.strip()}
        excel_row = startrow + 1 + r
        for obj_name in attended_set:
            d_col_name = f"{obj_name}_dist"
            a_col_name = f"{obj_name}_angle_deg"
            if d_col_name in col_idx:
                dcol = col_idx[d_col_name]
                val = df.iloc[r][d_col_name]
                if pd.isna(val):
                    ws.write_blank(excel_row, dcol, None, yellow_fmt)
                else:
                    ws.write_number(excel_row, dcol, float(val), workbook.add_format({'bg_color': '#FFF59D', 'num_format': '0.000'}))
            if a_col_name in col_idx:
                acol = col_idx[a_col_name]
                val = df.iloc[r][a_col_name]
                if pd.isna(val):
                    ws.write_blank(excel_row, acol, None, yellow_fmt)
                else:
                    ws.write_number(excel_row, acol, float(val), workbook.add_format({'bg_color': '#FFF59D', 'num_format': '0.00'}))

    # write formulas
    for (r, c), f in formulas.items():
        if full_df.columns[c] in {"TP", "FN", "FP", "TN", "Count GT", "Count Detected"}:
            ws.write_formula(r, c, f, int_fmt)
        elif full_df.columns[c] in {"closest_distance", "buffer", "Distance_threshold"} or full_df.columns[c].endswith("_dist"):
            ws.write_formula(r, c, f, num_fmt)
        elif full_df.columns[c] == "degree_threshold":
            ws.write_formula(r, c, f, workbook.add_format({'num_format': '0.00', 'font_color': '#800080'}))
        else:
            ws.write_formula(r, c, f, wrap_fmt)

    rows = len(full_df)
    cols = len(full_df.columns)
    ws.autofilter(startrow, 0, startrow + rows, cols - 1)

    # widths
    ws.set_column(0, 0, 16)
    ws.set_column(1, 4, 14)
    ws.set_column(4, cols - 1, 15)
    ws.set_column(col_idx["gt_attended_objects"], col_idx["gt_attended_objects"], 24)
    for name in ["predictions_distance", "predictionsDegree", "Predictions", "gt_attended_objects_check", "Detected"]:
        if name in col_idx:
            ws.set_column(col_idx[name], col_idx[name], 28, wrap_fmt)
    write_metrics_block(ws, workbook, full_df, startrow=startrow)

def write_metrics_block(ws, workbook, full_df, startrow=3):
    """
    Writes 4 metrics blocks inside the same sheet:
      1) parent - all rows
      2) child  - all rows
      3) parent - detected-only-per-object
      4) child  - detected-only-per-object

    'detected-only-per-object' means:
      for each object, only rows with ISNUMBER(<that_object>_dist) are evaluated.
    """
    header = list(full_df.columns)
    col_idx = {name: i for i, name in enumerate(header)}

    role_col = xl_col_to_name(col_idx["role"])
    pred_col = xl_col_to_name(col_idx["Predictions"])
    gt_col = xl_col_to_name(col_idx["gt_attended_objects_check"])

    first_data_excel_row = startrow + 2
    last_data_excel_row = startrow + 1 + len(full_df)

    metrics_start_col = len(header) + 2

    title_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9EAF7', 'border': 1})
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#E8F5E9', 'border': 1})
    header_fmt_detected = workbook.add_format({'bold': True, 'bg_color': '#FFE0B2', 'border': 1})
    int_fmt = workbook.add_format({'border': 1, 'num_format': '0'})
    float_fmt = workbook.add_format({'border': 1, 'num_format': '0.000'})
    text_fmt = workbook.add_format({'border': 1})

    def wrapped_contains_formula(range_ref, obj_name):
        obj_pat = obj_name.lower().replace('"', '""')
        return (
            f'ISNUMBER(SEARCH(";{obj_pat};",'
            f'";"&LOWER(SUBSTITUTE(SUBSTITUTE({range_ref},", ",";"),",",";"))&";"))'
        )

    def write_one_role_block(role_name, start_row_excel, detected_only=False):
        block_title = f"{role_name.upper()} object-wise metrics"
        hdr_fmt = header_fmt_detected if detected_only else header_fmt
        block_title += " (detected-only per object)" if detected_only else " (all rows)"

        ws.write(start_row_excel - 1, metrics_start_col, block_title, title_fmt)

        headers = ["Object", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"]
        for j, h in enumerate(headers):
            ws.write(start_row_excel, metrics_start_col + j, h, hdr_fmt)

        role_rng = f"${role_col}${first_data_excel_row}:${role_col}${last_data_excel_row}"
        pred_rng = f"${pred_col}${first_data_excel_row}:${pred_col}${last_data_excel_row}"
        gt_rng   = f"${gt_col}${first_data_excel_row}:${gt_col}${last_data_excel_row}"

        for i, obj in enumerate(OBJ_CLASSES):
            row_excel = start_row_excel + 1 + i
            row0 = row_excel - 1

            ws.write(row0, metrics_start_col + 0, obj, text_fmt)

            pred_hit = wrapped_contains_formula(pred_rng, obj)
            gt_hit = wrapped_contains_formula(gt_rng, obj)

            # object-specific detectability mask:
            # only rows where this object's dist cell is numeric are counted
            obj_dist_col = xl_col_to_name(col_idx[f"{obj}_dist"])
            obj_dist_rng = f"${obj_dist_col}${first_data_excel_row}:${obj_dist_col}${last_data_excel_row}"
            det_mask = f'ISNUMBER({obj_dist_rng})'

            if detected_only:
                base_args = f'--({role_rng}="{role_name}"),--({det_mask})'
            else:
                base_args = f'--({role_rng}="{role_name}")'

            tp_formula = f'=SUMPRODUCT({base_args},--({pred_hit}),--({gt_hit}))'
            fp_formula = f'=SUMPRODUCT({base_args},--({pred_hit}),--(1-({gt_hit})))'
            fn_formula = f'=SUMPRODUCT({base_args},--(1-({pred_hit})),--({gt_hit}))'
            tn_formula = f'=SUMPRODUCT({base_args},--(1-({pred_hit})),--(1-({gt_hit})))'

            tp_ref = xl_rowcol_to_cell(row0, metrics_start_col + 1)
            fp_ref = xl_rowcol_to_cell(row0, metrics_start_col + 2)
            fn_ref = xl_rowcol_to_cell(row0, metrics_start_col + 3)

            precision_formula = f'=IFERROR({tp_ref}/({tp_ref}+{fp_ref}),0)'
            recall_formula = f'=IFERROR({tp_ref}/({tp_ref}+{fn_ref}),0)'
            prec_ref = xl_rowcol_to_cell(row0, metrics_start_col + 5)
            rec_ref = xl_rowcol_to_cell(row0, metrics_start_col + 6)
            f1_formula = f'=IFERROR(2*{prec_ref}*{rec_ref}/({prec_ref}+{rec_ref}),0)'

            ws.write_formula(row0, metrics_start_col + 1, tp_formula, int_fmt)
            ws.write_formula(row0, metrics_start_col + 2, fp_formula, int_fmt)
            ws.write_formula(row0, metrics_start_col + 3, fn_formula, int_fmt)
            ws.write_formula(row0, metrics_start_col + 4, tn_formula, int_fmt)
            ws.write_formula(row0, metrics_start_col + 5, precision_formula, float_fmt)
            ws.write_formula(row0, metrics_start_col + 6, recall_formula, float_fmt)
            ws.write_formula(row0, metrics_start_col + 7, f1_formula, float_fmt)

    parent_all_start = 5
    child_all_start = parent_all_start + len(OBJ_CLASSES) + 4
    parent_det_start = child_all_start + len(OBJ_CLASSES) + 4
    child_det_start = parent_det_start + len(OBJ_CLASSES) + 4

    write_one_role_block("parent", parent_all_start, detected_only=False)
    write_one_role_block("child", child_all_start, detected_only=False)
    write_one_role_block("parent", parent_det_start, detected_only=True)
    write_one_role_block("child", child_det_start, detected_only=True)

    ws.set_column(metrics_start_col, metrics_start_col, 18)
    ws.set_column(metrics_start_col + 1, metrics_start_col + 4, 10)
    ws.set_column(metrics_start_col + 5, metrics_start_col + 7, 10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True,
                        help="Directory containing participant folders and model_outputs")
    args = parser.parse_args()

    processed_dir = args.basedir
    cv_dir = os.path.join(processed_dir, "model_outputs", "tinymlp")
    cv_summary_path = os.path.join(cv_dir, "summary.json")
    out_excel = os.path.join(cv_dir, "all_folds_summary_interactive.xlsx")


    degree_threshold = DEFAULT_DEGREE_THRESHOLD
    distance_margin = DEFAULT_DISTANCE_MARGIN

    with open(cv_summary_path, "r") as f:
        cv_summary = json.load(f)

    fold_indices = sorted({int(item["fold_idx"]) for item in cv_summary["folds"]})

    with pd.ExcelWriter(
        out_excel,
        engine="xlsxwriter",
        engine_kwargs={"options": {"nan_inf_to_errors": True}}
    ) as writer:
        workbook = writer.book
        all_best_test_rows = []

        for fold_idx in fold_indices:
            fold_dir = os.path.join(cv_dir, "predictions", f"fold_{fold_idx:02d}")

            child_train_records = load_records_json(os.path.join(fold_dir, "child_train_records.json"))
            child_test_records  = load_records_json(os.path.join(fold_dir, "child_test_records.json"))

            parent_train_records = load_records_json(os.path.join(fold_dir, "parent_train_records.json"))
            parent_test_records  = load_records_json(os.path.join(fold_dir, "parent_test_records.json"))

            train_records = child_train_records + parent_train_records
            test_records  = child_test_records + parent_test_records

            df_train = build_df_from_exact_records(train_records, processed_dir, fold_dir)
            df_test  = build_df_from_exact_records(test_records, processed_dir, fold_dir)

            write_sheet(writer, workbook, df_train, f"fold{fold_idx}_train", degree_threshold, distance_margin)
            write_sheet(writer, workbook, df_test, f"fold{fold_idx}_test", degree_threshold, distance_margin)
            train_grid_df = evaluate_grid_for_fold(df_train, DEGREE_GRID, MARGIN_GRID)

            write_grid_sheet_from_grid(writer, workbook, train_grid_df, f"fold{fold_idx}_train_grid")
            best_df = select_best_train_and_eval_test_from_grid(train_grid_df, df_test)

            if not best_df.empty:
                best_df.insert(0, "fold_idx", fold_idx)
                all_best_test_rows.append(best_df)

        if all_best_test_rows:
            final_best_df = pd.concat(all_best_test_rows, ignore_index=True)
            write_best_pairs_test_eval_sheet(writer, workbook, final_best_df, sheet_name="best_pairs_test_eval")
    print(f"Wrote interactive workbook: {out_excel}")

if __name__ == "__main__":
    main()
