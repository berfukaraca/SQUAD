import os
import json
import argparse
import random
from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict


""" Expected structure
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

# ---------------------------
# Joints index constants
# ---------------------------
LEYE, REYE, LSH, RSH, LHIP, RHIP, HEAD, NECK, NOSE = range(9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_head_pose_matrix_from_selected(selected9: np.ndarray):
    LE, RE, HD, NK = selected9[LEYE], selected9[REYE], selected9[HEAD], selected9[NECK]
    eye_center = (LE + RE) / 2.0
    head_center = (HD + NK) / 2.0

    x_axis = LE - RE
    x_axis = x_axis / np.linalg.norm(x_axis)

    v = eye_center - head_center
    v_proj = v - np.dot(v, x_axis) * x_axis
    z_axis = v_proj / np.linalg.norm(v_proj)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R, eye_center


def make_fpv_transform(R_head: np.ndarray, eye_center: np.ndarray):
    T = np.eye(4, dtype=np.float32)
    Rt = R_head.T
    T[:3, :3] = Rt
    T[:3, 3] = -Rt @ eye_center
    return T


def transform_to_fpv(point: np.ndarray, T: np.ndarray):
    ph = np.append(point, 1.0)
    qh = T @ ph
    return qh[:3]


def gt_angles_only(selected9: np.ndarray, obj_center: np.ndarray):
    R_head, eye_center = get_head_pose_matrix_from_selected(selected9)
    T = make_fpv_transform(R_head, eye_center)
    obj_fpv = transform_to_fpv(obj_center, T)
    x, y, z = float(obj_fpv[0]), float(obj_fpv[1]), float(obj_fpv[2])
    horiz = np.degrees(np.arctan2(x, z))
    vert = np.degrees(np.arctan2(y, z))
    return np.array([horiz, vert], dtype=np.float32)

# --------------------------
# JSON I/O
# ---------------------------
def load_inputs(path: str):
    with open(path, "r") as f:
        return json.load(f)

# ---------------------------
# PARTICIPANT-LEVEL FOLDS
# ---------------------------
def build_participant_folds(participants: List[str], n_folds: int, seed: int) -> List[List[str]]:
    participants = sorted([p for p in participants if p is not None and str(p).strip() != ""])
    rng = np.random.RandomState(seed)
    participants = participants.copy()
    rng.shuffle(participants)
    folds = [list(arr) for arr in np.array_split(participants, n_folds)]
    return folds


def get_train_test_participants_from_folds(folds: List[List[str]], fold_idx: int):
    test_participants = folds[fold_idx]
    train_participants = []
    for i, fold in enumerate(folds):
        if i != fold_idx:
            train_participants.extend(fold)
    return train_participants, test_participants


# ---------------------------
# Helpers for stored normalized features
# ---------------------------
def extract_five_tokens_from_stored_norm(norm_dict):
    """
    Reuse stored normalized body keypoints from annotated_inputs.json.

    Expected keys:
      nose, LEye, REye, LShoulder, RShoulder

    Returns:
      flat (15,) in the exact order used by old model:
      [nose, LEye, REye, LShoulder, RShoulder]
    """
    needed = ["nose", "LEye", "REye", "LShoulder", "RShoulder"]
    arrs = []
    for k in needed:
        if k not in norm_dict:
            raise KeyError(f"Missing stored norm key: {k}")
        v = np.asarray(norm_dict[k], dtype=np.float32).reshape(3)
        if not np.isfinite(v).all():
            raise ValueError(f"Non-finite stored norm for key: {k}")
        arrs.append(v)

    return np.stack(arrs, axis=0).reshape(-1).astype(np.float32)


# ---------------------------
# Dataset builder
# ---------------------------
def collect_role_from_annotated(basedir, participants, role, save_records=False, require_gt=True):
    """
    Uses:
        {basedir}/{participant}/data/inputs/{participant}_annotated_inputs.json

    Reuses existing info from JSON:
      - pitch_{role}
      - {role}_norm
      - {role}_attended_center

    Still recomputes target Y:
      - GT FPV angles from joints + stored attended center
    """
    Xf, Y = [], []
    kept = miss = bad = 0
    records = [] if save_records else None

    for pid in participants:
        pth = os.path.join(
            basedir,
            pid,
            "data",
            "inputs",
            f"{pid}_annotated_inputs.json"
        )
        if not os.path.exists(pth):
            print(f"[{role}] inputs file not found for {pid}: {pth}")
            continue

        data = load_inputs(pth)
        pitch_key = f"pitch_{role}"
        sel_key = f"{role}_joints"
        ctr_key = f"{role}_attended_center"
        norm_key = f"{role}_norm"

        for fid, frame in data.items():

            if pitch_key not in frame or sel_key not in frame or norm_key not in frame:
                miss += 1
                continue

            if frame[pitch_key] is None:
                miss += 1
                continue
            pitch = float(frame[pitch_key])


            if not np.isfinite(pitch):
                bad += 1
                continue

            if frame[sel_key] is None:
                miss += 1
                continue
            sel11 = np.asarray(frame[sel_key], dtype=np.float32)

            if sel11.shape != (11, 3) or not np.isfinite(sel11).all():
                bad += 1
                continue

            sel9 = sel11[:9].astype(np.float32)

            # use stored attended center directly if available
            stored_ctr = frame.get(ctr_key, None)
            ctr = None
            has_gt = False

            if stored_ctr is None:
                if require_gt:
                    miss += 1
                    continue
            else:
                ctr = np.asarray(stored_ctr, dtype=np.float32).reshape(3)
                if ctr.shape != (3,) or not np.isfinite(ctr).all():
                    if require_gt:
                        bad += 1
                        continue
                    ctr = None
                else:
                    has_gt=True

            # use stored normalized body keypoints directly
            stored_norm = frame.get(norm_key, None)
            if not isinstance(stored_norm, dict):
                miss += 1
                continue

            five_tokens_flat = extract_five_tokens_from_stored_norm(stored_norm)
            if five_tokens_flat is None:
                bad += 1
                continue

            # still compute GT target angles only if GT exists
            if has_gt:
                y_deg = gt_angles_only(sel9, ctr)
                if not np.isfinite(y_deg).all():
                    if require_gt:
                        bad += 1
                        continue
                    y_deg = np.array([np.nan, np.nan], dtype=np.float32)
            else:
                y_deg = np.array([np.nan, np.nan], dtype=np.float32)

            flat = np.concatenate([
                np.array([pitch], dtype=np.float32),
                five_tokens_flat
            ])

            Xf.append(flat)
            Y.append(y_deg.astype(np.float32))
            kept += 1

            if save_records:
                records.append({
                    "participant": pid,
                    "frame_id": int(fid) if str(fid).isdigit() else fid,
                    "role": role,
                    "pitch": float(pitch),
                    "attended_center": ctr.astype(np.float32).tolist() if ctr is not None else None,
                    "stored_norm": {
                        "nose": np.asarray(stored_norm["nose"], dtype=np.float32).tolist(),
                        "LEye": np.asarray(stored_norm["LEye"], dtype=np.float32).tolist(),
                        "REye": np.asarray(stored_norm["REye"], dtype=np.float32).tolist(),
                        "LShoulder": np.asarray(stored_norm["LShoulder"], dtype=np.float32).tolist(),
                        "RShoulder": np.asarray(stored_norm["RShoulder"], dtype=np.float32).tolist(),
                    },
                    "gt_angles_deg": y_deg.astype(np.float32).tolist(),
                    "features_flat_16": flat.astype(np.float32).tolist(),
                })

    print(f"[{role}] participants={len(participants)} | kept: {kept} | dropped(missing): {miss} | dropped(bad): {bad}")

    if kept == 0:
        return (None, None, []) if save_records else (None, None)

    if save_records:
        return np.stack(Xf).astype(np.float32), np.stack(Y).astype(np.float32), records
    return np.stack(Xf).astype(np.float32), np.stack(Y).astype(np.float32)

class TinyMLP(nn.Module):
    def __init__(self, in_dim=16, hidden=20, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def stats_flat(X: np.ndarray, eps: float = 1e-6):
    m = X.mean(axis=0).astype(np.float32)
    s = X.std(axis=0).astype(np.float32)
    s[s < eps] = eps
    return m, s


def norm_flat(X: np.ndarray, m: np.ndarray, s: np.ndarray):
    return (X - m) / s


class SmoothL1Circular(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    @staticmethod
    def short_diff_deg(pred, target):
        d = pred - target
        d = (d + 180.0) % 360.0 - 180.0
        return d

    def forward(self, pred_deg, target_deg):
        d = self.short_diff_deg(pred_deg, target_deg)
        absd = torch.abs(d)
        beta = self.beta
        loss = torch.where(absd < beta, 0.5 * (d ** 2) / beta, absd - 0.5 * beta)
        return loss.mean()


# ---------------------------
# Metrics
# ---------------------------
def short_diff_deg_np(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def compute_metrics(pred, gt):
    d = short_diff_deg_np(pred, gt)
    rmse_all = float(np.sqrt((d ** 2).mean()))
    mae_all = float(np.abs(d).mean())
    rmse_h = float(np.sqrt((d[:, 0] ** 2).mean()))
    rmse_v = float(np.sqrt((d[:, 1] ** 2).mean()))

    zeros = np.zeros_like(gt)
    db = short_diff_deg_np(zeros, gt)
    base_rmse_all = float(np.sqrt((db ** 2).mean()))
    base_rmse_h = float(np.sqrt((db[:, 0] ** 2).mean()))
    base_rmse_v = float(np.sqrt((db[:, 1] ** 2).mean()))

    return {
        "rmse_all": rmse_all,
        "mae_all": mae_all,
        "rmse_h": rmse_h,
        "rmse_v": rmse_v,
        "baseline_rmse_all": base_rmse_all,
        "baseline_rmse_h": base_rmse_h,
        "baseline_rmse_v": base_rmse_v,
    }


# ---------------------------
# Training / checkpoint I/O
# ---------------------------
def checkpoint_paths(checkpoint_root, fold_idx, role):
    fold_dir = os.path.join(checkpoint_root, f"fold_{fold_idx:02d}")
    model_path = os.path.join(fold_dir, f"tinymlp_{role}.pt")
    return fold_dir, model_path


def train_until_converged(
    Xf_tr, y_tr,
    checkpoint_root, role, fold_idx,
    lr=1e-3, min_lr=1e-7, batch_size=32, seed=42,
    weight_decay=1e-4, max_epochs=2000,
    target_train_rmse_deg=0.8, min_epochs=10,
    early_stop_patience=15, sched_patience=5
):
    f_m, f_s = stats_flat(Xf_tr)
    Xf_tr_n = norm_flat(Xf_tr, f_m, f_s)

    tr_ds = TensorDataset(torch.tensor(Xf_tr_n), torch.tensor(y_tr))
    g = torch.Generator().manual_seed(seed)
    tr_dl = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
    )

    model = TinyMLP(in_dim=16, hidden=20, out_dim=2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=sched_patience,
        min_lr=min_lr,
    )
    loss_ang = SmoothL1Circular(beta=0.5)

    best_state = None
    best_loss = float("inf")
    no_improve = 0
    epoch = 0
    eps = 1e-6

    while True:
        model.train()
        train_err_accum = 0.0
        n_train_err = 0
        train_loss_sum = 0.0
        n_train_samples = 0

        for xb_f, yb_deg in tr_dl:
            xb_f = xb_f.to(device)
            yb_deg = yb_deg.to(device)

            opt.zero_grad()
            out = model(xb_f)
            loss = loss_ang(out, yb_deg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            with torch.no_grad():
                d = (out - yb_deg + 180.0) % 360.0 - 180.0
                train_err_accum += (d ** 2).sum().item()
                n_train_err += d.numel()
                train_loss_sum += loss.item() * len(xb_f)
                n_train_samples += len(xb_f)

        train_rmse = float(np.sqrt(train_err_accum / max(1, n_train_err)))
        train_loss_mean = float(train_loss_sum / max(1, n_train_samples))
        sched.step(train_loss_mean)

        if train_loss_mean < best_loss - eps:
            print(
                f"[fold {fold_idx:02d}][{role}] train loss improved: "
                f"{best_loss:.4f} -> {train_loss_mean:.4f}, RMSE deg={train_rmse:.3f}"
            )
            best_loss = train_loss_mean
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        epoch += 1

        if (epoch >= min_epochs) and (no_improve >= early_stop_patience):
            print(f"[fold {fold_idx:02d}][{role}] early stop after {epoch} epochs.")
            break
        if (epoch >= min_epochs) and (train_rmse <= target_train_rmse_deg):
            print(
                f"[fold {fold_idx:02d}][{role}] early stop: train RMSE deg "
                f"{train_rmse:.3f} <= {target_train_rmse_deg:.3f}."
            )
            break
        if epoch >= max_epochs:
            print(f"[fold {fold_idx:02d}][{role}] hit max_epochs={max_epochs}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


    checkpoint_fold_dir, model_path = checkpoint_paths(
        checkpoint_root, fold_idx, role
    )
    os.makedirs(checkpoint_fold_dir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": torch.from_numpy(f_m),
        "feature_std": torch.from_numpy(f_s),
        "in_dim": 16,
        "hidden_dim": 20,
        "out_dim": 2,
        "fold_idx": fold_idx,
        "role": role,
    }, model_path)

    print(f"[fold {fold_idx:02d}][{role}] checkpoint saved: {model_path}")
    return model, (f_m, f_s)

def load_model_and_scaler(checkpoint_root, fold_idx, role):
    _, model_path = checkpoint_paths(checkpoint_root, fold_idx, role)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=device
        )

    model = TinyMLP(
        in_dim=checkpoint["in_dim"],
        hidden=checkpoint["hidden_dim"],
        out_dim=checkpoint["out_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = (
        checkpoint["feature_mean"].cpu().numpy().astype(np.float32),
        checkpoint["feature_std"].cpu().numpy().astype(np.float32),
    )

    return model, scaler
# ---------------------------
# Evaluation
# ---------------------------
def empty_metrics():
    return {
        "rmse_all": float("nan"),
        "mae_all": float("nan"),
        "rmse_h": float("nan"),
        "rmse_v": float("nan"),
        "baseline_rmse_all": float("nan"),
        "baseline_rmse_h": float("nan"),
        "baseline_rmse_v": float("nan"),
    }


def metrics_with_valid_gt(pred, gt):
    valid_mask = np.isfinite(gt).all(axis=1)
    if valid_mask.sum() == 0:
        return empty_metrics()
    return compute_metrics(pred[valid_mask], gt[valid_mask])


def evaluate_model(model, scaler, Xf_te, y_te):
    f_m, f_s = scaler
    Xn = norm_flat(Xf_te, f_m, f_s)

    model.eval()
    with torch.no_grad():
        pred = model(
            torch.as_tensor(Xn, dtype=torch.float32, device=device)
        ).cpu().numpy()

    metrics = metrics_with_valid_gt(pred, y_te)
    return pred, metrics

def print_test_metrics(prefix, metrics):
    print(prefix)
    print(f"  RMSE (horizontal) : {metrics['rmse_h']:.3f} deg")
    print(f"  RMSE (vertical)   : {metrics['rmse_v']:.3f} deg")
    print(f"  RMSE (overall)    : {metrics['rmse_all']:.3f} deg")
    print(f"  MAE  (overall)    : {metrics['mae_all']:.3f} deg")
    print(f"  Baseline RMSE H   : {metrics['baseline_rmse_h']:.3f} deg")
    print(f"  Baseline RMSE V   : {metrics['baseline_rmse_v']:.3f} deg")
    print(f"  Baseline RMSE All : {metrics['baseline_rmse_all']:.3f} deg")


# ---------------------------
# Output helpers
# ---------------------------
def save_per_participant_npzs(fold_dir, role, records, pred, gt):
    per_pid = defaultdict(lambda: {
        "frames": [],
        "gt_h": [],
        "gt_v": [],
        "pred_h": [],
        "pred_v": [],
    })

    for rec, pred_row, gt_row in zip(records, pred, gt):
        pid = rec["participant"]
        fid = rec["frame_id"]

        per_pid[pid]["frames"].append(int(fid) if str(fid).isdigit() else fid)
        per_pid[pid]["gt_h"].append(float(gt_row[0]))
        per_pid[pid]["gt_v"].append(float(gt_row[1]))
        per_pid[pid]["pred_h"].append(float(pred_row[0]))
        per_pid[pid]["pred_v"].append(float(pred_row[1]))

    for pid, d in per_pid.items():
        out_npz = os.path.join(
            fold_dir,
            f"{role}_{pid}_fullvideo_angles_pred_gt_angles.npz",
        )
        np.savez(
            out_npz,
            frames=np.asarray(d["frames"]),
            gt_h=np.asarray(d["gt_h"], dtype=np.float32),
            gt_v=np.asarray(d["gt_v"], dtype=np.float32),
            pred_h=np.asarray(d["pred_h"], dtype=np.float32),
            pred_v=np.asarray(d["pred_v"], dtype=np.float32),
        )


def save_train_and_test_outputs(
    run_dir, fold_idx, role,
    rec_tr, rec_te,
    pred_tr, y_tr,
    pred_te, y_te,
):
    fold_dir = os.path.join(run_dir, f"fold_{fold_idx:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    save_per_participant_npzs(fold_dir, role, rec_tr, pred_tr, y_tr)
    save_per_participant_npzs(fold_dir, role, rec_te, pred_te, y_te)

    np.savez(
        os.path.join(fold_dir, f"{role}_train_predictions.npz"),
        pred=pred_tr.astype(np.float32),
        gt=y_tr.astype(np.float32),
    )
    np.savez(
        os.path.join(fold_dir, f"{role}_test_predictions.npz"),
        pred=pred_te.astype(np.float32),
        gt=y_te.astype(np.float32),
    )

    with open(os.path.join(fold_dir, f"{role}_train_records.json"), "w") as f:
        json.dump(rec_tr, f, indent=2)

    with open(os.path.join(fold_dir, f"{role}_test_records.json"), "w") as f:
        json.dump(rec_te, f, indent=2)

def make_summary_base(args, participants, folds, run_name):
    return {
        "run_name": run_name,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "participants": participants,
        "participant_folds": folds,
        "folds": [],
    }


def add_overall_role_metrics(summary, role, role_all_pred, role_all_gt):
    if len(role_all_pred) == 0:
        return

    pred_all = np.concatenate(role_all_pred, axis=0)
    gt_all = np.concatenate(role_all_gt, axis=0)
    overall_metrics = metrics_with_valid_gt(pred_all, gt_all)

    print(f"\n========== OVERALL  CROSS-VALIDATION RESULTS | {role.upper()} ==========")
    print_test_metrics("", overall_metrics)

    summary[f"overall_{role}"] = overall_metrics


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basedir", type=str, required=True, help="Root directory containing the processed participant folders.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=5000)
    ap.add_argument("--target_train_rmse_deg", type=float, default=0.8)
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--sched_patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    basedir = args.basedir
    processed_dir = os.path.join(basedir, "processed_data")
    out_dir = os.path.join(
        basedir,
        "model_outputs",
        "tinymlp"
    )
    checkpoint_root = os.path.join(basedir, "checkpoints")
    predictions_dir = os.path.join(out_dir, "predictions")

    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    participants = []

    for participant in os.listdir(basedir):
        input_path = os.path.join(
            basedir,
            participant,
            "data",
            "inputs",
            f"{participant}_annotated_inputs.json"
        )

        if os.path.isfile(input_path):
            participants.append(participant)

    participants = sorted(participants)

    if len(participants) == 0:
        raise RuntimeError(
            f"No participant annotated input files found under {basedir}"
        )

    folds = build_participant_folds(
        participants,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    summary = make_summary_base(
        args, participants, folds, "tinymlp"
    )

    for role in ["child", "parent"]:
        print(f"\n==================== ROLE: {role.upper()} ====================")
        role_all_pred = []
        role_all_gt = []

        for fold_idx in range(len(folds)):
            train_parts, test_parts = get_train_test_participants_from_folds(
                folds, fold_idx
            )

            print(f"\n-----fold {fold_idx:02d} | role={role} -----")
            print(f"train participants: {len(train_parts)}")
            print(f"test participants : {len(test_parts)}")

            Xf_tr, y_tr, rec_tr = collect_role_from_annotated(
                basedir,
                train_parts,
                role,
                save_records=True,
                require_gt=True,
            )
            Xf_te, y_te, rec_te = collect_role_from_annotated(
                basedir,
                test_parts,
                role,
                save_records=True,
                require_gt=False,
            )

            if Xf_tr is None or Xf_te is None:
                print(
                    f"[fold {fold_idx:02d}][{role}] skipped because "
                    "train or test is empty."
                )
                continue

            model, scaler = train_until_converged(
                Xf_tr,
                y_tr,
                checkpoint_root=checkpoint_root,
                role=role,
                fold_idx=fold_idx,
                lr=args.lr,
                min_lr=args.min_lr,
                batch_size=args.batch_size,
                seed=args.seed,
                weight_decay=args.weight_decay,
                max_epochs=args.max_epochs,
                target_train_rmse_deg=args.target_train_rmse_deg,
                min_epochs=args.min_epochs,
                early_stop_patience=args.early_stop_patience,
                sched_patience=args.sched_patience,
            )

            pred_tr, metrics_tr = evaluate_model(model, scaler, Xf_tr, y_tr)
            pred_te, metrics_te = evaluate_model(model, scaler, Xf_te, y_te)

            save_train_and_test_outputs(
                predictions_dir,
                fold_idx,
                role,
                rec_tr,
                rec_te,
                pred_tr,
                y_tr,
                pred_te,
                y_te,
            )

            print_test_metrics(
                f"[fold {fold_idx:02d}][{role}] TEST RESULTS",
                metrics_te,
            )

            summary["folds"].append({
                "fold_idx": fold_idx,
                "role": role,
                "train_participants": train_parts,
                "test_participants": test_parts,
                "n_train_samples": int(len(Xf_tr)),
                "n_test_samples": int(len(Xf_te)),
                "metrics_train": metrics_tr,
                "metrics_test": metrics_te,
            })

            role_all_pred.append(pred_te)
            role_all_gt.append(y_te)

        add_overall_role_metrics(
            summary, role, role_all_pred, role_all_gt
        )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved CV summary: {summary_path}")

if __name__ == "__main__":
    main()
