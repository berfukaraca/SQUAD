import os
import cv2
import json
import pandas as pd
import numpy as np
import pickle
from bisect import bisect_left

# return an empty list with nans
def add_nans(n):
    return [np.nan for i in range(n)]

# return the index of the value closest to the given value in input list
def get_closest_index(ref_list, value):
    pos = bisect_left(ref_list, value)
    if pos == 0:
        return 0
    if pos == len(ref_list):
        return -1
    before_value = ref_list[pos - 1]
    after_value = ref_list[pos]
    if after_value - value < value - before_value:
        return pos
    else:
        return pos - 1

# compute Euclidean distance (displacement) between two points
def get_disp(xyc1, xyc2):
    x1, y1, _ = xyc1
    x2, y2, _ = xyc2
    return abs(np.sqrt((x1 - x2)**2 + (y1 - y2)**2))

# rearrange keypoints
def rearrange_kps(keypoints, kp_indices):
    """
    Rearranges keypoints from format [x, y, c, x, y, c, ...]
    to format [(x, y), (x, y), ...].
    """
    keypoints = list(zip(*[iter(keypoints)]*3))
    keypoints = [
        (x, y, c) for idx, (x, y, c) in enumerate(keypoints)
        if idx in kp_indices
    ]
    return keypoints

# filter out frame level data when too many overlapping keypoints
def filter_double_dets(dets, kp_indices, **kwargs):
    """
    Filtering based on displacement threshold (disp_thres) and 
    number of accepted keypoints above displacement threshold (kp_thres).
    """
    kps = [rearrange_kps(kps, kp_indices) for *_, kps, _ in dets]
    disps = [get_disp(xyc1, xyc2) for xyc1, xyc2 in zip(*kps)]
    filt_disps = [
        disp for disp in disps
        if disp < kwargs["duplicate_displacement_threshold"]
    ]
    if len(filt_disps) >= kwargs["n_duplicate_keypoints_threshold"]:
        confs = [np.median([c for *_, c in xyc]) for xyc in kps]
        best_idx = confs.index(max(confs))
        dets = [
            det if idx == best_idx else [] for idx, det in enumerate(dets)
        ]
        print("done filtered")
    return dets

# return assigned detections from a list of detections based id label list
def get_assigned_dets(dets, id_labels):
    assigned_det = [
        [frame_nr, bbox, kp, assigned_id]
        for frame_nr, bbox, kp, assigned_id in dets
        if assigned_id in id_labels
    ]
    return assigned_det

# compute displacement between two sets of keypoints of selected indices
def get_kp_disp(old_kps, new_kps, kp_indices):
    """Returns a list with displacement between keypoints for each index."""
    old_kps = rearrange_kps(old_kps, kp_indices)
    new_kps = rearrange_kps(new_kps, kp_indices)
    kps_disp = [
        get_disp(xyc_old, xyc_new)
        for xyc_old, xyc_new
        in zip(old_kps, new_kps)
    ]
    return kps_disp

# match track keypoints to detections based on a function
def match_kps(dets, tr_kps, kp_indices, match_func):
    """
    Takes in a list of detections, a list of tracklet keypoints, 
    a list of keypoint indices, and a matching function. Returns a list
    containing the index of the best tracklet match, median displacement
    to the keypoints of the best match, and full detection data.
    """
    det_matches = []
    for det in dets:
        _, _, kps, _ = det
        kps_disps = [
            [tr_idx, np.median(match_func(tr_kps, kps, kp_indices)), det]
            for tr_idx, tr_kps in enumerate(tr_kps)
            if tr_kps
        ]
        if kps_disps:
            det_matches.append(min(kps_disps, key=lambda x: x[1]))
    return det_matches

# compute spread within a set of keypoints of selected indices
def get_kp_spread(kps, kp_indices):
    """
    Returns the median spread of one set of keypoints of selected indices by
    first taking the median displacement between each keypoint and all other
    keypoints and then computing the median of that across all keypoints.
    
    """
    kps = rearrange_kps(kps, kp_indices)
    kps_spreads = []
    for xyc_ref in kps:
        kps_spread = [
            get_disp(xyc_ref, xyc_comp)
            for xyc_comp
            in [xy for xy in kps if not xy == xyc_ref]
        ]
        kps_spreads.append(np.median(kps_spread))
    return np.median(kps_spreads)

# start new tracklet
def tracklet_start(tracklets, dets, id_labels):
    """
    Needs a list of tracklets (can be empty), a list of detections,
    and a list of id labels. If at least one assigned detection exists,
    ends any previous tracklets and starts a new list of tracklets
    by appending the assigned detections to the list of tracklets
    and returning appended the list of tracklets.
    """
    assigned_dets = [
        [assigned_det]
        for assigned_det
        in get_assigned_dets(dets, id_labels)
    ]
    if assigned_dets:
        tracklets.append(assigned_dets)
        if len(assigned_dets) < 2:
            tracklets[-1].append([])
    return tracklets

# continue tracklet
def tracklet_continue(tracklets, dets, id_labels, kp_indices, **kwargs):
    """
    Needs a list of tracklets (cannot be empty), a list of detections,
    a list of id labels, and a list of keypoint indices. First looks for any
    assigned detections and matches them to the existing tracklets. If any
    unmatched tracklets remain, looks for matches in unassigned detections.
    Once matches have been determined, checks whether the tracklets are below
    a displacement threshold and whether the frame difference to
    the tracklets exceeds a frame threshold. If matched tracklets
    remain, appends them to the current tracklet. If no matched tracklets
    remain but assigned detections exist, starts a new tracklet with them.
    """
    assigned_dets = get_assigned_dets(dets, id_labels)
    tr_latest = [
        tr[-1] if tr else [[] for n in range(4)] for tr in tracklets[-1]
    ]
    frame_diffs = [
        abs(tr_fr - min([det_fr for det_fr, *_ in dets]))
        for tr_fr, *_ in tr_latest if tr_fr
    ]
    tr_kps = [kps for *_, kps, _ in tr_latest]
    assigned_matches = match_kps(
        assigned_dets, tr_kps, kp_indices, get_kp_disp)
    unmatched_tracklets = [
        track_idx for track_idx in [0, 1]
        if not track_idx in [idx for idx, *_ in assigned_matches]
    ]
    remaining_dets = [
        det for det in dets
        if not det in [det for *_, det in assigned_matches]
    ]
    remaining_matches = [
        [match_idx, disp, det] for match_idx, disp, det in match_kps(
            remaining_dets, tr_kps, kp_indices, get_kp_disp)
        if match_idx in unmatched_tracklets
    ]
    all_matches = assigned_matches + remaining_matches
    tr1_matches = [
        [match_idx, disp, det] for match_idx, disp, det in all_matches
        if match_idx == 0 if disp < kwargs["tracklet_displacement_threshold"]
    ]
    tr2_matches = [
        [match_idx, disp, det] for match_idx, disp, det in all_matches
        if match_idx == 1 if disp < kwargs["tracklet_displacement_threshold"]
    ]
    matched_dets = []
    matched_ids = []
    for idx, matches in enumerate([tr1_matches, tr2_matches]):
        if matches:
            *_, matched_det = min(matches, key=lambda x: x[1])
            *_, matched_id = matched_det
            matched_dets.append(matched_det)
            matched_ids.append(matched_id)
        else:
            matched_dets.append([])
    if not all(tracklets[-1]) and matched_dets and not all(matched_dets):
        remaining_assigned_dets = [
            [frame_nr, bbox, kps, assigned_id]
            for frame_nr, bbox, kps, assigned_id in assigned_dets
            if not assigned_id in matched_ids
        ]
        if remaining_assigned_dets:
            remaining_assigned_match = min(
                remaining_assigned_dets, key=lambda x: x[1])
            matched_dets = [
                matched_det if matched_det else remaining_assigned_match
                for matched_det in matched_dets
            ]
            spread1, spread2 = [
                get_kp_spread(kps, kp_indices) for *_, kps, _ in matched_dets
            ]
            if abs(spread1 - spread2) < kwargs["min_tracklet_spread_diff"]:
                matched_dets = [
                    matched_det
                    if not matched_det == remaining_assigned_match else []
                    for matched_det in matched_dets
                ]
    if all(matched_dets):
        matched_dets = filter_double_dets(matched_dets, kp_indices, **kwargs)
    if any(diff >= kwargs["tracklet_nan_threshold"] for diff in frame_diffs):
        if any(matched_dets):
            debug_info = "tracklet start with matched dets"
            tracklets.append([])
            for matched_det in matched_dets:
                if matched_det:
                    tracklets[-1].append([matched_det])
                else:
                    tracklets[-1].append([])
        elif assigned_dets:
            debug_info = "tracklet start with assigned dets"
            tracklets = tracklet_start(tracklets, dets, id_labels)
        else:
            debug_info = "frame data skipped, no matches or assigned dets"
    else:
        debug_info = f"tracklet continue with matched dets"
        for idx, matched_det in enumerate(matched_dets):
            if matched_det:
                tracklets[-1][idx].append(matched_det)
    if kwargs["debug_range"]:
        debug_start, debug_end = kwargs["debug_range"]
        det_frs = [det_fr for det_fr, *_ in dets]
        if any(debug_start <= fr < debug_end for fr in det_frs):
            print(
                f"---\ntracklet frames {det_frs} debug info:\n"
                f"decision: {debug_info}\n"
                f"{[i[:2] for i in all_matches]}\n"
                f"final matches: "
                f"{[idx for idx, det in enumerate(matched_dets) if det]}")
    return tracklets

# create tracklets from frame-level pose data
def create_tracklets(pose_data, id_labels, kp_indices, **kwargs):
    """
    Needs a list with frame-level pose data, a list of id labels, and a list
    of keypoint indices. Loops through pose data and looks for detections to
    start and continue tracklets (i.e., detections linked across frames)
    from them. Returns final list of tracklets.
    """
    tracklets = []
    for p in pose_data:
        frame_nr = int(p["image_id"].replace(".jpg", ""))
        frame_data = p["output_results"]
        bboxes = [i["box"] for i in frame_data]
        keypoints = [i["keypoints"] for i in frame_data]
        assigned_ids = [i["classification"] for i in frame_data]
        if frame_data:
            dets = [
                [frame_nr, [x, y, x + w, y + h], kp, assigned_id]
                for (x, y, w, h), kp, assigned_id
                in zip(bboxes, keypoints, assigned_ids)
            ]
            if dets:
                if tracklets:
                    tracklets = tracklet_continue(
                        tracklets,
                        dets,
                        id_labels,
                        kp_indices,
                        **kwargs)
                else:
                    tracklets = tracklet_start(
                        tracklets,
                        dets,
                        id_labels)
    return tracklets

# groups tracklets to form person tracks
def group_tracklets(tracklets, kp_indices, **kwargs):
    """
    Takes in a list of tracklets and a list of keypoint indices. First loops
    through all tracklets and computes the median within-keypoints spread for
    each. Then loops through all tracklets with two people detected, computes
    the difference between their respective spreads, retaining those that are
    long enough and have a large enough difference in within-keypoint spread
    between them (create_tracklets settings). Retained tracklets are ordered
    with the tracklet that has smaller within-keypoints spread first, forming
    an initial set of two tracks. Then computes the median within-keypoints
    spread for each track and assigns any remaining tracklets with only one
    person detected to the tracks based on difference between tracklet spread
    and median track spread. Returns the list of assigned tracklets.
    """
    tracks = []
    for tr1, tr2 in tracklets:
        kps_spreads = []
        for tracklet in [tr1, tr2]:
            if tracklet:
                kps_spread = [
                    get_kp_spread(kps, kp_indices) for *_, kps, _ in tracklet
                ]
                tr_start, *_ = tracklet[0]
                mid_point = tr_start + len(tracklet) // 2
                kps_spreads.append(
                    [np.median(kps_spread), tracklet, mid_point])
            else:
                kps_spreads.append([])
        tracks.append(kps_spreads)
    assigned_tracklets = []
    unassigned_tracklets = []
    for tr1, tr2 in tracks:
        if tr1 and tr2:
            tr1_spread, *_ = tr1
            tr2_spread, *_ = tr2
            tr_filt = [
                [spread, tracklet, mid_point]
                for spread, tracklet, mid_point in [tr1, tr2]
                if len(tracklet) > kwargs["min_tracklet_dur"]
            ]
            if len(tr_filt) == 2:
                spread_diff = abs(tr1_spread - tr2_spread)
                if spread_diff > kwargs["min_tracklet_spread_diff"]:
                    assigned_tracklets.append(
                        sorted([tr1, tr2], key=lambda x: x[0]))
            elif tr_filt:
                unassigned_tracklets.append(tr_filt[0])
        else:
            if tr1:
                unassigned_tracklets.append(tr1)
            elif tr2:
                unassigned_tracklets.append(tr2)
    tr1_assigned = [tr for tr, _ in assigned_tracklets]
    tr2_assigned = [tr for _, tr in assigned_tracklets]
    tr1_spreads = [spread for spread, *_ in tr1_assigned]
    tr2_spreads = [spread for spread, *_ in tr2_assigned]
    tr1_mid_points = [mid_point for *_, mid_point in tr1_assigned]
    tr2_mid_points = [mid_point for *_, mid_point in tr2_assigned]
    for tracklet in unassigned_tracklets:
        tr_spread, _, tr_mid_point = tracklet
        tr1_closest_idx = get_closest_index(tr1_mid_points, tr_mid_point)
        tr2_closest_idx = get_closest_index(tr2_mid_points, tr_mid_point)
        ref_spreads = [
            tr1_spreads[tr1_closest_idx], tr2_spreads[tr2_closest_idx]
        ]
        spread_diffs = [
            abs(tr_spread - ref_spread) for ref_spread in ref_spreads
        ]
        spread_match = spread_diffs.index(min(spread_diffs))
        if spread_match == 0:
            assigned_tracklets.append([tracklet, []])
        else:
            assigned_tracklets.append([[], tracklet])
    if kwargs["debug_range"]:
        print("---\ngroup_tracklets debug info:")
        for tracklets in assigned_tracklets:
            print("---")
            for idx, tr in enumerate(tracklets):
                if tr:
                    tr_spread, tr_data, _ = tr
                    tr_start_frame, *_ = tr_data[0]
                    tr_end_frame, *_ = tr_data[-1]
                    print(
                        f"matched to track {idx + 1}, "
                        f"spread {tr_spread:.2f}, "
                        f"from {tr_start_frame}, to {tr_end_frame}")
    return assigned_tracklets

# flatten list of tracklets
def flatten_tracklets(tracklets):
    """
    Takes in a list of tracks (grouped tracklets given by group_tracklets),
    and flattens them to separate lists of tracks, ordered by frame number
    and with the structure [frame_nr, bbox, keypoints, assigned_id].
    """
    track1 = []
    track2 = []
    for tr1, tr2 in tracklets:
        _, tr1_tracklet, _ = tr1 if tr1 else [_, [], _]
        _, tr2_tracklet, _ = tr2 if tr2 else [_, [], _]
        if tr1_tracklet:
            track1.extend(tr1_tracklet)
        if tr2_tracklet:
            track2.extend(tr2_tracklet)
    track1 = sorted(track1, key=lambda x: x[0])
    track2 = sorted(track2, key=lambda x: x[0])
    return track1, track2

# fill in missing frames in track
def add_empty_values(track, frame_range):
    """
    Fills in missing frames for track with NaN values for bbox,
    keypoints, and assigned_id based on a given frame range.
    """
    filled_track = []
    first_frame, last_frame = frame_range
    for idx, (frame_nr, bbox, keypoints, assigned_id) in enumerate(track):
        if idx == 0:
            if not frame_nr == 1:
                for i in range(first_frame, frame_nr):
                    filled_track.append(
                        [i, add_nans(4), add_nans(78), np.nan])
        if idx < len(track) - 1:
            next_frame_nr = track[idx + 1][0]
            frame_difference = next_frame_nr - frame_nr
            if frame_difference == 1:
                filled_track.append([frame_nr, bbox, keypoints, assigned_id])
            else:
                for i in range(frame_difference):
                    filled_track.append(
                        [frame_nr + i, add_nans(4), add_nans(78), np.nan])
        elif idx == len(track) - 1:
            frame_difference = last_frame - frame_nr
            for i in range(frame_difference):
                filled_track.append(
                    [frame_nr + i, add_nans(4), add_nans(78), np.nan])
        else:
            filled_track.append([frame_nr, bbox, keypoints, assigned_id])
    return filled_track

# interpolate given track
def run_interpolate(track, lim):
    """
    Interpolates each column of the track separately using pandas.
    For now set to use nearest-neigbor interpolation. The interpolation
    limit (set by lim) determines how many consecutive NaN values to
    interpolate. Returns the interpolated track with original format.
    """
    interpolated_track = []
    for j in range(len(track[0])):
        track_col = [i[j] for i in track]
        track_col = pd.Series(track_col)
        track_col.interpolate(
            method="nearest",
            limit=lim,
            limit_direction="both",
            limit_area="inside",
            inplace=True)
        interpolated_track.append(track_col)
    return interpolated_track

# interpolate bbox and keypoint tracks
def interpolate_track(track, **kwargs):
    """
    Interpolate the bbox and keypoint tracks of a given person track by
    passing it the run_interpolate function. Returns the interpolated track
    in its original format.
    """
    bboxes = [bbox for _, bbox, *_ in track]
    keypoints = [kps for *_, kps, _ in track]
    assigned_ids = [assigned_id for *_, assigned_id in track]
    output_track = [[frame_nr for frame_nr, *_ in track]]
    for tr in [bboxes, keypoints]:
        interpolated_track = run_interpolate(
            tr, kwargs["interpolation_limit"])
        output_track.append(list(zip(*interpolated_track)))
    output_track.append(assigned_ids)
    return list(zip(*output_track))

# create person tracks by running all the necessary steps
def create_all_tracks(
        video_dir, data_dir, kp_indices=[kp for kp in range(20)]):
    """
    Loop through all camera views in a given video directory and create
    person tracks for each camera view by running all the necessary steps
    in an order. Given keypoint indices (kp_indices) determine the keypoints
    that will be used when forming the tracks (excluding unstable keypoints
    may lead to better performance). Settings for the different steps are
    created here and passed on to the relevant functions. Returns a dictionary
    with an entry for each camera view that has entries for the frame numbers,
    bboxes, keypoints, and assigned ids that belong to it. The output is used
    for both camera movement detection and labeling tracks.
    """
    settings = {
        # minimum displacement threshold for creating tracklets (in pixels)
        "tracklet_displacement_threshold": 25,
        # number of allowed consecutive missing values in a tracklet
        "tracklet_nan_threshold": 10,
        # displacement threshold for what is considered a duplicate
        "duplicate_displacement_threshold": 5,
        # number of duplicate keypoints allowed
        "n_duplicate_keypoints_threshold": len(kp_indices) // 2,
        # minimum required tracklet duration to add to tracks
        "min_tracklet_dur": 25,
        # minimum required within-keypoints spread difference between tracks
        "min_tracklet_spread_diff": 10,
        # number of consecutive NaNs to interpolate for tracks
        "interpolation_limit": 25,
        # frame range for printing debugging information (e.g., (99, 999))
        "debug_range": (100, 120)
    }
    track_dict = {}
    for cam_nr in range(4):
        cam_dir = f"{video_dir[:-11]}_{cam_nr}"
        print(f"Forming tracks for {video_dir}_{cam_nr}...")
        pose_data_path = (
            f"{data_dir}/assigned_ids/{video_dir}/"
            f"{cam_dir}/assignedID_results.json"
        )
        with open(pose_data_path, "r") as json_data:
            pose_data = json.load(json_data)
        frame_range = [
            int(i["image_id"].replace(".jpg", ""))
            for i in [pose_data[0], pose_data[-1]]
        ]
        id_labels = ["parent", "child"]
        tracklets = create_tracklets(
            pose_data, id_labels, kp_indices, **settings)
        tracklets = group_tracklets(
            tracklets, kp_indices, **settings)
        track1, track2 = flatten_tracklets(tracklets)
        if not track1 or not track2:
            raise Exception("Insufficient data for two tracks")
        track1 = add_empty_values(track1, frame_range)
        track2 = add_empty_values(track2, frame_range)
        track1 = interpolate_track(track1, **settings)
        track2 = interpolate_track(track2, **settings)
        person_tracks = {
            "frame_nr": [],
            "bbox1": [],
            "keypoints1": [],
            "assigned_id1": [],
            "bbox2": [],
            "keypoints2": [],
            "assigned_id2": []
        }
        for track1_data, track2_data in zip(track1, track2):
            frame_nr, bbox1, kps1, id1 = track1_data
            _, bbox2, kps2, id2 = track2_data
            out_ord = [frame_nr, bbox1, kps1, id1, bbox2, kps2, id2]
            for idx, key in enumerate(person_tracks.keys()):
                person_tracks[key].append(out_ord[idx])
        track_dict[f"{video_dir}_{cam_nr}"] = person_tracks
    return track_dict

# draw video from tracks for validation purposes
def draw_validation_video(track_dict, data_dir, kp_indices):
    """
    Reads the original video frames and superimposes keypoints selected by
    kp_indices to their respective frames and writes the frames to a video.
    Can be used to validate person track data. 
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    video_dir = list(track_dict.keys())[0][:-2]
    video_name = video_dir[:-11]
    video_output_path = (
        f"{data_dir}/pose_tracks/{video_dir}_overlay.mp4")
    if os.path.exists(video_output_path):
        raise FileExistsError(
            f"Path {video_output_path} already exists, delete first")
    video_length = len(list(track_dict.values())[0]["frame_nr"])
    writer = []
    colors = [(0, 255, 255), (255, 255, 0)]
    track_keys = ["frame_nr", "keypoints1", "keypoints2"]
    cam_zips = [
        zip(*[cam_dict[tr_key] for tr_key in track_keys])
        for cam_dict in track_dict.values()
    ]
    cam_zips = [
        [[frame_nr, kps1, kps2] for frame_nr, kps1, kps2 in cam_zip]
        for cam_zip in cam_zips
    ]
    for frame_data in zip(*cam_zips):
        frames = []
        for cam_nr, (frame_nr, kps1, kps2) in enumerate(frame_data):
            img = cv2.imread(
                f"{data_dir}/extracted_frames/{video_dir}/"
                f"{video_name}_{cam_nr}/{frame_nr}.jpg")
            frame_height, frame_width, _ = img.shape
            frame_kps = [rearrange_kps(kps, kp_indices) for kps in [kps1, kps2]]
            for tr_idx, kps in enumerate(frame_kps):
                for x, y, _ in kps:
                    try:
                        cv2.circle(img, (int(x), int(y)), 5, colors[tr_idx], -1)
                    except ValueError:
                        pass
            text = f"Cam {cam_nr}"
            cv2.putText(
                img, text, (50, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
            frames.append(img)
        full_frame = np.hstack([np.vstack(frames[:2]), np.vstack(frames[2:])])
        if not writer:
            frame_height, frame_width, _ = full_frame.shape
            writer = cv2.VideoWriter(
                video_output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                25,
                (frame_width, frame_height))
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame: {frame_nr}"
        cv2.putText(
            full_frame, text, (50, 100), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        writer.write(full_frame)
        print(
            f"\rWriting overlaid video to file: {frame_nr}/{video_length}",
            end="",
            flush=True)
    writer.release()
    print("\nFinished")

# label tracks as parent and child track
def label_tracks(track_path, data_dir, draw_video_kps=[], ratio_check=False):
    """
    Labels tracks with the assumption that within-keypoints spread will be
    higher for the parent than for the child. First checks for match ratios
    for each track and raises an exception if both tracks have the same max
    ratio match for the labels "parent", "child", and "no match". Then loads
    in camera movement data and produces a list of binary frame-level output
    structured [frame_nr, parent_detected, child_detected, cam_stable].
    Returns a dictionary with an entry for each camera view that contains
    the frame numbers, parent keypoints, child keypoints, and binary
    output for the given camera view. This dictionary has the correct format
    for feeding data to DMMR (using prep_dmmr.py). Saves the dictionary to a
    pickle file. If keypoint indices are passed to draw_video_kps, also draws
    a validation video with superimposed track keypoints.
    """
    labeled_track_dict = {}
    with open(track_path, "rb") as tracks_in:
        tracks = pickle.load(tracks_in)
    for video_name, person_tracks in tracks.items():
        print(f"Assigning labels to {video_name} tracks...")
        video_dir = video_name[:-2]
        cam_nr = video_name[-1]
        cam_dir = f"{video_dir[:-11]}_{cam_nr}"
        id_labels = ["parent", "child", "no_match"]
        assigned_id_tracks = [
            person_tracks[key] for key in ["assigned_id1", "assigned_id2"]
        ]
        track_ratios = []
        for track in assigned_id_tracks:
            valid_ids = [
                assigned_id for assigned_id in track
                if assigned_id in id_labels
            ]
            label_ratios = []
            for id_label in id_labels:
                isolated_id = [
                    assigned_id for assigned_id in valid_ids
                    if assigned_id == id_label
                ]
                label_ratio = len(isolated_id) / len(valid_ids)
                label_ratios.append(label_ratio)
            track_ratios.append(label_ratios)
        for idx, (parent_ratio, child_ratio, no_match_ratio) in enumerate(
                track_ratios):
            print(
                f"Match ratios for {video_name} track{idx + 1} "
                f"(parent/child/no_match): "
                f"{parent_ratio:.2f}/{child_ratio:.2f}/{no_match_ratio:.2f}")
        if len({ratios.index(max(ratios)) for ratios in track_ratios}) == 1:
            if not ratio_check:
                raise Exception(
                    "Same max ratio for both tracks (CHECK FOR ISSUES!)")
        cam_movement_path = (
            f"{data_dir}/camera_movement/{video_dir}/"
            f"{cam_dir}/binary_frame_info_validated.json")
        with open(cam_movement_path, "r") as f:
            cam_movement = json.load(f)
        # **Create a frame-wise structured output**  
        person_dict_keys = ["frame_nr", "keypoints1", "keypoints2"]
        person_dict_data = [person_tracks[key] for key in person_dict_keys]
        labeled_tracks = {
            "frame_nr": [],
            "child_keypoints": [],
            "parent_keypoints": [],
            "binary_output": []
        }
        for tr_fr, keypoints1, keypoints2, (cam_fr, cam_mv, _) in zip(
                *person_dict_data, cam_movement):
            assert tr_fr == int(cam_fr.replace(".jpg", ""))
            child_det = 1 if not any(np.isnan(keypoints1)) else 0
            parent_det = 1 if not any(np.isnan(keypoints2)) else 0
            frame_binary = [tr_fr, parent_det, child_det, cam_mv]
            out_ord = [tr_fr, keypoints1, keypoints2, frame_binary]
            for idx, key in enumerate(labeled_tracks.keys()):
                labeled_tracks[key].append(out_ord[idx])
        labeled_track_dict[video_name] = labeled_tracks
    if draw_video_kps:
        draw_validation_video(
            tracks,
            data_dir,
            draw_video_kps)
    return labeled_track_dict
