import os
import shutil
import json
import pickle
import subprocess
import numpy as np
from natsort import natsorted

# look for breaks in a signal (used to find sequences)
def look_for_breaks(signal):
    """
    Assumes the signal to have at least [frame_nr, value] for every index.
    Return a list of signal entries where the value changes from one frame
    to the next.
    """
    breaks = [signal[0]]
    if len(signal) >= 2:
        for idx, value in enumerate(signal):
            if idx < len(signal) - 1:
                if not value[1] == signal[idx + 1][1]:
                    breaks.append(signal[idx + 1])
    return breaks

# isolate sequences from a signal based on breaks
def isolate_sequences(breaks, signal):
    """
    Uses a list of signal entries where the signal value changes from one
    frame to the next (given by look_for_breaks) to return an output list
    of sequences containing the start frame, value, and duration of each.
    """
    sequences = []
    for idx, value in enumerate(breaks):
        start_frame = value[0]
        frame_value = value[1]
        if len(signal[0]) == 2:
            last_frame = signal[-1][0]
        else:
            last_frame = signal[-1][0] + signal[-1][2]
        if idx < len(breaks) - 1:
            frame_dur = breaks[idx + 1][0] - start_frame
        elif idx == len(breaks) - 1:
            if start_frame < last_frame:
                frame_dur = last_frame - start_frame
            else:
                continue
        sequences.append([start_frame, frame_value, frame_dur])
    return sequences

# return found sequences with format
def find_sequences(input_signal):
    """
    Implements look_for_breaks and isolate_sequnces together to return
    a list of found sequences with format [start, value, duration].
    """
    if not input_signal:
        return []
    signal_breaks = look_for_breaks(input_signal)
    found_sequences = isolate_sequences(signal_breaks, input_signal)
    return found_sequences

# read in frame range and frame offset information
def read_video_info(video_name, **kwargs):
    offsets_path = (
        f"{kwargs['base_dir']}/data/frame_offsets/"
        f"{video_name}_offsets.pickle")
    with open(offsets_path, "rb") as f:
        video_info = pickle.load(f)
    return video_info

# prepare a sequence for DMMR
def prep_dmmr_data(
        track_dict, people, seq, camparams_path, cam_offsets, **kwargs):
    """
    Copies the frames and keypoint .json files to the DMMR input data
    folder (currently DMMR/data, specified in dmmr_subprocess) so that the
    data for the sequence can be processed by DMMR. Takes the camera offsets
    into account when copying the files so that the file names will match
    when fed to DMMR. Frame number adjustments are done with respect to
    camera view 0. Returns the offset-corrected sequence information.
    """
    seq_start, (seq_cameras, seq_code), seq_dur = seq
    offset0, *_ = cam_offsets
    seq_end = seq_start + seq_dur
    cam_offsets = [
        offset + abs(offset0) if offset0 < 0 else offset - abs(offset0)
        for offset in cam_offsets
    ]
    max_digits = len(str(seq_end))
    if not kwargs["debug"]:
        print(
            f"Copying frames and keypoints for sequence (with offsets)...")
    for key, cam_data in track_dict.items():
        video_name = key[:-2]
        cam_nr = int(key[-1])
        if cam_nr not in seq_cameras:
            continue
        cam_offset = cam_offsets[cam_nr]
        frames_in = (
            f"{kwargs['base_dir']}/data/extracted_frames/"
            f"{video_name}/{video_name[:-11]}_{cam_nr}")
        data_out = (
            f"{kwargs['base_dir']}/DMMR/data/"
            f"{video_name}_{seq_start}_{seq_code}_{seq_dur}")
        frames_out = f"{data_out}/images/{video_name}/Camera0{cam_nr}"
        keypoints_out = f"{data_out}/keypoints/{video_name}/Camera0{cam_nr}"
        camparams_out = f"{data_out}/camparams/{video_name}"
        for fld in [frames_out, keypoints_out, camparams_out]:
            if not os.path.exists(fld) and not kwargs["debug"]:
                os.makedirs(fld)
        seq_frames = [
            frame_nr
            for frame_nr
            in range(seq_start + cam_offset, seq_end + cam_offset)
        ]
        already_copied_frames = os.listdir(frames_out)
        frames_to_copy = len(seq_frames) - len(already_copied_frames)
        if frames_to_copy > 0:
            print(f"Copying {frames_to_copy} frames for Camera0{cam_nr}")
            for frame_nr in seq_frames:
                frame_src = f"{frames_in}/{frame_nr}.jpg"
                frame_dest = (
                    f"{frames_out}/"
                    f"{frame_nr - cam_offset:0{max_digits}d}.jpg")
                if not os.path.exists(frame_dest):
                    shutil.copy(frame_src, frame_dest)
        camparams_dest = f"{camparams_out}/camparams.txt"
        if not os.path.exists(camparams_dest) and not kwargs["debug"]:
            shutil.copy(camparams_path, camparams_dest)
        data_zip = [cam_data[label] for label in ["frame_nr"] + people]
        already_copied_keypoints = os.listdir(keypoints_out)
        keypoints_to_copy = len(seq_frames) - len(already_copied_keypoints)
        if keypoints_to_copy:
            print(
                f"Copying {keypoints_to_copy} keypoints for Camera0{cam_nr}")
            for frame_nr, *keypoints in zip(*data_zip):
                if seq_start + cam_offset <= frame_nr < seq_end + cam_offset:
                    if any(any(np.isnan(kps)) for kps in keypoints):
                        raise ValueError(
                            "NaN keypoints found, check sequence")
                    keypoints = [
                        {"pose_keypoints_2d": kps} for kps in keypoints
                    ]
                    json_out_dict = {"version": 1.1, "people": keypoints}
                    json_out_path = (
                        f"{keypoints_out}/"
                        f"{frame_nr - cam_offset:0{max_digits}d}"
                        "_keypoints.json")
                    if not os.path.exists(json_out_path):
                        with open(json_out_path, "w") as f:
                            json.dump(json_out_dict, f)
    return seq_start, (seq_cameras, seq_code), seq_dur

# process the camera binaries for each camera view
def process_binaries(cam_binaries):
    """
    Takes in the frame-level camera binaries for four views and returns them
    with the format [camera_index, detection_status] with detection_status
    formatted as a string with parent and child detection statuses separated
    by a comma (e.g., "1,1" for both detections present).
    """
    cam_binaries = [
        (idx, f"{p},{c}")
        for idx, (_, p, c, _)
        in enumerate(cam_binaries)
    ]
    return cam_binaries

# write processed sequence details in the dmmr log file
def write_log(seq_fld, opt, **kwargs):
    if not any(seq_fld in i for i in kwargs["log_data"]):
        with open(kwargs["log_path"], "a") as f:
            f.write(f"{kwargs['stable_period']}\t{seq_fld}\t{opt}\t")
            if opt:
                f.write(f"{seq_fld}\n")
            else:
                f.write(f"{kwargs['initial_seq_fld']}\n")

# save updates to config and run dmmr main.py on a sequence using subprocess
def dmmr_subprocess(video_name, seq, people, opt=True, **kwargs):
    """
    First outputs the relevant DMMR arguments, including whether to optimize
    camera parameters, the input data folder, the output data folder, and the
    number of people to a pickle file, saves that in the DMMR main directory,
    then uses the subprocess module to run DMMR/main.py with the given
    arguments. The settings file is read by DMMR/main.py.
    """
    seq_start, (_, seq_code), seq_dur = seq
    seq_fld = f"{video_name}_{seq_start}_{seq_code}_{seq_dur}"
    write_log(seq_fld, opt, **kwargs)
    if kwargs["debug"]:
        return seq_fld
    dmmr_results_dir = f"{kwargs['out_fld']}/{seq_fld}/results"
    if os.path.exists(dmmr_results_dir):
        if os.listdir(dmmr_results_dir):
            print(
                f"Folder '{seq_fld}/results' already exists "
                "and is not empty, skipping...")
            return seq_fld
    dmmr_conf = {
        "opt_cam": opt,
        "data_folder": f"{kwargs['in_fld']}/{seq_fld}",
        "output_folder": f"{kwargs['out_fld']}/{seq_fld}",
        "num_people": len(people[seq_code]),
        "people": people[seq_code]
    }
    with open(f"{kwargs['base_dir']}/DMMR/dmmr_settings.pkl", "wb") as f:
        pickle.dump(dmmr_conf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Feeding sequence of "
        f"{len(people[seq_code])} individual(s), "
        f"starting at {seq_start}, and lasting for {seq_dur} "
        "frames to DMMR")
    subprocess.run(
        "python main.py", cwd=f"{kwargs['base_dir']}/DMMR", shell=True)
    return seq_fld

# align tracks based on camera frame ranges
def align_tracks(track_dict, video_name, frame_ranges):
    """
    Aligns track data for each camera view according to the given camera
    offsets. Returns only the frames that fall within the given frame range.
    Frame numbers are kept unchanged.
    """
    trk_filt = {}
    for cam_nr, (start_frame, end_frame) in enumerate(zip(*frame_ranges)):
        cam_key = f"{video_name}_{cam_nr}"
        trk_filt[cam_key] = {
            "frame_nr": [],
            "parent_keypoints": [],
            "child_keypoints": [],
            "binary_output": []
        }
        for trk_idx, frame_nr in enumerate(track_dict[cam_key]["frame_nr"]):
            if start_frame <= frame_nr < end_frame:
                for k, v in trk_filt[cam_key].items():
                    v.append(track_dict[cam_key][k][trk_idx])
    return trk_filt

# find sequences of dets across four cams
def dets_across_cams(det_status, start, dur, **kwargs):
    """
    First takes a slice of the detection binary and looks for sequences
    in the slice. Then checks those sequences for any sequences with four
    cameras and two detections of at least a given duration of frames (as
    set by min_dur). Lastly, looks for any valid sequences with at least
    three cameras and one person, prioritizing sequences with two people
    over sequences with one person. Returns a list of all sequences with
    four cameras and two detections as well as a list of all valid
    sequences that can be processed by DMMR.
    """
    det_slice = [
        [frame_nr, dets]
        for frame_nr, dets in det_status
        if start <= frame_nr < start + dur
    ]
    det_sequences = find_sequences(det_slice)
    four_cam_sequences = [
        [seq_start, ([0, 1, 2, 3], 3), seq_dur]
        for seq_start, binaries, seq_dur
        in det_sequences
        if all(code == "1,1" for _, code in binaries)
        and seq_dur >= kwargs["min_dur"]
    ]
    valid_sequences = []
    for seq_start, dets, seq_dur in det_sequences:
        two_dets = [cam_idx for cam_idx, code in dets if code == "1,1"]
        parent_dets = [cam_idx for cam_idx, code in dets if code[0] == "1"]
        child_dets = [cam_idx for cam_idx, code in dets if code[2] == "1"]
        if len(two_dets) >= 3 and seq_dur >= kwargs["min_dur"]:
            valid_sequences.append([seq_start, (two_dets, 3), seq_dur])
        else:
            if len(parent_dets) >= 3 and seq_dur >= kwargs["min_dur"]:
                valid_sequences.append([seq_start, (parent_dets, 2), seq_dur])
            if len(child_dets) >= 3 and seq_dur >= kwargs["min_dur"]:
                valid_sequences.append([seq_start, (child_dets, 1), seq_dur])
    return four_cam_sequences, valid_sequences

# splits a sequence based on a max duration threshold
def split_long_seq(seq, max_dur=1046):
    """
    If the sequence duration exceeds a frame threshold given by max_dur,
    splits the sequence into two. The first split will have a duration of
    max_dur frames and the remaining frames will be assigned to the second
    split. DMMR splits sequences at 1046 frames so that is used as default.
    """
    seq_start, seq_code, seq_dur = seq
    if seq_dur > max_dur:
        first_split = [seq_start, seq_code, max_dur]
        second_split = [seq_start + max_dur, seq_code, seq_dur - max_dur]
        print(
            f"Split long initial sequence {seq} to: "
            f"{first_split} and {second_split}")
        return first_split, second_split
    else:
        return seq, []

# split track into sequences and feed them to DMMR
def feed_to_dmmr(track, **kwargs):
    """
    Takes in a track and first looks for periods in the track where the camera
    is stable. Then loops through those stable periods and within each looks
    for sequences in which all four cameras have two detections. If at least
    one such sequence with a duration exceeding the frame threshold (set by
    min_dur) is found, looks for any remaining sequences with at least three
    camera views and one person detected. The longest sequence with four
    cameras and two people is then treated as the initial sequence and fed to
    the DMMR functions with opt_cam=True. After that, any remaining valid
    sequences are fed to DMMR with opt_cam=False, using the camera parameters
    optimized for the initial sequence. Skips stable periods without any
    sequences consisting of four cameras and two detections. Data is output
    to data/dmmr_output with a separate folder for each sequence, with folder
    names containing sequence start frame, number of people in sequence,
    and sequence duration (in this order).
    """
    print(f"Preparing to feed {track} to DMMR...")
    with open(f"{kwargs['base_dir']}/data/pose_tracks/{track}", "rb") as f:
        trk = pickle.load(f)
    video_name = track.replace("_tracks_labeled.pickle", "")
    video_info = read_video_info(video_name, **kwargs)
    frame_ranges = video_info["condition_frames"]
    cam_offsets = video_info["frame_offsets"]
    trk = align_tracks(trk, video_name, frame_ranges)
    combined_views = list(
        zip(*[trk[key]["binary_output"] for key in trk.keys()]))
    camera_movements = []
    det_status = []
    for all_cams in combined_views:
        frame_nr, *_ = all_cams[0]
        det_status.append(
            [frame_nr, process_binaries(all_cams)])
        if all(cam_stable == 1 for *_, cam_stable in all_cams):
            camera_movements.append([frame_nr, 1])
        else:
            camera_movements.append([frame_nr, 0])
    camera_movements = find_sequences(camera_movements)
    stable_periods = [
        [seq_start, seq_code, seq_dur]
        for seq_start, seq_code, seq_dur in camera_movements
        if seq_code == 1
    ]
    kwargs["in_fld"] = f"{kwargs['base_dir']}/DMMR/data"
    kwargs["out_fld"] = f"{kwargs['base_dir']}/data/dmmr_output/{video_name}"
    if not os.path.exists(kwargs["out_fld"]):
        os.mkdir(kwargs["out_fld"])
    kwargs["log_path"] = f"{kwargs['out_fld']}/{video_name}_dmmr_log.txt"
    log_header = "stable_period\tseq_fld\topt_cam\tcamparams_fld\n"
    if not os.path.exists(kwargs["log_path"]):
        kwargs["log_data"] = []
        with open(kwargs["log_path"], "w") as f:
            f.write(log_header)
    else:
        with open(kwargs["log_path"], "r") as f:
            kwargs["log_data"] = [i for i in f if not i == log_header]
    for stable_period in stable_periods:
        period_start, _, period_dur = stable_period
        period_end = period_start + period_dur - 1
        kwargs["stable_period"] = f"{period_start}_to_{period_end}"
        print(
            f"Found stable period, starting at {period_start}, "
            f"lasting for {period_dur} frames")
        two_dets_four_cameras, valid_sequences = dets_across_cams(
            det_status, period_start, period_dur, **kwargs)
        if two_dets_four_cameras:
            initial_seq = max(two_dets_four_cameras, key=lambda x: x[2])
            initial_camparams_path = (
                f"{kwargs['base_dir']}/DMMR/YOUth_camparams/camparams.txt")
            remaining_sequences = [
                seq for seq in valid_sequences if seq != initial_seq
            ]
            initial_seq, remaining_initial_seq = split_long_seq(initial_seq)
            if remaining_initial_seq:
                remaining_sequences += [remaining_initial_seq]
            print(f"Found a valid initial sequence: {initial_seq}")
            people = [
                ["index_placeholder"],
                ["child_keypoints"],
                ["parent_keypoints"],
                ["parent_keypoints", "child_keypoints"]
            ]
            initial_seq = prep_dmmr_data(
                trk,
                people[initial_seq[1][1]],
                initial_seq,
                initial_camparams_path,
                cam_offsets,
                **kwargs)
            initial_seq_start, _, initial_seq_dur = initial_seq
            initial_seq_end = initial_seq_start + initial_seq_dur - 1
            initial_seq_fld = dmmr_subprocess(
                video_name, initial_seq, people, **kwargs)
            kwargs["initial_seq_fld"] = initial_seq_fld
            for remaining_seq in remaining_sequences:
                print(f"Found a remaining valid sequence: {remaining_seq}")
                optimized_camparams_path = (
                    f"{kwargs['out_fld']}/{initial_seq_fld}/"
                    f"camparams/{video_name}/{initial_seq_end}.txt")
                if not kwargs["debug"]:
                    with open(optimized_camparams_path, "r") as f:
                        camparam_lines = [i for i in f]
                        updated_camparams = []
                        for cam_nr, line in enumerate([0, 9, 18, 27]):
                            if cam_nr in remaining_seq[1][0]:
                                camparams = camparam_lines[line:line + 9]
                                updated_camparams.append(camparams)
                updated_camparams_path = optimized_camparams_path.replace(
                    ".txt", "_updated.txt")
                if not kwargs["debug"]:
                    with open(updated_camparams_path, "w") as f:
                        for cam_line in updated_camparams:
                            for param_line in cam_line:
                                f.write(param_line)
                remaining_seq = prep_dmmr_data(
                    trk,
                    people[remaining_seq[1][1]],
                    remaining_seq,
                    updated_camparams_path,
                    cam_offsets,
                    **kwargs)
                _ = dmmr_subprocess(
                    video_name, remaining_seq, people, opt=False, **kwargs)
        else:
            print(
                f"Did not find sequence of at least {kwargs['min_dur']} "
                "frames, skipping stable period...")
