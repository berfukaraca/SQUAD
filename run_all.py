import os
import pickle
from video_functions import extract_frames, run_alphapose, sync_video
from create_tracks import create_all_tracks, label_tracks
from camera_movement_detection import check_for_movement
from detection import object_detection, run_assign_bb
from prep_dmmr import feed_to_dmmr
import time
from validate_cammove import post_process_camera_movements
from pathlib import Path

start_time= time.time()

base_dir = Path(__file__).resolve().parent #Add SQUAD base directory path including run_all.py
data_dir = os.path.join(base_dir,'data')
alphapose_path = os.path.join(base_dir,"AlphaPose")

process_date1 = "13_03_2025" #IF USING A STEP OF PREVIOUS PROCESS, USE THAT PROCESS DATE. THE OUTPUT WILL BE THAT PROCESS DATE, DELETE THE PREVIOUS OR RENAME IT LIKE v_x
kp_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 19] # list of keypoint indices to use to create person tracks

# PHASE 1 
# Stage 1 - Single Frame Level

#STEP 1 --
# extract frames and output to data/extracted_frames
for video_file in os.listdir(f"{data_dir}/input_videos"):
    break
    video_name = video_file.replace(".mp4", "")
    extract_frames(process_date1,
        f"{data_dir}/input_videos/{video_file}",
        data_dir,
        video_section=("00:00:02", "00:07:30"))

# check for offsets, set condition time, and output to data/frame_offsets
for frame_dir in os.listdir(f"{data_dir}/extracted_frames"):
    break
    frame_timings = sync_video(
        f"{data_dir}/extracted_frames/{frame_dir}")
    with open(f"{data_dir}/frame_offsets/{frame_dir}_offsets.pickle", "wb") as f:
        pickle.dump(frame_timings, f, protocol=pickle.HIGHEST_PROTOCOL)


# STEP 2 -- Alphapose person detection and pose estimation
# run alphapose on extracted frames and output to data/alphapose_results
for frame_dir in os.listdir(f"{data_dir}/extracted_frames"):
    break
    run_alphapose(
        f"{data_dir}/extracted_frames/{frame_dir}",
        alphapose_path,
        f"{data_dir}/alphapose_results/{frame_dir}")
    
participant_list= os.listdir(f"{data_dir}/extracted_frames")

#Step 3 -- Object Detection
for participant in participant_list:
    break
    detection_dir= f"{data_dir}/detection"
    participant_path= os.path.join(f"{data_dir}/extracted_frames",participant)
    output_participant= os.path.join(detection_dir,participant)
    
    if not os.path.exists(output_participant):
        os.mkdir(output_participant)
    
    object_detection(base_dir, participant, output_participant)

#STEP 4 --  delete doll keypoint detections and assign initial person id to body boxes
for participant in participant_list:
    break
    assignedID_dir= f"{data_dir}/assigned_ids"
    participant_path= os.path.join(f"{data_dir}/alphapose_results",participant)
    output_participant= os.path.join(assignedID_dir,participant)

    if not os.path.exists(output_participant):
        os.mkdir(output_participant)

    run_assign_bb(data_dir,participant,output_participant)

# Stage 2 - Sequence of Frames Level

# STEP 5 -- create tracks from alphapose bboxes, output to data/pose_tracks
for alphapose_dir in os.listdir(f"{data_dir}/alphapose_results"):
    break
    video_tracks = create_all_tracks(
        alphapose_dir, data_dir, kp_indices)
    track_output_path = (
        f"{data_dir}/pose_tracks/{alphapose_dir}_tracks.pickle")
    with open(track_output_path, "wb") as f:
        pickle.dump(video_tracks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved tracks to {track_output_path}")

# STEP 6 -- Camera movement detection and binary frame info output ["image id", camera movement indicator] Camera movement condition: 1 if stable, 0 if moved
for frame_dir in os.listdir(f"{data_dir}/extracted_frames"):
    break
    for v in [0,1,2,3]:
        camera_movement = check_for_movement(frame_dir, v, base_dir)

#STEP 7 -- Camera Movement post-processing
for frame_dir in os.listdir(f"{data_dir}/extracted_frames"):
    break
    for v in [0,1,2,3]:
        post_process_camera_movements(data_dir,frame_dir,v)

# STEP 8 -- Assign labels to person tracks
for pose_track in os.listdir(f"{data_dir}/pose_tracks"):
    break
    if not ".pickle" in pose_track or "labeled" in pose_track:
        continue
    labeled_output_path = (
        f"{data_dir}/pose_tracks/"
        f"{pose_track.replace('.pickle', '_labeled.pickle')}")
    labeled_tracks = label_tracks(
        f"{data_dir}/pose_tracks/{pose_track}",
        data_dir,
        draw_video_kps=kp_indices)
    with open(labeled_output_path, "wb") as f:
        pickle.dump(labeled_tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

#STAGE 2 -- Lift 2D poses to 3D ==> 3D pose coordinates (joints), 3D body meshes, camera parameters 
# Step 9 -- Feed labeled track and camera view offsets to DMMR
for labeled_track in os.listdir(f"{data_dir}/pose_tracks"):
    # break
    if not "labeled" in labeled_track:
        continue
    feed_to_dmmr(labeled_track, base_dir=base_dir, min_dur=50, debug=False)

end_time = time.time()

# Calculate and print total duration
total_duration = end_time - start_time
hours, rem = divmod(total_duration, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nPipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
