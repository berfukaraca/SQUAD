import subprocess
import os
import cv2
import numpy as np
# crop video and outputs frames to images
def extract_frames(process_date, video_path, data_dir, video_section=[]):
    video_name = video_path.split("/")[-1].replace(".mp4", "")
    vid_cap = cv2.VideoCapture(video_path)
    cam_w = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
    cam_h = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
    vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)
    cam_corners = [(0, 0), (cam_w, 0), (0, cam_h), (cam_w, cam_h)]
    if not os.path.exists(video_name):
        os.mkdir(f"{data_dir}/extracted_frames/{video_name}_{process_date}")
    if video_section:
        start_time, end_time = video_section
    for cam_idx, (x_pos, y_pos) in enumerate(cam_corners):
        output_dir = (
            f"{data_dir}/extracted_frames/"
            f"{video_name}_{process_date}/{video_name}_{cam_idx}")
        if video_section:
            crop_command = (
                f"ffmpeg -ss {start_time} -to {end_time} "
                f"-i {video_path} -framerate {vid_fps} "
                f"-hide_banner -copyts -f image2 -frame_pts true "
                f"-filter:v \"crop={cam_w}:{cam_h}:{x_pos}:{y_pos}\" "
                f"-qscale:v 1 {output_dir}/%d.jpg")
        else:
            crop_command = (
                f"ffmpeg -i {video_path} -framerate {vid_fps} "
                f"-hide_banner -copyts -f image2 -frame_pts true "
                f"-filter:v \"crop={cam_w}:{cam_h}:{x_pos}:{y_pos}\" "
                f"-qscale:v 1 {output_dir}/%d.jpg")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        subprocess.run(crop_command, shell=True)

# run alphapose on all four camera views
def run_alphapose(video_dir, alphapose_dir, output_dir):
    cam_dirs = [i for i in os.listdir(video_dir)]
    os.chdir(alphapose_dir)
    for cam_dir in cam_dirs:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        results_dir = f"{output_dir}/{cam_dir}"
        if os.path.exists(f"{results_dir}/alphapose-results.json"):
            print(f"Results for {cam_dir} already exist, skipping")
            continue
        alphapose_command = (
            f"python {alphapose_dir}/scripts/demo_inference.py "
            "--cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml "
            "--checkpoint pretrained_models/halpe26_fast_res50_256x192.pth "
            "--debug "
            "--qsize 128 "
            "--detbatch 1 "
            "--posebatch 30 "
            # "--save_img "
            # "--detector yolox-x "
            "--pose_track "
            "--showbox "
            f"--indir {video_dir}/{cam_dir} "
            f"--outdir {results_dir}")
        subprocess.run(alphapose_command, cwd=alphapose_dir, shell=True)

# loop through frames to find offsets and annotate experimenter entry/exit
def sync_video(video_dir):
    cam_dirs = [
        cam_dir
        for cam_dir in os.listdir(video_dir)
    ]
    cam_frames = [
        int(i.replace(".jpg", ""))
        for i in
        os.listdir(f"{video_dir}/{cam_dirs[0]}")
    ]
    cam_dirs = [
        [cam_dir, min(cam_frames)]
        for cam_dir in cam_dirs
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_offsets = [0, 0, 0, 0]
    condition_frames = ["<not set>", "<not set>"]
    while True:
        frames_to_show = []
        for cam_dir, frame_pos in cam_dirs:
            frame = cv2.imread(f"{video_dir}/{cam_dir}/{frame_pos}.jpg")
            frame_height, frame_width, _ = frame.shape
            ratio = 0.75
            frame = cv2.resize(
                frame,
                (int(frame_width * ratio), int(frame_height * ratio)),
                interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                str(frame_pos),
                (10, 25),
                font,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA)
            frames_to_show.append(frame)
        first_row = cv2.hconcat(frames_to_show[:2])
        second_row = cv2.hconcat(frames_to_show[2:])
        full_view = cv2.vconcat([first_row, second_row])
        window_height, window_width, _ = full_view.shape
        status_bar = np.ones(shape=[60, window_width, 3], dtype=np.uint8)
        full_view = cv2.vconcat([full_view, status_bar])
        experimenter_text = (
            f"Condition start at: {condition_frames[0]}, "
            f"end at: {condition_frames[1]}")
        cv2.putText(
            full_view,
            experimenter_text,
            (10, window_height + 25),
            font,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA)
        commands_text = (
            "Keybinds-> Fast movement: Z/C; Slow movement: A/D; "
            "Individual views: 5/1+6/2+7/3+8/4; Timestamps: I/P; Quit: Q")
        cv2.putText(
            full_view,
            commands_text,
            (10, window_height + 45),
            font,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA)
        cv2.imshow("FullView", full_view)
        cam_keys_forward = [ord("1"), ord("2"), ord("3"), ord("4")]
        cam_keys_backward = [ord("5"), ord("6"), ord("7"), ord("8")]
        key = cv2.waitKey(0)
        if key == ord("q"):
            if "<not set>" in condition_frames:
                print("Set condition start and end")
            else:
                cv2.destroyAllWindows()
                break
        elif key == ord("d"):
            if all(pos < max(cam_frames) for cam, pos in cam_dirs):
                for idx, (cam_dir, frame_pos) in enumerate(cam_dirs):
                    cam_dirs[idx][1] += 1
            else:
                print("Next frame not available")
        elif key == ord("a"):
            if all(pos > min(cam_frames) for cam, pos in cam_dirs):
                for idx, (cam_dir, frame_pos) in enumerate(cam_dirs):
                    cam_dirs[idx][1] -= 1
            else:
                print("Previous frame not available")
        elif key == ord("c"):
            if all(pos < max(cam_frames) - 100 for cam, pos in cam_dirs):
                for idx, (cam_dir, frame_pos) in enumerate(cam_dirs):
                    cam_dirs[idx][1] += 100
            else:
                print("Next frame not available")
        elif key == ord("z"):
            if all(pos > min(cam_frames) + 100 for cam, pos in cam_dirs):
                for idx, (cam_dir, frame_pos) in enumerate(cam_dirs):
                    cam_dirs[idx][1] -= 100
            else:
                print("Previous frame not available")
        elif key == ord("i"):
            condition_frames[0] = [pos for cam, pos in cam_dirs]
        elif key == ord("p"):
            condition_frames[1] = [pos for cam, pos in cam_dirs]
        elif key in cam_keys_forward:
            key_idx = cam_keys_forward.index(key)
            if cam_dirs[key_idx][1] < max(cam_frames):
                cam_dirs[key_idx][1] += 1
                frame_offsets[key_idx] += 1
            else:
                print("Next frame not available")
        elif key in cam_keys_backward:
            key_idx = cam_keys_backward.index(key)
            if cam_dirs[key_idx][1] > min(cam_frames):
                cam_dirs[key_idx][1] -= 1
                frame_offsets[key_idx] -= 1
            else:
                print("Previous frame not available")
    output_dict = {
        "condition_frames": condition_frames,
        "frame_offsets": frame_offsets
    }
    return output_dict
