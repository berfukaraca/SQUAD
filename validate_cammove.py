import cv2
import numpy as np
import os
import json
import math
import re
import pickle
import random

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0][0] - p1[0][0]) ** 2 + (p2[0][1] - p1[0][1]) ** 2)

def extract_frame_number(filename):
    """Extract frame number from filename like '50.jpg'."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def create_mask(frame_shape, detection_data, tracker_data):
    """Creates a binary mask based on object detections and body bounding boxes."""
    mask = np.ones(frame_shape[:2], dtype=np.uint8)

    for obj in detection_data:
        x, y, w, h = map(int, obj[1:5])
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        mask[y1:y2, x1:x2] = 0

    for body in tracker_data:
        if any(np.isnan(float(b)) for b in body): 
            print(f"Skipping NaN tracker bounding box: {body}") 
            continue
        x1, y1, x2, y2 = map(int, body)
        mask[y1:y2, x1:x2] = 0 

    return mask

def load_detections(detection_file):
    """Loads detection data from JSON file."""
    with open(detection_file, 'r') as f:
        return json.load(f)

def find_movement_blocks(frame_data):
    """Finds movement blocks (consecutive movement frames)."""
    movement_blocks = []
    current_block = []

    for frame in frame_data:
        if frame['camera_moved'] == 0:  # Movement flagged
            if not current_block:
                current_block = [frame]
            elif frame['frame_id'] == current_block[-1]['frame_id'] + 1:
                current_block.append(frame)
            else:
                movement_blocks.append(current_block)
                current_block = [frame]
        else:
            if current_block:
                movement_blocks.append(current_block)
                current_block = []

    if frame_data[-1]['camera_moved'] == 0:
        last_range_end = movement_blocks[-1][-1]['frame_id'] if movement_blocks else None
        if last_range_end != frame_data[-1]['frame_id']:
            movement_blocks.append([frame_data[-1]])
            print(f"Last movement frame {frame_data[-1]['frame_id']}")
    return movement_blocks

def get_random_pairs(before_frames, after_frames, num_pairs=5):
    """Selects up to num_pairs random pairs from before and after frames."""
    num_pairs = min(num_pairs, len(before_frames), len(after_frames))
    return [(random.choice(before_frames), random.choice(after_frames)) for _ in range(num_pairs)]

def validate_camera_movement(frame_data, movement_blocks, frame_dir, detections, trackers, movement_threshold):
    """Validates camera movement by tracking points between randomly selected before-after frame pairs."""
    validated_data = frame_data.copy()

    min_frame_id = frame_data[0]['frame_id']
    max_frame_id = frame_data[-1]['frame_id']

    expanded_movement_ranges = []

    for i in range(len(movement_blocks)):
        current_movement = movement_blocks[i]

        start_frame = current_movement[0]['frame_id']
        end_frame = current_movement[-1]['frame_id']

        prev_end = movement_blocks[i - 1][-1]['frame_id'] if i > 0 else min_frame_id  

        #If there's no next movement, use max frame, BUT exclude last movement frame
        next_start = movement_blocks[i + 1][0]['frame_id'] if i + 1 < len(movement_blocks) else max_frame_id
        if next_start is None or next_start > max_frame_id:
            next_start = max_frame_id if end_frame < max_frame_id else None  # Exclude last movement

        print(f"Validating movement block: {start_frame}-{end_frame}")

        before_movement = [f['frame_name'] for f in frame_data if prev_end < f['frame_id'] < start_frame]

        after_movement = [f['frame_name'] for f in frame_data if end_frame < f['frame_id'] < next_start]

        if not before_movement or not after_movement:
            if start_frame == end_frame:  # Single-frame movement block
                # print(f"No valid before/after frames for {start_frame}, marking as FALSE POSITIVE.")
                for v_idx, v_frame in enumerate(validated_data):
                    if v_frame['frame_name'] == current_movement[0]['frame_name']:
                        validated_data[v_idx]['camera_moved'] = 1 
                        break
            else:
                print("Not enough pre/post frames available, skipping...")
                continue


        all_possible_pairs = [(b, a) for b in before_movement for a in after_movement]
        selected_pairs = random.sample(all_possible_pairs, k=min(len(all_possible_pairs), 10))

        total_movement = []

        for pre_frame_name, post_frame_name in selected_pairs:
            print(f"Checking random pair: {pre_frame_name} <-> {post_frame_name}")

            pre_gray, post_gray = load_gray_images(frame_dir, pre_frame_name, post_frame_name)
            pre_mask = get_frame_mask(pre_frame_name, detections, trackers, pre_gray.shape)
            post_mask = get_frame_mask(post_frame_name, detections, trackers, post_gray.shape)

            movement_distance = compute_optical_flow(pre_gray, post_gray, pre_mask, post_mask)
            if movement_distance is not None:
                total_movement.append(movement_distance)

        overall_mean_movement = np.mean(total_movement) if total_movement else 0
        print(f"📊 Overall mean movement: {overall_mean_movement:.2f}")

        if overall_mean_movement < movement_threshold:
            # print(f"Marking movement {start_frame}-{end_frame} as FALSE POSITIVE.")
            for frame in current_movement:
                for v_idx, v_frame in enumerate(validated_data):
                    if v_frame['frame_name'] == frame['frame_name']:
                        validated_data[v_idx]['camera_moved'] = 1
                        break
        else:
            print(f"✅ Movement {start_frame}-{end_frame} CONFIRMED.")

            # Expand the movement window by 2 frames before and after
            expanded_start = max(min_frame_id, start_frame - 2)
            expanded_end = min(max_frame_id, end_frame + 2)
            expanded_movement_ranges.append((expanded_start, expanded_end))

            print(f"Expanding movement window to: {expanded_start}-{expanded_end}")

            for frame_id in range(expanded_start, expanded_end + 1):
                frame_name = next((f['frame_name'] for f in frame_data if f['frame_id'] == frame_id), None)
                if frame_name:
                    for v_idx, v_frame in enumerate(validated_data):
                        if v_frame['frame_name'] == frame_name:
                            validated_data[v_idx]['camera_moved'] = 0  # Confirm movement
                            break
    
    validated_output = [[entry['frame_name'], entry['camera_moved'], entry['distance']] for entry in validated_data]

    return validated_output



def load_gray_images(frame_dir, pre_frame_name, post_frame_name):
    """Loads and converts frames to grayscale."""
    pre_frame = cv2.imread(os.path.join(frame_dir, pre_frame_name))
    post_frame = cv2.imread(os.path.join(frame_dir, post_frame_name))
    return cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(post_frame, cv2.COLOR_BGR2GRAY)

def get_frame_mask(frame_name, detections, trackers, frame_shape):
    """Creates a mask for the given frame using detections and tracker data."""
    detection_key = frame_name.replace('.jpg', '')
    first_detection = next(iter(detections))
    view_idx = first_detection.split('_')[1]
    detection_data = detections.get(f"{detection_key}_{view_idx}_processed", [])

    idx = trackers["frame_nr"].index(int(detection_key))
    tracker_data = [trackers["bbox1"][idx], trackers["bbox2"][idx]]

    return create_mask(frame_shape, detection_data, tracker_data)

def compute_optical_flow(pre_gray, post_gray, pre_mask, post_mask):
    """Computes the movement distance using optical flow."""
    points = cv2.goodFeaturesToTrack(pre_gray, mask=pre_mask, maxCorners=800, qualityLevel=0.005, minDistance=2, blockSize=7)
    if points is None:
        return None

    p1, st, err = cv2.calcOpticalFlowPyrLK(pre_gray, post_gray, points, None, winSize=(50, 50), maxLevel=4,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    if p1 is None:
        return None

    st = st.flatten()
    good_new, good_old = p1[st == 1], points[st == 1]
    distances = [calculate_distance(o, n) for o, n in zip(good_old, good_new)]
    return np.mean(distances) if distances else None

def post_process_camera_movements(data_dir,participant,v):

    video_name=participant.split('_')[0]
    json_file_path = f'{data_dir}/camera_movement/{participant}/{video_name}_{v}/binary_frame_info.json'  # Path to flagged camera movement JSON
    frame_directory = f'{data_dir}/extracted_frames/{participant}/{video_name}_{v}'  # Path to extracted frames
    detection_directory = f'{data_dir}/detection/{participant}/{video_name}_{v}/detections.json'  # Path to detection files for masking
    tracker_file = f'{data_dir}/pose_tracks/{participant}_tracks.pickle'  # Path to pose tracker file

    with open(json_file_path, 'r') as file:
        frame_data = json.load(file)

    frame_data = [
        {'frame_name': item[0], 'camera_moved': item[1], 'distance': item[2], 'frame_id': extract_frame_number(item[0])}
        for item in frame_data
    ]

    frame_data.sort(key=lambda x: x['frame_id'])

    detections = load_detections(detection_directory)

    with open(tracker_file, "rb") as pose_in:
        pose_data = pickle.load(pose_in)
    trackers = pose_data[f"{participant}_{v}"]

    movement_blocks = find_movement_blocks(frame_data)
    validated_data = validate_camera_movement(frame_data, movement_blocks, frame_directory, detections, trackers, movement_threshold=30)

    output_file = json_file_path.replace('.json', '_validated.json')
    with open(output_file, 'w') as f_out:
        json.dump(validated_data, f_out, separators=(',', ':'))

    print(f"Validated camera movement saved to {output_file}")
