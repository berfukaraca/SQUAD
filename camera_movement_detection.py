import numpy as np
import cv2
import os
import json
import math
import pickle
from natsort import natsorted
from ultralytics import YOLO
from detection import select_most_probable_by_class


def detect_objects(frame, base_dir):
    """Run object detection on a frame."""
    model_path = f"{base_dir}/train33/weights/best.pt"
    model = YOLO(model_path)
    results = model.predict(source=frame, save=False)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        predictions = [{'bbox': box, 'confidence': score, 'class': obj_class}
                       for box, score, obj_class in zip(boxes, scores, classes)]

        final_predictions = select_most_probable_by_class(predictions)

    bounding_boxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for result in final_predictions for box in [result['bbox']]]

    return bounding_boxes

def create_combined_mask(frame_shape, person_boxes, object_boxes):
    """Create a mask combining both person and object bounding boxes."""
    mask = np.ones(frame_shape[:2], dtype=np.uint8)
    all_boxes = object_boxes + person_boxes

    for box in all_boxes:
        if any(np.isnan(float(b)) for b in box):
            # print("Skipping box with NaN:", box) 
            continue
        x1,y1,x2,y2 = [max(0, min(int(float(b)), frame_shape[1] if i % 2 == 0 else frame_shape[0])) for i, b in enumerate(box)]
        mask[y1:y2, x1:x2] = 0  

    return mask

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def init_new_features(gray_frame, bounding_box_mask):
    """Find and visualize new feature points."""
    bounding_box_mask = bounding_box_mask.astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(gray_frame, mask=bounding_box_mask, maxCorners=800, qualityLevel=0.005, minDistance=2, blockSize=7)

    visualized_masked_input = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    if corners is not None:
        corners = np.float32(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(visualized_masked_input, (int(x), int(y)), 5, (0, 255, 0), -1)  
    return corners

def generate_binary_output(cam_movement, cam_dir):
    """Generates a binary output for each frame: 1 if =the camera did not move, otherwise 0."""
    binary_output = []
    # Convert cam_movement list to a dictionary for fast lookup {frame_id: cam_moved}
    cam_movement_dict = {
        frame_id: {'moved': moved, 'dist': dist}
        for frame_id, moved, dist in cam_movement
    }
    frames= natsorted(os.listdir(cam_dir))
    for idx, frame_name in enumerate(frames):  
        frame_id = int(frame_name.replace(".jpg", ""))

        movement_data = cam_movement_dict.get(frame_id, {'moved': 1,'dist': 0.0}) # Determine camera movement from precomputed results
        cam_moved= movement_data['moved']  # Camera movement condition: 1 if stable, 0 if moved
        dist =round(movement_data['dist'], 2)
        camera_movement_condition= 1-cam_moved
        binary_output.append([frame_name,camera_movement_condition, dist])

    return binary_output

def read_annotations(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        coords = {}
        for line in lines:
            parts = line.strip().split()
            obj_id = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)
            coords[obj_id] = [x1, y1, x2, y2] 
        return coords
    
def check_for_movement(video_name, cam_nr, base_dir):
    """Detect camera movement using optical flow and masked bounding box tracking."""
    video=video_name.split('_')[0]
    participant_path= f"{base_dir}/data/camera_movement/{video_name}"
    os.makedirs(participant_path, exist_ok=True)
    cam_dir = f"{base_dir}/data/extracted_frames/{video_name}/{video}_{cam_nr}"
    cam_frames = natsorted(os.listdir(cam_dir))

    output_dir = f"{participant_path}/{video}_{cam_nr}"
    if not os.path.exists(output_dir):
        os.mkdir(f"{output_dir}")
    else:
        raise FileExistsError(f"Data already exists at {output_dir}. Delete it to re-run.")    
    old_frame = cv2.imread(f"{cam_dir}/{cam_frames[0]}")
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    bounding_box_mask = np.ones(old_gray.shape, dtype=np.uint8)
    corners = init_new_features(old_gray, bounding_box_mask)

    camera_movement = []
    with open(f"{base_dir}/data/pose_tracks/{video_name}_tracks.pickle", "rb") as pose_in:
        pose_data = pickle.load(pose_in)
    pose_data = pose_data[f"{video_name}_{cam_nr}"]

    def mask_size(bbox_list):
        bbox_size = [
            (abs(x2 - x1), abs(y2 - y1))
            for x1, y1, x2, y2
            in bbox_list
        ]
        mean_w, mean_h = np.nanmean(bbox_size, axis=0)
        sd_w, sd_h = np.nanstd(bbox_size, axis=0)
        box_w = mean_w + sd_w
        box_h = mean_h + sd_h
        return box_w, box_h

    mask_sizes = [
        mask_size(pose_data[key])
        for key
        in ["bbox1", "bbox2"]
    ]
    bbox_params = []

    detection_path = f"{base_dir}/data/detection/{video_name}/{video}_{cam_nr}/detections.json"
    with open (detection_path,'r') as f:
        detection_v= json.load(f)

    for idx, frame_nr in enumerate(pose_data["frame_nr"][:-1]):
        current_frame = cv2.imread(f"{cam_dir}/{frame_nr}.jpg")
        next_frame = cv2.imread(f"{cam_dir}/{frame_nr+1}.jpg")

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        frame_bboxes = [pose_data["bbox1"][idx], pose_data["bbox2"][idx]] 
        detection_key = f"{frame_nr}_{cam_nr}_processed"

        frame_detection = detection_v.get(detection_key, None)
        coords = []
        for line in frame_detection:
            x, y, w, h = map(float, line[1:5])
            x1 = x - (w / 2)
            y1 = y - (h / 2)
            x2 = x + (w / 2)
            y2 = y + (h / 2)
            coords.append([x1, y1, x2, y2])
        
         # Updated logic to compare each bounding box size
        updated_bboxes = []
        for (x1, y1, x2, y2), (mask_w, mask_h) in zip(frame_bboxes, mask_sizes):
        # Calculate original bounding box width and height
            bb_width = abs(x2 - x1)
            bb_height = abs(y2 - y1)

            # If original bbox is larger than mask size, use it directly
            if bb_width > mask_w or bb_height > mask_h:
                updated_bboxes.append([x1, y1, x2, y2])
            else:
                # Use the mask size centered at the same position
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                new_x1 = center_x - mask_w / 2
                new_y1 = center_y - mask_h / 2
                new_x2 = center_x + mask_w / 2
                new_y2 = center_y + mask_h / 2
                updated_bboxes.append([new_x1, new_y1, new_x2, new_y2])

        final_mask = create_combined_mask(current_frame.shape, updated_bboxes, coords)
        masked_frame = current_frame.copy()
        masked_frame[final_mask == 0] = (0, 0, 0)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, next_gray, corners, None, winSize=(50, 50), maxLevel=4,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        if p1 is None or len(p1) == 0:
            new_corners = init_new_features(next_gray, final_mask, frame_nr, output_dir)
            if new_corners is not None:
                corners = np.vstack((corners, new_corners))
            continue

        st = st.flatten()
        valid_idx = np.where(st == 1)[0]
        if len(valid_idx) > 0:
            good_new = p1[valid_idx]
            good_old = corners[valid_idx]
        else:
            good_new = good_old = np.array([])

        cam_moved = False
        distances = []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = np.clip(new.ravel(), [0, 0], [final_mask.shape[1] - 1, final_mask.shape[0] - 1])
            c, d = np.clip(old.ravel(), [0, 0], [final_mask.shape[1] - 1, final_mask.shape[0] - 1])

            if final_mask[int(b), int(a)] == 0:
                continue

            distance = calculate_distance(a, b, c, d)
            distances.append(distance)

        if np.mean(distances) > 2:
            cam_moved = True
            print(np.mean(distances))

        if cam_moved:
            print(f'Camera moved between frames {frame_nr} and {frame_nr + 1}')
            corners = init_new_features(next_gray, final_mask)

        old_gray = next_gray.copy()
        camera_movement.append([frame_nr, cam_moved, np.mean(distances)])
    binary_results = generate_binary_output(camera_movement,cam_dir)
    binary_output_path = os.path.join(output_dir, "binary_frame_info.json")
    with open(binary_output_path, "w") as f:
        json.dump(binary_results, f, separators=(',', ':'))
    
    return camera_movement, binary_results
