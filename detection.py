import json
import cv2
import os
from natsort import natsorted

from ultralytics import YOLO


class_names = {
    0: "doll",
    1: "car",
    2: "jump box",
    3: "shape box",
    4: "flower",
    5: "book",
    6: "bottle",
    7: "green star",
    8: "yellow cylinder",
    9: "blue cube",
    10: "red triangle",
    11: "shape box lid",
    12: "head parent",
    13: "head child"
}

def calculate_iou(boxA, boxB):
    """
    Compute IoU between two bounding boxes.
    boxA and boxB are in [x1, y1, x2, y2] format.
    """

    # Determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area
    union_area = boxA_area + boxB_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def get_image_size(image_path):
    """
    Extract the width and height of an image.
    """
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    return image_width, image_height

def convert_pixel_to_xyxy(box):
    """
    pixel value input
    Convert YOLO format (x_center, y_center, width, height) to xyxy format (x1, y1, x2, y2).
    
    Args:
        box (list): [x_center, y_center, width, height] in absolute pixel values.
    
    Returns:
        list: [x1, y1, x2, y2] bounding box.
    """
    x_center, y_center, width, height = box

    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)

    return [x1, y1, x2, y2]

def convert_to_xyxy(x, y, width, height):
    # Convert width-height to xyxy format
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return [x1, y1, x2, y2]

def extract_relevant_keypoints(flattened_keypoints):
    """
    Extract keypoints for Nose, Left Ear, Right Ear, Left Shoulder, Right Shoulder, and Head.
    Args:
        flattened_keypoints (list): Flattened keypoints array in the format 
                                    [x1, y1, conf1, x2, y2, conf2, ...].
    Returns:
        dict: A dictionary containing the relevant keypoints and their coordinates.
    """
    relevant_indices = {
        "nose": 0,     
        "left_eye":1,   
        "right_eye":2,
        "left_ear": 3,       
        "right_ear": 4,      
        "head": 17,           
        "neck": 18
    }

    keypoints = {}
    for name, index in relevant_indices.items():
        x = flattened_keypoints[index * 3]
        y = flattened_keypoints[index * 3 + 1]
        conf = flattened_keypoints[index * 3 + 2]
        keypoints[name] = (x, y, conf)
    
    return keypoints

def generate_bbox_from_keypoints(keypoints):
    """
    Generate a bounding box that frames all the relevant keypoints.
    Args:
        keypoints (dict): A dictionary containing relevant keypoints and their coordinates.
                          Example:
                          {
                              "nose": (x1, y1, conf1),
                              "left_eye": (x2, y2, conf2),
                              ...
                          }
    Returns:
        list: Bounding box in xyxy format [x_min, y_min, x_max, y_max].
    """

    x_coords = [kp[0] for kp in keypoints.values()]
    y_coords = [kp[1] for kp in keypoints.values()]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    return [x_min, y_min, x_max, y_max]

def assign_person_ids(object_detections_lines, body_data):
    """
    Assign person IDs to body bounding boxes based on maximum IoU with head boxes.
    This version includes both body and head bounding box information in the output.
    """
    
    object_detections=[]
    for line in object_detections_lines:
        obj_id = line[0]
        box = line[1:5]
        confidence= line[5]

        object_detections.append({"id": obj_id, "box": box, 'confidence': confidence})
    
    for object_detection in object_detections:
        object_detection['box'] = convert_pixel_to_xyxy(object_detection['box'])

    head_boxes = [obj for obj in object_detections if obj['id'] in [12, 13]]

    body_detections = []

    for body_entry in body_data:
        body_bb = body_entry["box"]
        body_keypoints = body_entry["keypoints"]
        body_detections.append({"box": body_bb, "keypoints": body_keypoints})

    if object_detections:

        filtered_body_data = []

        for body_entry in body_detections:
            body_bb = body_entry["box"]
            keep_body = True

            for obj in object_detections:
                obj_bb = obj["box"]
                x, y, width, height = body_bb
                body_bb_c= convert_to_xyxy(x, y, width, height)
                iou = calculate_iou(body_bb_c, obj_bb)
                
                if iou > 0.80:
                    keep_body = False
                    break 

            if keep_body:
                filtered_body_data.append(body_entry)

        # Update body_data to only contain the filtered results
        body_data = filtered_body_data

    body_keypoints=[]
    for body_entry in body_data:
        full_keypoints = body_entry["keypoints"]
        relevant_kp = extract_relevant_keypoints(full_keypoints)
        box= body_entry["box"]
        body_keypoints.append({
            "keypoints": full_keypoints,
            "box": box,
            "relevant_keypoints": relevant_kp
        })

    generated_head_boxes= []
    for entry in body_keypoints:
        kps, bbox, kp = entry["keypoints"], entry["box"], entry["relevant_keypoints"]

        generated_head_boxes.append({
            "keypoints":kps,
            "box": bbox,
            "head_boxes":generate_bbox_from_keypoints(kp)
        })
    results = []  

    # Iterate through body boxes and match with head boxes
    for entry in generated_head_boxes:
        kps, box, generated_hb = entry["keypoints"], entry["box"], entry["head_boxes"]
        best_iou = 0
        best_head_box = None
        best_head_id = None

        for head_box in head_boxes:
            
            iou = calculate_iou(head_box["box"], generated_hb)
            # print("iou", iou)
            
            # Update the best match based on IoU
            if iou > best_iou:
                best_iou = iou
                best_head_box = head_box["box"]
                best_head_id = head_box["id"]  # Extract the head_box obj_id (12 or 13)

        # Classify based on object ID
        if best_head_box:
            if best_head_id == 12:
                classification = "parent"
            elif best_head_id == 13:
                classification = "child"
            else:
                classification = "unknown"
            
            x1,y1,x2,y2=generated_hb
            generated_hb_conv=x1,y1,(x2-x1),(y2-y1) # converted again to match with Json files

            results.append({
                "best_head_id": best_head_id,
                "classification": classification,
                "keypoints": kps,
                "box": box,
                "iou": best_iou,
                "generated_head_box": generated_hb_conv,  
                "head_box": best_head_box  
            })
        else:
            x1,y1,x2,y2=generated_hb
            generated_hb_conv=x1,y1,(x2-x1),(y2-y1)

            # No matching head box for this body box
            results.append({
                "best_head_id": None,
                "classification": "no_match",
                "keypoints": kps,
                "box": box,
                "iou": 0,
                "generated_head_box": generated_hb_conv,
                "head_box": None 
            })

    return results

def draw_bounding_boxes_from_results(frame, id_results):
    
    for result in id_results:
        classification = result["classification"] 
        best_head_id = result["best_head_id"]  
        iou = result["iou"]                     
        generated_head_box = result["generated_head_box"]
        detected_head_box = result["head_box"] 

        color_generated = (0, 255, 0)  # Green for generated head boxes
        color_detected = (255, 0, 0)   # Blue for detected head boxes

        if generated_head_box:
            x, y, width, height = map(int, generated_head_box)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color_generated, 2)
            label = f"Generated {classification}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_generated, 2)

        if detected_head_box:
            x1, y1, x2, y2 = map(int, detected_head_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_detected, 2)
            label = f"Detected ID: {best_head_id}, IoU: {iou:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_detected, 2)

    return frame

def select_most_probable_by_class(predictions, conf_threshold=0.20):
    # Filter predictions with confidence above the threshold
    filtered_predictions = [pred for pred in predictions if pred['confidence'] >= conf_threshold]

    # Group predictions by object class
    predictions_by_class = {}
    for pred in filtered_predictions:
        obj_class = pred['class']
        if obj_class not in predictions_by_class:
            predictions_by_class[obj_class] = []
        predictions_by_class[obj_class].append(pred)

    # Select the most probable prediction for each class
    final_predictions = []
    for obj_class, preds in predictions_by_class.items():
        best_prediction = max(preds, key=lambda x: x['confidence'])
        final_predictions.append(best_prediction)

    return final_predictions

def draw_boxes(image, predictions):
    for pred in predictions:
        bbox = pred['bbox']
        obj_class = int(pred['class'])
        confidence = pred['confidence']
        class_name = class_names.get(obj_class, 'Unknown')

        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        label = f"{obj_class} {class_name} {confidence:.2f}"
        cv2.putText(image, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def remove_duplicate_annotations(object_detections, image_width, image_height):
    """
    Removes duplicate annotations based on IoU threshold (>95%) by keeping the one with the higher confidence.
    
    Args:
        detections (list): List of detection dictionaries with 'box' and 'id'.
    
    Returns:
        list: Filtered detections with duplicates removed.
    """
    filtered_detections = object_detections[:]
    
    i = 0
    while i < len(filtered_detections):
        j = i + 1
        while j < len(filtered_detections):

            boxA = filtered_detections[i]["bbox"]
            boxB = filtered_detections[j]["bbox"]


            iou = calculate_iou(boxA, boxB)

            if iou > 0.95:
                # Check which box has lower confidence
                confA = filtered_detections[i]["confidence"]
                confB = filtered_detections[j]["confidence"]
                print(f"Due to high overlap between {filtered_detections[i]} and {filtered_detections[j]} lower confidence object is being deleted")

                if confA > confB:
                    del filtered_detections[j]  # Remove boxB
                else:
                    del filtered_detections[i]  # Remove boxA
                    i -= 1  # Adjust index since earlier element is removed
                continue  # Re-check the new element at position j

            j += 1
        i += 1

    return filtered_detections

def object_detection(model_dir,participant, output_participant):
    model_path=f"{model_dir}/train33/weights/best.pt"
    model = YOLO(model_path)
    data_dir=os.path.join(model_dir,'data')

    for v in [0,1,2,3]:
        view_dict = {}
        video_name=participant.split('_')[0]
        output_view=f"{output_participant}/{video_name}_{v}"
        if not os.path.exists(output_view):
            os.mkdir(output_view)

        participant_frames=os.path.join(f"{data_dir}/extracted_frames",participant,f"{video_name}_{v}") 
        frames=natsorted(os.listdir(participant_frames))
        
        for fr_fname in frames: 
            fr = fr_fname.replace(".jpg", "")
            frame=f"{participant_frames}/{fr}.jpg"

            results = model.predict(source=frame, save=False, save_txt=False) 

            view_dict[f"{fr}_{v}"] = []
            view_dict[f"{fr}_{v}_processed"] = []
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, lab = box
                # Write YOLO format: class x_center y_center width height
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                view_dict[f"{fr}_{v}"].append([
                        int(lab),
                        float(x_center),
                        float(y_center),
                        float(width),
                        float(height),
                        float(conf)
                    ])

            for i, result in enumerate(results):
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                predictions = [{'bbox': box, 'confidence': score, 'class': obj_class}
                            for box, score, obj_class in zip(boxes, scores, classes)]

                image = cv2.imread(frame)
                image_height, image_width = image.shape[:2]

                # Apply post-processing
                final_predictions = select_most_probable_by_class(predictions)
                filtered_predictions = remove_duplicate_annotations(final_predictions,image_width,image_height)

                for pred in filtered_predictions:
                    bbox = pred['bbox']
                    obj_class = int(pred['class'])
                    conf = pred['confidence']

                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = (bbox[2] - bbox[0])
                    height = (bbox[3] - bbox[1])
                    view_dict[f"{fr}_{v}_processed"].append([
                        int(obj_class),
                        round(float(x_center), 6),
                        round(float(y_center), 6),
                        round(float(width), 6),
                        round(float(height), 6),
                        round(float(conf), 6)
                    ])
                    
        with open(f"{output_view}/detections.json", "w") as f:
            json.dump(view_dict, f)
            
def run_assign_bb(data_dir,participant,output_participant):
    
    for v in [0,1,2,3]:
        all_results= []
        video_name=participant.split('_')[0]
        detection_path = f"{data_dir}/detection/{participant}/{video_name}_{v}/detections.json"
        with open (detection_path,'r') as f:
            detection_v= json.load(f)
        output_view=f"{output_participant}/{video_name}_{v}"
        if not os.path.exists(output_view):
            os.mkdir(output_view)

        participant_frames=os.path.join(f"{data_dir}/extracted_frames",participant,f"{video_name}_{v}") 
        json_dir=os.path.join(f"{data_dir}/alphapose_results",participant,f"{video_name}_{v}")
        frames=natsorted(os.listdir(participant_frames))
        json_f = f"{json_dir}/alphapose-results.json"
        
        with open(json_f, 'r') as f:
            alphapose_result = json.load(f)
        
        for fr_fname in frames:

            fr = int(fr_fname.replace(".jpg", ""))            
            img_result = [item for item in alphapose_result if item.get('image_id') == f"{fr}.jpg"]
            detection_key = f"{fr}_{v}_processed"

            frame_detection = detection_v.get(detection_key, None)            
            id_results = assign_person_ids(frame_detection, img_result)
            all_results.append({"image_id": f"{fr}.jpg", "output_results": id_results})

        json_output_file = f"{output_view}/assignedID_results.json"
        with open(json_output_file, 'w') as f:
            json.dump(all_results, f, separators=(',', ':'))

            print(f"Assigned ids saved to {json_output_file}")

