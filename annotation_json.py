import os
import json
from pathlib import Path

annotation_type = "attention" # Process touch separately
# Define your object name list
youth_classes = [
    'doll', 'car', 'jumpbox', 'shapebox', 'flower', 'book', 'bottle',
    'green_star', 'yellow_cylinder', 'blue_cube', 'red_triangle',
    'shapebox_lid', 'head_parent', 'head_child'
]

def convert_darklabel_to_json(annotations_dir, participant):
    """
    Convert all DarkLabel-style .txt files of one participant into a single JSON.
    Each line in txt: obj_id att_id x y w h
    Output JSON: { "fr_id": { "objects": [ { "obj_id": ..., "object_name": ..., "attention_id": ... }, ... ] }, ... }
    """
    participant_dir = f"{annotations_dir}/{participant}"
    data = {}
    txt_dir = f"{participant_dir}/{annotation_type}"
    for txt_file in sorted(os.listdir(txt_dir)):
        dr = f"{txt_dir}/{txt_file}"
        with open(dr, "r") as f:
            lines = f.readlines()

        frame_id = str(int(txt_file.split(".txt")[0]))
        objects = []
        for line in lines:
            parts = line.strip().split()

            obj_id = int(parts[0])
            att_id = int(parts[1])
            object_name = youth_classes[obj_id] 

            objects.append({
                "obj_id": obj_id,
                "object_name": object_name,
                f"{annotation_type}_id": att_id
            })

        data[frame_id] = {"objects": objects}

    output_path = os.path.join(annotations_dir, f"{participant}_{annotation_type}.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    basedir = Path(__file__).resolve().parent
    DATA_DIR = os.path.join(basedir, "data")
    participant_dir = f"{DATA_DIR}/extracted_frames"
    participant = next(
        name for name in os.listdir(participant_dir)
        if os.path.isdir(os.path.join(participant_dir, name))
    )

    annotations_dir = basedir /"data"/"annotations"
    convert_darklabel_to_json(annotations_dir, participant)