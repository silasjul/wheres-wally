from sahi.scripts.slice_coco import slice_coco
import json
import os
import shutil
import random

def slice_coco_dataset():
    # This takes your large images and labels and creates the 640x640 tiles
    slice_coco(
        coco_annotation_file_path="data/train/_annotations.coco.json",
        image_dir="data/train/",
        output_dir="data/train/sliced/",
        output_coco_annotation_file_name="sliced_annotations.coco.json",
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,  # 20% overlap
        overlap_width_ratio=0.2,
        min_area_ratio=0.8,  # Keep boxes even if only 80% of Waldo is in the tile
    )

def filter_sliced_dataset():
    # 1. Load the sliced COCO JSON
    with open("data/train/sliced/sliced_annotations.coco.json_coco.json", "r") as f:
        data = json.load(f)

    # 2. Find which image IDs actually have annotations
    images_with_waldo = {ann['image_id'] for ann in data['annotations']}

    # 3. Filter the image list to only include those with Waldo
    new_images = [img for img in data['images'] if img['id'] in images_with_waldo]
    new_annotations = [ann for ann in data['annotations'] if ann['image_id'] in images_with_waldo]

    # 4. Save the filtered JSON
    filtered_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data['categories']
    }

    with open("data/train/sliced/filtered_annotations.coco.json", "w") as f:
        json.dump(filtered_data, f)

    # 5. (Optional) Move only the needed image files to a clean directory
    os.makedirs("data/train/sliced_filtered/", exist_ok=True)
    for img_info in new_images:
        shutil.copy(
            os.path.join("data/train/sliced/", img_info['file_name']),
            os.path.join("data/train/sliced_filtered/", img_info['file_name'])
        )

    print(f"Done! Kept {len(new_images)} images containing Waldo.")

def balance_dataset():
    ORIGINAL_SLICED_JSON = "data/train/sliced/sliced_annotations.coco.json_coco.json"
    FILTERED_JSON_PATH = "data/train/sliced_filtered/filtered_annotations.coco.json"
    IMAGE_SRC_DIR = "data/train/sliced/"
    IMAGE_DST_DIR = "data/train/sliced_filtered/"
    TARGET_BG_PERCENTAGE = 0.10  # 10%

    # 1. Load the current Waldo-only data
    with open(FILTERED_JSON_PATH, "r") as f:
        filtered_data = json.load(f)
    
    # 2. Load all sliced data to find potential backgrounds
    with open(ORIGINAL_SLICED_JSON, "r") as f:
        original_data = json.load(f)

    pos_count = len(filtered_data["images"])
    print(f"Current Waldo images: {pos_count}")

    # 3. Calculate how many background images are needed
    # Formula: BG = (Target% * Pos) / (1 - Target%)
    # For 10%, it's essentially Pos / 9
    n_bg_needed = int((TARGET_BG_PERCENTAGE * pos_count) / (1 - TARGET_BG_PERCENTAGE))
    print(f"Target background images needed (10%): {n_bg_needed}")

    # 4. Identify all purely empty images
    annotated_ids = {ann['image_id'] for ann in original_data['annotations']}
    empty_pool = [img for img in original_data['images'] if img['id'] not in annotated_ids]
    
    if len(empty_pool) < n_bg_needed:
        print(f"Warning: Only found {len(empty_pool)} empty images. Using all of them.")
        n_bg_needed = len(empty_pool)

    # 5. Randomly pick the background images
    bg_samples = random.sample(empty_pool, n_bg_needed)

    # 6. Update JSON and move files
    for img_info in bg_samples:
        # Add image info to the filtered JSON
        filtered_data["images"].append(img_info)
        
        # Copy the physical file
        src_path = os.path.join(IMAGE_SRC_DIR, img_info['file_name'])
        dst_path = os.path.join(IMAGE_DST_DIR, img_info['file_name'])
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    # 7. Save updated JSON
    with open(FILTERED_JSON_PATH, "w") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Successfully added {n_bg_needed} background images.")
    print(f"Final dataset size: {len(filtered_data['images'])} images.")

if __name__ == "__main__":
    balance_dataset()