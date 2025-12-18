from sahi.scripts.slice_coco import slice_coco
import json
import os
import shutil
import random

def slice_coco_dataset(prune_background: bool = True) -> None:
    # Train
    slice_coco(
        coco_annotation_file_path="data/train/_annotations.coco.json",
        image_dir="data/train",
        output_dir="data/train_sliced",
        output_coco_annotation_file_name="_annotations",
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,  # 20% overlap
        overlap_width_ratio=0.2,
        min_area_ratio=0.8,  # Keep boxes even if only 80% of Waldo is in the tile
    )

    # Validation
    slice_coco(
        coco_annotation_file_path="data/valid/_annotations.coco.json",
        image_dir="data/valid",
        output_dir="data/valid_sliced",
        output_coco_annotation_file_name="_annotations",
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        min_area_ratio=0.8,
    )

    # Test
    slice_coco(
        coco_annotation_file_path="data/test/_annotations.coco.json",
        image_dir="data/test",
        output_dir="data/test_sliced",
        output_coco_annotation_file_name="_annotations",
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        min_area_ratio=0.8,
    )

def balance_dataset() -> None:
    # Define the directories and annotation files created by your slice_coco_dataset function
    dataset_splits = [
        {"dir": "data/train_sliced", "json": "data/train_sliced/_annotations_coco.json"},
        {"dir": "data/valid_sliced", "json": "data/valid_sliced/_annotations_coco.json"},
        {"dir": "data/test_sliced", "json": "data/test_sliced/_annotations_coco.json"}
    ]

    TARGET_BG_RATIO = 0.10  # We want 10% of the TOTAL images to be background

    for split in dataset_splits:
        json_path = split["json"]
        img_dir = split["dir"]

        if not os.path.exists(json_path):
            print(f"Skipping {json_path}: File not found.")
            continue

        print(f"Processing {img_dir}...")

        # 1. Load the sliced COCO JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        # 2. Separate Waldo images (positives) from empty ones (backgrounds)
        annotated_image_ids = {ann['image_id'] for ann in data['annotations']}
        all_images = data['images']
        
        positive_images = [img for img in all_images if img['id'] in annotated_image_ids]
        background_images = [img for img in all_images if img['id'] not in annotated_image_ids]

        n_pos = len(positive_images)
        n_bg_original = len(background_images)

        # 3. Calculate target background count
        # Formula: BG = (Ratio * Pos) / (1 - Ratio)
        # For 10%, this is (0.1 * n_pos) / 0.9, or simply n_pos / 9
        n_bg_target = int((TARGET_BG_RATIO * n_pos) / (1 - TARGET_BG_RATIO))

        if n_bg_original <= n_bg_target:
            print(f"  Already balanced! ({n_bg_original} backgrounds for {n_pos} positives).")
            continue

        # 4. Pick which backgrounds to KEEP and which to DELETE
        bg_to_keep = random.sample(background_images, n_bg_target)
        keep_ids = {img['id'] for img in bg_to_keep} | annotated_image_ids
        
        images_to_delete = [img for img in background_images if img['id'] not in keep_ids]

        print(f"  Positives: {n_pos} | Backgrounds: {n_bg_original} -> Target: {n_bg_target}")
        print(f"  Deleting {len(images_to_delete)} excess background images...")

        # 5. Physically delete the files
        for img in images_to_delete:
            file_path = os.path.join(img_dir, img['file_name'])
            if os.path.exists(file_path):
                os.remove(file_path)

        # 6. Update the JSON data to match the surviving files
        data['images'] = [img for img in all_images if img['id'] in keep_ids]

        # 7. Save the updated JSON in-place
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"  Successfully pruned {img_dir}. Final total: {len(data['images'])} images.")


if __name__ == "__main__":
    slice_coco_dataset()
    balance_dataset()