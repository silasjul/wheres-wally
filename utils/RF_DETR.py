import supervision as sv
from rfdetr import RFDETRMedium, RFDETRLarge
from PIL import Image
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision
import torch
import cv2
import numpy as np

class RF_DETR:
    def __init__(self, model_size: str = "medium"):
        self.model = self._load_model(model_size)
        self.images_dir = "data/test"
        self.annotations_path = "data/test/_annotations.coco.json"
        self.color_palette = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
            "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
        ])

    def _load_model(self, model_size: str = "medium"):
        print(f"Loading {model_size} model...")
        
        if model_size == "medium":
            model = RFDETRMedium(num_classes=5)
            weights_path = "models/checkpoint_best_total_medium.pth"
        elif model_size == "large":
            model = RFDETRLarge(num_classes=5)
            # This file MUST be the one with 5 classes
            weights_path = "models/checkpoint_best_total_large.pth" 
        else:
            raise ValueError(f"Invalid model size: {model_size}")
        
        # 1. Resize the model's brain to 5 classes
        try:
            model.model.model.reinitialize_detection_head(num_classes=5)
            print("✅ Resized detection head to 5 classes.")
        except AttributeError:
            print("⚠️ Could not resize head. Proceeding...")

        # 2. Load the file
        print(f"📂 Reading: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cuda", weights_only=False)
        
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        # 3. Filter and Match Keys (THE FIX)
        model_state_dict = model.model.model.state_dict()
        
        # Prepare the new dict
        filtered_state_dict = {}
        mismatch_count = 0
        
        for k, v in state_dict.items():
            # Fix prefix issues (remove "model." if it exists in file but not in code)
            if k.startswith("model.") and k[6:] in model_state_dict:
                key_name = k[6:]
            elif k in model_state_dict:
                key_name = k
            else:
                continue # Skip keys that don't exist in your model at all
            
            # CHECK SHAPE: Only load if the shape matches perfectly
            if v.shape == model_state_dict[key_name].shape:
                filtered_state_dict[key_name] = v
            else:
                # This catches the 91 vs 5 class mismatch
                print(f"⚠️ Skipping layer {key_name}: Shape mismatch {v.shape} vs {model_state_dict[key_name].shape}")
                mismatch_count += 1

        # 4. Load with strict=False
        # We use strict=False because we INTENTIONALLY skipped the mismatching head layers
        model.model.model.load_state_dict(filtered_state_dict, strict=False)
        
        if mismatch_count > 0:
            print(f"✅ Loaded matching weights. Skipped {mismatch_count} mismatched layers (likely the class head).")
            print("👉 NOTE: The detection head is now UNTRAINED. If you expected a trained model, check your file path!")
        else:
            print("✅ Perfect match! All weights loaded.")

        model.model.model.to("cuda")
        model.model.model.eval()
        
        return model

    def _make_slicer(self, conf: float = 0.1) -> sv.InferenceSlicer:
        # RF-DETR expects RGB, but OpenCV gives BGR
        def slicer_callback(image_slice: np.ndarray) -> sv.Detections:
            image_rgb = cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB)
            return self.model.predict(image_rgb, conf=conf)

        # overlap_wh expects pixel overlap, so we approximate 20% of 640px
        overlap_wh = (128, 128)

        return sv.InferenceSlicer(
            callback=slicer_callback,
            slice_wh=(640, 640),
            overlap_wh=overlap_wh,
            iou_threshold=0.5
        )

    def _get_class_names(self) -> list[str]:
        ds = sv.DetectionDataset.from_coco(
            images_directory_path=self.images_dir,
            annotations_path=self.annotations_path,
        )
        class_names = ds.classes
        if isinstance(class_names, dict):
            class_names = list(class_names.values())
        return class_names

    def predict(self, image_path: str) -> None:
        """
        Predict on a single large image using the same slicing strategy
        as the test dataset and show the annotated prediction.
        """
        slicer = self._make_slicer()
        class_names = self._get_class_names()

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        detections = slicer(image)

        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(image.shape[1], image.shape[0]))
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(image.shape[1], image.shape[0]))

        bbox_annotator = sv.BoxAnnotator(color=self.color_palette, thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            color=self.color_palette, text_color=sv.Color.BLACK, text_scale=text_scale
        )

        detections_labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        annotated_pred = image.copy()
        annotated_pred = bbox_annotator.annotate(annotated_pred, detections)
        annotated_pred = label_annotator.annotate(annotated_pred, detections, detections_labels)

        sv.plot_image(image=annotated_pred)

    def metrics_test(self):
        slicer = self._make_slicer(conf=0.0)

        ds = sv.DetectionDataset.from_coco(
            images_directory_path=self.images_dir,
            annotations_path=self.annotations_path,
        )

        targets = []
        predictions = []

        for _, image, annotations in tqdm(ds):
            detections = slicer(image)

            targets.append(annotations)
            predictions.append(detections)

        map_metric = MeanAveragePrecision()
        map_result = map_metric.update(predictions, targets).compute()
        print(map_result)

    def visual_comparison_test(self):
        slicer = self._make_slicer()

        # --- 3. LOAD DATASET ---
        ds = sv.DetectionDataset.from_coco(
            images_directory_path=self.images_dir,
            annotations_path=self.annotations_path,
        )
        
        # Get class names
        class_names = ds.classes
        if isinstance(class_names, dict):
            class_names = list(class_names.values())
            
        print(f"🔎 Dataset Classes found: {class_names}")
        print(f"🔎 Total classes: {len(class_names)}")

        # --- 4. RUN INFERENCE ---
        num_images = len(ds)
        print(f"Found {num_images} test images. Running visual comparison...")

        for idx in range(num_images):
            path, image, annotations = ds[idx]
            
            print(f"[{idx + 1}/{num_images}] Slicing and scanning: {path}...")
            
            # RUN THE SLICER
            detections = slicer(image)

            # --- 5. VISUALIZATION ---
            print("Generating visualization...")
            
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=(image.shape[1], image.shape[0]))
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(image.shape[1], image.shape[0]))

            bbox_annotator = sv.BoxAnnotator(color=self.color_palette, thickness=thickness)
            label_annotator = sv.LabelAnnotator(
                color=self.color_palette, text_color=sv.Color.BLACK, text_scale=text_scale
            )

            # --- Prepare Labels (THE FIX IS HERE) ---
            # 1. Ground Truth Labels
            annotations_labels = []
            for class_id in annotations.class_id:
                if class_id < len(class_names):
                    annotations_labels.append(f"{class_names[class_id]}")
                else:
                    annotations_labels.append(f"GT-ID:{class_id}?")

            # 2. Model Prediction Labels (Safe Mode)
            detections_labels = []
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                if class_id < len(class_names):
                    label = f"{class_names[class_id]} {confidence:.2f}"
                else:
                    # If ID is out of range, show the raw number so we can debug it
                    label = f"UNK-ID:{class_id} {confidence:.2f}" 
                detections_labels.append(label)

            # --- Annotate ---
            annotated_gt = image.copy()
            annotated_gt = bbox_annotator.annotate(annotated_gt, annotations)
            annotated_gt = label_annotator.annotate(annotated_gt, annotations, annotations_labels)

            annotated_pred = image.copy()
            annotated_pred = bbox_annotator.annotate(annotated_pred, detections)
            annotated_pred = label_annotator.annotate(annotated_pred, detections, detections_labels)

            # --- 6. DISPLAY ---
            # Save strictly to avoid blocking popup windows if running headless
            # But sv.plot_images_grid is fine if you are in a notebook/desktop
            sv.plot_images_grid(
                images=[annotated_gt, annotated_pred], 
                grid_size=(1, 2), 
                titles=["True Annotations", "Prediction"]
            )

if __name__ == "__main__":
    rf_detr = RF_DETR(model_size="medium")
    rf_detr.metrics_test()
    rf_detr.visual_comparison_test()

    # Test images
    rf_detr.predict("original_images/9.jpg")
    rf_detr.predict("original_images/10.jpg")

    # Valid images
    rf_detr.predict("original_images/7.jpg")
    rf_detr.predict("original_images/11.jpg")

    # Train images
    """ rf_detr.predict("original_images/1.jpg")
    rf_detr.predict("original_images/2.jpg")
    rf_detr.predict("original_images/3.jpg")
    rf_detr.predict("original_images/4.jpg")
    rf_detr.predict("original_images/5.jpg")
    rf_detr.predict("original_images/6.jpg")
    rf_detr.predict("original_images/8.jpg")
    rf_detr.predict("original_images/12.jpg") """