import supervision as sv
from rfdetr import RFDETRMedium
from PIL import Image
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision
import torch
import cv2
import numpy as np

class RF_DETR:
    def __init__(self):
        self.model = self._load_model()
        self.images_dir = "data/test"
        self.annotations_path = "data/test/_annotations.coco.json"
        self.color_palette = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
            "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
        ])

    def _load_model(self):
        print("Loading model...")
        model = RFDETRMedium(num_classes=5)
        weights_path = "models/checkpoint_best_total.pth"
        checkpoint = torch.load(weights_path, map_location="cuda", weights_only=False)

        try:
            model.model.reinitialize_detection_head(num_classes=5)
        except AttributeError:
            pass 

        model.model.model.load_state_dict(checkpoint['model'])
        model.model.model.to("cuda")
        model.model.model.eval()
        print("Model loaded!")

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
        slicer = self._make_slicer(conf=0.1)

        # --- 3. LOAD DATASET (To get Ground Truth) ---
        ds = sv.DetectionDataset.from_coco(
            images_directory_path=self.images_dir,
            annotations_path=self.annotations_path,
        )
        
        # Get class names
        class_names = ds.classes
        if isinstance(class_names, dict):
            class_names = list(class_names.values())

        # --- 4. RUN INFERENCE ON ALL TEST IMAGES ---
        num_images = len(ds)
        print(f"Found {num_images} test images. Running visual comparison on all...")

        for idx in range(num_images):
            # path: file path, image: huge numpy array, annotations: ground truth
            path, image, annotations = ds[idx]
            
            print(f"[{idx + 1}/{num_images}] Slicing and scanning massive image: {path}...")
            
            # RUN THE SLICER (This takes a moment as it scans the whole map)
            detections = slicer(image)

            # --- 5. VISUALIZATION (Per Image) ---
            print("Generating visualization...")
            
            # Calculate scale based on the current image size
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=(image.shape[1], image.shape[0]))
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(image.shape[1], image.shape[0]))

            bbox_annotator = sv.BoxAnnotator(color=self.color_palette, thickness=thickness)
            label_annotator = sv.LabelAnnotator(
                color=self.color_palette, text_color=sv.Color.BLACK, text_scale=text_scale
            )

            # --- Prepare Labels ---
            # Ground Truth Labels
            annotations_labels = [f"{class_names[class_id]}" for class_id in annotations.class_id]

            # Model Prediction Labels
            detections_labels = [
                f"{class_names[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            # --- Annotate ---
            # Left: Ground Truth
            annotated_gt = image.copy()
            annotated_gt = bbox_annotator.annotate(annotated_gt, annotations)
            annotated_gt = label_annotator.annotate(annotated_gt, annotations, annotations_labels)

            # Right: Sliced Inference Result
            annotated_pred = image.copy()
            annotated_pred = bbox_annotator.annotate(annotated_pred, detections)
            annotated_pred = label_annotator.annotate(annotated_pred, detections, detections_labels)

            # --- 6. DISPLAY ---
            sv.plot_images_grid(
                images=[annotated_gt, annotated_pred], 
                grid_size=(1, 2), 
                titles=["True Annotations", "Prediction"]
            )

if __name__ == "__main__":
    rf_detr = RF_DETR()
    rf_detr.metrics_test()
    rf_detr.visual_comparison_test()