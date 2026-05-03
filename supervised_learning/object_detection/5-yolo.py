#!/usr/bin/env python3
"""
YOLO v3 Object Detection
"""

import numpy as np
from tensorflow import keras


class Yolo:
    """
    Uses the YOLO v3 algorithm to perform object detection.

    Attributes:
        model       (keras.Model): Loaded Darknet Keras model.
        class_names (list[str]):   Ordered list of class names.
        class_t     (float):       Box-score threshold for initial filtering.
        nms_t       (float):       IoU threshold for non-max suppression.
        anchors     (np.ndarray):  Anchor boxes,
                                   shape (outputs, anchor_boxes, 2).
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialise a Yolo detector.

        Args:
            model_path   (str):        Path to the Darknet Keras model (.h5).
            classes_path (str):        Path to the file that lists class names
                                       (one name per line, in index order).
            class_t      (float):      Box score threshold used to discard
                                       low-confidence detections.
            nms_t        (float):      IoU threshold used during non-max
                                       suppression to remove overlapping boxes.
            anchors      (np.ndarray): Anchor boxes with shape
                                       (outputs, anchor_boxes, 2) where the
                                       last dimension is [width, height].
        """
        self.model = keras.models.load_model(model_path)

        with open(classes_path, "r") as fh:
            self.class_names = [
                line.strip() for line in fh.readlines() if line.strip()
            ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process raw predictions from the Darknet model for a single image.

        YOLOv3 encodes bounding boxes as offsets (t_x, t_y, t_w, t_h)
        relative to each grid cell and anchor box. This method decodes those
        offsets into absolute pixel coordinates expressed relative to the
        original (pre-resized) image dimensions.

        Decoding equations (from the YOLOv3 paper):
            b_x = sigmoid(t_x) + c_x
            b_y = sigmoid(t_y) + c_y
            b_w = p_w * exp(t_w)
            b_h = p_h * exp(t_h)

        where (c_x, c_y) is the top-left corner of the grid cell (in grid
        units) and (p_w, p_h) is the prior anchor size (in input-image px).
        The decoded values are then scaled to the original image size.

        Args:
            outputs (list[np.ndarray]): Raw model outputs, one array per
                detection scale. Each array has shape:
                (grid_h, grid_w, anchor_boxes, 5 + classes)
                where the last axis is
                [t_x, t_y, t_w, t_h, box_conf, *class_probs].
            image_size (np.ndarray): Original image dimensions
                [image_h, image_w].

        Returns:
            tuple:
                boxes (list[np.ndarray]): Decoded boxes in original-image
                    pixel coordinates (x1, y1, x2, y2), one array of shape
                    (grid_h, grid_w, anchor_boxes, 4) per output scale.
                box_confidences (list[np.ndarray]): Objectness scores after
                    sigmoid, shape (grid_h, grid_w, anchor_boxes, 1) per
                    scale.
                box_class_probs (list[np.ndarray]): Per-class probabilities
                    after sigmoid, shape
                    (grid_h, grid_w, anchor_boxes, classes) per scale.
        """
        image_height, image_width = image_size

        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        boxes = []
        box_confidences = []
        box_class_probs = []

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # 1. Isolate the raw fields
            t_xy = output[..., :2]       # (grid_h, grid_w, ab, 2)
            t_wh = output[..., 2:4]      # (grid_h, grid_w, ab, 2)
            t_conf = output[..., 4:5]    # (grid_h, grid_w, ab, 1)
            t_probs = output[..., 5:]    # (grid_h, grid_w, ab, classes)

            # 2. Build grid-cell offsets c_x (col index) and c_y (row index)
            col_offsets = np.arange(grid_width)
            row_offsets = np.arange(grid_height)
            c_x, c_y = np.meshgrid(col_offsets, row_offsets)

            # Reshape to (grid_h, grid_w, 1, 1) for broadcasting
            c_x = c_x.reshape(grid_height, grid_width, 1, 1)
            c_y = c_y.reshape(grid_height, grid_width, 1, 1)
            cell_offsets = np.concatenate([c_x, c_y], axis=-1)

            # 3. Decode centre coordinates (in grid units), then normalise
            #    b_x = sigmoid(t_x) + c_x
            #    b_y = sigmoid(t_y) + c_y
            b_xy = self._sigmoid(t_xy) + cell_offsets
            b_xy_norm = b_xy / np.array([grid_width, grid_height])

            # 4. Decode width / height, normalised by the model input size
            #    b_w = p_w * exp(t_w)
            #    b_h = p_h * exp(t_h)
            anchors_for_output = self.anchors[idx]
            b_wh = (
                anchors_for_output * np.exp(t_wh)
            ) / np.array([input_width, input_height])

            # 5. Convert (centre_x, centre_y, w, h) -> (x1, y1, x2, y2)
            box_xy1 = b_xy_norm - b_wh / 2    # top-left corner
            box_xy2 = b_xy_norm + b_wh / 2    # bottom-right corner

            # 6. Scale to original image pixel coordinates
            box_xy1 *= np.array([image_width, image_height])
            box_xy2 *= np.array([image_width, image_height])

            decoded_boxes = np.concatenate([box_xy1, box_xy2], axis=-1)
            boxes.append(decoded_boxes)

            # 7. Sigmoid on confidence and class probabilities
            box_confidences.append(self._sigmoid(t_conf))
            box_class_probs.append(self._sigmoid(t_probs))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter bounding boxes by score threshold (self.class_t).

        The score for each box is computed as:
            score = box_confidence * class_probability

        Only boxes whose highest class score meets or exceeds self.class_t
        are kept.

        Args:
            boxes (list[np.ndarray]): Decoded boundary boxes, one array of
                shape (grid_h, grid_w, anchor_boxes, 4) per output scale.
            box_confidences (list[np.ndarray]): Objectness scores, one array
                of shape (grid_h, grid_w, anchor_boxes, 1) per scale.
            box_class_probs (list[np.ndarray]): Class probabilities, one
                array of shape (grid_h, grid_w, anchor_boxes, classes)
                per scale.

        Returns:
            tuple:
                filtered_boxes (np.ndarray): Shape (?, 4) — coordinates of
                    every box that survived the threshold cut.
                box_classes (np.ndarray): Shape (?,) — index of the highest-
                    scoring class for each surviving box.
                box_scores (np.ndarray): Shape (?,) — the corresponding
                    highest class score for each surviving box.
        """
        all_boxes = []
        all_classes = []
        all_scores = []

        for boxes_s, confs_s, probs_s in zip(
            boxes, box_confidences, box_class_probs
        ):
            # Compute per-class scores: confidence * class probability
            # confs_s shape: (grid_h, grid_w, ab, 1)
            # probs_s shape: (grid_h, grid_w, ab, classes)
            # scores  shape: (grid_h, grid_w, ab, classes)
            scores = confs_s * probs_s

            # Best class and its score for every anchor position
            # Both resulting arrays: (grid_h, grid_w, ab)
            box_classes = np.argmax(scores, axis=-1)
            box_scores = np.max(scores, axis=-1)

            # Boolean mask of positions that exceed the threshold
            mask = box_scores >= self.class_t

            # Apply the mask to select surviving boxes
            # boxes_s shape: (grid_h, grid_w, ab, 4)
            all_boxes.append(boxes_s[mask])       # (kept, 4)
            all_classes.append(box_classes[mask])  # (kept,)
            all_scores.append(box_scores[mask])    # (kept,)

        # Concatenate results from all output scales into flat arrays
        filtered_boxes = np.concatenate(all_boxes, axis=0)
        box_classes = np.concatenate(all_classes, axis=0)
        box_scores = np.concatenate(all_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Max Suppression (NMS) per class.

        For each class, boxes are sorted by descending score.  The highest-
        scoring box is always kept; any remaining box whose IoU with a kept
        box exceeds self.nms_t is suppressed as a likely duplicate detection.
        This repeats until no boxes remain for that class.

        Args:
            filtered_boxes (np.ndarray): Shape (?, 4) — (x1, y1, x2, y2)
                boundary boxes that passed the score threshold.
            box_classes (np.ndarray): Shape (?,) — class index for each box.
            box_scores (np.ndarray): Shape (?,) — confidence score for each
                box.

        Returns:
            tuple:
                box_predictions (np.ndarray): Shape (?, 4) — surviving boxes
                    ordered by class then descending score.
                predicted_box_classes (np.ndarray): Shape (?,) — class index
                    for each surviving box, same ordering.
                predicted_box_scores (np.ndarray): Shape (?,) — score for
                    each surviving box, same ordering.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            # --- Isolate all boxes that belong to this class ----------------
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]    # (n, 4)
            cls_scores = box_scores[cls_mask]        # (n,)

            # --- Sort by descending score -----------------------------------
            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            # --- Greedy NMS loop --------------------------------------------
            while len(cls_boxes) > 0:
                # The first box (highest score) is always kept
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                # Compute IoU of the kept box against all remaining boxes
                ious = self._iou(cls_boxes[0], cls_boxes[1:])

                # Keep only boxes whose IoU is below the threshold
                keep = ious < self.nms_t
                cls_boxes = cls_boxes[1:][keep]
                cls_scores = cls_scores[1:][keep]

        box_predictions = np.array(box_predictions)           # (?, 4)
        predicted_box_classes = np.array(predicted_box_classes)  # (?,)
        predicted_box_scores = np.array(predicted_box_scores)    # (?,)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def _iou(box, boxes):
        """
        Compute IoU between a single box and an array of boxes.

        All boxes are in (x1, y1, x2, y2) format.

        Args:
            box (np.ndarray): Shape (4,) — the reference box.
            boxes (np.ndarray): Shape (n, 4) — boxes to compare against.

        Returns:
            np.ndarray: Shape (n,) — IoU scores in [0, 1].
        """
        # Coordinates of the intersection rectangle
        inter_x1 = np.maximum(box[0], boxes[:, 0])
        inter_y1 = np.maximum(box[1], boxes[:, 1])
        inter_x2 = np.minimum(box[2], boxes[:, 2])
        inter_y2 = np.minimum(box[3], boxes[:, 3])

        # Intersection area (zero when boxes do not overlap)
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Areas of both boxes
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (
            boxes[:, 3] - boxes[:, 1]
        )

        # Union area and IoU
        union_area = box_area + boxes_area - inter_area
        return inter_area / union_area

    @staticmethod
    def _sigmoid(x):
        """Numerically stable element-wise sigmoid."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def load_images(folder_path):
        """
        Load every image found directly inside folder_path.

        Supported formats are those recognised by OpenCV (JPEG, PNG, BMP,
        TIFF, WebP, etc.). Non-image files and sub-directories are silently
        skipped.

        Args:
            folder_path (str): Path to the directory that holds the images.

        Returns:
            tuple:
                images (list[np.ndarray]): Loaded images in BGR format, one
                    array per file, in the same order as image_paths.
                image_paths (list[str]): Paths to each successfully loaded
                    image file, in sorted filename order.
        """
        import cv2
        import os

        valid_ext = {
            ".jpg", ".jpeg", ".png", ".bmp",
            ".tiff", ".tif", ".webp"
        }

        images = []
        image_paths = []

        for filename in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_ext:
                continue

            full_path = os.path.join(folder_path, filename)
            if not os.path.isfile(full_path):
                continue

            img = cv2.imread(full_path)
            if img is None:
                continue

            images.append(img)
            image_paths.append(full_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resize and normalise images to match the Darknet model's input spec.

        Each image is:
          1. Resized to (input_h, input_w) using bicubic interpolation, which
             preserves edge sharpness better than bilinear or nearest-neighbour
             when upscaling and produces smoother results when downscaling.
          2. Normalised from uint8 [0, 255] to float32 [0, 1] by dividing by
             255.

        Args:
            images (list[np.ndarray]): Raw BGR images as loaded by OpenCV,
                each of arbitrary height and width.

        Returns:
            tuple:
                pimages (np.ndarray): Shape (ni, input_h, input_w, 3) —
                    stacked preprocessed images in float32.
                image_shapes (np.ndarray): Shape (ni, 2) — original
                    (height, width) of each image before resizing, in the
                    same order as pimages.
        """
        import cv2

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            # Record original dimensions before any transformation
            image_shapes.append(img.shape[:2])   # (height, width)

            # Convert BGR (OpenCV) to RGB (Darknet model expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to model input size using inter-cubic interpolation
            resized = cv2.resize(
                img_rgb,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            # Normalise pixel values from [0, 255] to [0, 1]
            normalised = resized.astype(np.float32) / 255.0

            pimages.append(normalised)

        # Stack into a single 4-D array: (ni, input_h, input_w, 3)
        pimages = np.array(pimages)

        # Convert shapes list to array: (ni, 2)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
