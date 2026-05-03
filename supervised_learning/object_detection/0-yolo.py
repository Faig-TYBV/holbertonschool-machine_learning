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
        anchors     (np.ndarray):  All anchor boxes, shape
        (outputs, anchor_boxes, 2).
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
        # ── Load the pre-trained Darknet model ──────────────────────────────
        self.model = keras.models.load_model(model_path)

        # ── Load class names (strip whitespace / blank lines) ───────────────
        with open(classes_path, "r") as fh:
            self.class_names = [
                line.strip() for line in fh.readlines() if line.strip()
            ]

        # ── Store scalar hyper-parameters ───────────────────────────────────
        self.class_t = class_t
        self.nms_t = nms_t

        # ── Store anchor boxes ───────────────────────────────────────────────
        self.anchors = anchors
