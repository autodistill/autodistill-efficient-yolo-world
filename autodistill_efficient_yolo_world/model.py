import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.helpers import load_image
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill_yolo_world import YOLOWorldModel
from autodistill_efficientsam import EfficientSAM
import numpy as np

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EfficientYOLOWorld(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.detection_model = YOLOWorldModel(ontology=ontology)
        self.segmentation_model = EfficientSAM(None)

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input)

        result = self.detection_model.predict(load_image(input), confidence=0.1).with_nms()

        result.mask = np.array([None] * len(result.xyxy))

        for i, [x_min, y_min, x_max, y_max] in enumerate(result.xyxy):
            y_min, y_max = int(y_min), int(y_max)
            x_min, x_max = int(x_min), int(x_max)
            input_image = image[y_min:y_max, x_min:x_max]
            mask = self.segmentation_model.predict(input_image)
            full_mask = np.zeros((image.shape[0], image.shape[1]))

            full_mask[y_min:y_max, x_min:x_max] = mask.mask[0]
            result.mask[i] = full_mask.astype(bool)

        return result