import os
import json
import torch
import io
from PIL import Image
from cog import BasePredictor, Input
import cv2
from models.detr_model import load_detr_model, detect_objects
from models.glpn_model import load_glpn_model, estimate_depth
from models.lstm_model import LSTMModel
from utils.depth_processing import overlap_depth_with_boxes
from utils.visualization import plot_overlapping_boxes
from config import CONFIG
from utils.classes import CLASSES



class Predictor(BasePredictor):
    def setup(self):
        # Load the models
        global device, detr_model, detr_processor, glpn_model, glpn_extractor, distance_model
        device = CONFIG['device']
        detr_model, detr_processor = load_detr_model(device)
        glpn_model, glpn_extractor = load_glpn_model(device)
        distance_model = LSTMModel(CONFIG['lstm_model_path'])

    def predict(self,
                image: bytes = Input(description="Image to process"),
                scale: float = Input(description="Factor to scale image by", default=1.5)
    ) -> dict:
        # Load the image from bytes
        sample_image = Image.open(io.BytesIO(image))  # Convert image bytes to PIL Image

        # Scale the image
        original_size = sample_image.size  # (width, height)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        scaled_image = sample_image.resize(new_size, Image.BILINEAR)  # Resize the image with bilinear interpolation

        # DETR Object Detection
        scores, boxes, img_shape, detected_classes = detect_objects(scaled_image, detr_model, detr_processor, device)

        # GLPN Depth Estimation
        depth_map = estimate_depth(scaled_image, glpn_model, glpn_extractor, new_size, device)

        # Resize depth map to match the scaled image size
        depth_map_resized = cv2.resize(depth_map, new_size, interpolation=cv2.INTER_LINEAR)

        # Overlap depth map and bounding boxes, and get depth features
        depth_features_matrix = overlap_depth_with_boxes(depth_map_resized, boxes)

        # Prepare JSON output for each object
        output_json = []
        for i, features in enumerate(depth_features_matrix):
            features = torch.tensor(features).unsqueeze(0).unsqueeze(0)  # Convert features to tensor and add batch and sequence dims
            distance = distance_model.predict(features)  # Predict distance for each object
            
            object_info = {
                "class": CLASSES[detected_classes[i].item()],  # Convert class tensor to scalar value
                "distance_estimated": distance.item(),  # Get predicted distance
                "features": {
                    "xmin": features[0][0][0].item(),
                    "ymin": features[0][0][1].item(),
                    "xmax": features[0][0][2].item(),
                    "ymax": features[0][0][3].item(),
                    "mean_depth": features[0][0][4].item(),
                    "min_depth": features[0][0][5].item(),
                    "max_depth": features[0][0][6].item(),
                    "width": features[0][0][7].item(),
                    "height": features[0][0][8].item()
                }
            }
            output_json.append(object_info)

        return output_json