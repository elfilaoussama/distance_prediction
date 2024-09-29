import os
import json
from PIL import Image
import cv2
from models.detr_model import load_detr_model, detect_objects
from models.glpn_model import load_glpn_model, estimate_depth
from models.lstm_model import LSTMModel
from utils.depth_processing import overlap_depth_with_boxes
from utils.visualization import plot_overlapping_boxes
from config import CONFIG
from utils.classes import CLASSES

if __name__ == "__main__":
    # Load the models
    device = CONFIG['device']
    detr_model, detr_processor = load_detr_model(device)
    glpn_model, glpn_extractor = load_glpn_model(device)
    distance_model = LSTMModel(CONFIG['lstm_model_path'])

    # Load a sample image
    sample_image = Image.open(CONFIG['image_path'])

    # DETR Object Detection
    scores, boxes, img_shape, detected_classes = detect_objects(sample_image, detr_model, detr_processor, device)

    # GLPN Depth Estimation
    depth_map = estimate_depth(sample_image, glpn_model, glpn_extractor, img_shape, device)

    # Resize depth map to match the original image size
    depth_map_resized = cv2.resize(depth_map, sample_image.size, interpolation=cv2.INTER_LINEAR)

    # Overlap depth map and bounding boxes, and get depth features
    depth_features_matrix = overlap_depth_with_boxes(depth_map_resized, boxes)

    # Prepare JSON output for each object
    output_json = []
    for i, features in enumerate(depth_features_matrix):
        features = features.unsqueeze(0).unsqueeze(0)  # Add extra dimension for batch and sequence
        distance = distance_model.predict(features)  # Predict distance for each object
        """object_info = {
            "class": CLASSES[detected_classes[i].item()],
            "distance_estimated": distance.item(),
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
        output_json.append(object_info)"""

    # Save results to JSON
    with open('results/output.json', 'w') as f:
        json.dump(output_json, f)
