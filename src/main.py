import os
from PIL import Image
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from io import BytesIO
from models.detr_model import load_detr_model, detect_objects
from models.glpn_model import load_glpn_model, estimate_depth
from models.lstm_model import LSTMModel
from utils.depth_processing import overlap_depth_with_boxes
from config import CONFIG
from utils.classes import CLASSES

app = FastAPI()

# Load the models
device = CONFIG['device']
detr_model, detr_processor = load_detr_model(device)
glpn_model, glpn_extractor = load_glpn_model(device)
distance_model = LSTMModel(CONFIG['lstm_model_path'])

@app.get("/")
def read_root():
    file_path = os.path.join(os.path.dirname(__file__), "api_documentation.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Documentation file not found")
    with open(file_path) as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load the image
    image = Image.open(file.file).convert('RGB')

    # DETR Object Detection
    scores, boxes, img_shape, detected_classes = detect_objects(image, detr_model, detr_processor, device)

    # GLPN Depth Estimation
    depth_map = estimate_depth(image, glpn_model, glpn_extractor, img_shape, device)

    # Resize depth map to match the original image size
    depth_map_resized = cv2.resize(depth_map, image.size, interpolation=cv2.INTER_LINEAR)

    # Overlap depth map and bounding boxes, and get depth features
    depth_features_matrix = overlap_depth_with_boxes(depth_map_resized, boxes)

    # Prepare JSON output for each object
    output_json = []
    for i, features in enumerate(depth_features_matrix):
        distance = distance_model.predict(features.unsqueeze(0).unsqueeze(0))  # Predict distance for each object
        object_info = {
            "class": CLASSES[detected_classes[i].item()],
            "distance_estimated": distance.item(),
            "features": {
                "xmin": features[0].item(),
                "ymin": features[1].item(),
                "xmax": features[2].item(),
                "ymax": features[3].item(),
                "mean_depth": features[6].item(),
                "min_depth": features[7].item(),
                "max_depth": features[8].item(),
                "width": features[4].item(),
                "height": features[5].item()
            }
        }
        output_json.append(object_info)

    return {"objects": output_json}

@app.websocket("/ws/predict/")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            image = Image.open(BytesIO(data)).convert('RGB')

            # DETR Object Detection
            scores, boxes, img_shape, detected_classes = detect_objects(image, detr_model, detr_processor, device)

            # GLPN Depth Estimation
            depth_map = estimate_depth(image, glpn_model, glpn_extractor, img_shape, device)

            # Resize depth map to match the original image size
            depth_map_resized = cv2.resize(depth_map, image.size, interpolation=cv2.INTER_LINEAR)

            # Overlap depth map and bounding boxes, and get depth features
            depth_features_matrix = overlap_depth_with_boxes(depth_map_resized, boxes)

            # Prepare JSON output for each object
            output_json = []
            for i, features in enumerate(depth_features_matrix):
                distance = distance_model.predict(features.unsqueeze(0).unsqueeze(0))  # Predict distance for each object
                object_info = {
                    "class": CLASSES[detected_classes[i].item()],
                    "distance_estimated": distance.item(),
                    "features": {
                        "xmin": features[0].item(),
                        "ymin": features[1].item(),
                        "xmax": features[2].item(),
                        "ymax": features[3].item(),
                        "mean_depth": features[6].item(),
                        "min_depth": features[7].item(),
                        "max_depth": features[8].item(),
                        "width": features[4].item(),
                        "height": features[5].item()
                    }
                }
                output_json.append(object_info)

            await websocket.send_json({"objects": output_json})

    except WebSocketDisconnect:
        print("Client disconnected")


