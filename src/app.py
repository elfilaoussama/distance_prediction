import os
import json
import pickle
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2
from models.detr_model import DETR
from models.glpn_model import GLPDepth
from models.lstm_model import LSTM_Model
from utils.JSON_output import generate_output_json
from utils.processing import PROCESSING
from config import CONFIG

app = FastAPI(
    title="WebSocket Image Upload API",
    description="API for uploading images via WebSocket and receiving object detection and depth estimation results."
)

# Load the models and configurations
device = CONFIG['device']
print("Loading models...")

try:
    detr = DETR()
    print("DETR model loaded.")
    
    glpn = GLPDepth()
    print("GLPDepth model loaded.")
    
    zlocE = LSTM_Model()
    print("LSTM model loaded.")
    
    scaler = pickle.load(open(CONFIG['lstm_scaler_path'], 'rb'))
    print("Scaler loaded.")
    
    processing = PROCESSING()
    print("Processing utilities loaded.")

except Exception as e:
    print(f"Error loading models or utilities: {e}")


# Serve the HTML documentation
@app.get("/", response_class=HTMLResponse)
async def get_docs():
    html_path = os.path.join(os.path.dirname(__file__), "docs.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="docs.html file not found", status_code=404)
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


# WebSocket endpoint for receiving image and returning predictions
@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            try:
                # Receive raw bytes (image data)
                image_bytes = await websocket.receive_bytes()
                
                # Convert bytes to a NumPy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                
                # Decode the image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Failed to decode image")

                # Resize the image (if necessary) and convert to RGB
                frame = cv2.resize(frame, (1280, 640))
                color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(color_converted)
                img_shape = color_converted.shape[0:2]  # (height, width)

                # DETR Object Detection
                scores, boxes = detr.detect(pil_image)

                # GLPN Depth Estimation
                depth_map = glpn.predict(pil_image, img_shape)

                # Process bounding boxes and overlap them with depth map
                pdata = processing.process_detections(scores, boxes, depth_map, detr)

                # Generate the output JSON
                output_json = generate_output_json(pdata, zlocE, scaler)


                # Visualize the image with bounding boxes
                img_base64 = detr.visualize(pil_image, scores, boxes)

                # Add the base64 image to the output JSON
                output_json['img_base64'] = img_base64

                # Send the output back to the client (JSON result)
                await websocket.send_text(json.dumps(output_json))

            except ValueError as ve:
                await websocket.send_text(f"Image processing error: {ve}")
            except Exception as e:
                await websocket.send_text(f"Unexpected error: {str(e)}")

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()
