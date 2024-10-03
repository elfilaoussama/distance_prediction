import os
from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers import GLPNForDepthEstimation, GLPNFeatureExtractor
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cv2.setNumThreads(1)

# Colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'detr_model_path': 'facebook/detr-resnet-101',
    'glpn_model_path': 'vinvino02/glpn-kitti',
    'lstm_model_path': 'models/pretrained_lstm.pth',
    'confidence_threshold': 0.7
}


# Predefined classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Load the models
def load_detr_model():
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101', revision="no_timm")
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101', revision="no_timm")
    model.to(device)  # Move DETR model to CPU
    model.eval()
    return model, processor

def load_glpn_model():
    feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-kitti")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
    model.to(device)  # Move GLPN model to CPU
    model.eval()
    return model, feature_extractor

# Bounding box conversion functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)

    # Move the tensor for size to the same device as out_bbox
    device = out_bbox.device
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    
    return b

# Function to compute depth features for each bounding box (min, max, mean depth)
def compute_depth_features(box, depth_map):
    xmin, ymin, xmax, ymax = map(int, box)
    depth_patch = depth_map[ymin:ymax, xmin:xmax]
    
    if depth_patch.size > 0:  # Ensure the depth patch is not empty
        mean_depth = depth_patch.mean().item()
        min_depth = depth_patch.min().item()
        max_depth = depth_patch.max().item()
    else:
        mean_depth = min_depth = max_depth = 0.0
    
    return mean_depth, min_depth, max_depth

# Function to compute depth features and overlap bounding boxes with depth map
def overlap_depth_with_boxes(depth_map, boxes):
    depth_features_list = []  # Store depth features for each object
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        # Extract depth patch corresponding to the bounding box
        mean_depth, min_depth, max_depth = compute_depth_features(box, depth_map)
        width, height = xmax - xmin, ymax - ymin
        # Store bounding box coordinates, size, and depth features
        # Only include the first 9 features expected by the LSTM model
        features = [xmin, ymin, xmax, ymax, min_depth, mean_depth, max_depth, width, height] #removed extra features, added a zero to keep same size
        
        depth_features_list.append(features)
    
    # Return depth features for all objects in the image
    return torch.tensor(depth_features_list, dtype=torch.float32)

# Separate function to plot overlapping bounding boxes and depth map
def plot_overlapping_boxes(image, depth_map, boxes, detected_classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)  # Display original image
    ax = plt.gca()
    
    # Move boxes to CPU and convert to numpy
    boxes_np = boxes.detach().cpu().numpy()
    
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes_np):
        depth_patch = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)]  # Extract depth patch
        mean_depth = depth_patch.mean() if depth_patch.size else 0.0
        
        # Draw bounding box on the image
        color = COLORS[i % len(COLORS)]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        class_idx = detected_classes[i].item()
        class_name = CLASSES[class_idx]
        ax.text(xmin, ymin, f'{class_name}: Mean Depth {mean_depth:.2f}m', fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        
        # Display the depth patch as an overlay
        plt.imshow(depth_map, cmap='plasma', alpha=0.6)  # Overlay depth map with some transparency

    plt.axis('off')
    plt.show()


# z-location estimator
class Zloc_Estimaotor(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        
        layersize = [306, 154, 76]
        layerlist = []
        n_in = hidden_dim
        for i in layersize:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU())
            n_in = i           
        layerlist.append(nn.Linear(layersize[-1], 1))
        
        self.fc = nn.Sequential(*layerlist)

    def forward(self, x):
        out, hn = self.rnn(x)
        output = self.fc(out[:, -1])
        return output


class LSTMModel():
    def __init__(self, path):
        self.input_dim = 9  # Adjust according to your input features
        self.hidden_dim = 612
        self.layer_dim = 3

        self.model = Zloc_Estimaotor(self.input_dim, self.hidden_dim, self.layer_dim)
        self.model.load_state_dict(torch.load(path, map_location=CONFIG['device']), strict=False)
        self.model.to(device)  # Move the model to the correct device

    def predict(self, input_data):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            input_data = input_data.to(device)  # Move input data to the correct device
            return self.model(input_data)  # Pass the data through the model


# Object Detection using DETR
def detect_objects(img, detr_model, detr_processor):
    img = img.convert('RGB')
    inputs = detr_processor(images=img, return_tensors="pt")
    img_tensor = inputs.pixel_values.to(device)  # Ensure this is on the correct device

    img_shape = img_tensor.shape[-2:]

    outputs = detr_model(img_tensor)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].detach(), img.size)

    # Get class indices of the detected objects
    class_indices = probas.argmax(-1)[keep]

    return probas[keep], bboxes_scaled, img_shape, class_indices


# Depth Estimation using GLPN
def estimate_depth(img, glpn_model, glpn_extractor, img_shape):
    img = img.convert('RGB')  # Ensure the image is in RGB format
    with torch.no_grad():
        pixel_values = glpn_extractor(img, return_tensors="pt").pixel_values.to(device)
        outputs = glpn_model(pixel_values)
        predicted_depth = outputs.predicted_depth

        # Resize prediction to original image shape
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_shape[:2],
            mode="bicubic",
            align_corners=False
        )
        return prediction.squeeze().cpu().numpy()
    
    

class Predictor(BasePredictor):
    def setup(self):
        # Load the models
        global device, detr_model, detr_processor, glpn_model, glpn_extractor, distance_model
        device = CONFIG['device']
        detr_model, detr_processor = load_detr_model(device)
        glpn_model, glpn_extractor = load_glpn_model(device)
        distance_model = LSTMModel(CONFIG['lstm_model_path'])

    def predict(self,
                image: Path = Input(description="Image to process"),  # Change to Path type
                scale: float = Input(description="Factor to scale image by", default=1.5)
    ) -> dict:
        # Load the image from the file path
        sample_image = Image.open(image)  # Image file is opened from the given path

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

   
