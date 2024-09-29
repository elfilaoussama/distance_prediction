import torch
import cv2

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
