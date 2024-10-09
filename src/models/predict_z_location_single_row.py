import torch
import numpy as np

def predict_z_location_single_row(row, ZlocE, scaler):
    """
    Preprocess bounding box coordinates, depth information, and class type 
    to predict Z-location using the LSTM model for a single row.
    
    Parameters:
    - row: A single row of DataFrame with bounding box coordinates, depth info, and class type.
    - ZlocE: Pre-loaded LSTM model for Z-location prediction.
    - scaler: Scaler for normalizing input data.
    
    Returns:
    - z_loc_prediction: Predicted Z-location for the given row.
    """
    # One-hot encoding of class type
    class_type = row['class']
    
    if class_type == 'bicycle':
        class_tensor = torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.float32)
    elif class_type == 'car':
        class_tensor = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32)
    elif class_type == 'person':
        class_tensor = torch.tensor([[0, 0, 0, 1, 0, 0]], dtype=torch.float32)
    elif class_type == 'train':
        class_tensor = torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.float32)
    elif class_type == 'truck':
        class_tensor = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    else :
        class_tensor = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32)

    # Prepare input data (bounding box + depth info)
    input_data = np.array([row[['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'depth_mean', 'depth_median', 'depth_mean_trim']].values], dtype=np.float32)
    input_data = torch.from_numpy(input_data)

    # Concatenate class information
    input_data = torch.cat([input_data, class_tensor], dim=-1)

    # Scale the input data
    scaled_input = torch.tensor(scaler.transform(input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Use the LSTM model to predict the Z-location
    z_loc_prediction = ZlocE.predict(scaled_input).detach().numpy()[0]

    return z_loc_prediction
