import torch
import numpy as np

def predict_z_location(data, ZlocE, scaler):
    """
    Preprocess bounding box coordinates, depth information, and class type 
    to predict Z-location using the LSTM model.
    
    Parameters:
    - data: DataFrame with bounding box coordinates, depth information, and class type.
    - ZlocE: Pre-loaded LSTM model for Z-location prediction.
    - scaler: Scaler for normalizing input data.
    
    Returns:
    - z_locations: List of predicted Z-locations.
    """
    z_locations = []

    for k in data.index:
        # Get class type and convert to one-hot encoding
        classes = data.iloc[k, -2]
        if classes == 'bicycle':
            class_tensor = torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.float32)
        elif classes == 'car':
            class_tensor = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32)
        elif classes == 'person':
            class_tensor = torch.tensor([[0, 0, 0, 1, 0, 0]], dtype=torch.float32)
        elif classes == 'train':
            class_tensor = torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.float32)
        elif classes == 'truck':
            class_tensor = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.float32)
        else :
            class_tensor = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32)

        # Create input data by concatenating bounding box coordinates and depth information
        input_data = np.array(data.iloc[[k], 0:9].values, dtype=np.float32)
        input_data = torch.from_numpy(input_data)
        input_data = torch.cat([input_data, class_tensor], axis=1)

        # Scale the input data
        scaled_input = torch.tensor(scaler.transform(input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Use LSTM model to predict Z-location
        z_loc_prediction = ZlocE.predict(scaled_input).detach().numpy()[0]
        z_locations.append(z_loc_prediction)

    return z_locations
