from models.predict_z_location_single_row import predict_z_location_single_row

def generate_output_json(data, ZlocE, scaler):
    """
    Predict Z-location for each object in the data and prepare the JSON output.
    
    Parameters:
    - data: DataFrame with bounding box coordinates, depth information, and class type.
    - ZlocE: Pre-loaded LSTM model for Z-location prediction.
    - scaler: Scaler for normalizing input data.
    
    Returns:
    - JSON structure with object class, distance estimated, and relevant features.
    """
    output_json = []
    
    # Iterate over each row in the data
    for i, row in data.iterrows():
        # Predict distance for each object using the single-row prediction function
        distance = predict_z_location_single_row(row, ZlocE, scaler)
        
        # Create object info dictionary
        object_info = {
            "class": row["class"],  # Object class (e.g., 'car', 'truck')
            "distance_estimated": float(distance),  # Convert distance to float (if necessary)
            "features": {
                "xmin": float(row["xmin"]),  # Bounding box xmin
                "ymin": float(row["ymin"]),  # Bounding box ymin
                "xmax": float(row["xmax"]),  # Bounding box xmax
                "ymax": float(row["ymax"]),  # Bounding box ymax
                "mean_depth": float(row["depth_mean"]),  # Depth mean
                "depth_mean_trim": float(row["depth_mean_trim"]),  # Depth mean trim
                "depth_median": float(row["depth_median"]),  # Depth median
                "width": float(row["width"]),  # Object width
                "height": float(row["height"])  # Object height
            }
        }
        
        # Append each object info to the output JSON list
        output_json.append(object_info)
    
    # Return the final JSON output structure
    return {"objects": output_json}
