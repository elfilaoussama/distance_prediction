"""
Created on Sat Apr  9 04:08:02 2022
@author: Admin_with ODD Team

Edited by our team : Sat Oct 5 10:00 2024

references: https://github.com/vinvino02/GLPDepth 
"""

import torch
from transformers import GLPNForDepthEstimation, GLPNFeatureExtractor
from PIL import Image
from config import CONFIG


# GLPDepth Model Class
class GLPDepth:
    def __init__(self):
        self.feature_extractor = GLPNFeatureExtractor.from_pretrained(CONFIG['glpn_model_path'])
        self.model = GLPNForDepthEstimation.from_pretrained(CONFIG['glpn_model_path'])
        self.model.to(CONFIG['device'])
        self.model.eval()


    def predict(self, img: Image.Image, img_shape : tuple):
        """Predict the depth map of the input image.

        Args:
            img (PIL.Image): Input image for depth estimation.
            img_shape (tuple): Original image size (height, width).

        Returns:
            np.ndarray: The predicted depth map in numpy array format.
        """
        with torch.no_grad():
            # Preprocess image and move to the appropriate device
            pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values.to(CONFIG['device'])
            # Get model output
            outputs = self.model(pixel_values)
            predicted_depth = outputs.predicted_depth
            
            # Resize depth prediction to original image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img_shape[:2],  # Interpolate to original image size
                mode="bicubic",
                align_corners=False,
            )
            prediction = prediction.squeeze().cpu().numpy()  # Convert to numpy array (shape: (H, W))
        
        return prediction