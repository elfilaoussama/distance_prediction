import torch
from transformers import GLPNForDepthEstimation, GLPNFeatureExtractor
from config import CONFIG

def load_glpn_model(device):
    feature_extractor = GLPNFeatureExtractor.from_pretrained(CONFIG['glpn_model_path'])
    model = GLPNForDepthEstimation.from_pretrained(CONFIG['glpn_model_path'])
    model.to(device)
    model.eval()
    return model, feature_extractor

def estimate_depth(img, glpn_model, glpn_extractor,img_shape ,device):
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

