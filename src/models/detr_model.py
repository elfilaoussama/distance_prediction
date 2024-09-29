import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from config import CONFIG
from utils.bounding_boxes import rescale_bboxes

def load_detr_model(device):
    processor = DetrImageProcessor.from_pretrained(CONFIG['detr_model_path'], revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(CONFIG['detr_model_path'], revision="no_timm")
    model.to(device)
    model.eval()
    return model, processor

def detect_objects(img, detr_model, detr_processor, device):
    img = img.convert('RGB')
    inputs = detr_processor(images=img, return_tensors="pt")
    img_tensor = inputs.pixel_values.to(device)  # Ensure this is on the correct device

    img_shape = img_tensor.shape[-2:]

    outputs = detr_model(img_tensor)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].detach(), img.size, device)

    # Get class indices of the detected objects
    class_indices = probas.argmax(-1)[keep]

    return probas[keep], bboxes_scaled, img_shape, class_indices
