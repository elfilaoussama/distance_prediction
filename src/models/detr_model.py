import torch
from config import CONFIG
from torchvision import transforms
from matplotlib import pyplot as plt
from transformers import DetrForObjectDetection, DetrImageProcessor

class DETR:
    def __init__(self):

        self.CLASSES = [
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
        
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], 
                       [0.929, 0.694, 0.125], [0, 0, 1], [0.466, 0.674, 0.188], 
                       [0.301, 0.745, 0.933]]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = DetrForObjectDetection.from_pretrained(CONFIG['detr_model_path'], revision="no_timm")
        self.model.to(CONFIG['device'])
        self.model.eval()

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1).to(CONFIG['device'])

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(CONFIG['device'])
        return b

    
    def detect(self, im):
        img = self.transform(im).unsqueeze(0).to(CONFIG['device'])
        assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'Image too large'
        outputs = self.model(img)
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        return probas[keep], bboxes_scaled
    
    def visualize(self, im, probas, bboxes):
        """
        Visualizes the detected bounding boxes and class probabilities on the image.

        Parameters:
            im (PIL.Image): The original input image.
            probas (Tensor): Class probabilities for detected objects.
            bboxes (Tensor): Bounding boxes for detected objects.
        """
        # Convert image to RGB format for matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(im)
        ax = plt.gca()
        
        # Iterate over detections and draw bounding boxes and labels
        for p, (xmin, ymin, xmax, ymax), color in zip(probas, bboxes, self.COLORS * 100):
            # Detach tensors and convert to float
            xmin, ymin, xmax, ymax = map(lambda x: x.detach().cpu().numpy().item(), (xmin, ymin, xmax, ymax))
            
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=color, linewidth=3))
            cl = p.argmax()
            text = f'{self.CLASSES[cl]}: {p[cl].detach().cpu().numpy():0.2f}'  # Detach probability as well
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.axis('off')
        plt.show()



