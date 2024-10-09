import torch

CONFIG = {
    'device': torch.device('cpu'),
    'detr_model_path': 'facebook/detr-resnet-101',
    'glpn_model_path': 'vinvino02/glpn-kitti',
    'lstm_model_path': '../data/models/pretrained_lstm.pth',
    'image_path': 'data/images/crosswalk.png',
    'confidence_threshold': 0.7
}
