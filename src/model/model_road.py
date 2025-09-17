import torch
import torchvision
torch.serialization.add_safe_globals([torchvision.models.detection.ssd.SSD])
model = torch.load('ssd_calculator_merge_model4.1.10.pth', map_location='cpu',weights_only=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.head.classification_head.num_classes = 16
model.eval()