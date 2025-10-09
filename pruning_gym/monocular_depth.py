
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import cv2

class DepthAnything:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("isl-org/Depth-Anything-Small")
        self.model = AutoModelForDepthEstimation.from_pretrained("isl-org/Depth-Anything-Small").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_depth(self, frame: np.ndarray):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        depth = self.model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False)
        depth = depth.squeeze().cpu().numpy()
        # Normalize for visualization
        depth_vis = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)
        depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        return depth_vis_rgb
