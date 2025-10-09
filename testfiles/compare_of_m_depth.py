import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import cv2
import torch as th
import numpy as np
from torchvision.utils import flow_to_image

from pruning_sb3.pruning_gym.optical_flow import OpticalFlow

def process_video(input_path, output_path, resize_to=(240, 240)):
    optical_flow_model = OpticalFlow(size=resize_to)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = resize_to[0], resize_to[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 3, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        return

    prev_frame = cv2.resize(prev_frame, resize_to)
    prev_tensor = th.from_numpy(prev_frame).permute(2, 0, 1).float() / 255.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize_to)
        curr_tensor = th.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        flow = optical_flow_model.calculate_optical_flow(curr_tensor, prev_tensor)  # shape: [1, 2, H, W]
        # flow_img = flow_to_image(flow)[0].permute(1, 2, 0).byte().cpu().numpy()  # shape: (H, W, 3)
        #
        # # Concatenate side-by-side: [RGB | Flow]
        # combined = np.concatenate((frame, flow_img), axis=1)
        flow_np = flow[0].cpu().numpy()  # shape: (2, H, W)
        flow_x = flow_np[0]
        flow_y = flow_np[1]

        # Normalize to 0–255 for visualization
        def normalize_and_colorize(flow_component):
            flow_min, flow_max = np.min(flow_component), np.max(flow_component)
            norm = (flow_component - flow_min) / (flow_max - flow_min + 1e-8)
            norm_uint8 = (norm * 255).astype(np.uint8)
            return cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

        flow_x_img = normalize_and_colorize(flow_x)
        flow_y_img = normalize_and_colorize(flow_y)

        # Resize RGB to match (just in case)
        rgb_frame = frame.copy()

        # Final layout: [RGB | Flow-X | Flow-Y]
        combined = np.concatenate((rgb_frame, flow_x_img, flow_y_img), axis=1)
        out.write(combined)

        prev_tensor = curr_tensor.clone()

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from a ROS 2 bag file and create a video")
    parser.add_argument('--input', type=str,default="2025_pruning_trials/videos/", help="Path to input video file")
    parser.add_argument('--output', type=str, default="2025_pruning_trials/videos/envy_1_0/", help="Path to output video file")
    parser.add_argument('--name', type=str, default="comparison", help="Name of the output video file")
    args = parser.parse_args()
    for i in os.listdir(args.input):
        input_file = os.path.join(args.input, i, "output_video.mp4")
        output_name = os.path.join(args.input, i, args.name + ".mp4")
        print(f"Processing {input_file} to {output_name}")
        process_video(input_file, output_name)
