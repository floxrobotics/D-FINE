"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import glob
import numpy as np
from PIL import Image, ImageDraw

import sys
import os
import cv2  # Added for video processing
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import YAMLConfig
import cv2

import cv2

def draw(images, labels, boxes, scores, output_dir, thrh=0.4):
    class_labels = {
        0: 'Elk/Moose',
        1: 'Deer/Reindeer',
        2: 'Boar',
        3: 'Human',
        4: 'Vehicle',
        5: 'Animal'
    }

    class_colors = {
        'Elk/Moose': (0, 255, 0),
        'Deer/Reindeer': (255, 0, 0),
        'Boar': (128, 0, 128),
        'Human': (0, 0, 255),
        'Vehicle': (0, 255, 255),
        'Animal': (255, 165, 0)
    }

    for i, im in enumerate(images):
        image_path = images[i]
        frame = cv2.imread(image_path)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            label = class_labels[lab[j].item()]
            color = class_colors[label]
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thickness=3)
            cv2.putText(frame, f"{label} {round(scrs[j].item(), 2)}",
                        (int(b[0]), int(b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, frame)


def process_images(model, device, input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_files = glob.glob(os.path.join(input_dir, '*'))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for file_path in image_files:
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        im_data = transforms(im_pil).unsqueeze(0).to(device)
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        draw([file_path], labels, boxes, scores, output_dir)


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Handle directory input
    input_dir = args.input
    output_dir = os.path.join(input_dir, '../output_images')  # Assuming the output folder is next to the input

    # Process all images in the directory
    process_images(model, device, input_dir, output_dir)
    print("Image processing complete for all images in directory.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
