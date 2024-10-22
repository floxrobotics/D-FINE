import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import YAMLConfig

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]


# Function to draw bounding boxes on the image using OpenCV
def draw_boxes_opencv(frame, labels, boxes, scores, thrh=0.6):
    h, w, _ = frame.shape

    # Remove the batch dimension (assuming batch size is 1)
    scores = scores[0]
    boxes = boxes[0]
    labels = labels[0]

    # Filter out boxes, labels, and scores below the threshold
    indices = scores > thrh
    filtered_scores = scores[indices]
    filtered_boxes = boxes[indices]
    filtered_labels = labels[indices]
    # Draw the filtered boxes and labels
    for i, box in enumerate(filtered_boxes):
        score = filtered_scores[i].item()  # Get the confidence score
        label_idx = filtered_labels[i].item()  # Get the class index
        label_name = COCO_CLASSES[int(label_idx)]  # Get the class name from the index
        # print(label_name)
        # Convert the normalized box coordinates to pixel values
        x1, y1, x2, y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Draw the rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Put the label and score text on the frame
        text = f"{label_name} {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def preprocess_frame(frame, device):
    # Convert the frame from OpenCV BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to the required input size (640x640 in this case)
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    # Convert the frame to a tensor and normalize
    frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0  # Change to [C, H, W]
    frame_tensor = frame_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device

    return frame_tensor

def main(args):
    """Main function to perform real-time object detection"""
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

    model = Model().to(args.device)

    # Open the webcam
    if args.video:
        cap = cv2.VideoCapture(args.video)
        output_name = os.path.splitext(os.path.basename(args.video))[0] + '_result.mp4'
        output_path = os.path.join('test_videos', output_name)
        if not os.path.exists('test_videos'):
            os.makedirs('test_videos')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    else:
        cap = cv2.VideoCapture(0)  # Use the webcam if no video path is provided
        out = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Get original size for postprocessing
        h, w, _ = frame.shape
        orig_size = torch.tensor([w, h])[None].to(args.device)

        # Preprocess the frame
        im_data = preprocess_frame(frame, args.device)

        # Perform inference
        with torch.no_grad():
            output = model(im_data, orig_size)

        # Parse the outputs (assuming output has the structure [labels, boxes, scores])
        labels, boxes, scores = output

        # Draw bounding boxes and labels on the frame
        frame_with_boxes = draw_boxes_opencv(frame, labels, boxes, scores)

        # Save the frame if writing to video
        if out:
            out.write(frame_with_boxes)

        # Display the resulting frame
        cv2.imshow('Real-Time Object Detection', frame_with_boxes)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-r', '--resume', type=str)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-v', '--video', type=str, help="Optional path to a video file")
    args = parser.parse_args()
    main(args)
