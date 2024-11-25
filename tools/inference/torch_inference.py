import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import YAMLConfig

class TorchInference:
    def __init__(self, config_path, weights_path, device='cpu', classes=None):
        """
        Initialize the TorchInference module.

        Args:
            config_path (str): Path to the YAML config file.
            weights_path (str): Path to the model weights.
            device (str): Device for inference ('cpu' or 'cuda').
            classes (list): List of class names for inference.
        """
        self.device = device
        self.classes = classes or []
        
        # Load model configuration
        self.cfg = YAMLConfig(config_path, resume=weights_path)
        if 'HGNetv2' in self.cfg.yaml_cfg:
            self.cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        # Load model weights
        if weights_path:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        else:
            raise ValueError("Weights file is required for inference.")

        self.cfg.model.load_state_dict(state)
        self.model = self._create_model().to(self.device)

    def _create_model(self):
        """
        Create and wrap the model with the postprocessor.
        """
        class Model(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        return Model(self.cfg)

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for inference.
        
        Args:
            frame (np.ndarray): Input frame in BGR format.

        Returns:
            torch.Tensor: Preprocessed frame tensor.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
        return frame_tensor.unsqueeze(0).to(self.device)

    def draw_boxes(self, frame, labels, boxes, scores, threshold=0.6):
        """
        Draw bounding boxes on the frame.

        Args:
            frame (np.ndarray): Frame to draw on.
            labels (torch.Tensor): Detected class labels.
            boxes (torch.Tensor): Detected bounding boxes.
            scores (torch.Tensor): Confidence scores.
            threshold (float): Score threshold for filtering detections.

        Returns:
            np.ndarray: Frame with drawn boxes.
        """
        h, w, _ = frame.shape
        scores, boxes, labels = scores[0], boxes[0], labels[0]

        indices = scores > threshold
        filtered_scores = scores[indices]
        filtered_boxes = boxes[indices]
        filtered_labels = labels[indices]

        for i, box in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, box)
            label_idx = int(filtered_labels[i].item())
            label_name = self.classes[label_idx] if self.classes else str(label_idx)
            score = filtered_scores[i].item()

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

    def infer_frame(self, frame):
        """
        Perform inference on a single frame.

        Args:
            frame (np.ndarray): Input frame in BGR format.

        Returns:
            tuple: Detected labels, bounding boxes, and scores.
        """
        orig_size = torch.tensor([[frame.shape[1], frame.shape[0]]]).to(self.device)
        preprocessed_frame = self.preprocess_frame(frame)

        with torch.no_grad():
            output = self.model(preprocessed_frame, orig_size)

        return output  # (labels, boxes, scores)
