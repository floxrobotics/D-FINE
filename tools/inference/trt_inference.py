import os
import time
import collections
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import tensorrt as trt
import cv2
import contextlib


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference:
    def __init__(self, engine_path, device='cuda:0', max_batch_size=32, verbose=False):
        """
        Initialize the TensorRT inference engine.

        Args:
            engine_path (str): Path to the TensorRT engine file.
            device (str): Device for inference ('cpu' or 'cuda:0').
            max_batch_size (int): Maximum batch size for the engine.
            verbose (bool): If True, enables verbose logging.
        """
        self.device = device
        self.max_batch_size = max_batch_size
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self._get_bindings(self.engine, self.context, max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self._get_input_names()
        self.output_names = self._get_output_names()
        self.time_profile = TimeProfiler()

    def _load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_input_names(self):
        return [name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]

    def _get_output_names(self):
        return [name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

    def _get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_inference(self, blob):
        """
        Run inference on the input data.

        Args:
            blob (dict): Input blob containing 'images' and 'orig_target_sizes'.

        Returns:
            dict: Output data containing 'labels', 'boxes', and 'scores'.
        """
        for n in self.input_names:
            if blob[n].dtype is not self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        return {n: self.bindings[n].data for n in self.output_names}

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for inference.

        Args:
            frame (np.ndarray): Input frame in BGR format as a NumPy array.

        Returns:
            dict: Preprocessed frame as a blob ready for inference.
        """
        # Get the original dimensions of the frame
        h, w = frame.shape[:2]
        orig_size = torch.tensor([[w, h]]).to(self.device)

        # Resize the frame to the required input size (640x640) and normalize
        frame_resized = cv2.resize(frame, (640, 640))
        frame_normalized = frame_resized.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

        # Convert the frame to a tensor and rearrange dimensions to [C, H, W]
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        return {
            'images': frame_tensor,
            'orig_target_sizes': orig_size
        }


    def draw_boxes(self, frame, labels, boxes, scores, classes, threshold=0.4):
        """
        Draw bounding boxes on the frame.

        Args:
            frame (np.ndarray): Frame to draw on.
            labels (torch.Tensor): Detected class labels.
            boxes (torch.Tensor): Detected bounding boxes.
            scores (torch.Tensor): Confidence scores.
            classes (list): List of class names.
            threshold (float): Minimum confidence score for drawing boxes.

        Returns:
            np.ndarray: Frame with drawn bounding boxes.
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
            label_name = classes[label_idx] if classes else str(label_idx)
            score = filtered_scores[i].item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame
