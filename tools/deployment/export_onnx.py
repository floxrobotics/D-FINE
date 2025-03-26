"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from src.core import YAMLConfig


def main(args):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Disable pretrained weights for HGNetv2 if present.
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # Load checkpoint if provided.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)
    else:
        print('No checkpoint provided; using default initialization...')

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    # IMPORTANT: Move the model to GPU so that both weights and inputs are on the same device.
    model = model.cuda()

    # ---------------------------------------------------------
    # Prepare example inputs with matching batch dimensions.
    # Here we choose B = 3 (dummy input) so that the exported model's batch dimension is 3.
    # The dynamic axes below will then allow any batch size (e.g. 1,2,3,...) at runtime.
    # ---------------------------------------------------------
    B = 3
    data = torch.rand(B, 3, 640, 640).cuda()          # Shape: [3, 3, 640, 640]
    size = torch.tensor([[640, 640]] * B, device='cuda')  # Shape: [3, 2]

    # Run a dry forward pass to ensure the model works.
    _ = model(data, size)

    # ---------------------------------------------------------
    # Set dynamic axes so that the batch dimension is flexible.
    # This allows the exported model to accept a different batch size at runtime (e.g., 1, 2, or 3).
    # ---------------------------------------------------------
    dynamic_axes = {
        'images': {0: 'batch_size'},
        'orig_target_sizes': {0: 'batch_size'}
    }

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    # ---------------------------------------------------------
    # Export the model to ONNX with the dynamic batch dimension.
    # ---------------------------------------------------------
    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        input_shapes = {
            'images': [B, 3, 640, 640],
            'orig_target_sizes': [B, 2]
        } if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str,
                        help='Path to config file.')
    parser.add_argument('--resume', '-r', type=str,
                        help='Path to the .pth checkpoint.')
    parser.add_argument('--check', action='store_true', default=True,
                        help='Whether to run ONNX checker on the exported file.')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Whether to simplify the ONNX file with onnxsim.')
    args = parser.parse_args()
    main(args)
