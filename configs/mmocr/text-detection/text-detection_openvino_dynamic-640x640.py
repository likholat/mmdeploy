_base_ = ['./text-detection_dynamic.py', '../../_base_/backends/openvino.py']

onnx_config = dict(input_shape=None)

import torch

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 640, 640]))],
    ovc_options=({
            # 'mean_values': {"input": [0.5, 0.5, 0.5]},
            # 'scale_values': {"input": [255, 255, 255]},
            # 'target_layout': {"input": "NCHW"},
            'compress_to_fp16': False,
            # 'example_input': torch.rand(1,3,640,640)
        })
)
