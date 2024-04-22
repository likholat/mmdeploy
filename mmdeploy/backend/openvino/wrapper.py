# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


@BACKEND_WRAPPER.register_module(Backend.OPENVINO.value)
class OpenVINOWrapper(BaseWrapper):
    """OpenVINO wrapper for inference in CPU.

    Args:
        ir_model_file (str): Input OpenVINO IR model file.
        output_names (Sequence[str] | None): Names of model outputs in order.
            Defaults to `None` and the wrapper will load the output names from
            model.

    Examples:
        >>> from mmdeploy.backend.openvino import OpenVINOWrapper
        >>> import torch
        >>>
        >>> ir_model_file = 'model.xml'
        >>> model = OpenVINOWrapper(ir_model_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224, device='cpu'))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 ir_model_file: str,
                 output_names: Optional[Sequence[str]] = None,
                 **kwargs):

        import openvino as ov
        self.core = ov.Core()
        self.model = self.core.read_model(model=ir_model_file)
        for input_obj in self.model.inputs:
            shape = input_obj.get_partial_shape()
            batch_size = shape[0]
            dims = len(shape)
            # if input is a image, it has (B,C,H,W) channels,
            # need batch_size==1
            assert not dims == 4 or batch_size == 1, \
                'Only batch 1 is supported.'
        self.device = 'cpu'
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device.upper())

        # TODO: Check if output_names can be read
        if output_names is None:
            output_names = [name for name in self.net.outputs]

        super().__init__(output_names)

    def __update_device(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Updates the device type to 'self.device' (cpu) for the input
        tensors.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Returns:
            Dict[str, torch.Tensor]: The output name and tensor pairs
                with updated device type.
        """
        updated_inputs = {
            name: data.to(torch.device(self.device)).contiguous()
            for name, data in inputs.items()
        }
        return updated_inputs

    def __reshape(self, inputs: Dict[str, torch.Tensor]):
        """Reshape the model for the shape of the input data.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.
        """
        input_shapes = {name: data.shape for name, data in inputs.items()}
        reshape_needed = False

        for input_name, input_shape in input_shapes.items():
            self.model.input(input_name).get_shape()
            blob_shape = self.model.input(input_name).get_shape()
            if not np.array_equal(input_shape, blob_shape):
                reshape_needed = True
                break
        if reshape_needed:
            self.model.reshape(input_shapes)
            self.compiled_model = self.core.compile_model(
                                            model=self.model,
                                            device_name=self.device.upper())

    def __process_outputs(
            self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Converts tensors from 'torch' to 'numpy' and fixes the names of the
        outputs.

        Args:
            outputs Dict[str, torch.Tensor]: The output name and tensor pairs.

        Returns:
            Dict[str, torch.Tensor]: The output name and tensor pairs
                after processing.
        """
        outputs = {
            name: torch.from_numpy(tensor)
            for name, tensor in outputs.items()
        }
        cleaned_outputs = {}

        for const_output, value in outputs.items():
            name = const_output.get_any_name()
            if '.' in name:
                new_output_name = name.split('.')[0]
                cleaned_outputs[new_output_name] = value
            else:
                cleaned_outputs[name] = value
        return cleaned_outputs

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Returns:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        inputs = self.__update_device(inputs)
        self.__reshape(inputs)
        outputs = self.__openvino_execute(inputs)
        outputs = self.__process_outputs(outputs)
        return outputs

    @TimeCounter.count_time(Backend.OPENVINO.value)
    def __openvino_execute(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference with OpenVINO IE.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Returns:
            Dict[str, numpy.ndarray]: The output name and tensor pairs.
        """
        outputs = self.compiled_model.infer_new_request(inputs)
        return outputs
