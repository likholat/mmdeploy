# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Optional, Sequence, Union

import mmengine
import onnx
import openvino as ov

from mmdeploy.utils import get_root_logger


def get_output_model_file(onnx_path: str, work_dir: str) -> str:
    """Returns the path to the .xml file with export result.

    Args:
        onnx_path (str): The path to the onnx model.
        work_dir (str): The path to the directory for saving the results.

    Returns:
        str: The path to the file where the export result will be located.
    """
    mmengine.mkdir_or_exist(osp.abspath(work_dir))
    file_name = osp.splitext(osp.split(onnx_path)[1])[0]
    model_xml = osp.join(work_dir, file_name + '.xml')
    return model_xml


# Available options for model conversion
ovc_base_options={'example_input', 'extension', 'verbose', 'compress_to_fp16'}
preprocess_params = {'target_layout', 'source_layout', 'layout', 'mean_values', 'scale_values'}


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_info: Union[list, dict, str] = None,
              output_names: Union[str, Sequence[str]] = None,
              ovc_options: Optional[dict] = None):
    
    """Convert ONNX to OpenVINO.

    Examples:
        >>> from mmdeploy.apis.openvino import from_onnx
        >>> input_info = {'input': [1,3,800,1344]}
        >>> output_names = ['dets', 'labels']
        >>> onnx_path = 'work_dir/end2end.onnx'
        >>> output_dir = 'work_dir'
        >>> from_onnx(onnx_path, output_dir, input_info, output_names)

    Args:
        onnx_model (str|ModelProto): The onnx model or its path.
        output_file_prefix (str): The path to the directory for saving
            the results.
        input_info (list|dict|str):
            Information of model input required for model conversion.
            Input can be set by a list of tuples or a dictionary. Each tuple can contain optionally input name (string),
            input type (ov.Type, numpy.dtype) or input shape (ov.Shape, ov.PartialShape, list, tuple).
            Example: input=("op_name", PartialShape([-1, 3, 100, 100]), ov.Type.f32).
            Alternatively input can be set by a dictionary, where key - input name,
            value - tuple with input parameters (shape or type).
            Example 1: input={"op_name_1": ([1, 2, 3], ov.Type.f32), "op_name_2": ov.Type.i32}
            Example 2: input=[("op_name_1", [1, 2, 3], ov.Type.f32), ("op_name_2", ov.Type.i32)]
            Example 3: input=[([1, 2, 3], ov.Type.f32), ov.Type.i32]
            The order of inputs in converted model will match the order of specified inputs.
            If data type is not specified explicitly data type is taken from the original node data type.
        output_names (str|Sequence[str]): Output names. Example:
            'output' or ['dets', 'labels'].
        ovc_options (dict): The dictionary with
            additional arguments for the OpenVINO Model Conversion.
    """

    if ovc_options is None:
        ovc_options = {}
    else:
        unsupported_options = ovc_options.keys() - (ovc_base_options | preprocess_params)
        assert len(unsupported_options) == 0, f"Unsupported options for OpenVINO model conversion: {unsupported_options}"

    logger = get_root_logger()

    if isinstance(onnx_model, str):
        onnx_path = onnx_model
    else:
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)

    ovc_options['example_input'] = ovc_options.get('example_input')
    ovc_options['extension'] = ovc_options.get('extension')
    ovc_options['verbose'] = ovc_options.get('verbose') or False
    ovc_options['compress_to_fp16'] = ovc_options.get('compress_to_fp16') or True

    ov_model = ov.convert_model(onnx_path, input=input_info, output=output_names, example_input=ovc_options['example_input'], extension=ovc_options['extension'], verbose=ovc_options['verbose'])
    model_xml = get_output_model_file(onnx_path, output_file_prefix)

    if len(ovc_options.keys() & preprocess_params) > 0:
        prep = ov.preprocess.PrePostProcessor(ov_model)
        
        if 'source_layout' in ovc_options.keys():
            for input_name, layout in ovc_options['source_layout'].items():
                prep.input(input_name).model().set_layout(ov.Layout(layout))

        if 'target_layout' in ovc_options.keys():
            for input_name, layout in ovc_options['target_layout'].items():
                prep.input(input_name).tensor().set_layout(ov.Layout(layout))

        if 'mean_values' in ovc_options.keys():
            for input_name, mean_val in ovc_options['mean_values'].items():
                prep.input(input_name).preprocess().mean(mean_val)

        if 'scale_values' in ovc_options.keys():
            for input_name, scale_val in ovc_options['scale_values'].items():
                prep.input(input_name).preprocess().scale(scale_val)

        ov_model = prep.build()

    ov.save_model(ov_model, model_xml, compress_to_fp16=ovc_options['compress_to_fp16'])

    logger.info(f'Successfully exported OpenVINO model: {model_xml}')
