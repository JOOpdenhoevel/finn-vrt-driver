import torch
from brevitas_examples import bnn_pynq
from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.util.data_packing as dpk

# Load the example CNV model from brevitas
model = bnn_pynq.cnv_2w2a(pretrained=True)
ishape = (1, 3, 32, 32)
model_file = "cnv_2w2a.onnx"
export_qonnx(model, torch.randn(ishape), model_file, opset_version=13)

# Build the design
cfg = build.DataflowBuildConfig(
    board="V80",
    output_dir="build",
    synth_clk_period_ns=20.0,
    generate_outputs=[
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.CPP_DRIVER,
    ],
    shell_flow_type=build_cfg.ShellFlowType.SLASH_ALVEO,
    folding_config_file="folding_config.json"
)
build.build_dataflow_cfg(model_file, cfg)
