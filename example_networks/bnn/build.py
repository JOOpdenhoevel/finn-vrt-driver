# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Condensed build script for the BNN-PYNQ CNV network (wbits=2, abits=2) targeting
# the V80 board. The build steps and their ordering are derived from the end-to-end
# integration test at:
# https://github.com/Xilinx/finn/blob/92a6f5a79c5372a019c2ebb6c8ba79e52d4e7a01/tests/end2end/test_end2end_bnn_pynq.py

import os
import torch
import shutil
from brevitas_examples import bnn_pynq
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.alveo_build import PrepareForLinking, SlashLink
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.make_driver import MakeCPPDriver
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.pytorch import ToTensor

# Build configuration
target_clk_ns = 20
fpga_part = "xcv80-lsva4737-2MHP-e-s"
build_output_dir = "build"
os.makedirs(build_output_dir, exist_ok=True)

# Load the example CNV w2a2 model from brevitas
model = bnn_pynq.cnv_2w2a(pretrained=True)
ishape = (1, 3, 32, 32)
model_file = "cnv_2w2a.onnx"
export_qonnx(model, torch.randn(ishape), model_file, opset_version=13)

# --- Step 1: Export cleanup (test_export) ---
print("Step 1: Export cleanup")
qonnx_cleanup(model_file, out_file=model_file)
model = ModelWrapper(model_file)
model = model.transform(ConvertQONNXtoFINN())
model.save(build_output_dir + "/01_export.onnx")

# --- Step 2: Tidy up (test_import_and_tidy) ---
print("Step 2: Tidy up")
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())
model.save(build_output_dir + "/02_tidy.onnx")

# --- Step 3: Pre- and post-processing (test_add_pre_and_postproc) ---
print("Step 3: Pre- and post-processing")
global_inp_name = model.get_first_global_in()
preproc_onnx = build_output_dir + "/preproc.onnx"
export_qonnx(ToTensor(), torch.randn(ishape), preproc_onnx, opset_version=13)
qonnx_cleanup(preproc_onnx, out_file=preproc_onnx)
pre_model = ModelWrapper(preproc_onnx)
pre_model = pre_model.transform(ConvertQONNXtoFINN())
pre_model = pre_model.transform(InferShapes())
pre_model = pre_model.transform(FoldConstants())
model = model.transform(MergeONNXModels(pre_model))
model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
model = model.transform(InsertTopK(k=1))
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())
model.save(build_output_dir + "/03_pre_post.onnx")

# --- Step 4: Streamline (test_streamline, CNV path) ---
print("Step 4: Streamline")
model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(Streamline())
model = model.transform(LowerConvsToMatMul())
model = model.transform(MakeMaxPoolNHWC())
model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(Streamline())
model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save(build_output_dir + "/04_streamline.onnx")

# --- Step 5: Convert to HW layers (test_convert_to_hw_layers, CNV path) ---
print("Step 5: Convert to HW layers")
model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
model = model.transform(to_hw.InferLabelSelectLayer())
model = model.transform(to_hw.InferThresholdingLayer())
model = model.transform(to_hw.InferPool())
model = model.transform(to_hw.InferConvInpGen())
model = model.transform(RemoveCNVtoFCFlatten())
model = model.transform(absorb.AbsorbConsecutiveTransposes())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(InferDataLayouts())
model.save(build_output_dir + "/05_hw_layers.onnx")

# --- Step 6: Specialize layers (test_specialize_layers) ---
print("Step 6: Specialize layers")
model = model.transform(SpecializeLayers(fpga_part))
model = model.transform(GiveUniqueNodeNames())
model.save(build_output_dir + "/06_specialize.onnx")

# --- Step 7: Create dataflow partition (test_create_dataflow_partition) ---
print("Step 7: Create dataflow partition")
parent_model = model.transform(CreateDataflowPartition())
parent_model.save(build_output_dir + "/07_dataflow_parent.onnx")
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node_inst = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node_inst.get_nodeattr("model")
model = ModelWrapper(dataflow_model_filename)
model.save(build_output_dir + "/07_dataflow_model.onnx")

# --- Step 8: Fold (test_fold, fold_cnv_small for wbits=2, abits=2) ---
print("Step 8: Fold")
fc_layers = model.get_nodes_by_op_type("MVAU_hls")
folding = [
    (8, 3, "distributed"),
    (16, 16, "distributed"),
    (8, 16, "auto"),
    (8, 16, "distributed"),
    (4, 8, "auto"),
    (1, 8, "auto"),
    (1, 2, "block"),
    (2, 2, "auto"),
    (5, 1, "distributed"),
]
for fcl, (pe, simd, ramstyle) in zip(fc_layers, folding):
    fcl_inst = getCustomOp(fcl)
    fcl_inst.set_nodeattr("PE", pe)
    fcl_inst.set_nodeattr("SIMD", simd)
    fcl_inst.set_nodeattr("ram_style", ramstyle)
    fcl_inst.set_nodeattr("mem_mode", "internal_decoupled")
    fcl_inst.set_nodeattr("resType", "lut")
swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
for i in range(len(swg_layers)):
    swg_inst = getCustomOp(swg_layers[i])
    if not swg_inst.get_nodeattr("depthwise"):
        simd = folding[i][1]
        swg_inst.set_nodeattr("SIMD", simd)
    swg_inst.set_nodeattr("ram_style", "distributed")
inp_qnt_node = model.get_nodes_by_op_type("Thresholding_rtl")[0]
inp_qnt = getCustomOp(inp_qnt_node)
inp_qnt.set_nodeattr("depth_trigger_uram", 32000)
inp_qnt.set_nodeattr("depth_trigger_bram", 32000)
model.save(build_output_dir + "/08_fold.onnx")

# --- Step 9: Minimize bit width (test_minimize_bit_width) ---
print("Step 9: Minimize bit width")
model = model.transform(MinimizeWeightBitWidth())
model = model.transform(MinimizeAccumulatorWidth())
model = model.transform(RoundAndClipThresholds())
model = model.transform(MinimizeWeightBitWidth())
model.save(build_output_dir + "/09_minimize_bit_width.onnx")

# --- Step 10: IP generation / HLS synthesis (test_ipgen) ---
print("Step 10: IP generation / HLS synthesis")
model = model.transform(GiveUniqueNodeNames())
model = model.transform(PrepareIP(fpga_part, target_clk_ns))
model = model.transform(HLSSynthIP())
model.save(build_output_dir + "/10_ipgen.onnx")

# --- Step 11: Set FIFO depths (test_set_fifo_depths) ---
print("Step 11: Set FIFO depths")
model = model.transform(InsertAndSetFIFODepths(fpga_part, target_clk_ns))
model.save(build_output_dir + "/11_fifodepth.onnx")

# --- Step 12: Prepare for linking + link (test_prepare_for_linking, test_linking) ---
print("Step 12: Prepare for linking + link")
model = model.transform(PrepareForLinking(fpga_part, target_clk_ns, "slash-vrt"))
model.save(build_output_dir + "/12_prepare_linking.onnx")
model = model.transform(SlashLink())
model.save(build_output_dir + "/12_linked.onnx")

# --- Step 13: Prepare the driver configuration files
print("Step 13: Prepare the driver configuration files")
model = model.transform(MakeCPPDriver("slash-vrt", version="main"))
model.save(build_output_dir + "/13_driver.onnx")

# --- Step 14: Export the results
bitfile_path =  model.get_metadata_prop("bitfile")
print("Step 14: Export the results")
shutil.copy(bitfile_path, build_output_dir + "/finn.vbin")

cpp_driver_dir = model.get_metadata_prop("cpp_driver_dir")
shutil.copy(cpp_driver_dir + "/acceleratorconfig.json", build_output_dir + "/acceleratorconfig.json")
shutil.copy(cpp_driver_dir + "/AcceleratorDatatypes.h", build_output_dir + "/AcceleratorDatatypes.h")
