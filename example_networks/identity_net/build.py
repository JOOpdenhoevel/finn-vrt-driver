import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.util.data_packing as dpk

cfg = build.DataflowBuildConfig(
    board="V80",
    output_dir="build",
    synth_clk_period_ns=5.0,
    generate_outputs=[
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.CPP_DRIVER,
    ],
    shell_flow_type=build_cfg.ShellFlowType.SLASH_ALVEO,
    enable_hw_sim=True
)
model_file = "ident.onnx"
build.build_dataflow_cfg(model_file, cfg)