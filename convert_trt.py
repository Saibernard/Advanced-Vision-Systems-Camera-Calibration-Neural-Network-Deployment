import torch
import onnx
import tensorrt as trt
import os
from yolo import F110_YOLO

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def convert_torch_to_onnx(input_model_path, output_onnx_path):
    # load the model
    model = F110_YOLO().cuda()
    model.load_state_dict(torch.load(input_model_path))
    sample_input = torch.zeros(1, 3, 180, 320, dtype=torch.float,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.onnx.export(model, sample_input, output_onnx_path, input_names=['input'], output_names=['output'],
                      export_params=True)
    print('Exported to onnx')

    # check conversion
    model_onnx = onnx.load(output_onnx_path)
    onnx.checker.check_model(model_onnx)
    print('Conversion successful')


def convert_onnx_to_trt(input_onnx_path, output_rt_path):
    TRT_LOGGER = trt.Logger()

    # init TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    builder_config = builder.create_builder_config()
    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder_config.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    # if builder.platform_has_fast_fp16:
    #     builder.fp16_mode = True
    # if builder_config.use_fp16:
    # builder_config.set_flag(trt.BuilderFlag.FP16)
    builder_config.set_flag(trt.BuilderFlag.FP32)
    with open(input_onnx_path, 'rb') as model:
        parser.parse(model.read())
    engine = builder.build_engine(network, builder_config)
    with open(input_onnx_path, 'rb') as f:
        f.write(bytearray(engine.serialize()))


def load_trt(path):
    TRT_LOGGER = trt.Logger()
    runtime = trt.Runtime(TRT_LOGGER)
    engine, context = None, None
    with open(path, 'rb') as f:
        engine_ = f.read()
        engine = runtime.deserialize_cuda_engine(engine_)
        context = engine.create_execution_context()
        print('Finished loading TensorRT engine')
    return engine, context

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    TRT_LOGGER = trt.Logger()
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # config.set_flag(trt.BuilderFlag.FP16)
            # config.set_flag(trt.BuilderFlag.FP32)
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

if __name__ == "__main__":
    # convert_torch_to_onnx('model/model_60.pt', 'model/model.onnx')
    # convert_onnx_to_trt('model/model.onnx', 'model/model.trt')
    get_engine('./model/model.onnx', './model/model_32.trt')