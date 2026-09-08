"""Export Depth Anything V2 Small (DPT) to ONNX for edge deployment.

Run this on any machine with the HuggingFace model cached, then use the
resulting .onnx file with onnxruntime (CPU here) or TensorRT (on Jetson).
"""

import torch
from transformers import AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
OUT_PATH = "depth_anything_v2_small.onnx"
INPUT_SIZE = (518, 518)


class DepthWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


def main():
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).eval()
    wrapper = DepthWrapper(model).eval()

    dummy = torch.randn(1, 3, *INPUT_SIZE)

    torch.onnx.export(
        wrapper,
        dummy,
        OUT_PATH,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"Exported {OUT_PATH}")


if __name__ == "__main__":
    main()
