#!/usr/bin/env python3
"""OFFLINE export of a PE-Core checkpoint to ONNX + weight-only int8 (ADR-4 recipe).

Runs in a THROWAWAY venv — torch is ~2GB and must never enter the runtime env (ADR-7,
B23 asserts `import torch` FAILS there):

    mkdir -p .scratch/pecore && cd .scratch/pecore
    uv venv --python 3.12 .venv
    VIRTUAL_ENV=$PWD/.venv uv pip install torch open_clip_torch onnx onnxruntime pillow numpy
    .venv/bin/python ../../scripts/export_pecore.py hf-hub:timm/PE-Core-T-16-384 pecore-t16-384 384 32

Writes <out>/<tag>-{vision,text}.onnx plus -int8 variants. Every int8 artifact must then
pass the B24 fidelity gate (cos >=0.98 AND top-1 NN ranking agreement >=0.90 vs its own
fp32) before it is trusted — official exports included (the Xenova precedent).
"""

import sys
from pathlib import Path

HUB = sys.argv[1] if len(sys.argv) > 1 else "hf-hub:timm/PE-Core-S-16-384"
TAG = sys.argv[2] if len(sys.argv) > 2 else "pecore-s16-384"
RES = int(sys.argv[3]) if len(sys.argv) > 3 else 384
CTX = int(sys.argv[4]) if len(sys.argv) > 4 else 32
OUT = Path(sys.argv[5]) if len(sys.argv) > 5 else Path(__file__).resolve().parents[1] / "models"


def main() -> int:
    import open_clip
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model, _, preprocess = open_clip.create_model_and_transforms(HUB)
    model.eval()
    print(f"{HUB}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params; preprocess={preprocess}")

    class Tower(torch.nn.Module):
        def __init__(self, m, fn):
            super().__init__()
            self.m, self.fn = m, fn

        def forward(self, x):
            return getattr(self.m, self.fn)(x)

    with torch.no_grad():
        for fn, name, sample, inp, outp in (
            ("encode_image", "vision", torch.randn(1, 3, RES, RES), "pixel_values", "image_embeds"),
            ("encode_text", "text", torch.randint(0, 49407, (1, CTX), dtype=torch.int64), "input_ids", "text_embeds"),
        ):
            dst = OUT / f"{TAG}-{name}.onnx"
            torch.onnx.export(
                Tower(model, fn), (sample,), str(dst),
                input_names=[inp], output_names=[outp],
                dynamic_axes={inp: {0: "batch"}, outp: {0: "batch"}},  # dynamic axis, fixed batch at run time
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
            # ADR-4 quantization recipe: weight-only int8, MatMul only, activations fp32.
            quantize_dynamic(
                str(dst), str(dst.with_name(f"{TAG}-{name}-int8.onnx")),
                weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul"],
                extra_options={"MatMulConstBOnly": True},
            )
            print(f"exported {dst.name} + int8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
