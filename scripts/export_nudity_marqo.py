#!/usr/bin/env python3
"""OFFLINE export of Marqo/nsfw-image-detection-384 (Apache-2.0) to ONNX.

Runs in a THROWAWAY venv — torch/timm must never enter the runtime env (ADR-7, B23):

    mkdir -p .scratch/nudity && cd .scratch/nudity
    uv venv --python 3.12 .venv
    VIRTUAL_ENV=$PWD/.venv uv pip install timm onnx onnxruntime
    .venv/bin/python ../../scripts/export_nudity_marqo.py

Writes models/moderation/nudity-marqo-384.onnx (fp32, dynamic batch) and prints the
sha256 + a parity check (torch vs ORT logits on random input) so the export is trusted
by measurement, not by assumption.
"""

import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "models" / "moderation")
HUB = "hf_hub:Marqo/nsfw-image-detection-384"
NAME = "nudity-marqo-384.onnx"


def main() -> int:
    import hashlib

    import numpy as np
    import onnxruntime as ort
    import timm
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / NAME

    model = timm.create_model(HUB, pretrained=True).eval()
    cfg = model.pretrained_cfg
    print(f"cfg: input_size={cfg['input_size']} mean={cfg['mean']} std={cfg['std']} "
          f"interp={cfg['interpolation']} crop_pct={cfg['crop_pct']} labels={model.pretrained_cfg.get('label_names')}")

    dummy = torch.randn(2, 3, 384, 384)
    torch.onnx.export(
        model, dummy, str(dst),
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )

    # torch 2.13's dynamo exporter externalizes weights (.onnx.data). We ship ONE file so
    # the sha256 covers the weights too — a hash over the 80KB graph alone proves nothing.
    import onnx

    m = onnx.load(str(dst), load_external_data=True)
    onnx.save(m, str(dst), save_as_external_data=False)
    for junk in OUT.glob(f"{NAME}.data"):
        junk.unlink()

    # Parity: torch fp32 vs ORT fp32 on the same input. Export is trusted only if it matches.
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(
        None, {"pixel_values": dummy.numpy()})[0]
    dmax = float(np.abs(ref - got).max())
    print(f"parity: max|torch-ort| = {dmax:.3e}  (must be < 1e-3)")

    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    (dst.with_suffix(".onnx.sha256")).write_text(f"{sha}  {NAME}\n")
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB)\nsha256 {sha}")
    return 0 if dmax < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
