import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os, time
src = '/tmp/imodels/clipb32_vis_fp32.onnx'
outs = {}
for tag, wt, per_ch in [('dynU8_perch', QuantType.QUInt8, True),
                        ('dynS8_perch', QuantType.QInt8, True),
                        ('dynU8_pertensor', QuantType.QUInt8, False)]:
    dst = f'/tmp/imodels/b32_{tag}.onnx'
    if not os.path.exists(dst):
        t = time.perf_counter()
        try:
            quantize_dynamic(src, dst, weight_type=wt, per_channel=per_ch,
                             op_types_to_quantize=['MatMul'], extra_options={'MatMulConstBOnly': True})
            print(f"{tag}: quantized in {time.perf_counter()-t:.1f}s  size={os.path.getsize(dst)/1e6:.1f}MB")
        except Exception as e:
            print(f"{tag}: FAILED {type(e).__name__} {str(e)[:200]}")
