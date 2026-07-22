import onnxruntime as ort, numpy as np, glob, os
from PIL import Image
ort.set_default_logger_severity(4)
M = '/tmp/imodels/'
files = sorted(glob.glob('/tmp/realimgs/*.jpg'))[:24]

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def load(f, S):
    im = Image.open(f); im.draft('RGB', (S, S))
    im = im.convert('RGB').resize((S, S), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1)

def embed(path, S, norm=True):
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    X = np.stack([load(f, S) for f in files]).astype(np.float32)
    out = s.run(None, {'pixel_values': X})[0].astype(np.float32)
    if norm: out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out

def report(name, a, b):
    cos = (a * b).sum(1)
    # rank agreement: image-image similarity matrix top-3 overlap
    Sa, Sb = a @ a.T, b @ b.T
    np.fill_diagonal(Sa, -9); np.fill_diagonal(Sb, -9)
    ta = np.argsort(-Sa, 1)[:, :3]; tb = np.argsort(-Sb, 1)[:, :3]
    ov = np.mean([len(set(x) & set(y)) / 3 for x, y in zip(ta, tb)])
    top1 = np.mean(ta[:, 0] == tb[:, 0])
    print(f"{name}: per-image cos(fp32,quant) mean={cos.mean():.4f} min={cos.min():.4f} | "
          f"top-3 neighbour overlap={ov:.3f} | top-1 neighbour match={top1:.3f}")

print("=== CLIP ViT-B/32 (224) ===")
a = embed(M + 'clipb32_vis_fp32.onnx', 224)
b = embed(M + 'clipb32_vis_int8.onnx', 224)
report("int8 vs fp32", a, b)

print("=== MobileCLIP-S0 (256) ===")
c = embed(M + 'mcs0_vis_fp32.onnx', 256)
d = embed(M + 'mcs0_vis_int8.onnx', 256)
report("int8 vs fp32", c, d)
e = embed(M + 'mcs0_vis_q4.onnx', 256)
report("q4   vs fp32", c, e)

# fp16 STORAGE (not compute) fidelity — what we'd store in the index
for nm, x in (("b32", a), ("mcs0", c)):
    h = x.astype(np.float16).astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    print(f"fp16-STORAGE {nm}: cos={np.mean((x*h).sum(1)):.6f}")
    q = np.clip(np.round(x * 127), -127, 127).astype(np.int8).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    print(f"int8-STORAGE {nm}: cos={np.mean((x*q).sum(1)):.6f}")
