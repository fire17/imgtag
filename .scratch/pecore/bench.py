"""Verify + benchmark PE-Core-S16-384 ONNX on CPU. Throwaway spike."""
import json, os, time, glob
import numpy as np, onnxruntime as ort
from PIL import Image

M = "/Users/magic/Creations/ImgTag/models"
DATA = "/Users/magic/Creations/ImgTag/data/quick500"
RES, DIM = 384, 512


def sess(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def prep(p):
    im = Image.open(p).convert("RGB").resize((RES, RES), Image.BILINEAR)  # squash
    a = np.asarray(im, np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return a.transpose(2, 0, 1)


def timeit(s, feed, n=10, warm=3):
    for _ in range(warm):
        s.run(None, feed)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); s.run(None, feed); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2], min(ts)


print(f"onnxruntime {ort.__version__} | providers {ort.get_available_providers()}")
imgs = sorted(glob.glob(f"{DATA}/images/*.jpg"))[:8]
X = np.stack([prep(p) for p in imgs])
print(f"{len(imgs)} images loaded, batch shape {X.shape}")

rows = []
for tag, path in [("fp32", f"{M}/pecore-s16-384-vision.onnx"), ("int8", f"{M}/pecore-s16-384-vision-int8.onnx")]:
    if not os.path.exists(path):
        continue
    for th in (4, 16):
        s = sess(path, th)
        m1, b1 = timeit(s, {"pixel_values": X[:1]})
        m8, b8 = timeit(s, {"pixel_values": X}, n=5, warm=2)
        rows.append((tag, th, m1, b1, m8, m8 / 8, 1000 / (m8 / 8)))
        print(f"VISION {tag} th={th:2d}  1img {m1:7.1f}ms (best {b1:6.1f})  b8 {m8:8.1f}ms = {m8/8:6.1f}ms/img  {1000/(m8/8):6.1f} img/s")

# text tower
tok_rows = []
for th in (4, 16):
    st = sess(f"{M}/pecore-s16-384-text.onnx", th)
    ids = np.zeros((1, 32), np.int64)
    m1, _ = timeit(st, {"input_ids": ids})
    ids8 = np.zeros((8, 32), np.int64)
    m8, _ = timeit(st, {"input_ids": ids8}, n=5, warm=2)
    tok_rows.append((th, m1, m8))
    print(f"TEXT   fp32 th={th:2d}  1txt {m1:7.2f}ms   b8 {m8:7.2f}ms = {m8/8:5.2f}ms/txt")

# correctness: dim + L2 norm + dynamic batch
sv = sess(f"{M}/pecore-s16-384-vision.onnx", 16)
e = sv.run(None, {"pixel_values": X})[0]
print(f"\nvision embeds shape {e.shape} dtype {e.dtype}  L2 norms {np.linalg.norm(e,axis=1)[:4].round(4)}")
st = sess(f"{M}/pecore-s16-384-text.onnx", 16)
te = st.run(None, {"input_ids": np.zeros((3, 32), np.int64)})[0]
print(f"text embeds shape {te.shape}  L2 norms {np.linalg.norm(te,axis=1).round(4)}")
np.save("/Users/magic/Creations/ImgTag/.scratch/pecore/onnx_img_embeds.npy", e)
