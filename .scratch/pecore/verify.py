"""ONNX-vs-torch parity + real caption retrieval sanity + thread sweep."""
import json, glob, time
import numpy as np, torch, open_clip, onnxruntime as ort
from PIL import Image

M = "/Users/magic/Creations/ImgTag/models"
DATA = "/Users/magic/Creations/ImgTag/data/quick500"


def prep(p):
    im = Image.open(p).convert("RGB").resize((384, 384), Image.BILINEAR)
    a = (np.asarray(im, np.float32) / 255.0 - 0.5) / 0.5
    return a.transpose(2, 0, 1)


def sess(path, th):
    o = ort.SessionOptions(); o.intra_op_num_threads = th
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


# ---- pick 3 images with distinctive COCO categories ----
d = json.load(open(f"{DATA}/instances_quick500.json"))
cats = {c["id"]: c["name"] for c in d["categories"]}
by_img = {}
for a in d["annotations"]:
    by_img.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
have = {int(p.split("/")[-1].split(".")[0]): p for p in glob.glob(f"{DATA}/images/*.jpg")}
picks = []
for iid, p in sorted(have.items()):
    c = by_img.get(iid, set())
    if len(c) == 1 and list(c)[0] not in [x[2] for x in picks]:
        picks.append((iid, p, list(c)[0]))
    if len(picks) == 3:
        break
print("picks:", [(i, c) for i, _, c in picks])

captions = [f"a photo of a {c}" for _, _, c in picks]
X = np.stack([prep(p) for _, p, _ in picks])

# ---- torch reference ----
model, _, _ = open_clip.create_model_and_transforms("hf-hub:timm/PE-Core-S-16-384")
model.eval()
tok = open_clip.get_tokenizer("hf-hub:timm/PE-Core-S-16-384")
ids = tok(captions).numpy().astype(np.int64)
print("token ids shape", ids.shape)
with torch.no_grad():
    tv = model.encode_image(torch.from_numpy(X)).numpy()
    tt = model.encode_text(torch.from_numpy(ids)).numpy()

sv, st = sess(f"{M}/pecore-s16-384-vision.onnx", 4), sess(f"{M}/pecore-s16-384-text.onnx", 4)
ov = sv.run(None, {"pixel_values": X})[0]
ot = st.run(None, {"input_ids": ids})[0]
sq = sess(f"{M}/pecore-s16-384-vision-int8.onnx", 4)
qv = sq.run(None, {"pixel_values": X})[0]

n = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
print(f"\nPARITY fp32-onnx vs torch  vision cos={np.sum(n(ov)*n(tv),1).round(6)}  max|Δ|={np.abs(ov-tv).max():.2e}")
print(f"PARITY fp32-onnx vs torch  text   cos={np.sum(n(ot)*n(tt),1).round(6)}  max|Δ|={np.abs(ot-tt).max():.2e}")
print(f"PARITY int8-onnx vs torch  vision cos={np.sum(n(qv)*n(tv),1).round(6)}")

print("\nRETRIEVAL sanity (rows=images, cols=captions), fp32 onnx:")
S = n(ov) @ n(ot).T
for i, (_, _, c) in enumerate(picks):
    print(f"  img[{c:12s}] " + "  ".join(f"{captions[j].split()[-1]}={S[i,j]:+.4f}" for j in range(3)) + f"   argmax={captions[int(S[i].argmax())].split()[-1]} {'OK' if S[i].argmax()==i else 'FAIL'}")
Sq = n(qv) @ n(ot).T
print("int8 argmax:", [captions[int(r.argmax())].split()[-1] for r in Sq], "->", "OK" if (Sq.argmax(1) == np.arange(3)).all() else "FAIL")

print("\nTHREAD SWEEP (vision, single image, median of 10):")
for tag, path in [("fp32", f"{M}/pecore-s16-384-vision.onnx"), ("int8", f"{M}/pecore-s16-384-vision-int8.onnx")]:
    out = []
    for th in (1, 2, 4, 6, 8, 12, 16):
        s = sess(path, th)
        f = {"pixel_values": X[:1]}
        for _ in range(3):
            s.run(None, f)
        ts = sorted((time.perf_counter(), s.run(None, f), time.perf_counter())[::2] for _ in range(10))
        ts = sorted([(b - a) * 1000 for a, b in ts])
        out.append(f"th{th}={ts[5]:.0f}ms")
    print(f"  {tag}: " + "  ".join(out))
