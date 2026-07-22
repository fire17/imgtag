import onnxruntime as ort, numpy as np, glob, time, os
from PIL import Image
ort.set_default_logger_severity(4)
M='/tmp/imodels/'
files=sorted(glob.glob('/tmp/realimgs/*.jpg'))[:24]
MEAN=np.array([0.48145466,0.4578275,0.40821073],np.float32); STD=np.array([0.26862954,0.26130258,0.27577711],np.float32)
def load(f,S=224):
    im=Image.open(f); im.draft('RGB',(S,S)); im=im.convert('RGB').resize((S,S),Image.BICUBIC)
    a=(np.asarray(im,np.float32)/255.0-MEAN)/STD
    return a.transpose(2,0,1)
X=np.stack([load(f) for f in files]).astype(np.float32)

def run(path,threads=4):
    so=ort.SessionOptions(); so.intra_op_num_threads=threads; so.inter_op_num_threads=1
    s=ort.InferenceSession(path,so,providers=['CPUExecutionProvider'])
    e=s.run(None,{'pixel_values':X})[0].astype(np.float32)
    e/=np.linalg.norm(e,axis=1,keepdims=True)
    xb=X[:4]
    for _ in range(2): s.run(None,{'pixel_values':xb})
    t=time.perf_counter()
    for _ in range(6): s.run(None,{'pixel_values':xb})
    ips=4*6/(time.perf_counter()-t)
    return e, ips

base,ips0 = run(M+'clipb32_vis_fp32.onnx')
print(f"{'variant':22s} {'MB':>7} {'img/s':>8} {'cos vs fp32':>12} {'min cos':>9} {'top1-NN match':>14}")
print(f"{'fp32 (reference)':22s} {os.path.getsize(M+'clipb32_vis_fp32.onnx')/1e6:7.1f} {ips0:8.1f} {'1.0000':>12} {'-':>9} {'1.00':>14}")
Sb=base@base.T; np.fill_diagonal(Sb,-9); tb=np.argmax(Sb,1)
for tag,f in [('xenova int8','clipb32_vis_int8.onnx'),
              ('mine dynU8 per-chan','b32_dynU8_perch.onnx'),
              ('mine dynS8 per-chan','b32_dynS8_perch.onnx'),
              ('mine dynU8 per-tensor','b32_dynU8_pertensor.onnx')]:
    p=M+f
    if not os.path.exists(p): print(f"{tag:22s} MISSING"); continue
    try:
        e,ips=run(p)
        cos=(base*e).sum(1)
        Se=e@e.T; np.fill_diagonal(Se,-9); te=np.argmax(Se,1)
        print(f"{tag:22s} {os.path.getsize(p)/1e6:7.1f} {ips:8.1f} {cos.mean():12.4f} {cos.min():9.4f} {np.mean(te==tb):14.2f}")
    except Exception as ex:
        print(f"{tag:22s} ERR {type(ex).__name__} {str(ex)[:80]}")
