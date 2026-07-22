import onnxruntime as ort, numpy as np, glob
from PIL import Image
ort.set_default_logger_severity(4)
M = '/tmp/imodels/'
files = sorted(glob.glob('/tmp/realimgs/*.jpg'))[:24]
MEAN = np.array([0.48145466,0.4578275,0.40821073],np.float32)
STD  = np.array([0.26862954,0.26130258,0.27577711],np.float32)

def load(f,S,norm_mode):
    im=Image.open(f); im.draft('RGB',(S,S)); im=im.convert('RGB').resize((S,S),Image.BICUBIC)
    a=np.asarray(im,dtype=np.float32)/255.0
    if norm_mode=='clip': a=(a-MEAN)/STD
    return a.transpose(2,0,1)

def embed(path,S,nm):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    s=ort.InferenceSession(path,so,providers=['CPUExecutionProvider'])
    X=np.stack([load(f,S,nm) for f in files]).astype(np.float32)
    return s.run(None,{'pixel_values':X})[0].astype(np.float32)

def stats(tag,E):
    N=E/np.linalg.norm(E,axis=1,keepdims=True)
    S=N@N.T; np.fill_diagonal(S,np.nan)
    print(f"{tag:34s} |emb| mean={np.linalg.norm(E,axis=1).mean():8.3f}  "
          f"pairwise-cos mean={np.nanmean(S):.4f} std={np.nanstd(S):.4f}  "
          f"(collapse if mean~1.0 & std~0)")
    return N

for nm in ('clip','01'):
    print(f"--- preprocessing normalization = {nm}")
    a=stats(f'b32   fp32 [{nm}]', embed(M+'clipb32_vis_fp32.onnx',224,nm))
    b=stats(f'b32   int8 [{nm}]', embed(M+'clipb32_vis_int8.onnx',224,nm))
    print(f"   cos(fp32,int8) mean={np.mean((a*b).sum(1)):.4f}")
    c=stats(f'mcs0  fp32 [{nm}]', embed(M+'mcs0_vis_fp32.onnx',256,nm))
    d=stats(f'mcs0  int8 [{nm}]', embed(M+'mcs0_vis_int8.onnx',256,nm))
    print(f"   cos(fp32,int8) mean={np.mean((c*d).sum(1)):.4f}")
