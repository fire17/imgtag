"""End-to-end: decode+resize+normalize+encode, N worker processes x 1-2 ORT threads."""
import os, sys, time, glob
import numpy as np
NW = int(sys.argv[1]); THR = int(sys.argv[2]); MODEL = sys.argv[3]
os.environ['OMP_NUM_THREADS'] = str(THR)
FILES = sorted(glob.glob('/tmp/realimgs/*.jpg')) * 12   # 288 images
MEAN = np.array([0.48145466,0.4578275,0.40821073],np.float32)
STD  = np.array([0.26862954,0.26130258,0.27577711],np.float32)
S = 224; BATCH = 8

def work(chunk):
    import onnxruntime as ort
    from PIL import Image
    ort.set_default_logger_severity(4)
    so = ort.SessionOptions(); so.intra_op_num_threads = THR; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    n = 0
    for i in range(0, len(chunk), BATCH):
        b = chunk[i:i+BATCH]
        arr = np.empty((len(b),3,S,S), np.float32)
        for j,f in enumerate(b):
            im = Image.open(f); im.draft('RGB',(S,S))
            im = im.convert('RGB').resize((S,S), Image.BILINEAR)
            arr[j] = ((np.asarray(im,np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)
        sess.run(None, {'pixel_values': arr}); n += len(b)
    return n

if __name__ == '__main__':
    import multiprocessing as mp
    chunks = [FILES[i::NW] for i in range(NW)]
    work(chunks[0][:16])   # warm this proc (cache files)
    t = time.perf_counter()
    with mp.Pool(NW) as p:
        tot = sum(p.map(work, chunks))
    dt = time.perf_counter() - t
    print(f"workers={NW} ort_threads={THR} model={os.path.basename(MODEL):24s} "
          f"-> {tot/dt:7.1f} img/s   ({tot} imgs in {dt:.2f}s)  "
          f"| 10k imgs ETA = {10000/(tot/dt)/60:.1f} min")
