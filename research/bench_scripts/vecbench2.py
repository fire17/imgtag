import numpy as np, time, os
D=512
def timeit(f,reps=20):
    for _ in range(3): f()
    t=time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter()-t)/reps*1000

rng=np.random.default_rng(0)
for N in (10_000, 100_000, 1_000_000):
    X32=rng.standard_normal((N,D),dtype=np.float32); X32/=np.linalg.norm(X32,axis=1,keepdims=True)
    X16=X32.astype(np.float16)
    Xi8=np.clip(np.round(X32*127),-127,127).astype(np.int8)
    q32=X32[0].copy()
    print(f"--- N={N} mem: fp32={X32.nbytes/1e6:.0f}MB fp16={X16.nbytes/1e6:.0f}MB int8={Xi8.nbytes/1e6:.0f}MB")
    print(f"  fp32 X@q + top10        : {timeit(lambda: np.argpartition(X32@q32,-10)[-10:]):8.3f} ms")
    print(f"  fp16 store->fp32 cast@q : {timeit(lambda: np.argpartition(X16.astype(np.float32)@q32,-10)[-10:]):8.3f} ms")
    print(f"  int8 store->fp32 cast@q : {timeit(lambda: np.argpartition(Xi8.astype(np.float32)@q32,-10)[-10:]):8.3f} ms")
    print(f"  int8 int32 matmul       : {timeit(lambda: np.argpartition(Xi8@q32.astype(np.int8) if False else Xi8.astype(np.int16)@q32.astype(np.int16),-10)[-10:]):8.3f} ms")
    Q=rng.standard_normal((8,D),dtype=np.float32)
    print(f"  fp32 batch8 queries     : {timeit(lambda: (X32@Q.T)):8.3f} ms")
    del X32,X16,Xi8
