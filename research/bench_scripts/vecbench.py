import numpy as np, time
def bench(N, D, dtype, reps=20):
    rng=np.random.default_rng(0)
    if dtype==np.float32:
        X=rng.standard_normal((N,D),dtype=np.float32); X/=np.linalg.norm(X,axis=1,keepdims=True)
        q=rng.standard_normal((D,),dtype=np.float32); q/=np.linalg.norm(q)
    elif dtype==np.float16:
        X=rng.standard_normal((N,D)).astype(np.float16)
        q=rng.standard_normal((D,)).astype(np.float16)
    else: # int8
        X=(rng.standard_normal((N,D))*40).astype(np.int8)
        q=(rng.standard_normal((D,))*40).astype(np.int8)
    # warm
    for _ in range(3): s=X@q
    t=time.perf_counter()
    for _ in range(reps):
        s=X@q
        top=np.argpartition(s,-10)[-10:]
    dt=(time.perf_counter()-t)/reps
    return dt*1000
for D in (512,):
  for N in (10_000,100_000,1_000_000):
    for dt_,name in ((np.float32,'fp32'),(np.float16,'fp16'),(np.int8,'int8')):
        try:
            ms=bench(N,D,dt_)
            print(f"N={N:>9} D={D} {name}: {ms:8.3f} ms  ({N/ (ms/1000)/1e6:.1f} M vec/s)")
        except Exception as e: print(N,D,name,"ERR",e)
