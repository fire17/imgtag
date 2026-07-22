import onnxruntime as ort, numpy as np, time, os
ort.set_default_logger_severity(3)
NCPU=os.cpu_count()
def bench(path, shape, threads, batch, reps=8, dtype=np.float32, name='pixel_values'):
    so=ort.SessionOptions(); so.intra_op_num_threads=threads; so.inter_op_num_threads=1
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s=ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    x=np.random.rand(batch,*shape).astype(dtype)
    for _ in range(3): s.run(None,{name:x})
    t=time.perf_counter()
    for _ in range(reps): s.run(None,{name:x})
    dt=(time.perf_counter()-t)/reps
    return dt/batch*1000, batch/dt   # ms/img, img/s
M='/tmp/imodels/'
print(f"cpu_count={NCPU}")
rows=[]
for lbl,f,shp,dt in [
  ('mobileclip_s0 fp32','mcs0_vis_fp32.onnx',(3,256,256),np.float32),
  ('mobileclip_s0 int8','mcs0_vis_int8.onnx',(3,256,256),np.float32),
  ('mobileclip_s0 q4  ','mcs0_vis_q4.onnx',(3,256,256),np.float32),
  ('clip-ViT-B/32 fp32','clipb32_vis_fp32.onnx',(3,224,224),np.float32),
  ('clip-ViT-B/32 int8','clipb32_vis_int8.onnx',(3,224,224),np.float32),
  ('clip-ViT-B/32 fp16','clipb32_vis_fp16.onnx',(3,224,224),np.float16),
]:
    for threads in (1,4,8):
        for batch in (1,8):
            try:
                ms,ips=bench(M+f,shp,threads,batch,dtype=dt)
                print(f"{lbl} | thr={threads} bs={batch}: {ms:7.2f} ms/img  {ips:7.1f} img/s")
            except Exception as e:
                print(f"{lbl} thr={threads} bs={batch} ERR {type(e).__name__}: {str(e)[:120]}")
