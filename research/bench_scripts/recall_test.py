import numpy as np, time
D, K, N = 512, 10, 100_000
rng = np.random.default_rng(0)
# clustered, low-intrinsic-dim data ~ real CLIP embeddings
C = rng.standard_normal((200, D), np.float32)
lab = rng.integers(0, 200, N)
X = C[lab] * 1.0 + rng.standard_normal((N, D), np.float32) * 0.55
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = X[rng.choice(N, 50, replace=False)] + rng.standard_normal((50, D), np.float32) * 0.15
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
gt = [set(np.argsort(-(X @ q))[:K].tolist()) for q in Q]

def recall(fn):
    return np.mean([len(gt[i] & set(fn(Q[i]))) / K for i in range(len(Q))])
def lat(fn, reps=20):
    for _ in range(3): fn(Q[0])
    t = time.perf_counter()
    for _ in range(reps):
        for q in Q[:5]: fn(q)
    return (time.perf_counter() - t) / (reps * 5) * 1000

print(f"clustered data N={N} D={D} (200 clusters) — realistic embedding geometry")
bf = lambda q: np.argpartition(X @ q, -K)[-K:].tolist()
print(f"  numpy brute force      : {lat(bf):7.3f} ms  recall@{K}={recall(bf):.3f}")
from usearch.index import Index
for dt, M_, ef in (('f32', 16, 64), ('f32', 32, 128), ('i8', 32, 128)):
    idx = Index(ndim=D, metric='ip', dtype=dt, connectivity=M_, expansion_add=128, expansion_search=ef)
    t0 = time.perf_counter(); idx.add(np.arange(N), X, log=False); b = time.perf_counter() - t0
    f = lambda q: np.asarray(idx.search(q, K).keys).tolist()
    print(f"  usearch HNSW {dt} M={M_:<2} ef={ef:<3}: {lat(f):7.3f} ms  recall@{K}={recall(f):.3f}  build={b:.1f}s  mem={idx.memory_usage/1e6:.0f}MB")
    del idx
