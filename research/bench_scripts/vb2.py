import numpy as np, time, sys
D, K = 512, 10
LIB = sys.argv[1]; N = int(sys.argv[2])
def t(f, reps=20):
    for _ in range(3): f()
    s = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - s) / reps * 1000
rng = np.random.default_rng(0)
X = rng.standard_normal((N, D), dtype=np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
q = X[0].copy()
exact_ids = set(np.argsort(-(X@q))[:K].tolist())

if LIB == 'numpy':
    print(f"N={N} numpy fp32 matmul+top{K}   : {t(lambda: np.argpartition(X@q,-K)[-K:]):8.3f} ms  [exact] mem={X.nbytes/1e6:.0f}MB")
elif LIB == 'usearch-exact':
    from usearch.index import search
    print(f"N={N} usearch exact SIMD        : {t(lambda: search(X, q, K, exact=True)):8.3f} ms  [exact]")
elif LIB.startswith('usearch-hnsw'):
    from usearch.index import Index
    dt = LIB.split('-')[-1]
    idx = Index(ndim=D, metric='ip', dtype=dt, connectivity=16)
    t0=time.perf_counter(); idx.add(np.arange(N), X, log=False); b=time.perf_counter()-t0
    got = set(np.asarray(idx.search(q,K).keys).tolist())
    print(f"N={N} usearch HNSW {dt:3s}         : {t(lambda: idx.search(q,K)):8.3f} ms  build={b:6.2f}s recall@{K}={len(exact_ids&got)/K:.2f} mem={idx.memory_usage/1e6:.0f}MB")
elif LIB == 'faiss-flat':
    import faiss; faiss.omp_set_num_threads(4)
    f_ = faiss.IndexFlatIP(D); f_.add(X)
    print(f"N={N} faiss IndexFlatIP          : {t(lambda: f_.search(q.reshape(1,-1),K)):8.3f} ms  [exact]")
elif LIB == 'faiss-hnsw':
    import faiss; faiss.omp_set_num_threads(4)
    h = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
    t0=time.perf_counter(); h.add(X); b=time.perf_counter()-t0
    _,I = h.search(q.reshape(1,-1),K); got=set(I[0].tolist())
    print(f"N={N} faiss HNSW16               : {t(lambda: h.search(q.reshape(1,-1),K)):8.3f} ms  build={b:6.2f}s recall@{K}={len(exact_ids&got)/K:.2f}")
elif LIB == 'hnswlib':
    import hnswlib
    h = hnswlib.Index(space='ip', dim=D); h.init_index(max_elements=N, ef_construction=100, M=16)
    t0=time.perf_counter(); h.add_items(X, np.arange(N)); b=time.perf_counter()-t0
    h.set_ef(64)
    got=set(h.knn_query(q,k=K)[0][0].tolist())
    print(f"N={N} hnswlib M16 ef64           : {t(lambda: h.knn_query(q,k=K)):8.3f} ms  build={b:6.2f}s recall@{K}={len(exact_ids&got)/K:.2f}")
