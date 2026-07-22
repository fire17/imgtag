"""Validate the lock-free append + atomic-manifest pattern for search-while-indexing.

Writer:  appends fixed-size fp32 rows to shard.f32, fsyncs, then atomically
         replaces manifest.json with the new committed row count.
Reader:  reads manifest count -> np.memmap the shard -> matmul over [0:count].
No locks anywhere. Correctness rests on: rows are only ever APPENDED, and the
count is only published AFTER the bytes are durable (publish-after-write).
"""
import numpy as np, os, json, time, multiprocessing as mp, tempfile, sys

D = 512
DIR = os.environ.setdefault('IMGTAG_CONC_DIR', tempfile.mkdtemp(prefix='imgtag_conc_'))
SHARD = os.path.join(DIR, 'shard.f32')
MAN = os.path.join(DIR, 'manifest.json')

def commit(n):
    tmp = MAN + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'count': n, 'dim': D}, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, MAN)          # atomic on POSIX + Windows

def writer(total, batch):
    rng = np.random.default_rng(1)
    with open(SHARD, 'wb', buffering=0) as f:
        written = 0
        while written < total:
            b = min(batch, total - written)
            # row i has value i/total in every dim -> trivially verifiable
            rows = np.tile(((np.arange(written, written + b) + 1) / total).astype(np.float32)[:, None], (1, D))
            f.write(rows.tobytes()); os.fsync(f.fileno())
            written += b
            commit(written)
            time.sleep(0.002)

def reader(q, total):
    seen, torn, bad, polls = 0, 0, 0, 0
    t0 = time.perf_counter()
    lat = []
    while seen < total and time.perf_counter() - t0 < 30:
        polls += 1
        try:
            n = json.load(open(MAN))['count']
        except Exception:
            continue
        if n == 0: continue
        if os.path.getsize(SHARD) < n * D * 4:
            torn += 1; continue
        mm = np.memmap(SHARD, dtype=np.float32, mode='r', shape=(n, D))
        t = time.perf_counter()
        q_vec = np.full(D, 1.0, np.float32)
        s = mm @ q_vec
        top = int(np.argmax(s))
        lat.append((time.perf_counter() - t) * 1000)
        if top != n - 1: bad += 1      # largest value is always the newest row
        exp = (n / total) * D
        if abs(s[n - 1] - exp) > 1e-2 * D: bad += 1
        seen = n
        del mm
    q.put(dict(seen=seen, torn=torn, bad=bad, polls=polls,
               lat_ms_med=float(np.median(lat)) if lat else -1,
               lat_ms_max=float(np.max(lat)) if lat else -1))

if __name__ == '__main__':
    TOTAL, BATCH = 20000, 250
    q = mp.Queue()
    r = mp.Process(target=reader, args=(q, TOTAL)); r.start()
    w = mp.Process(target=writer, args=(TOTAL, BATCH)); w.start()
    w.join(); r.join()
    res = q.get()
    print(f"lock-free append + atomic manifest: rows_visible={res['seen']}/{TOTAL} "
          f"torn_reads={res['torn']} wrong_results={res['bad']} reader_polls={res['polls']}")
    print(f"  concurrent search latency over growing index: median={res['lat_ms_med']:.3f} ms "
          f"max={res['lat_ms_max']:.3f} ms  (search ran WHILE writer appended)")
    print(f"  final shard size = {os.path.getsize(SHARD)/1e6:.1f} MB for {TOTAL} x {D} fp32")
