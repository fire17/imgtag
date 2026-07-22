/* imgtag app — vanilla JS, zero build, zero dependencies.
 *
 * Four views over ONE API (IA.md view tiers): fleet · search results · dataset gallery ·
 * jobs & health. Everything on screen comes from the daemon; there is no placeholder data
 * anywhere in this file — when the daemon is down or a dataset is empty you get a designed,
 * honest state instead of invented numbers.
 *
 * B4 instrumentation (MANDATORY, the bench measures through it):
 *   performance.mark('imgtag:key')      — on every search keystroke
 *   performance.mark('imgtag:painted')  — post-commit rAF after the results paint
 *   performance.measure('imgtag:key-to-paint', 'imgtag:key', 'imgtag:painted')
 * B14: grid is hand-rolled virtualization (scroll math + node reuse). DOM stays < 5000 nodes.
 */
'use strict';

// ── tiny helpers ───────────────────────────────────────────────────────────
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) if (k != null) n.append(k);
  return n;
};
const esc = (s) => String(s ?? '');
const nf = new Intl.NumberFormat();
const n0 = (v) => (Number.isFinite(v) ? nf.format(Math.round(v)) : '—');
const pct = (a, b) => (b > 0 ? Math.max(0, Math.min(100, (a / b) * 100)) : 0);
const bytes = (v) => {
  if (!Number.isFinite(v) || v < 0) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${u[i]}`;
};
const dur = (s) => {
  if (!Number.isFinite(s) || s < 0) return '—';
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${Math.round(s % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
};
// accepts epoch seconds, epoch millis, or an ISO-8601 string (the daemon sends ISO)
const ago = (ts) => {
  if (!ts) return '';
  const t = typeof ts === 'string' ? Date.parse(ts) / 1000 : ts > 1e12 ? ts / 1000 : ts;
  if (!Number.isFinite(t)) return '';
  const s = Date.now() / 1000 - t;
  return s < 60 ? 'just now' : s < 3600 ? `${Math.round(s / 60)}m ago`
    : s < 86400 ? `${Math.round(s / 3600)}h ago` : `${Math.round(s / 86400)}d ago`;
};

// parent directory (shrinkable) + filename (never shrinks)
const splitPath = (p) => {
  if (!p) return { dir: '', file: '' };
  const parts = String(p).split('/');
  const file = parts.pop();
  const parent = parts.pop();
  return { dir: parent ? `…/${parent}/` : '', file };
};
// "why this matched" — straight from the API payload, never inferred client-side
const whyText = (why) => {
  if (!why || !why.path) return '';
  if (why.tag) return `tag · ${why.tag}`;
  return why.path === 'text' ? 'text match' : String(why.path);
};

// ── API layer ──────────────────────────────────────────────────────────────
// One adapter per endpoint. Field fallbacks are deliberate: the daemon lane (b-daemon) owns
// the exact JSON keys; if a name differs at integration, it is fixed HERE and nowhere else.
const API = {
  base: '',
  online: null,
  async get(path, { signal } = {}) {
    const r = await fetch(this.base + path, { signal, headers: { accept: 'application/json' } });
    if (!r.ok) {
      const err = new Error(`HTTP ${r.status}`);
      err.status = r.status;
      try { err.body = await r.json(); } catch { /* non-JSON error body */ }
      throw err;
    }
    return r.json();
  },
  async datasets() {
    const j = await this.get('/api/datasets');
    const list = Array.isArray(j) ? j : (j.datasets || []);
    return list.map(normDataset);
  },
  async search(q, { dataset, k = 100, signal } = {}) {
    const p = new URLSearchParams({ q, k: String(k) });
    if (dataset) p.set('dataset', dataset);
    const t0 = performance.now();
    const j = await this.get(`/api/search?${p}`, { signal });
    return {
      query: j.query ?? q,
      tookMs: Number.isFinite(j.tookMs) ? j.tookMs : null,
      rttMs: performance.now() - t0,
      coverage: j.coverage || null,
      noMatch: j.no_match === true || (j.hits || []).length === 0,
      hits: (j.hits || []).map(normHit),
    };
  },
  async jobs() {
    const j = await this.get('/api/jobs');
    return (Array.isArray(j) ? j : (j.jobs || [])).map(normJob);
  },
  async status() { return this.get('/api/status'); },
  async images(dataset, offset, limit, signal) {
    const p = new URLSearchParams({ dataset, offset: String(offset), limit: String(limit) });
    const j = await this.get(`/api/images?${p}`, { signal });
    return { total: j.total ?? j.count ?? 0, items: (j.items || j.images || []).map(normHit) };
  },
  thumb(dataset, id, s = 256) {
    return `${this.base}/api/thumb/${encodeURIComponent(dataset)}/${encodeURIComponent(id)}?s=${s}`;
  },
};

function normDataset(d) {
  const count = d.count ?? d.indexed ?? d.manifest?.count ?? 0;
  const total = d.total ?? d.files ?? d.total_files ?? count;
  return {
    slug: d.dataset || d.slug || d.name || '?',
    root: d.root_path || d.root || '',
    count, total,
    bytes: d.index_bytes ?? d.bytes ?? d.size ?? null,
    model: d.model_id || d.model || '',
    dim: d.dim ?? null,
    updated: d.updated ?? d.created ?? null,
  };
}
function normHit(h) {
  return {
    id: h.image_id ?? h.id ?? '',
    path: h.path ?? '',
    dataset: h.dataset ?? h.dataset_slug ?? '',
    score: Number.isFinite(h.score) ? h.score : null,
    p: Number.isFinite(h.p) ? h.p : null,
    exists: h.exists !== false,
    why: h.why || null,
    w: h.w ?? null, h: h.h ?? null,
  };
}
function normJob(j) {
  return {
    id: j.job_id || j.id || '',
    dataset: j.dataset || '',
    state: j.state || 'queued',
    total: j.total ?? 0,
    done: j.done ?? 0,
    inFlight: j.in_flight ?? 0,
    failed: j.failed ?? 0,
    failures: j.failures || [],
    imgS: j.img_s ?? 0,
    etaS: j.eta_s ?? null,
    elapsedS: j.elapsed_s ?? null,
    updated: j.updated ?? null,
  };
}

// ── shared state ───────────────────────────────────────────────────────────
const S = {
  datasets: [],
  jobs: new Map(),          // job_id -> job  (live via SSE)
  lastQuery: '',
  recent: [],               // {q, tookMs, rttMs, hits, at} — observability, real data only
  rss: null,
  view: null,               // current view controller {destroy?}
};
const activeJobs = () => [...S.jobs.values()].filter((j) => j.state === 'running' || j.state === 'queued');

// ── B4 marks ───────────────────────────────────────────────────────────────
let keyPending = false;
function markKey() {
  performance.clearMarks('imgtag:key');
  performance.mark('imgtag:key');
  keyPending = true;
}
function markPainted() {
  if (!keyPending) return;
  keyPending = false;
  requestAnimationFrame(() => requestAnimationFrame(() => {   // post-commit
    performance.clearMarks('imgtag:painted');
    performance.mark('imgtag:painted');
    try { performance.measure('imgtag:key-to-paint', 'imgtag:key', 'imgtag:painted'); } catch { /* mark evicted */ }
  }));
}

// ── virtualized grid (hand-rolled: scroll math + node reuse, no libraries) ──
class VirtualGrid {
  /** @param scroller scrolling element  @param mount element inside it that owns the grid */
  constructor(scroller, mount, { min = 190, gap = 10, overscan = 2, cap = 44, hideDataset = false, onOpen, onFill } = {}) {
    this.scroller = scroller; this.mount = mount;
    this.min = min; this.gap = gap; this.overscan = overscan; this.cap = cap;
    this.hideDataset = hideDataset;   // inside one dataset, repeating its name on every tile is noise
    this.onOpen = onOpen; this.onFill = onFill;   // onFill(offset, limit) -> lazy page loader
    this.items = []; this.total = 0;
    this.nodes = new Map();                        // index -> element (reused across scrolls)
    this.sel = -1;
    this.win = el('div', { className: 'grid__win', role: 'listbox', tabIndex: -1 });
    this.win.setAttribute('aria-label', 'Images');
    this.mount.append(this.win);
    // coalesce bursts of scroll events into one render per frame (keeps every task short)
    this._raf = 0;
    this._onScroll = () => {
      if (this._raf) return;
      this._raf = requestAnimationFrame(() => { this._raf = 0; this.render(); });
    };
    this.scroller.addEventListener('scroll', this._onScroll, { passive: true });
    this._ro = new ResizeObserver(() => this.measure());
    this._ro.observe(this.mount);
    this.measure();
  }
  destroy() {
    this.scroller.removeEventListener('scroll', this._onScroll);
    this._ro.disconnect();
  }
  setItems(items, total = items.length) {
    this.items = items; this.total = total;
    for (const [, n] of this.nodes) n.remove();
    this.nodes.clear();
    this.sel = -1;
    this.measure();
  }
  measure() {
    const w = this.mount.clientWidth;
    if (!w) return;
    this.cols = Math.max(1, Math.floor((w + this.gap) / (this.min + this.gap)));
    this.tw = Math.floor((w - this.gap * (this.cols - 1)) / this.cols);
    this.th = this.tw + this.cap;                   // uniform cells ⇒ O(1) row math
    const rows = Math.ceil(this.total / this.cols);
    this.mount.style.height = `${Math.max(0, rows * (this.th + this.gap) - this.gap)}px`;
    this.render();
  }
  render() {
    if (!this.cols || !this.total) return;
    const rowH = this.th + this.gap;
    const top = Math.max(0, this.scroller.scrollTop - this.mount.offsetTop);
    const first = Math.max(0, Math.floor(top / rowH) - this.overscan);
    const last = Math.min(
      Math.ceil(this.total / this.cols) - 1,
      Math.floor((top + this.scroller.clientHeight) / rowH) + this.overscan,
    );
    const from = first * this.cols;
    const to = Math.min(this.total - 1, last * this.cols + this.cols - 1);

    for (const [i, node] of this.nodes) {
      if (i < from || i > to) { node.remove(); this.nodes.delete(i); }
    }
    let missing = null;
    for (let i = from; i <= to; i++) {
      const item = this.items[i];
      if (!item) { (missing ||= []).push(i); continue; }
      let node = this.nodes.get(i);
      if (!node) { node = this.tile(item, i); this.nodes.set(i, node); this.win.append(node); }
      const r = Math.floor(i / this.cols), c = i % this.cols;
      node.style.width = `${this.tw}px`;
      node.style.height = `${this.th}px`;
      node.style.transform = `translate(${c * (this.tw + this.gap)}px, ${r * rowH}px)`;
    }
    if (missing && this.onFill) this.onFill(missing[0], missing[missing.length - 1] - missing[0] + 1);
  }
  tile(item, i) {
    const t = el('button', { className: 'tile', type: 'button', tabIndex: -1, role: 'option' });
    t.dataset.i = String(i);
    t.setAttribute('aria-selected', String(i === this.sel));
    t.title = `${item.path || item.id}\n${item.dataset} · ${item.id}`;
    const frame = el('div', { className: 'tile__img' });
    if (item.exists === false) {
      t.classList.add('tile--gone');
      frame.append(el('span', { className: 'tile__x', textContent: 'file missing on disk' }));
    } else {
      const img = el('img', {
        src: API.thumb(item.dataset, item.id, 256),
        alt: item.path ? item.path.split('/').pop() : item.id,
        loading: 'lazy', decoding: 'async', draggable: false,
      });
      // a thumbnail that will not decode (truncated / hostile file) says so — never a blank square
      img.addEventListener('error', () => {
        t.classList.add('tile--gone');
        frame.replaceChildren(el('span', { className: 'tile__x', textContent: 'no thumbnail' }));
      }, { once: true });
      frame.append(img);
    }
    t.append(frame);
    if (item.score != null || item.p != null) {
      // confidence rail — width IS the calibrated probability (falls back to the raw score)
      const v = item.p != null ? item.p : Math.max(0, Math.min(1, item.score));
      t.append(el('div', { className: 'tile__rail' }, el('i', { style: `width:${(v * 100).toFixed(1)}%` })));
    }
    const cap = el('div', { className: 'tile__cap' });
    const shortId = item.id ? item.id.slice(0, 8) : '';
    if (!this.hideDataset) {
      const l1 = el('div', { className: 'tile__l1' },
        el('span', { className: 'tile__ds', textContent: item.dataset || '—' }),
        el('span', { className: 'tile__id', textContent: shortId }));
      if (item.score != null) l1.append(el('span', { className: 'tile__score', textContent: item.score.toFixed(3) }));
      cap.append(l1);
    }
    // the directory absorbs the squeeze; the FILENAME never shrinks — truncation must not
    // eat the one string you actually read
    const { dir, file } = splitPath(item.path);
    cap.append(el('div', { className: 'tile__l2' },
      dir && !this.hideDataset ? el('span', { className: 'tile__dir', textContent: dir }) : null,
      el('span', { className: 'tile__file', textContent: file || item.id }),
      this.hideDataset ? el('span', { className: 'tile__id', textContent: shortId }) : null));
    const why = whyText(item.why);
    if (why) cap.append(el('div', { className: 'tile__why', textContent: why }));
    t.append(cap);
    t.addEventListener('click', () => { this.select(i); this.onOpen?.(this.items[i]); });
    return t;
  }
  select(i) {
    if (i < 0 || i >= this.total) return;
    this.nodes.get(this.sel)?.setAttribute('aria-selected', 'false');
    this.sel = i;
    this.nodes.get(i)?.setAttribute('aria-selected', 'true');
    this.scrollTo(i);
  }
  scrollTo(i) {
    const rowH = this.th + this.gap;
    const r = Math.floor(i / this.cols);
    const y = this.mount.offsetTop + r * rowH;
    const view = this.scroller.scrollTop;
    if (y < view) this.scroller.scrollTop = y - this.gap;
    else if (y + this.th > view + this.scroller.clientHeight) this.scroller.scrollTop = y + this.th - this.scroller.clientHeight + this.gap;
    this.render();
    this.nodes.get(i)?.setAttribute('aria-selected', 'true');
  }
  move(d) {
    const next = this.sel < 0 ? 0 : this.sel + d;
    this.select(Math.max(0, Math.min(this.total - 1, next)));
  }
  current() { return this.items[this.sel] || null; }
}

// ── detail overlay (native <dialog>) ───────────────────────────────────────
const dlg = $('#detail');
function openDetail(item) {
  if (!item) return;
  const side = el('div', { className: 'detail__side' });
  const kv = (k, v, mono = true) => el('dl', { className: 'kv' },
    el('dt', { textContent: k }), el('dd', { textContent: v, style: mono ? '' : 'font-family:var(--font)' }));
  side.append(
    el('h2', { textContent: item.path ? item.path.split('/').pop() : item.id }),
    kv('dataset', item.dataset),
    kv('image id', item.id),
    kv('path', item.path || '—'),
  );
  if (item.w && item.h) side.append(kv('dimensions', `${item.w} × ${item.h}`));
  if (item.score != null) side.append(kv('score', item.score.toFixed(4)));
  if (item.p != null) side.append(kv('calibrated p', item.p.toFixed(3)));
  if (item.why) {
    side.append(kv('why it matched', item.why.tag ? `${item.why.path}: ${item.why.tag}` : String(item.why.path || '—')));
  }
  if (item.exists === false) side.append(kv('status', 'indexed, but the file is no longer on disk'));
  const close = el('button', { className: 'detail__close', textContent: 'Close  ·  Esc', type: 'button' });
  close.addEventListener('click', () => dlg.close());
  side.append(close);

  const box = el('div', { className: 'detail__box' },
    el('div', { className: 'detail__img' },
      item.exists === false
        ? el('p', { className: 'sub', style: 'padding:2rem', textContent: 'file missing on disk' })
        : el('img', { src: API.thumb(item.dataset, item.id, 1400), alt: item.path || item.id, decoding: 'async' })),
    side);
  dlg.replaceChildren(box);
  dlg.showModal();
}

// ── status strip (honest numbers only) ─────────────────────────────────────
function paintStatus({ latency, hits, coverage } = {}) {
  if (latency !== undefined) $('#stLatency').innerHTML = latency ?? '';
  if (hits !== undefined) $('#stHits').textContent = hits ?? '';
  if (coverage !== undefined) $('#stCoverage').textContent = coverage ?? '';
  const act = activeJobs();
  const rate = act.reduce((a, j) => a + (j.imgS || 0), 0);
  $('#stJobs').textContent = act.length
    ? `${act.length} job${act.length > 1 ? 's' : ''} · ${rate.toFixed(1)} img/s`
    : '';
  $('#jobsDot').hidden = act.length === 0;
  const d = $('#stDaemon');
  d.className = 'status__cell ' + (API.online === true ? 'ok' : API.online === false ? 'bad' : '');
  d.textContent = API.online === true ? (S.rss != null ? `daemon ok · rss ${bytes(S.rss)}` : 'daemon ok')
    : API.online === false ? 'daemon unreachable' : 'connecting…';
}

// ── SSE: live job progress (≤1s freshness per B10) ─────────────────────────
let es = null, sseRepaint = 0;
function connectEvents() {
  if (es) es.close();
  es = new EventSource(`${API.base}/api/events`);
  es.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    const list = Array.isArray(data) ? data : data.jobs ? data.jobs : [data];
    for (const raw of list) {
      if (!raw || (!raw.job_id && !raw.id)) continue;
      const j = normJob(raw);
      S.jobs.set(j.id, j);
    }
    if (data && Number.isFinite(data.rss)) S.rss = data.rss;
    API.online = true;
    const now = performance.now();
    if (now - sseRepaint > 250) {                 // coalesce: never more than 4 repaints/s
      sseRepaint = now;
      paintStatus();
      S.view?.onJobs?.();
    }
  };
  es.onerror = () => { API.online = false; paintStatus(); };
}

// ── views ──────────────────────────────────────────────────────────────────
const main = $('#main');
function mountView(node, controller = {}) {
  S.view?.destroy?.();
  S.view = controller;
  main.replaceChildren(node);
}

function daemonDownView(err) {
  return el('div', { className: 'view' }, el('div', { className: 'pad' },
    el('div', { className: 'empty' },
      el('h2', { textContent: 'The imgtag daemon is not answering' }),
      el('p', { textContent: `The app is a window onto a local daemon; nothing here is cached or faked, so there is nothing to show until it is up. (${err?.message || 'connection refused'})` }),
      el('p', { className: 'sub' }, 'Start it, then this view recovers on its own:'),
      el('p', null, el('code', { className: 'k', textContent: 'imgtag daemon --tcp 127.0.0.1:8787' })),
    )));
}

// (1) fleet / home
async function viewFleet() {
  const root = el('div', { className: 'view' });
  const pad = el('div', { className: 'pad' });
  root.append(pad);
  pad.append(el('div', { className: 'head' }, el('h1', { textContent: 'Datasets' }),
    el('span', { className: 'sub', id: 'fleetSum', textContent: 'reading manifests…' })));
  const grid = el('div', { className: 'fleet' });
  for (let i = 0; i < 3; i++) grid.append(el('div', { className: 'skel', style: 'height:148px' }));
  pad.append(grid);
  mountView(root, { onJobs: () => paintFleet() });

  let ds;
  try { ds = await API.datasets(); API.online = true; } catch (e) { API.online = false; mountView(daemonDownView(e)); paintStatus(); return; }
  S.datasets = ds;
  paintStatus({ latency: '', hits: '', coverage: '' });

  function paintFleet() {
    const total = S.datasets.reduce((a, d) => a + d.count, 0);
    const files = S.datasets.reduce((a, d) => a + d.total, 0);
    const sum = $('#fleetSum');
    if (sum) sum.textContent = S.datasets.length
      ? `${n0(total)} images searchable across ${S.datasets.length} dataset${S.datasets.length > 1 ? 's' : ''}${files > total ? ` · ${n0(files - total)} still to index` : ''}`
      : '';
    grid.replaceChildren();
    if (!S.datasets.length) {
      grid.replaceChildren(el('div', { className: 'empty' },
        el('h2', { textContent: 'No datasets indexed yet' }),
        el('p', { textContent: 'Point imgtag at a folder of images. Indexing is incremental and searchable while it runs — you do not have to wait for it to finish.' }),
        el('p', null, el('code', { className: 'k', textContent: 'imgtag index ~/Pictures --dataset pictures' })),
      ));
      return;
    }
    for (const d of S.datasets) {
      const job = [...S.jobs.values()].find((j) => j.dataset === d.slug && (j.state === 'running' || j.state === 'queued'));
      const done = job ? Math.max(d.count, job.done) : d.count;
      const total_ = job ? Math.max(d.total, job.total) : d.total;
      const card = el('a', { className: 'ds', href: `#/d/${encodeURIComponent(d.slug)}` });
      card.append(el('div', { className: 'ds__top' },
        el('span', { className: 'ds__name', textContent: d.slug }),
        d.root ? el('span', { className: 'ds__root', textContent: d.root, title: d.root }) : null));
      card.append(el('div', { className: 'ds__stats' },
        el('div', { className: 'stat' }, el('b', { textContent: n0(done) }), el('span', { textContent: 'indexed' })),
        el('div', { className: 'stat' }, el('b', { textContent: total_ > 0 ? `${Math.round(pct(done, total_))}%` : '—' }), el('span', { textContent: 'of dataset' })),
        el('div', { className: 'stat' }, el('b', { textContent: d.bytes != null ? bytes(d.bytes) : '—' }), el('span', { textContent: 'index size' }))));
      const prog = el('div', { className: 'ds__prog' });
      prog.append(el('i', { className: job ? 'live' : '', style: `width:${pct(done, total_)}%` }));
      if (job?.inFlight) prog.append(el('i', { className: 'ghost', style: `width:${pct(job.inFlight, total_)}%` }));
      card.append(prog);
      const foot = el('div', { className: 'ds__stats', style: 'margin-top:12px;gap:8px;flex-wrap:wrap' });
      if (job) foot.append(el('span', { className: 'chip chip--live', textContent: `● ${job.imgS.toFixed(1)} img/s · eta ${dur(job.etaS)}` }));
      if (d.model) foot.append(el('span', { className: 'chip', textContent: d.model }));
      if (d.updated) foot.append(el('span', { className: 'chip', textContent: ago(d.updated) }));
      card.append(foot);
      grid.append(card);
    }
  }
  paintFleet();
}

// (2) search results
let searchAbort = null, searchSeq = 0;
async function viewSearch(q, dataset) {
  const seq = ++searchSeq;   // only the newest keystroke may paint (responses can land out of order)
  $('#q').value = q;
  S.lastQuery = q;
  const root = el('div', { className: 'view' });
  const pad = el('div', { className: 'pad' });
  root.append(pad);
  const bannerSlot = el('div');
  const filterSlot = el('div');
  const head = el('div', { className: 'head' },
    el('h1', { textContent: q ? `“${q}”` : 'Search' }),
    el('span', { className: 'sub', id: 'resSum' }));
  const gridMount = el('div', { className: 'grid' });
  pad.append(head, bannerSlot, filterSlot, gridMount);

  let grid = null, lastRes = null;
  const ctl = {
    destroy: () => grid?.destroy(),
    onJobs: () => paintBanner(lastRes),
    key: (e) => {
      if (!grid) return false;
      const cols = grid.cols || 1;
      const map = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols, Home: -1e9, End: 1e9 };
      if (e.key in map) { grid.move(map[e.key]); return true; }
      if (e.key === 'Enter' && grid.sel >= 0) { openDetail(grid.current()); return true; }
      return false;
    },
  };
  mountView(root, ctl);

  if (!q) {
    pad.replaceChildren(head, el('div', { className: 'empty' },
      el('h2', { textContent: 'Type to search every indexed image' }),
      el('p', { textContent: 'Search is semantic, not keyword: “vehicle” finds the cars and the motorcycles too. Results stream from whatever is indexed right now, including datasets still being processed.' })));
    paintStatus({ latency: '—', hits: '', coverage: '' });
    return;
  }

  searchAbort?.abort();
  searchAbort = new AbortController();
  let res;
  try {
    res = await API.search(q, { dataset, k: 200, signal: searchAbort.signal });
    API.online = true;
  } catch (e) {
    if (e.name === 'AbortError' || seq !== searchSeq) return;
    API.online = false;
    mountView(daemonDownView(e));
    paintStatus();
    return;
  }
  if (seq !== searchSeq) return;   // superseded: never paint into a view the user has left
  lastRes = res;
  S.recent.unshift({ q, tookMs: res.tookMs, rttMs: res.rttMs, hits: res.hits.length, at: Date.now() });
  S.recent.length = Math.min(S.recent.length, 20);

  function paintBanner(r) {
    if (!r) return;
    bannerSlot.replaceChildren();
    const cov = r.coverage;
    const running = activeJobs();
    if (cov && cov.total > cov.indexed) {
      bannerSlot.append(el('div', { className: 'banner' },
        el('span', null, 'Partial coverage — searching '),
        el('b', { textContent: n0(cov.indexed) }), el('span', { textContent: ' of ' }),
        el('b', { textContent: n0(cov.total) }), el('span', { textContent: ' images indexed so far.' }),
        running.length ? el('span', { className: 'chip chip--live', style: 'margin-left:auto', textContent: `● ${running.reduce((a, j) => a + j.imgS, 0).toFixed(1)} img/s` }) : null,
      ));
    }
    const covPct = cov ? pct(cov.indexed, cov.total) : 100;
    const covBar = $('#qCov');
    covBar.hidden = !cov || covPct >= 99.999;
    covBar.firstElementChild.style.width = `${covPct}%`;
  }

  function paintFilters(r) {
    const counts = new Map();
    for (const h of r.hits) counts.set(h.dataset, (counts.get(h.dataset) || 0) + 1);
    if (counts.size < 2 && !dataset) { filterSlot.replaceChildren(); return; }
    const row = el('div', { className: 'filters' });
    const mk = (slug, label, on) => {
      const b = el('button', { type: 'button', textContent: label });
      b.setAttribute('aria-pressed', String(on));
      b.addEventListener('click', () => go(`#/search?q=${encodeURIComponent(q)}${slug ? `&d=${encodeURIComponent(slug)}` : ''}`));
      return b;
    };
    row.append(mk('', `all · ${n0(r.hits.length)}`, !dataset));
    for (const [slug, c] of [...counts].sort((a, b) => b[1] - a[1])) row.append(mk(slug, `${slug} · ${n0(c)}`, dataset === slug));
    filterSlot.replaceChildren(row);
  }

  paintBanner(res);
  paintFilters(res);
  $('#resSum').textContent = res.hits.length
    ? `${n0(res.hits.length)} ${res.hits.length === 1 ? 'hit' : 'hits'}${res.tookMs != null ? ` · ${res.tookMs.toFixed(1)} ms server` : ''}`
    : '';

  if (!res.hits.length) {
    gridMount.replaceChildren();
    gridMount.classList.remove('grid');
    gridMount.append(el('div', { className: 'empty' },
      el('h2', { textContent: `Nothing scored above the match threshold for “${q}”` }),
      el('p', { textContent: 'This is an answer, not a failure: the engine looked at every indexed image and refused to return weak matches, because a wrong hit costs you more than an empty page.' }),
      el('ul', null,
        el('li', { textContent: 'Try a broader word — “vehicle” instead of “tuk-tuk”; the semantic index expands hypernyms, not typos.' }),
        el('li', { textContent: `Searched ${res.coverage ? `${n0(res.coverage.indexed)} of ${n0(res.coverage.total)}` : 'all'} indexed images${activeJobs().length ? ' — indexing is still running, so re-run this in a moment' : ''}.` }),
        dataset ? el('li', null, 'Filtered to one dataset — ', el('a', { href: `#/search?q=${encodeURIComponent(q)}`, textContent: 'search all datasets' }), '.') : null,
      )));
  } else {
    gridMount.classList.add('grid');
    grid = new VirtualGrid(root, gridMount, { onOpen: openDetail, cap: 60 }); // 3 caption lines
    ctl.grid = grid;
    grid.setItems(res.hits);
    grid.select(0);
  }

  paintStatus({
    latency: res.tookMs != null
      ? `<b>${res.tookMs.toFixed(1)} ms</b> server · ${res.rttMs.toFixed(0)} ms round-trip`
      : `${res.rttMs.toFixed(0)} ms round-trip`,
    hits: `${n0(res.hits.length)} ${res.hits.length === 1 ? 'hit' : 'hits'}`,
    coverage: res.coverage ? `coverage ${n0(res.coverage.indexed)}/${n0(res.coverage.total)}` : '',
  });
  markPainted();
}

// (3) dataset gallery
async function viewDataset(slug) {
  const root = el('div', { className: 'view' });
  const pad = el('div', { className: 'pad' });
  root.append(pad);
  const head = el('div', { className: 'head' },
    el('h1', { textContent: slug }),
    el('span', { className: 'sub', id: 'dsSum', textContent: 'loading…' }));
  const form = el('form', { className: 'filters' });
  const inp = el('input', {
    type: 'search', placeholder: `search inside ${slug}`, className: 'q__input',
    style: 'max-width:340px;height:30px', spellcheck: false,
  });
  form.append(inp);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (inp.value.trim()) go(`#/search?q=${encodeURIComponent(inp.value.trim())}&d=${encodeURIComponent(slug)}`);
  });
  const prog = el('div');
  const gridMount = el('div', { className: 'grid' });
  pad.append(head, form, prog, gridMount);

  let grid = null;
  const ctl = {
    destroy: () => grid?.destroy(),
    onJobs: () => paintProgress(),
    key: (e) => {
      if (!grid) return false;
      const cols = grid.cols || 1;
      const map = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols };
      if (e.key in map) { grid.move(map[e.key]); return true; }
      if (e.key === 'Enter' && grid.sel >= 0) { openDetail(grid.current()); return true; }
      return false;
    },
  };
  mountView(root, ctl);

  function paintProgress() {
    const job = [...S.jobs.values()].find((j) => j.dataset === slug && (j.state === 'running' || j.state === 'queued'));
    prog.replaceChildren();
    if (!job) return;
    prog.append(el('div', { className: 'banner' },
      el('span', { className: 'chip chip--live', textContent: `● indexing · ${job.imgS.toFixed(1)} img/s` }),
      el('span', null, `${n0(job.done)} of ${n0(job.total)} · eta ${dur(job.etaS)}`),
      job.failed ? el('a', { href: '#/jobs', className: 'sub', style: 'margin-left:auto', textContent: `${n0(job.failed)} skipped — see the ledger` }) : null,
    ));
  }

  // Paged image listing. The gallery needs a paged listing endpoint; if the daemon does not
  // expose one yet, say so plainly rather than inventing a grid.
  let loading = new Set();
  const PAGE = 300;
  let items = [];
  async function page(offset, limit) {
    const start = Math.max(0, Math.floor(offset / PAGE) * PAGE);
    if (loading.has(start)) return;
    loading.add(start);
    try {
      const r = await API.images(slug, start, Math.max(PAGE, limit), null);
      for (let i = 0; i < r.items.length; i++) items[start + i] = r.items[i];
      grid?.render();
      return r;
    } finally { loading.delete(start); }
  }

  try {
    const first = await API.images(slug, 0, PAGE, null);
    API.online = true;
    items = new Array(first.total);
    for (let i = 0; i < first.items.length; i++) items[i] = first.items[i];
    $('#dsSum').textContent = `${n0(first.total)} images indexed`;
    grid = new VirtualGrid(root, gridMount, { onOpen: openDetail, cap: 22, min: 170, hideDataset: true, onFill: (o, l) => page(o, l) });
    ctl.grid = grid;
    grid.setItems(items, first.total);
  } catch (e) {
    gridMount.classList.remove('grid');
    if (e.status === 404 || e.status === 501) {
      $('#dsSum').textContent = '';
      gridMount.replaceChildren(el('div', { className: 'empty' },
        el('h2', { textContent: 'Gallery listing is not served yet' }),
        el('p', { textContent: 'The daemon has no paged image-listing endpoint on this build, so this view has nothing real to render. Search over this dataset works now:' }),
        el('p', null, el('code', { className: 'k', textContent: `GET /api/images?dataset=${slug}&offset=0&limit=300` })),
      ));
    } else {
      API.online = false;
      mountView(daemonDownView(e));
    }
  }
  paintProgress();
  paintStatus({ latency: '', hits: '', coverage: '' });
}

// (4) jobs & health
async function viewJobs() {
  const root = el('div', { className: 'view' });
  const pad = el('div', { className: 'pad' });
  root.append(pad);
  pad.append(el('div', { className: 'head' }, el('h1', { textContent: 'Jobs & health' }),
    el('span', { className: 'sub', textContent: 'live from the daemon — progress events, never polled guesses' })));
  const metrics = el('div', { className: 'cards' });
  const jobsWrap = el('div');
  const queries = el('div');
  const failures = el('div');
  pad.append(metrics, jobsWrap, queries, failures);
  mountView(root, { onJobs: () => paint() });

  try {
    for (const j of await API.jobs()) S.jobs.set(j.id, j);
    API.online = true;
  } catch (e) { API.online = false; mountView(daemonDownView(e)); paintStatus(); return; }
  try {
    const st = await API.status();
    if (Number.isFinite(st?.rss)) S.rss = st.rss;
    if (Array.isArray(st?.datasets)) S.datasets = st.datasets.map(normDataset);
  } catch { /* /api/status is optional; the jobs table stands on its own */ }
  if (!S.datasets.length) { try { S.datasets = await API.datasets(); } catch { /* fleet unavailable */ } }

  function paint() {
    const jobs = [...S.jobs.values()].sort((a, b) => (b.updated || 0) - (a.updated || 0));
    const act = jobs.filter((j) => j.state === 'running');
    const idxBytes = S.datasets.reduce((a, d) => a + (d.bytes || 0), 0);
    metrics.replaceChildren(
      el('div', { className: 'metric' }, el('b', { textContent: act.length ? act.reduce((a, j) => a + j.imgS, 0).toFixed(1) : '0' }), el('span', { textContent: 'images / second, right now' })),
      el('div', { className: 'metric' }, el('b', { textContent: n0(S.datasets.reduce((a, d) => a + d.count, 0)) }), el('span', { textContent: 'images searchable' })),
      el('div', { className: 'metric' }, el('b', { textContent: idxBytes ? bytes(idxBytes) : '—' }), el('span', { textContent: 'index on disk' })),
      el('div', { className: 'metric' }, el('b', { textContent: S.rss != null ? bytes(S.rss) : '—' }), el('span', { textContent: 'daemon RSS' })),
    );

    jobsWrap.replaceChildren(el('h2', { textContent: 'Index jobs' }));
    if (!jobs.length) {
      jobsWrap.append(el('p', { className: 'sub', style: 'margin-top:8px' }, 'No jobs have run yet on this daemon.'));
    } else {
      const tb = el('tbody');
      for (const j of jobs) {
        const bar = el('div', { className: 'bar-mini' },
          // ember means LIVE; a finished job's bar is neutral — colour carries state, not decoration
          el('i', { className: j.state === 'running' ? '' : 'idle', style: `width:${pct(j.done, j.total)}%` }),
          j.inFlight ? el('i', { className: 'ghost', style: `width:${pct(j.inFlight, j.total)}%` }) : null);
        tb.append(el('tr', null,
          el('td', null, el('a', { href: `#/d/${encodeURIComponent(j.dataset)}`, textContent: j.dataset, style: 'color:inherit' })),
          el('td', null, el('span', { className: `state state--${j.state}`, textContent: j.state })),
          el('td', { style: 'width:26%' }, bar),
          el('td', { className: 'n', textContent: `${n0(j.done)} / ${n0(j.total)}` }),
          el('td', { className: 'n', textContent: j.inFlight ? `+${n0(j.inFlight)}` : '' }),
          el('td', { className: 'n', textContent: j.imgS ? j.imgS.toFixed(1) : '—' }),
          el('td', { className: 'n', textContent: j.state === 'running' ? dur(j.etaS) : dur(j.elapsedS) }),
          el('td', { className: 'n', textContent: j.failed ? n0(j.failed) : '0' }),
        ));
      }
      jobsWrap.append(el('table', null,
        el('thead', null, el('tr', null,
          ...['dataset', 'state', 'progress'].map((h) => el('th', { textContent: h })),
          ...['done', 'in flight', 'img/s', 'eta / elapsed', 'skipped'].map((h) => el('th', { className: 'n', textContent: h })))),
        tb));
    }

    queries.replaceChildren();
    if (S.recent.length) {
      const tb = el('tbody');
      for (const r of S.recent) {
        tb.append(el('tr', null,
          el('td', { textContent: r.q }),
          el('td', { className: 'n', textContent: r.tookMs != null ? `${r.tookMs.toFixed(1)} ms` : '—' }),
          el('td', { className: 'n', textContent: `${r.rttMs.toFixed(0)} ms` }),
          el('td', { className: 'n', textContent: n0(r.hits) }),
          el('td', { className: 'n', textContent: ago(r.at / 1000) })));
      }
      queries.append(el('h2', { textContent: 'Recent queries', style: 'margin-top:32px' }),
        el('table', null, el('thead', null, el('tr', null,
          el('th', { textContent: 'query' }),
          ...['server', 'round-trip', 'hits', 'when'].map((h) => el('th', { className: 'n', textContent: h })))), tb));
    }

    const fails = jobs.flatMap((j) => (j.failures || []).map((f) => ({ ...f, dataset: j.dataset })));
    failures.replaceChildren();
    if (fails.length) {
      const tb = el('tbody');
      for (const f of fails.slice(0, 200)) {
        tb.append(el('tr', null,
          el('td', { className: 'mono', textContent: f.path }),
          el('td', { textContent: f.reason }),
          el('td', { textContent: f.dataset })));
      }
      failures.append(el('h2', { textContent: 'Skip ledger', style: 'margin-top:32px' }),
        el('p', { className: 'sub', style: 'margin:4px 0 12px' }, 'Every file the indexer refused, with the reason. A silent skip would be a bug.'),
        el('table', null, el('thead', null, el('tr', null,
          ...['path', 'reason', 'dataset'].map((h) => el('th', { textContent: h })))), tb));
    }
  }
  paint();
  paintStatus({ latency: '', hits: '', coverage: '' });
}

// ── router ─────────────────────────────────────────────────────────────────
let suppressRoute = false;
function go(hash) {
  if (location.hash === hash) return route();
  location.hash = hash;
}
function route() {
  const h = location.hash.replace(/^#/, '') || '/';
  const [path, qs] = h.split('?');
  const p = new URLSearchParams(qs || '');
  for (const a of document.querySelectorAll('.nav a')) a.removeAttribute('aria-current');
  if (path === '/' || path === '') {
    $('[data-nav="fleet"]').setAttribute('aria-current', 'page');
    $('#qCov').hidden = true;
    viewFleet();
  } else if (path === '/search') {
    viewSearch((p.get('q') || '').trim(), p.get('d') || '');
  } else if (path.startsWith('/d/')) {
    viewDataset(decodeURIComponent(path.slice(3)));
  } else if (path === '/jobs') {
    $('[data-nav="jobs"]').setAttribute('aria-current', 'page');
    viewJobs();
  } else {
    go('#/');
  }
}
window.addEventListener('hashchange', () => { if (suppressRoute) { suppressRoute = false; return; } route(); });

// ── search box wiring (B4 path: keydown → request → paint → mark) ──────────
const qInput = $('#q');
const openSelected = () => {
  const g = S.view?.grid;
  if (g && g.sel >= 0) { openDetail(g.current()); return true; }
  return false;
};
qInput.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { qInput.value = ''; go('#/'); return; }
  // arrows hand the grid the keyboard without losing the query; Enter opens what is selected
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); main.focus(); return; }
  if (e.key === 'Enter') { e.preventDefault(); openSelected(); return; }
  markKey();     // B4: t0 is the keystroke itself, before any work
});
qInput.addEventListener('input', () => {
  const q = qInput.value.trim();
  // No debounce on purpose: the daemon is local and B4 budgets keystroke→painted at 150 ms p95.
  // Superseded requests are aborted in viewSearch, so at most one is ever in flight.
  const hash = q ? `#/search?q=${encodeURIComponent(q)}` : '#/';
  history.replaceState(null, '', hash);
  route();
});
$('#searchForm').addEventListener('submit', (e) => { e.preventDefault(); openSelected(); });

// ── global keyboard ────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== qInput) {
    e.preventDefault(); qInput.focus(); qInput.select(); return;
  }
  if (e.key === 'Escape' && dlg.open) { dlg.close(); return; }
  if (dlg.open) return;
  if (e.target === qInput && !['ArrowDown', 'ArrowUp'].includes(e.key)) return;
  if (S.view?.key?.(e)) e.preventDefault();
});

// Read-only hook for the bench harness (B4/B14/B18): last queries with their honest
// server-vs-round-trip split, and the live grid geometry. No setters — the app owns its state.
window.__imgtag = {
  recent: () => S.recent.slice(),
  jobs: () => [...S.jobs.values()],
  grid: () => (S.view && S.view.grid ? { tiles: S.view.grid.nodes.size, total: S.view.grid.total } : null),
};

// ── boot ───────────────────────────────────────────────────────────────────
connectEvents();
route();
qInput.focus();      // search is never more than zero keystrokes away
paintStatus();
