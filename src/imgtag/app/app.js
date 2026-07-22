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
// ALL → SOME → ANY: the user's own spectrum (VISION-ADDENDA 12:12Z)
const tierOf = (t) => (t.m >= t.n ? 'all' : t.m <= 1 ? 'any' : 'some');
const tierLabel = (m, n) => (m >= n ? `all ${n} terms` : m <= 1 ? 'one term only' : `${m} of ${n} terms`);

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
      terms: normTerms(j.terms),
      calibration: j.calibration || null,   // "fitted" | "unfitted" — gates every confidence word
      servedBy: j.served_by || null,        // "tag-table" | "warm-tower" | "cold-load"
      towerLoadMs: Number.isFinite(j.text_tower_load_ms) ? j.text_tower_load_ms : null,
      // deliberately surfaced, not swallowed: the engine collapses duplicate ids but reports
      // how many it collapsed, so an indexer writing repeats stays visible downstream
      collapsedDuplicates: Number.isFinite(j.collapsed_duplicates) ? j.collapsed_duplicates : null,
      noMatch: j.no_match === true || (j.hits || []).length === 0,
      hits: (j.hits || []).map(normHit),
    };
  },
  async moderation({ source, dataset } = {}) {
    const p = new URLSearchParams();
    if (source) p.set('source', source);   // both "stored" and "current-scan" accepted by the daemon
    if (dataset) p.set('dataset', dataset);
    const qs = p.toString();
    return this.get(`/api/moderation${qs ? `?${qs}` : ''}`);
  },
  // every track's confidence for ONE image, on demand when the detail overlay opens
  // (VISION-ADDENDA 14:16Z). Returns null if the endpoint is not on this build — the caller
  // then falls back to whatever flags the hit already carries.
  async imageTracks(dataset, id, { signal } = {}) {
    try {
      const j = await this.get(`/api/image/${encodeURIComponent(dataset)}/${encodeURIComponent(id)}/tracks`, { signal });
      return (j.tracks || j || []).map(normTrackScore);
    } catch (e) {
      if (e.status === 404 || e.status === 501) return null;
      throw e;
    }
  },
  async track(category, { tier, k = 300, signal } = {}) {
    const p = new URLSearchParams({ track: category, k: String(k) });
    if (tier) p.set('tier', tier);
    const t0 = performance.now();
    const j = await this.get(`/api/search?${p}`, { signal });
    return {
      hits: (j.hits || []).map(normHit),
      tookMs: Number.isFinite(j.tookMs) ? j.tookMs : null,
      rttMs: performance.now() - t0,
      enforcementReady: j.enforcement_ready ?? null,
      // the track's own state, now the SAME value /api/moderation reports for this category
      calibration: j.track_calibration || j.calibration || null,
      // what the track's spec claims for itself — kept apart, because a spec claiming
      // "proxy-fitted" while the engine refuses to gate on it is exactly the gap to show
      specCalibration: j.spec_calibration || null,
      coverage: j.coverage || null,
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
// Top-level `terms` is a BARE ARRAY of the parsed terms, absent when n<2 (b-daemon, pinned).
// A quoted span or a greedy vocabulary match is ONE element — one element, one chip.
function normTerms(t) {
  return Array.isArray(t) && t.length > 1 ? { list: t, n: t.length } : null;
}
// Per-hit coverage rides inside `why.terms` and is absent for single-term queries.
// `via` maps an internal tag to the user words that reached it through a hypernym/synonym —
// the chips always print the user's own word, never the internal tag.
function normHitTerms(why) {
  const t = why && why.terms;
  if (!t || !(t.n > 1)) return null;
  return {
    matched: t.matched || [], missed: t.missed || [],
    m: t.m ?? (t.matched || []).length, n: t.n,
    meanP: Number.isFinite(t.mean_p) ? t.mean_p : null,
    via: t.via || null,
  };
}
// ADR-14 moderation flags. TWO SOURCES that must never be added together or shown as one
// number: `flags` is the CURRENT SCAN (computed now from today's prompt sets) and
// `flags_stored` is what b-engine wrote INTO THE INDEX when the batch ran. They may
// legitimately disagree — that is the detector version changing, not a bug — so the label
// travels with the number everywhere in this app.
function normFlags(h) {
  const list = Array.isArray(h.flags) ? h.flags
    : h.category ? [{ category: h.category, tier: h.tier, p: h.p, kind: h.kind }]   // track-browse row
      : null;
  // kind is "moderation" | "content"; when absent, a content-only tier (match) still reads
  // as content so a descriptive label never renders like a safety flag
  return list ? list.map((f) => ({
    category: f.category, tier: f.tier || 'review', p: f.p ?? null,
    kind: f.kind || (f.tier === 'match' ? 'content' : 'moderation'),
  })) : null;
}
// one track's confidence for one image. `scored:false` (or missing p) = head not warm / no
// sidecar → "score pending", NOT a real zero. tier null = scored but below every threshold.
function normTrackScore(t) {
  const scored = t.scored !== false && Number.isFinite(t.p);
  return {
    category: t.category, label: t.label || t.category,
    kind: t.kind || (t.tier === 'match' ? 'content' : 'moderation'),
    p: scored ? t.p : null,
    tier: t.tier || null,
    scored,
    calibration: t.calibration || null,
    specCalibration: t.spec_calibration || null,
    labelValue: t.label_value || null,   // the argmax concept: "handgun", "tennis"
    via: t.via || null,
  };
}
function normStoredFlags(f) {
  if (Array.isArray(f)) return f.map((x) => ({ category: x.category, tier: x.tier || null, p: x.p ?? null }));
  // legacy {category: p} map — no tier was recorded, so this claims none
  if (f && typeof f === 'object') return Object.entries(f).map(([category, p]) => ({ category, tier: null, p }));
  return null;
}
function normHit(h) {
  return {
    terms: normHitTerms(h.why),
    flags: normFlags(h),
    flagsStored: normStoredFlags(h.flags_stored),
    id: h.image_id ?? h.id ?? '',
    path: h.path ?? '',
    dataset: h.dataset ?? h.dataset_slug ?? '',
    score: Number.isFinite(h.score) ? h.score : null,
    p: Number.isFinite(h.p) ? h.p : null,
    exists: h.exists !== false,
    paths: Array.isArray(h.paths) && h.paths.length > 1 ? h.paths : null,  // same bytes, many files
    // same content hash living in OTHER datasets, folded to one hit (B18-safe cross-dataset collapse)
    alsoIn: Array.isArray(h.also_in) && h.also_in.length ? h.also_in : null,
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
  models: null,             // models_loaded from /api/status; [] at idle is CORRECT (ADR-5)
  calibration: null,        // "measured-default" | "fitted" — gates the confidence wording
  textTower: null,
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
const MAX_NEW_TILES_PER_FRAME = 6;   // decode budget: see the note in render()
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
    this.groups = null;
    this.items = items; this.total = total;
    this.clearNodes();
    this.measure();
  }
  /** Tier bands (ALL → SOME → ANY). Each band opens with a label cell and is padded to a
   *  whole row, so tiers are perceivable while scrolling WITHOUT breaking the uniform-row
   *  math the virtualizer depends on. Re-laid out whenever the column count changes. */
  setGroups(groups) {
    this.groups = groups;
    this.items = []; this.total = 0;
    this.clearNodes();
    this.laidOutFor = -1;
    this.measure();
  }
  clearNodes() {
    for (const [, n] of this.nodes) n.remove();
    this.nodes.clear();
    this.sel = -1;
  }
  layout() {
    if (!this.groups || this.laidOutFor === this.cols) return;
    if (this.laidOutFor !== -1) this.clearNodes();   // indices shift when columns change
    this.laidOutFor = this.cols;
    const items = [];
    this.bandRow = new Set();          // indices sharing a row with a band label
    for (const g of this.groups) {
      const start = items.length;
      for (let c = 0; c < this.cols; c++) this.bandRow.add(start + c);
      items.push({ kind: 'band', ...g });
      items.push(...g.hits);
      while (items.length % this.cols !== 0) items.push(null);   // pad: next band starts a row
    }
    this.items = items; this.total = items.length;
  }
  isImage(i) { const it = this.items[i]; return !!it && it.kind !== 'band'; }
  measure() {
    const w = this.mount.clientWidth;
    if (!w) return;
    this.cols = Math.max(1, Math.floor((w + this.gap) / (this.min + this.gap)));
    this.layout();
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
    // Creating every newly-visible tile in one frame makes the browser decode a burst of
    // thumbnails in a single task — measured at ~2.9s of long task over a fast scroll, which
    // blows B14. Cap the new nodes per frame and finish the rest on the next one: the work is
    // identical, spread across frames instead of blocking one.
    let budget = MAX_NEW_TILES_PER_FRAME;
    let deferred = false;
    for (let i = from; i <= to; i++) {
      const item = this.items[i];
      if (!item) { (missing ||= []).push(i); continue; }
      let node = this.nodes.get(i);
      if (!node) {
        if (budget <= 0) { deferred = true; continue; }
        budget--;
        node = item.kind === 'band' ? this.band(item) : this.tile(item, i);
        this.nodes.set(i, node); this.win.append(node);
      }
      const r = Math.floor(i / this.cols), c = i % this.cols;
      node.style.width = `${this.tw}px`;
      node.style.height = `${this.th}px`;
      node.style.transform = `translate(${c * (this.tw + this.gap)}px, ${r * rowH}px)`;
    }
    if (deferred && !this._raf) {
      this._raf = requestAnimationFrame(() => { this._raf = 0; this.render(); });
    }
    if (missing && this.onFill) this.onFill(missing[0], missing[missing.length - 1] - missing[0] + 1);
  }
  band(g) {
    return el('div', { className: `band band--${g.tier}`, role: 'presentation' },
      el('div', { className: 'band__m', textContent: `${g.m}/${g.n}` }),
      el('div', { className: 'band__label', textContent: g.label }),
      el('div', { className: 'band__sub', textContent: `${nf.format(g.hits.length)} ${g.hits.length === 1 ? 'image' : 'images'}` }));
  }
  tile(item, i) {
    const t = el('button', { className: 'tile', type: 'button', tabIndex: -1, role: 'option' });
    if (this.bandRow?.has(i)) t.classList.add('tile--band-row');   // carries the band's rule across
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
    if (item.terms) t.dataset.tier = tierOf(item.terms);   // ALL reads primary, ANY reads weakest
    if (!this.hideDataset) {
      const l1 = el('div', { className: 'tile__l1' },
        el('span', { className: 'tile__ds', textContent: item.dataset || '—' }),
        el('span', { className: 'tile__id', textContent: shortId }));
      if (item.terms) {
        // "scored present", never "verified present": measured against COCO ground truth the
        // ALL tier is recall-heavy and precision-poor (92% / 36%) while τ is unfitted
        l1.append(el('span', {
          className: 'tile__cov', textContent: `${item.terms.m}/${item.terms.n}`,
          title: S.calibration === 'fitted'
            ? tierLabel(item.terms.m, item.terms.n)
            : `${tierLabel(item.terms.m, item.terms.n)} — scored present, thresholds unfitted`,
        }));
      }
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
    // one photo indexed into more than one dataset folded to a single hit — name the others
    if (item.alsoIn) {
      cap.append(el('div', { className: 'tile__also' },
        el('span', { textContent: `also in ${item.alsoIn.map((a) => a.dataset).join(', ')}`,
          title: item.alsoIn.map((a) => `${a.dataset} · ${a.path}`).join('\n') })));
    }
    // flag chips: violation and review are visually distinct because they mean different
    // things — review is "a human should look", never "this is bad"
    if (item.flags || item.flagsStored) {
      const row = el('div', { className: 'tile__flags' });
      for (const f of (item.flags || []).slice(0, 2)) {
        const content = f.kind === 'content';
        row.append(el('span', {
          className: `flag flag--${content ? 'content' : f.tier}`,
          textContent: `${f.category}${f.p != null ? ` ${f.p.toFixed(2)}` : ''}`,
          title: content ? `content label · current scan`
            : `${f.tier === 'alert' ? 'safety alert' : f.tier === 'violation' ? 'matched the violation prompt set' : 'needs a human look'} · current scan`,
        }));
      }
      for (const f of (item.flagsStored || []).slice(0, 2)) {
        row.append(el('span', {
          className: 'flag flag--stored', textContent: f.category,
          title: 'flagged at indexing (stored in the index, not this scan)',
        }));
      }
      cap.append(row);
    }
    if (item.terms) {
      // the terms themselves ARE the explanation: matched plain, missed struck through
      const row = el('div', { className: 'tile__terms' });
      const via = item.terms.via || {};
      for (const term of item.terms.matched) {
        const through = Object.keys(via).find((tag) => (via[tag] || []).includes(term));
        const chip = el('span', { className: 'term', textContent: term });
        if (through) { chip.classList.add('term--via'); chip.title = `matched through “${through}”`; }
        row.append(chip);
      }
      for (const term of item.terms.missed) row.append(el('span', { className: 'term term--missed', textContent: term }));
      cap.append(row);
    } else {
      const why = whyText(item.why);
      if (why) cap.append(el('div', { className: 'tile__why', textContent: why }));
    }
    t.append(cap);
    t.addEventListener('click', () => { this.select(i); this.onOpen?.(this.items[i]); });
    return t;
  }
  select(i) {
    if (i < 0 || i >= this.total || !this.isImage(i)) return;
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
    let next = this.sel < 0 ? 0 : this.sel + d;
    next = Math.max(0, Math.min(this.total - 1, next));
    // band labels and row padding are not selectable — keep walking the way we were headed
    const step = d >= 0 ? 1 : -1;
    while (next >= 0 && next < this.total && !this.isImage(next)) next += step;
    if (next < 0 || next >= this.total) {          // fell off the end: take the nearest image back
      next = this.sel < 0 ? 0 : this.sel;
      while (next >= 0 && next < this.total && !this.isImage(next)) next -= step;
    }
    this.select(next);
  }
  current() { return this.items[this.sel] || null; }
}

// ── detail overlay (native <dialog>) ───────────────────────────────────────
const dlg = $('#detail');
let detailSeq = 0;   // guards the async track fetch against a fast image-switch

// The ranked track-confidence panel (VISION-ADDENDA 14:16Z): ALL tracks for one image,
// ranked p DESC, coloured by tier (alert > violation > review > match/content > none),
// bar width = p. Unfitted p is labelled, never dressed as authoritative; a head that is
// not warm shows "score pending", never a fake 0.
function trackPanel(tracks, { partial } = {}) {
  const panel = el('div', { className: 'tracks' });
  // partial = the full set is still loading. The FIRST open per dataset derives scores live
  // (b-daemon: ~7s cold, then ~8ms cached), so this needs a real in-progress affordance, not
  // just static text — a 7s wait under a plain label reads as frozen.
  panel.append(el('div', { className: 'tracks__head' },
    el('h3', { textContent: 'Track confidences' }),
    partial ? el('span', { className: 'spin', role: 'progressbar', 'aria-label': 'scoring all tracks' }) : null,
    partial ? el('span', { className: 'sub', textContent: 'scoring all tracks — first open scores the whole dataset, then it’s instant' }) : null));
  if (!tracks.length) {
    panel.append(el('p', { className: 'sub', textContent: 'No tracks are defined.' }));
    return panel;
  }
  // a real fired tier — NOT "none" (below threshold) and NOT null (not scored)
  const fired = (t) => t.tier && t.tier !== 'none';
  // Rank WITHIN kind, never across (b-daemon: moderation p is a margin scale, content p is
  // cosine — cross-kind comparison is a lie). Moderation group first (it answers "is this a
  // problem"), content second; inside each: scored before pending, fired-severity, then p DESC.
  const kindOrder = (t) => (t.kind === 'content' ? 1 : 0);
  const ranked = tracks.slice().sort((a, b) =>
    (kindOrder(a) - kindOrder(b))
    || (b.scored - a.scored)
    || (tierRank(a.tier) - tierRank(b.tier))
    || ((b.p ?? -1) - (a.p ?? -1)));
  const top = ranked.find(fired);   // highlight the strongest FIRING track (never a "none")
  for (const t of ranked) {
    const tier = t.kind === 'content' ? 'content' : (fired(t) ? t.tier : 'none');
    const row = el('div', { className: `trk trk--${tier}${t === top ? ' trk--top' : ''}` });
    // the concrete matched concept (handgun / tennis) rides alongside the track name
    row.append(el('span', { className: 'trk__name' },
      el('span', { textContent: t.label }),
      t.labelValue ? el('em', { className: 'trk__val', textContent: t.labelValue }) : null));
    const bar = el('span', { className: 'trk__bar' });
    if (t.scored) bar.append(el('i', { style: `width:${Math.round(Math.max(0, Math.min(1, t.p)) * 100)}%` }));
    row.append(bar);
    row.append(el('span', { className: 'trk__p', textContent: t.scored ? t.p.toFixed(2) : 'pending' }));
    // tier / maturity annotation — honest about unfitted thresholds, quiet about "none"
    const notes = [];
    if (fired(t)) notes.push(TIER_LABEL[t.tier] || t.tier);
    else if (t.scored) notes.push('below threshold');
    else notes.push('not scored yet');
    if (t.scored && t.calibration && t.calibration !== 'fitted') notes.push(t.calibration);
    row.append(el('span', { className: 'trk__note', textContent: notes.join(' · ') }));
    row.title = t.scored
      ? `${t.label}: p=${t.p.toFixed(3)} ${fired(t) ? `(${t.tier})` : '— below every threshold'}`
        + `${t.calibration && t.calibration !== 'fitted' ? ` · ${t.calibration}, not authoritative` : ''}`
      : `${t.label}: score pending — the head for this track is not warm`;
    panel.append(row);
  }
  const anyUnfitted = ranked.some((t) => t.scored && t.calibration && t.calibration !== 'fitted');
  if (anyUnfitted) {
    panel.append(el('p', { className: 'hint hint--caveat', style: 'margin:8px 0 0' },
      'Scores are unfitted: a high confidence means the image resembled that track’s prompt set, not a verified finding.'));
  }
  return panel;
}
// provisional tracks built from the flags a hit already carries, so the panel is never empty
// while the full set loads (or if the endpoint is absent on this build)
function tracksFromHit(item) {
  const out = [];
  for (const f of (item.flags || [])) {
    out.push({ category: f.category, label: f.category, kind: f.kind,
      p: f.p, tier: f.tier, scored: f.p != null, calibration: S.calibration, via: null });
  }
  return out;
}

function openDetail(item) {
  if (!item) return;
  const seq = ++detailSeq;
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
  if (item.p != null) {
    // never call it calibrated while the engine is still on its measured default
    side.append(kv(S.calibration === 'fitted' ? 'calibrated p' : 'p (uncalibrated default)', item.p.toFixed(3)));
  }
  if (item.terms) {
    side.append(kv(`term coverage — ${tierLabel(item.terms.m, item.terms.n)}`,
      [item.terms.matched.join(', ') || '—',
        item.terms.missed.length ? `missed: ${item.terms.missed.join(', ')}` : null].filter(Boolean).join(' · ')));
    for (const [tag, words] of Object.entries(item.terms.via || {})) {
      side.append(kv('matched through', `${words.join(', ')} → ${tag}`));
    }
  }
  if (item.why) {
    side.append(kv('why it matched', item.why.tag ? `${item.why.path}: ${item.why.tag}` : String(item.why.path || '—')));
  }
  if (item.paths) {
    side.append(kv(`also on disk at ${item.paths.length - 1} other path${item.paths.length > 2 ? 's' : ''}`,
      item.paths.slice(1).join('\n')));
  }
  if (item.alsoIn) {
    // same content hash, different dataset — provenance genuinely differs, so it is listed
    side.append(kv(`also indexed in ${item.alsoIn.length} other dataset${item.alsoIn.length > 1 ? 's' : ''}`,
      item.alsoIn.map((a) => `${a.dataset} · ${a.path}`).join('\n')));
  }
  if (item.exists === false) side.append(kv('status', 'indexed, but the file is no longer on disk'));

  // ranked track-confidence panel — provisional from the hit's flags now, full set on load
  const seed = tracksFromHit(item);
  const tracksSlot = trackPanel(seed, { partial: true });
  side.append(tracksSlot);
  API.imageTracks(item.dataset, item.id).then((tracks) => {
    if (seq !== detailSeq) return;                 // user moved on — don't paint stale
    if (tracks && tracks.length) tracksSlot.replaceWith(trackPanel(tracks));
    else tracksSlot.replaceWith(trackPanel(seed));  // endpoint absent: keep the provisional, drop "loading"
  }).catch(() => { if (seq === detailSeq) tracksSlot.replaceWith(trackPanel(seed)); });

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

// RSS and resident-model state come from /api/status — the SSE stream stays job-only so it
// costs nothing when nothing is indexing (b-daemon's design; do not ask events for rss).
async function refreshStatus() {
  try {
    const st = await API.status();
    if (Number.isFinite(st?.rss)) S.rss = st.rss;
    if (Array.isArray(st?.models_loaded)) S.models = st.models_loaded;
    if (st?.text_tower) S.textTower = st.text_tower;
    // text_ttl_s is null when no timer is armed — "no timer" and "evict now" are not the same
    S.eviction = st?.eviction_policy || (st?.text_ttl_s ? `text tower evicted after ${dur(st.text_ttl_s)} idle` : null);
    if (Array.isArray(st?.datasets) && st.datasets.length) S.datasets = st.datasets.map(normDataset);
    paintStatus();
  } catch { /* older daemons have no /api/status; every tile fed by it reads "—" */ }
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

// Group hits into the ALL → SOME → ANY spectrum. The engine guarantees the bands arrive
// CONTIGUOUS (m DESC → mean_p DESC → p DESC → image_id, byte-identical across runs per B18e),
// so this walks the list in the engine's order and never re-sorts — band boundaries cannot
// jitter between renders. Null means "render a plain grid": one term, one tier, or coverage
// missing from any hit.
function groupByCoverage(res) {
  if (!res.terms || !res.hits.length || res.hits.some((h) => !h.terms)) return null;
  const groups = [];
  for (const h of res.hits) {
    const last = groups[groups.length - 1];
    if (last && last.m === h.terms.m) last.hits.push(h);
    else groups.push({ m: h.terms.m, n: res.terms.n, hits: [h] });
  }
  if (groups.length < 2) return null;
  for (const g of groups) {
    g.tier = tierOf(g);
    g.label = tierLabel(g.m, g.n);
  }
  return groups;
}

// The syntax is taught until it is used, then it gets out of the way for good.
const SYNTAX_KEY = 'imgtag:multi-term-known';
const syntaxKnown = () => { try { return !!localStorage.getItem(SYNTAX_KEY); } catch { return false; } };
const markSyntaxKnown = () => { try { localStorage.setItem(SYNTAX_KEY, '1'); } catch { /* private mode */ } };
function syntaxHint() {
  if (syntaxKnown()) return null;
  return el('p', { className: 'hint' },
    'Space adds a tag — ', el('code', { className: 'k', textContent: 'red car night' }),
    ' puts images matching all three first, then two, then one. Quote an exact phrase: ',
    el('code', { className: 'k', textContent: '"night city"' }), '.');
}

// (2) search results
let searchAbort = null, searchSeq = 0;
async function viewSearch(q, dataset) {
  const seq = ++searchSeq;   // only the newest keystroke may paint (responses can land out of order)
  // Never clobber what the user is typing: the hash carries a TRIMMED query, so a blind
  // assignment would delete the trailing space the moment it is typed — which made multi-term
  // search impossible to type at all. Only sync when the field genuinely disagrees.
  if (qInput.value.trim() !== q) qInput.value = q;
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
      el('p', { textContent: 'Search is semantic, not keyword: “vehicle” finds the cars and the motorcycles too. Results stream from whatever is indexed right now, including datasets still being processed.' }),
      el('p', { className: 'hint' },
        'Space adds a tag — ', el('code', { className: 'k', textContent: 'red car night' }),
        ' puts images matching all three first, then two, then one. Quote an exact phrase: ',
        el('code', { className: 'k', textContent: '"night city"' }), '.')));
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
  if (typeof res.calibration === 'string') S.calibration = res.calibration;
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
    if (r.collapsedDuplicates) {
      // the engine folded repeats into one image each — say so rather than let the count
      // silently disagree with k
      bannerSlot.append(el('p', { className: 'hint hint--caveat' },
        el('b', { textContent: n0(r.collapsedDuplicates) }),
        ' duplicate rows were collapsed into single images. Identity is the file’s content '
        + 'hash, so the same bytes indexed more than once are one result — the count is '
        + 'shown because repeated rows mean something upstream wrote them twice.'));
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
  if (res.terms) markSyntaxKnown();          // used it once — stop teaching it
  else { const h = syntaxHint(); if (h) filterSlot.append(h); }
  if (res.terms && S.calibration !== 'fitted') {
    filterSlot.append(el('p', { className: 'hint hint--caveat' },
      'Coverage is unfitted: a ', el('b', { textContent: `${res.terms.n}/${res.terms.n}` }),
      ' means every term scored present, not that it was verified present. Measured against '
      + 'ground truth this tier finds nearly everything it should and over-claims about two '
      + 'thirds of the time, until the calibration fit lands.'));
  }
  const allTier = res.terms ? res.hits.filter((h) => h.terms && h.terms.m >= h.terms.n).length : 0;
  // free-text is recall-first + uncalibrated: it can return weak matches (even for gibberish)
  // rather than a false empty, so the ranking is flagged uncalibrated — the scores are real
  // distances, the CONFIDENCE is not yet calibrated. Never let a weak result read as authoritative.
  const uncal = res.calibration && res.calibration !== 'fitted';
  $('#resSum').textContent = res.hits.length
    ? [
      `${n0(res.hits.length)} ${res.hits.length === 1 ? 'hit' : 'hits'}`,
      res.terms ? `${n0(allTier)} match all ${res.terms.n} terms` : null,
      uncal ? 'uncalibrated ranking' : null,
      res.tookMs != null ? `${res.tookMs.toFixed(1)} ms server` : null,
    ].filter(Boolean).join(' · ')
    : '';
  $('#resSum').title = uncal
    ? 'Confidence thresholds are not fitted yet: results are ranked by raw similarity and may include weak matches — the order is honest, the certainty is not.'
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
    // one extra caption line when any hit carries moderation flags — rows stay uniform
    // caption height scales to the tallest caption in the set so rows stay uniform: base 3
    // lines, +1 for a flag row, +1 for an also-in line
    const extra = (res.hits.some((h) => h.flags || h.flagsStored) ? 18 : 0)
      + (res.hits.some((h) => h.alsoIn) ? 16 : 0);
    grid = new VirtualGrid(root, gridMount, { onOpen: openDetail, cap: 60 + extra });
    ctl.grid = grid;
    const groups = groupByCoverage(res);
    if (groups) grid.setGroups(groups); else grid.setItems(res.hits);
    grid.move(0);   // land on the first real image, skipping any band label
  }

  // served_by makes a slow first query legible: a cold load is a one-time model load, not the
  // engine being slow — say which, rather than letting the number look like the steady state
  const servedNote = res.servedBy === 'cold-load'
    ? ` · cold model load${res.towerLoadMs ? ` ${Math.round(res.towerLoadMs)} ms` : ''}`
    : res.servedBy ? ` · ${res.servedBy}` : '';
  paintStatus({
    latency: res.tookMs != null
      ? `<b>${res.tookMs.toFixed(1)} ms</b> server · ${res.rttMs.toFixed(0)} ms round-trip${servedNote}`
      : `${res.rttMs.toFixed(0)} ms round-trip${servedNote}`,
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

// (5) moderation — ADR-14. Two sources, never one number; tiers never merged.
// Categories and tiers are read from the payload, never hard-coded: the safety track
// (people lying down + injury/destruction escalation, VISION-ADDENDA 13:20Z) adds both a
// new category and an `alert` tier, and this view must render them the day they ship.
const CATS = ['nudity', 'weapons', 'drugs'];
const TIER_ORDER = ['alert', 'violation', 'review'];   // most severe first
const tierRank = (t) => { const i = TIER_ORDER.indexOf(t); return i === -1 ? 99 : i; };
const TIER_LABEL = { alert: 'alert', violation: 'violation', review: 'needs review' };
async function viewModeration() {
  const root = el('div', { className: 'view' });
  const pad = el('div', { className: 'pad' });
  root.append(pad);
  const head = el('div', { className: 'head' }, el('h1', { textContent: 'Moderation' }),
    el('span', { className: 'sub', id: 'modSum', textContent: 'scanning…' }));
  const note = el('div');
  const body = el('div');
  pad.append(head, note, body);
  mountView(root);

  // Default to the STORED flags — what the indexer recorded when each batch ran, the numbers
  // the user's own batch summaries referenced. The current scan is one click away and is
  // always labelled; the two are never averaged, summed, or shown under one heading.
  let source = 'stored';
  async function load() {
    let m;
    try { m = await API.moderation({ source }); API.online = true; } catch (e) { API.online = false; mountView(daemonDownView(e)); paintStatus(); return; }
    source = m.source || source;   // trust what the daemon says it gave us, not what we asked
    const labels = m.datasets?.[0]?.labels || {};
    // the per-category calibration map is authoritative; track pages read it from here
    if (m.calibration && typeof m.calibration === 'object') S.modCalibration = m.calibration;
    if (m.totals) S.modCounts = m.totals;      // also tells track pages which tiers exist
    // whatever tiers the engine reports, most severe first — not a fixed pair
    const tiers = [...new Set(Object.values(m.totals || {}).flatMap((t) => Object.keys(t)))]
      .sort((a, b) => tierRank(a) - tierRank(b));
    const cats = Object.keys(m.totals || {}).length ? Object.keys(m.totals) : CATS;
    $('#modSum').textContent = `${n0(m.indexed)} images scanned across ${m.datasets?.length || 0} datasets`;

    const stored = source === 'stored';
    note.replaceChildren(
      el('div', { className: 'banner' },
        el('span', null, 'Showing '),
        el('b', { textContent: stored ? 'flagged at indexing' : 'the current scan' }),
        el('span', {
          textContent: stored
            ? ' — what the indexer recorded into each image when its batch ran. These are the numbers your batch summaries reported.'
            : ' — recomputed just now from today’s prompt sets. It can differ from what was stored: that means the detector changed, not that one is wrong.',
        }),
        el('button', {
          className: 'btn', type: 'button',
          textContent: stored ? 'Re-scan now' : 'Show what was stored',
          onclick: (e) => {
            e.currentTarget.disabled = true;
            e.currentTarget.textContent = stored ? 'scanning…' : 'loading…';
            source = stored ? 'current-scan' : 'stored';
            load();
          },
        })),
      el('p', { className: 'hint hint--caveat' },
        'Nothing here is a confirmed finding: a violation row means the image resembled the '
        + 'violation prompt set more than the review set, and sampling by eye found both correct '
        + 'and incorrect calls. Each category carries its own threshold state below — they are '
        + 'not all at the same maturity — and a category is only safe to act on automatically '
        + 'once it reads enforcement-ready.'),
      // the other source, named so the two numbers are never conflated
      el('p', { className: 'hint' },
        'The other source — ', el('b', { textContent: stored ? 'the current scan' : 'flagged at indexing' }),
        stored
          ? ' — recomputes every category now from today’s prompt sets, and can legitimately disagree with what was stored.'
          : ' — is what each image carries inside the index, and shows on individual results as a stored chip.',
        ' The two are never added together.'),
    );

    // An empty stored map means "nothing was RECORDED at index time" — the batch ran without
    // --moderation. It does NOT mean "nothing was found", and a table of zeros would say
    // exactly the wrong thing, so the zeros are refused and the reason is named instead.
    if (stored && !tiers.length) {
      body.replaceChildren(el('div', { className: 'empty' },
        el('h2', { textContent: 'Nothing was recorded at indexing time' }),
        el('p', { textContent: 'These datasets were indexed without moderation enabled, so there are no stored flags to total. This is an absence of records, not a finding of nothing — the images have never been checked at index time.' }),
        el('p', { className: 'sub' }, 'Scan the existing index now:'),
        el('p', null, el('button', {
          className: 'btn', type: 'button', style: 'margin:0',
          textContent: 'Run the current scan',
          onclick: () => { source = 'current-scan'; load(); },
        })),
        el('p', { className: 'sub', style: 'margin-top:16px' }, 'Or record flags for future batches:'),
        el('p', null, el('code', { className: 'k', textContent: 'imgtag index <path> --dataset <name> --moderation' })),
      ));
      paintStatus({ latency: '', hits: '', coverage: 'source: stored · no flags recorded at index time' });
      return;
    }
    const cell = (cat, tier, count) => {
      if (!count) return el('td', { className: 'n', textContent: '0' });
      const a = el('a', { href: `#/track/${encodeURIComponent(cat)}?tier=${tier}`, textContent: n0(count) });
      return el('td', { className: 'n' }, a);
    };
    const tb = el('tbody');
    for (const cat of cats) {
      const t = (m.totals || {})[cat] || { violation: 0, review: 0 };
      const ready = (m.enforcement_ready || {})[cat];
      // calibration is PER CATEGORY (nudity may be unfitted while drugs is proxy-fitted),
      // so it is stated per row — a single page-level label would be false for some row
      const calib = typeof m.calibration === 'string' ? m.calibration : (m.calibration || {})[cat];
      tb.append(el('tr', null,
        el('td', null, el('a', { href: `#/track/${encodeURIComponent(cat)}`, textContent: labels[cat] || cat, style: 'color:inherit' })),
        ...tiers.map((tier) => cell(cat, tier, t[tier])),
        el('td', null,
          el('span', {
            className: `chip ${ready ? '' : 'chip--warn'}`,
            textContent: ready ? 'enforcement-ready' : 'not enforcement-ready',
          }),
          calib ? el('span', { className: 'chip', style: 'margin-left:8px', textContent: `τ ${calib}` }) : null)));
    }
    body.replaceChildren(
      el('h2', { textContent: 'Across every dataset' }),
      el('table', { style: 'margin-top:12px' },
        el('thead', null, el('tr', null,
          el('th', { textContent: 'category' }),
          ...tiers.map((tier) => el('th', { className: 'n', textContent: TIER_LABEL[tier] || tier })),
          el('th', { textContent: '' }))),
        tb));

    const dtb = el('tbody');
    for (const d of m.datasets || []) {
      for (const cat of cats) {
        const c = (d.counts || {})[cat] || {};
        if (!tiers.some((tier) => c[tier])) continue;
        dtb.append(el('tr', null,
          el('td', null, el('a', { href: `#/d/${encodeURIComponent(d.dataset)}`, textContent: d.dataset, style: 'color:inherit' })),
          el('td', { textContent: labels[cat] || cat }),
          ...tiers.map((tier) => el('td', { className: 'n', textContent: n0(c[tier] || 0) })),
          el('td', { className: 'n', textContent: n0(d.indexed) })));
      }
    }
    if (dtb.children.length) {
      body.append(el('h2', { textContent: 'Per dataset', style: 'margin-top:32px' }),
        el('table', { style: 'margin-top:12px' },
          el('thead', null, el('tr', null,
            el('th', { textContent: 'dataset' }), el('th', { textContent: 'category' }),
            ...tiers.map((tier) => el('th', { className: 'n', textContent: TIER_LABEL[tier] || tier })),
            el('th', { className: 'n', textContent: 'scanned' }))),
          dtb));
    }
    // content tracks (kind:"content", e.g. a "sports" match) live in their OWN map and are
    // never mixed into moderation totals — a content match must not read as a safety concern
    const cc = m.content_counts || {};
    const ccats = Object.keys(cc);
    if (ccats.length) {
      const ctiers = [...new Set(Object.values(cc).flatMap((t) => Object.keys(t)))].sort((a, b) => tierRank(a) - tierRank(b));
      const ctb = el('tbody');
      for (const cat of ccats) {
        const t = cc[cat] || {};
        ctb.append(el('tr', null,
          el('td', null, el('a', { href: `#/track/${encodeURIComponent(cat)}`, textContent: labels[cat] || cat, style: 'color:inherit' })),
          ...ctiers.map((tier) => cell(cat, tier, t[tier]))));
      }
      body.append(
        el('h2', { textContent: 'Content tracks', style: 'margin-top:32px' }),
        el('p', { className: 'hint', style: 'margin:4px 0 12px' },
          'Descriptive labels, not moderation — counted separately and never added to the totals above.'),
        el('table', null,
          el('thead', null, el('tr', null,
            el('th', { textContent: 'category' }),
            ...ctiers.map((tier) => el('th', { className: 'n', textContent: TIER_LABEL[tier] || tier })))),
          ctb));
    }
    const thr = m.datasets?.[0]?.threshold;
    paintStatus({
      latency: Number.isFinite(thr) ? `threshold ${thr.toFixed(3)}` : '',
      hits: '',
      coverage: `source: ${m.source || 'current-scan'} · calibration stated per category`,
    });
  }
  load();
}

// (6) one moderation track — browse what fired, tier by tier
async function viewTrack(category, tier) {
  const root = el('div', { className: 'view' });
  const pad = el('div', { className: 'pad' });
  root.append(pad);
  pad.append(el('div', { className: 'head' },
    el('h1', { textContent: category }),
    el('span', { className: 'sub', id: 'trackSum', textContent: 'loading…' })));
  const chips = el('div', { className: 'filters' });
  // tiers come from the last moderation scan when we have it, so a new tier (alert) appears
  // here the moment the engine reports it — nothing about this list is hard-coded
  const known = Object.keys((S.modCounts || {})[category] || {}).sort((a, b) => tierRank(a) - tierRank(b));
  const tierChoices = [['', 'most severe first'], ...(known.length ? known : ['violation', 'review'])
    .map((t) => [t, `${TIER_LABEL[t] || t} only`])];
  for (const [t, label] of tierChoices) {
    const b = el('button', { type: 'button', textContent: label });
    b.setAttribute('aria-pressed', String((tier || '') === t));
    b.addEventListener('click', () => go(`#/track/${encodeURIComponent(category)}${t ? `?tier=${t}` : ''}`));
    chips.append(b);
  }
  const caveat = el('p', { className: 'hint hint--caveat' },
    'Not a confirmed finding. A violation row resembled the violation prompt set more than the '
    + 'review set; a review row is one a human should look at. Thresholds are unfitted.');
  const gridMount = el('div', { className: 'grid' });
  pad.append(chips, caveat, gridMount);

  let grid = null;
  const ctl = {
    destroy: () => grid?.destroy(),
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

  let res;
  try { res = await API.track(category, { tier }); API.online = true; } catch (e) {
    API.online = false; mountView(daemonDownView(e)); paintStatus(); return;
  }
  if (typeof res.calibration === 'string') S.calibration = res.calibration;   // per-category on tracks
  $('#trackSum').textContent = res.hits.length
    ? `${n0(res.hits.length)} flagged · ${res.tookMs != null ? `${res.tookMs.toFixed(1)} ms server` : ''}`
    : '';
  // This track's own maturity, straight from the track response — it now carries the same
  // per-category values /api/moderation reports, so the two screens cannot disagree.
  const pick = (v) => (v && typeof v === 'object' ? v[category] : v);
  const ready = pick(res.enforcementReady);
  const cal = pick(res.calibration) || (S.modCalibration || {})[category];
  const spec = pick(res.specCalibration);
  caveat.append(el('span', { className: `chip ${ready ? '' : 'chip--warn'}`, style: 'margin-left:8px', textContent: ready ? 'enforcement-ready' : 'not enforcement-ready' }));
  if (cal) caveat.append(el('span', { className: 'chip', style: 'margin-left:8px', textContent: `τ ${cal}` }));
  // a spec claiming more maturity than the engine will act on is worth seeing, not hiding
  if (spec && spec !== cal) {
    caveat.append(el('span', {
      className: 'chip', style: 'margin-left:8px', textContent: `spec claims ${spec}`,
      title: `This track's spec reports ${spec}, but the engine gates on ${cal} — the stricter of the two wins.`,
    }));
  }
  if (!res.hits.length) {
    gridMount.classList.remove('grid');
    gridMount.replaceChildren(el('div', { className: 'empty' },
      el('h2', { textContent: `Nothing fired in ${category}${tier ? ` at the ${tier} tier` : ''}` }),
      el('p', { textContent: 'No indexed image resembled this prompt set closely enough to be flagged. That is a real answer about this corpus, not an error.' })));
  } else {
    grid = new VirtualGrid(root, gridMount, { onOpen: openDetail, cap: 60 });
    ctl.grid = grid;
    grid.setItems(res.hits);
    grid.move(0);
  }
  paintStatus({
    latency: res.tookMs != null ? `<b>${res.tookMs.toFixed(1)} ms</b> server · ${res.rttMs.toFixed(0)} ms round-trip` : '',
    hits: `${n0(res.hits.length)} flagged`,
    coverage: 'source: current scan · not enforcement-ready',
  });
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
  await refreshStatus();
  if (!S.datasets.length) { try { S.datasets = await API.datasets(); } catch { /* fleet unavailable */ } }

  function paint() {
    const jobs = [...S.jobs.values()].sort((a, b) => (b.updated || 0) - (a.updated || 0));
    const act = jobs.filter((j) => j.state === 'running');
    const idxBytes = S.datasets.reduce((a, d) => a + (d.bytes || 0), 0);
    metrics.replaceChildren(
      el('div', { className: 'metric' }, el('b', { textContent: act.length ? act.reduce((a, j) => a + j.imgS, 0).toFixed(1) : '0' }), el('span', { textContent: 'images / second, right now' })),
      el('div', { className: 'metric' }, el('b', { textContent: n0(S.datasets.reduce((a, d) => a + d.count, 0)) }), el('span', { textContent: 'images searchable' })),
      el('div', { className: 'metric' }, el('b', { textContent: idxBytes ? bytes(idxBytes) : '—' }), el('span', { textContent: 'index on disk' })),
      el('div', { className: 'metric', title: S.eviction || '' },
        el('b', { textContent: S.rss != null ? bytes(S.rss) : '—' }),
        el('span', {
          // an empty model list at idle is the design, not a fault (ADR-5): nothing stays
          // resident until a free-text query needs it
          textContent: S.models == null ? 'daemon RSS'
            : S.models.length ? `daemon RSS · ${S.models.join(', ')} resident`
              : 'daemon RSS · no model resident (loads on demand)',
        })),
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
  // the coverage hairline belongs to the search field only — never leave it stale on another view
  if (path !== '/search') $('#qCov').hidden = true;
  if (path === '/' || path === '') {
    $('[data-nav="fleet"]').setAttribute('aria-current', 'page');
    viewFleet();
  } else if (path === '/search') {
    viewSearch((p.get('q') || '').trim(), p.get('d') || '');
  } else if (path.startsWith('/d/')) {
    viewDataset(decodeURIComponent(path.slice(3)));
  } else if (path === '/moderation') {
    $('[data-nav="moderation"]').setAttribute('aria-current', 'page');
    viewModeration();
  } else if (path.startsWith('/track/')) {
    $('[data-nav="moderation"]').setAttribute('aria-current', 'page');
    viewTrack(decodeURIComponent(path.slice(7)), p.get('tier') || '');
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
    // focus on the NEXT tick: moving focus inside the keydown still lets Chromium deliver
    // the "/" to the newly focused field
    e.preventDefault();
    setTimeout(() => { qInput.focus(); qInput.select(); }, 0);
    return;
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
refreshStatus();
route();
qInput.focus();      // search is never more than zero keystrokes away
paintStatus();
