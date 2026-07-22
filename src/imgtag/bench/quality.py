"""CORPUS-A quality metrics for the candidate matrix: B6, B5, B17, B7.

Every metric here is a RANKING metric computed from L2-normalized embeddings, so it is
engine-independent and comparable across candidates on one fixed corpus.

Definitions (written down because a metric without its protocol is not a number):
- B6 precision@k, k = min(10, N_pos) per category — p@10 is ill-defined for toaster (8
  positives in val2017) and hair drier (9).
- B5 hypernym: query the SUPERCATEGORY word. precision@100 = fraction of the top-100 that
  contain ANY child. per-child recall@R with R = |union of all children's positives| —
  the number of images an oracle would have to return to get them all; per-child recall is
  that child's positives found inside the top-R.
- B17 text->image R@k over COCO captions. Corpus = val2017 5k. NOT the Karpathy test
  split (see `retrieval()` docstring) — valid for the RELATIVE +12pt-vs-control gate,
  not for the absolute-vs-model-card clause.
- B7 negatives: both sides in ONE run. tau is fitted on THIS run as the highest threshold
  that still holds mean recall@10 >= 0.70 over present categories; leakage is then the
  fraction of absent queries with any hit above it. Candidate-selection proxy only — the
  SHIPPED tau is fitted on the held-out CAL-SET (ADR-3 calibration contract).
"""
from __future__ import annotations

import numpy as np

from . import corpus as X
from . import textsets as T


def _rank(img: np.ndarray, txt: np.ndarray) -> np.ndarray:
    """Descending image indices per text row. img [N,D], txt [Q,D], both L2-normed."""
    return np.argsort(-(txt @ img.T), axis=1)


def category_precision(img: np.ndarray, txt_cat: np.ndarray, names: list[str],
                       pos: dict[str, list[int]]) -> dict:
    """B6: per-category precision@min(10, N_pos)."""
    order = _rank(img, txt_cat)
    rows = []
    for i, name in enumerate(names):
        p = set(pos[name])
        k = min(10, len(p))
        if not k:
            continue
        hits = sum(1 for j in order[i, :k] if j in p)
        rows.append({"category": name, "n_pos": len(p), "k": k, "p_at_k": hits / k})
    vals = [r["p_at_k"] for r in rows]
    return {
        "rows": sorted(rows, key=lambda r: r["p_at_k"]),
        "mean": float(np.mean(vals)), "min": float(np.min(vals)),
        "zeros": [r["category"] for r in rows if r["p_at_k"] == 0.0],
        "pass": bool(np.mean(vals) >= 0.90 and np.min(vals) >= 0.70
                     and not any(r["p_at_k"] == 0.0 for r in rows)),
    }


def hypernym(img: np.ndarray, txt_sup: np.ndarray, supers: dict[str, list[str]],
             pos: dict[str, list[int]]) -> dict:
    """B5: supercategory precision@100 + per-child recall@R."""
    order = _rank(img, txt_sup)
    out, all_p100, all_child = [], [], []
    for i, (sup, children) in enumerate(supers.items()):
        union = set().union(*(set(pos[c]) for c in children))
        R = len(union)
        top100 = order[i, :100]
        p100 = sum(1 for j in top100 if j in union) / 100
        topR = set(order[i, :R].tolist())
        top100s = set(top100.tolist())
        childs = []
        for c in children:
            p = set(pos[c])
            childs.append({"child": c, "n_pos": len(p),
                           "recall_at_R": len(p & topR) / len(p) if p else 0.0,
                           "in_top100": len(p & top100s)})
        out.append({"supercat": sup, "R": R, "precision_at_100": p100, "children": childs})
        all_p100.append(p100)
        all_child += [c["recall_at_R"] for c in childs]
    missing = [c["child"] for s in out for c in s["children"] if c["in_top100"] == 0]
    return {
        "rows": out,
        "mean_p_at_100": float(np.mean(all_p100)),
        "mean_child_recall": float(np.mean(all_child)),
        "min_child_recall": float(np.min(all_child)),
        "children_absent_from_top100": missing,
        "pass": bool(np.mean(all_p100) >= 0.85 and np.mean(all_child) >= 0.55
                     and np.min(all_child) >= 0.35 and not missing),
    }


def retrieval(img: np.ndarray, txt_cap: np.ndarray, cap_img_idx: list[int]) -> dict:
    """B17 text->image R@1/5/10.

    CORPUS NOTE (honest): this is COCO **val2017** (5,000 images / 25,014 captions), NOT
    the Karpathy 5k test split — val2017 is the 2017 minival, a different 5k than
    Karpathy's test partition, so BUDGETS' image-id intersection check fails BY
    CONSTRUCTION and no karpathy json is on disk. The default-vs-control delta (the +12pt
    gate) is measured on one identical corpus and is therefore valid; the
    "within 2 pts of the model card" clause is NOT satisfiable until the Karpathy json is
    fetched. Reported, never silently substituted.
    """
    sims = txt_cap @ img.T
    gt = np.asarray(cap_img_idx)
    ranks = (sims > sims[np.arange(len(gt)), gt][:, None]).sum(1)
    return {
        "corpus": "coco-val2017-5k (NOT Karpathy test split)",
        "n_queries": int(len(gt)),
        "R@1": float((ranks < 1).mean() * 100),
        "R@5": float((ranks < 5).mean() * 100),
        "R@10": float((ranks < 10).mean() * 100),
        "median_rank": float(np.median(ranks) + 1),
    }


def negatives(img: np.ndarray, txt_cat: np.ndarray, txt_abs: np.ndarray,
              names: list[str], pos: dict[str, list[int]], abs_names: list[str]) -> dict:
    """B7 both sides in one run: fit tau on present recall@10, then measure leakage."""
    s_pres = txt_cat @ img.T           # [80, N]
    s_abs = txt_abs @ img.T            # [A,  N]
    order = np.argsort(-s_pres, axis=1)[:, :10]

    def recall_at_10(tau: float) -> float:
        rs = []
        for i, name in enumerate(names):
            p = set(pos[name])
            if not p:
                continue
            kept = [j for j in order[i] if s_pres[i, j] >= tau]
            rs.append(len(p & set(kept)) / min(10, len(p)))
        return float(np.mean(rs))

    lo, hi = float(s_pres.min()), float(s_pres.max())
    if recall_at_10(lo) < 0.70:
        # B7(b) is unreachable at ANY threshold: the model's own top-10 already misses
        # 30%+ of positives. Report that honestly instead of a meaningless tau.
        return {"tau": None, "unfittable": True,
                "recall_at_10_ceiling": recall_at_10(lo),
                "n_absent_queries": len(abs_names),
                "leakage_rate": None,
                "margin_present_minus_absent": float(s_pres.max(1).mean()
                                                     - s_abs.max(1).mean()),
                "pass": False}
    for _ in range(40):  # highest tau still holding B7(b) recall >= 0.70
        mid = (lo + hi) / 2
        if recall_at_10(mid) >= 0.70:
            lo = mid
        else:
            hi = mid
    tau = lo
    leaked = [{"query": q, "n_above_tau": int((s_abs[i] >= tau).sum()),
               "max_sim": float(s_abs[i].max())}
              for i, q in enumerate(abs_names)]
    n_leak = sum(1 for r in leaked if r["n_above_tau"] > 0)
    # Separability: how far absent-query peaks sit below present-query peaks.
    pres_top = s_pres.max(1)
    abs_top = s_abs.max(1)
    return {
        "tau": tau,
        "recall_at_10_at_tau": recall_at_10(tau),
        "n_absent_queries": len(abs_names),
        "leakage_rate": n_leak / max(1, len(abs_names)),
        "margin_present_minus_absent": float(pres_top.mean() - abs_top.mean()),
        "worst_leaks": sorted(leaked, key=lambda r: -r["n_above_tau"])[:5],
        "pass": bool(n_leak / max(1, len(abs_names)) <= 0.02
                     and recall_at_10(tau) >= 0.70),
    }


# ── one-call driver ───────────────────────────────────────────────────────────
def text_sets() -> dict:
    """Every string this suite scores, built once, deterministic."""
    a = X.corpus_a()
    names = list(a["pos"].keys())
    sups = list(a["supers"].keys())
    absent = list(X.absent_concepts()) + list(X.ABSURD)
    return {
        "cat_names": names, "cat_prompts": [T.prompt(n) for n in names],
        "sup_names": sups, "sup_prompts": [T.prompt(s) for s in sups],
        "abs_names": absent,
        "abs_prompts": [a if " " in a and len(a.split()) > 4 else T.prompt(a)
                        for a in absent],
        "captions": [c for _, c in a["captions"]],
        "caption_img_idx": [i for i, _ in a["captions"]],
    }


def score_all(img: np.ndarray, txt: dict, ts: dict, gt: dict | None = None) -> dict:
    """img: [N,D] corpus embeddings. txt: {'cat','sup','abs','cap'} text embeddings.

    gt (aligned ground truth: pos/supers) defaults to corpus_a — pass the snapshot-aligned
    version (corpus.align_to_ids) when scoring an already-indexed dataset whose row order is
    the indexer's, not corpus_a's.
    """
    a = gt or X.corpus_a()
    out = {
        "b6_category_precision": category_precision(img, txt["cat"], ts["cat_names"],
                                                    a["pos"]),
        "b5_hypernym": hypernym(img, txt["sup"], a["supers"], a["pos"]),
        "b7_negatives": negatives(img, txt["cat"], txt["abs"], ts["cat_names"], a["pos"],
                                  ts["abs_names"]),
    }
    if txt.get("cap") is not None:
        out["b17_retrieval"] = retrieval(img, txt["cap"], ts["caption_img_idx"])
    return out
