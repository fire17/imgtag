#!/usr/bin/env python3
"""Build the STATIC hypernym table (ADR-3: precomputed offline, no nltk/WordNet at runtime).

Sources, all already on disk (no egress):
  data/coco/annotations/instances_val2017.json   supercategory -> category names
  data/openimages/bbox_labels_600_hierarchy.json + oidv7-class-descriptions-boxable.csv

Output: src/imgtag/data/hierarchy.json = {"version", "sources", "children": {parent: [child…]}}
Names are lowercased. LVIS synset closure is b-bench's later job (this is the minimal table).

    uv run python scripts/build_hierarchy.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "imgtag" / "data" / "hierarchy.json"


def coco(children: dict[str, set[str]]) -> str | None:
    p = ROOT / "data" / "coco" / "annotations" / "instances_val2017.json"
    if not p.is_file():
        return None
    for c in json.loads(p.read_bytes())["categories"]:
        sup, name = c["supercategory"].lower().strip(), c["name"].lower().strip()
        if sup and sup != name:
            children.setdefault(sup, set()).add(name)
    return str(p.relative_to(ROOT))


def openimages(children: dict[str, set[str]]) -> str | None:
    h = ROOT / "data" / "openimages" / "bbox_labels_600_hierarchy.json"
    d = ROOT / "data" / "openimages" / "oidv7-class-descriptions-boxable.csv"
    if not (h.is_file() and d.is_file()):
        return None
    with open(d, newline="") as f:
        names = {row[0]: row[1].lower().strip() for row in csv.reader(f) if len(row) >= 2}

    def walk(node: dict) -> None:
        parent = names.get(node.get("LabelName", ""), "")
        for kid in node.get("Subcategory", []) or []:
            name = names.get(kid.get("LabelName", ""), "")
            if parent and name and parent != name and parent != "entity":
                children.setdefault(parent, set()).add(name)
            walk(kid)

    walk(json.loads(h.read_bytes()))
    return str(h.relative_to(ROOT))


def main() -> None:
    children: dict[str, set[str]] = {}
    sources = [s for s in (coco(children), openimages(children)) if s]
    if not sources:
        raise SystemExit("no hierarchy sources on disk (data/coco, data/openimages)")
    OUT.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": sources,
                "note": "hypernym -> direct hyponyms, lowercase; LVIS closure pending (b-bench)",
                "children": {k: sorted(v) for k, v in sorted(children.items())},
            },
            indent=1,
        )
    )
    print(f"{OUT.relative_to(ROOT)}: {len(children)} parents, {sum(len(v) for v in children.values())} edges")


if __name__ == "__main__":
    main()
