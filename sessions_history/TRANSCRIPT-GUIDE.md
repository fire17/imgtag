# TRANSCRIPT-GUIDE — mining the founding session

`2026-07-22-founding-session-a4a879e7.jsonl` is the complete, verbatim conversation that
built this project in one day: ~7.3MB of JSONL, one JSON object per line (message/tool
call/tool result). It is the answer to every "but WHY is it like this?" — the decisions
are all in there with their evidence, in the order they actually happened.

## How to read it efficiently

Don't read it linearly. Grep for anchors:

```bash
J=2026-07-22-founding-session-a4a879e7.jsonl
grep -o '"NEW PROJECT IMGTAG[^"]*"' $J | head -1     # the founding vision
grep -c '"type":"tool_use"' $J                        # scale of the build
python3 -c "import json,sys; [print(json.loads(l).get('type')) for l in open('$J')]" | sort | uniq -c
```

Useful anchor strings (each marks a story worth reading a few KB around):

| grep for | you'll find |
|---|---|
| `VISION-ADDENDA` | every mid-mission user directive, verbatim, as it arrived |
| `MODEL: claude` | agent-lane reports (the multi-agent choreography; ~20 lanes) |
| `Login expired` | the auth outage that killed the whole swarm mid-mission |
| `tripwire` | spawned-model drift caught by the MODEL-first-line protocol (twice) |
| `git stash` / `git add -A` | the shared-worktree incidents and the hygiene laws they forged |
| `quiet window` | the load-gated benchmarking protocol evolving through failure |
| `dedupe(` / `also_in` | the user-reported duplicate-results bug, live root-cause |
| `weapo` | the best bug of the day: a partial word out-searching the full word |
| `0.281` | the discovery that multi-object precision has an architectural ceiling |
| `nn_agree` | mean-cosine parity lying about int8 text (the R@10 74.2 story) |
| `WITHHOLD` | the safety lane refusing to ship a weak alert tier, with measurements |
| `18 violation / 4 review / 178 none` | the reader-parity gate closing the drugs lane |
| `filter-repo` | the 619MB→2.4MB history rewrite at publish |

## The day's arc (so you know where you are)

1. Vision → budgets → 5-lane SOTA research → adversarial design review (79 findings).
2. Build wave: engine/daemon/app/bench/skill lanes in parallel; e2e verified same day.
3. Moderation wave: 7 tracks, policy rulings from the user, TP-probe datasets,
   separation measurements, fitted thresholds.
4. Feel-test wave: the user's live reports (duplicates, panel, weapo/weapon) each
   root-caused and fixed within minutes — good examples of verify-through-the-user's-surface.
5. Publish: history rewrite, honest-numbers gate (int8-text), README/LICENSE, release.

## Reading advice

The most transferable material is not the code (that's in the repo) — it's watching
*how problems were diagnosed*: ground-truth checks before beliefs, measurements before
rulings, honest retractions when a number didn't survive contact. If you adopt one habit
from this transcript, make it that one.
