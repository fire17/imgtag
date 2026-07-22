# sessions_history — the founding session, and everything it learned

This folder is a **handover, not an archive**. It exists so that a future agent (or human)
can take everything this project built — the working code, the measured decisions, the
hard-won tricks — and be productive in minutes instead of re-deriving days of work.

**The standing encouragement, from the team that built this:** everything here is
*operational and optimized as it stands*. Use the files as they exist. Patch **your**
system to call this one; make minor edits only where your platform genuinely differs.
The temptation to rewrite is how the lessons get lost — resist it, and enjoy this from
the box.

## What's in this folder

| file | read it when |
|---|---|
| [`HANDOVER.md`](HANDOVER.md) | **Start here.** System map, run-it-now commands, verify-it-yourself commands, honest state. |
| [`TRICKS-AND-OPTIMIZATIONS.md`](TRICKS-AND-OPTIMIZATIONS.md) | Before changing ANY performance- or quality-relevant code. Every non-obvious choice, with the measurement that earned it. |
| [`ADAPTING.md`](ADAPTING.md) | You're integrating this into a social network / marketplace / CMS / any platform. The cookbook. |
| [`TRANSCRIPT-GUIDE.md`](TRANSCRIPT-GUIDE.md) | You want the full story or need to settle "why is it like this?" — how to mine the transcript efficiently. |
| `2026-07-22-founding-session-a4a879e7.jsonl` | The complete, **verbatim** founding-session conversation (7.3MB JSONL). One build day: vision → research → build → measure → publish. |

## The project's own constitution (repo root — the handover's other half)

- **`ORACLE.md`** — every architecture decision (ADR-1..15) with the *why*, symptom-keyed
  playbooks, the incident field log. The single most valuable file in the repo.
- **`BUDGETS.md`** — every performance/quality claim as a numbered, tested budget with its
  exact verification command. Nothing here is vibes.
- **`TRACKS.md`** — the laws that make moderation scale to 100 tracks at ~zero marginal
  cost, including READER PARITY (the law three separate bugs taught us).
- **`VISION.md` / `VISION-ADDENDA.md`** — the founding intent, verbatim, plus every
  mid-mission directive. Executors who understand intent degrade gracefully.
- **`DARWIN.md`** — the honest list of what's NOT done (D1–D13), each with its measured
  justification and success criterion. Your best roadmap is our unfinished business.
- **`research/`** — per-track lane reports with full measured tables and honest-limits
  sections. `research/candidates.md` is the model-selection matrix.

## Provenance

Built 2026-07-22 in one continuous session by a conductor agent + ~20 specialist agent
lanes (the transcript shows the whole choreography, including the coordination failures
and how they were caught). Published the same day: https://github.com/fire17/imgtag

Transcript integrity: copied from the live session record at handover time (a session
transcript is inherently a point-in-time prefix — the final commit/publish exchange is
its own tail). Every line validates as JSON. **Exactly 3 redactions** were applied, all
of the same string class: two private email addresses that appeared only inside the
conductor's own redaction-planning discussion were masked as `[redacted-email]`.
Everything else — every word of the user, every agent report, every measurement — is
byte-verbatim. No keys, no tokens (pattern-swept).
