# VISION-ADDENDA.md — later founding constraints, verbatim

> VISION.md is sealed (sha256-pinned). User constraints arriving after kickoff land here
> VERBATIM, dated, never rewritten. Derived docs (BUDGETS/ORACLE/UNKNOWNS) cite them.

## 2026-07-22 ~10:45Z — deployment target (mid-turn message, verbatim)

the final system will be running on a linux server without a gpu for the mostpast - so any optimizations should be focused for such systems rather than for this machine (maybe ill ask for optimizations for this machine later - but keep everything generic and ready for a 8gb ram (not powerful) linux server (that also has other things running and we cant slow down the server while we are doing both processing and infrence work) - hope all of that makes sense and will be addressed

## 2026-07-22 ~12:12Z — multi-tag search spectrum (verbatim)

in the search i should be able to use space to enter more  tags (show results that have all tags first, then decending for higher n/m found tags number, then finally results from each tag (like any) - does that makes sense - its like a specturm between ALL-SOME-ANY in tag-search - does all of that makes sense ?

## 2026-07-22 ~12:33Z — content moderation tracks + generic metadata (verbatim)

i want to be able to use this one some public sites and to enforce good behavior and following the rules of our sites, we dont want images with nudity, weapons or drugs
these are very important to indentify correctly - please make sure to create 3 tracks to specialize on each of these in a dedicated way - at the end we should be able to find (and later flag these) by searching, and also when indexing, so on every batch of images processed - i will be able to see something like "Found 10 images with drugs, 7 with weapons, 5 with nudity" alongside the images and metadata (like the filepaths etc, and we want to be able when indexing to also save more generic metadata that can hold account ids dates etc) - or check see these details accross the entire gallery, or by each dataset - does all of that makes sense ?

## 2026-07-22 ~12:50Z — moderation policy rulings (structured answers)

| Boundary | Ruling |
|---|---|
| Nude mannequin / statue / non-person figure | **Don't flag** |
| Swimwear / lingerie on a person | **Flag for review** |
| Toy / replica weapons | **Flag for review** |
| Tobacco / vaping / smoking imagery | **Flag for review** |

→ Two-tier flag model: `violation` (human nudity/explicit, real weapons, illegal drugs +
paraphernalia) vs `review` (swimwear, toy weapons, tobacco/vape). Counts and views report
both tiers distinctly.

## 2026-07-22 ~13:20Z — safety track: people lying down + danger escalation (verbatim)

make another track to identify people lying down (even if part of their body is obstructed) and even higher flagging if either detecting injury, things broken, distruction distress high stress or anything dangorous - does that makes sense ?

## 2026-07-22 ~13:23Z — sports content track (verbatim)

make another dedicated track to identify all sports related images

## 2026-07-22 ~13:26Z — track confidence + scaling invariant (verbatim)

very important - make sure this is true and to also right it in the project's dir somewhere where it matters - for each track so for or any that we add in the future i want a confidence score for each track for every image - ideally each track is specialized and possibly autoresearch auto improved onto itself but we must makes sure that even if we have 100 tracks, the times for indexing and inferencing should remain relatively the same so this system can continue to scale

## 2026-07-22 ~13:28Z — people/face counting track (verbatim)

i want track to be able to categorize images if they have 1 person in them (even if its their back with no face), more then one person, 1 visible face - and more than one visible face (even at angles for any)

## 2026-07-22 ~13:29Z — violence/abuse track (verbatim)

and one track for general violence or abuse

## 2026-07-22 ~13:42Z — track self-improvement protocol + agents-never-operate law (verbatim)

tell the agent that did the drugs track to keep going (ill add a /goal to it) to continue on expanding and improving it's track, do autoresearch loop to improve correctness of confidence levels, and to deepend different categories or subcategories of drugs and to be able to detect them all better - 

think about the process of how you give that subagnet to imrpove its own track and create a mini skill in this projects folder (not global, or remind me to make global later, we need to undersand how to make it more generic for other usecases where tracks are a thing), and it would be called to tell the track subagent to improve itself - and that skill in itself should have and run a darwin skill and autoresearch improvement cycle (or several) to improve the way we tell subagents how to improve their own tracks, measureing our success based on the change in rate of improvement of the subagents (inner loop, we are the outerloop observers) - does all of that makes sense?

and i hope this goes without saying - the agents can check (sample selective) images and see them themselves in order for them to conduct tests - but they should never be used for actually doing the process of categorization or whatever the track does - everything has to be programatic and be able to run the same with no agents afterwards - agents shouldnt waste their tokens on verifying images for tracks themselves if that makes sense - make sure that all working subagents are working on improving the system and or at indentifying or improving their tracks and that they are not really wasting tokens

## 2026-07-22 ~13:58Z — weapons + nudity true positives, subcategory depth, ratio thresholds (verbatim)

> "the identifying weapons track needs to be improved, and also the nudity, i dont see true
> positibes in the datasets and all findings are false positives, we must make sure that both
> bring these so we can truely test them and like the drug discovery, need to also go deep and
> and make sure that all handguns firearms - make a list of subitems in each of these
> categories - so that we can make sure that our monitoring really works at identifying and
> flagging them at high confidence - and whatever is scored now should - if working correct -
> be at lower confidence results compared to the true positive results and then we will be
> able to set the ratio thershould for auto flagging for each track - does all of that makes
> sense ?"

Derived (💭): per-track TP probe corpora indexed as real datasets (weaponprobe / nudityprobe,
pattern = drugprobe) · subcategory taxonomies as versioned data · TP-vs-current-FP confidence
separation per subcategory · fitted per-tier τ = the "ratio threshold for auto flagging".
Nudity bounded by the EVAL DATA LAW: review-tier TPs local-sourced; violation tier stays
benchmark-cited, honestly labeled. Dispatched: track-weapons3 + track-nudity3 (opus).

## 2026-07-22 ~14:05Z — per-track progress report + continue (verbatim)

> "please ask each track for their progress and delta and give me a visually pleasing report
> (rich table with progressbars and etas and current status and next phases and what each
> track does) - do this now then continue everything else please includeing addressing the
> tracks i asked for in a way that will result in what i asked previously - make sure you
> dont miss a thing"

Also this hour (user, live feel-test): "im seeing duplicate results when searching" → fixed
(cross-dataset collapse in dedupe(), also_in provenance, verified over HTTP).

## 2026-07-22 ~14:16Z — per-track confidences in the image detail view (verbatim)

> "Every image when i click on it in the gallery it must show the confidences for each one of
> our tracks. Ranked and highlighted by confidence score."

Derived (💭): detail overlay gains a tracks panel — ALL tracks, every image, ranked by p
descending, highlighted by score/tier (alert > violation > review > match > none). Powered by
ADR-15 sidecars / live matvec fallback. Routed: b-daemon (per-image all-tracks payload),
b-app (ranked highlighted panel).
