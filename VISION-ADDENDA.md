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
