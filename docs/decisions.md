# Project Decisions Log

Short, dated records of decisions that aren't obvious from the code
alone — what was decided, why, and what it means for later phases.

## 0001 — Currency handling for the target variable

**Date:** 2026-08-01
**Status:** Decided

**Decision:** The target variable `price_usd` stays in **USD**
throughout cleaning, EDA, and modeling. It is not converted to AED at
any point in the pipeline.

**Why:**
- The source dataset (`secondary_sales.csv`) natively reports prices in
  USD — converting introduces an FX-rate assumption (which rate? fixed
  historical peg, or a rate at time of listing?) that has nothing to do
  with the modeling problem itself.
- Keeping the internal representation identical to the raw data
  preserves reproducibility: anyone re-running the pipeline against the
  same raw file gets numerically identical results, with no external
  FX lookup or hardcoded conversion constant to go stale.
- The original project blueprint (`PROJECT_SPECIFICATION.md`) assumes
  AED throughout its README/business-framing language, since it was
  written before the actual dataset's currency was known. That framing
  is a presentation concern, not a modeling concern.

**How this applies downstream:**
- All EDA visualizations, axis labels, and written observations that
  reference price must say **USD** explicitly (e.g. "Price (USD)", not
  "Price"), so nothing is ambiguous or silently assumed to be AED.
- All modeling-phase metrics (MAE, RMSE, etc. in `03_modeling.ipynb`)
  will be reported in USD.
- **AED conversion is deferred to the README/business-interpretation
  layer only** (Part 10 of the blueprint), if and when it's needed for
  a UAE-facing audience. If that happens, it should be a single
  documented conversion applied for display purposes only, using a
  clearly stated, dated FX rate — never applied to the modeling
  pipeline itself.
