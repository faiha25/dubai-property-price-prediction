/**
 * Real reported results for the trained model, transcribed from README.md
 * and docs/model_card.md. Nothing here is estimated or invented — every
 * figure traces back to an executed notebook cell.
 */

export const MODEL_COMPARISON = [
  {
    model: "Baseline (predict mean)",
    trainR2: "0.000",
    testR2: "-0.000",
    testRmse: "$1,206,741",
    testMae: "$809,911",
  },
  {
    model: "Linear Regression",
    trainR2: "0.819",
    testR2: "0.834",
    testRmse: "$491,101",
    testMae: "$210,191",
  },
  {
    model: "Random Forest",
    trainR2: "0.971",
    testR2: "0.849",
    testRmse: "$468,818",
    testMae: "$191,514",
  },
  {
    model: "XGBoost (untuned)",
    trainR2: "0.939",
    testR2: "0.884",
    testRmse: "$411,465",
    testMae: "$167,670",
  },
  {
    model: "XGBoost (tuned)",
    trainR2: "0.913",
    testR2: "0.888",
    testRmse: "$403,817",
    testMae: "$162,554",
    isBest: true,
  },
] as const;

export const HEADLINE_STATS = [
  { value: "0.888", label: "Test R², tuned XGBoost" },
  { value: "$162,554", label: "Test MAE (USD)" },
  { value: "50,000", label: "Listings analyzed" },
  { value: "24", label: "Engineered features" },
] as const;

// Gain-based feature importance, notebooks/03_modeling.ipynb Section 11
// (also docs/model_card.md Section 2). SHAP analysis (src/explain.py)
// ranks log_area_sqft above bedrooms — the two measures disagree on the
// #1 feature, which is expected and reported rather than hidden.
export const FEATURE_IMPORTANCE = [
  { feature: "bedrooms", importance: 0.271 },
  { feature: "location_tier (highest price tier)", importance: 0.149 },
  { feature: "size_category: very large", importance: 0.099 },
  { feature: "log_area_sqft", importance: 0.089 },
  { feature: "location_tier (tier 3)", importance: 0.073 },
] as const;

export const KEY_FINDINGS = [
  "Location produces roughly a 30x price spread across zones — the single largest categorical price driver in the data.",
  "Property type produces a roughly 21x spread independent of location, from studios to 6-bedroom villas.",
  "Villas price far above apartments overall, and the gap holds within every major zone — location and property-category effects are additive.",
  "Bedroom count is the single strongest individual feature, but not a clean linear driver: 4BR penthouses out-price 4-bedroom villas.",
  "Floor position shows almost no relationship with price, counter to common real-world intuition about height and view premiums.",
  "Sea-view listings price more than 2.5x above community-view listings.",
  "Prediction error grows for high-end properties — the model is least reliable exactly where the financial stakes are largest.",
] as const;

export const LIMITATIONS = [
  "The target is the listed sale price, not a confirmed transaction price — actual sale prices may differ by some margin this project cannot measure.",
  "There is no building or developer field. Two identically-specified units in different buildings are treated identically, though real buyer behavior differentiates them.",
  "Location is captured via a 4-tier zone encoding plus straight-line distance to one landmark (Burj Khalifa) — it does not capture street-level or building-level variation.",
  "This dataset shows signs of being synthetically generated (zero duplicate rows, only 6 distinct construction years, a zone-price hierarchy that doesn't fully mirror the real Dubai market). Model behavior describes this dataset's structure, not a validated claim about the real market.",
  "Error variance increases for high-end properties, where a wrong estimate matters most financially.",
] as const;
