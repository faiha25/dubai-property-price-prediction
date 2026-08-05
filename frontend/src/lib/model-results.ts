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
