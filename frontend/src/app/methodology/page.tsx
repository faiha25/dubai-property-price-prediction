import type { Metadata } from "next";
import Image from "next/image";
import Reveal from "@/components/reveal";
import {
  MODEL_COMPARISON,
  FEATURE_IMPORTANCE,
  KEY_FINDINGS,
  LIMITATIONS,
} from "@/lib/model-results";

export const metadata: Metadata = {
  title: "Methodology — Dubai Property Valuation",
};

const maxImportance = Math.max(...FEATURE_IMPORTANCE.map((f) => f.importance));

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-16 sm:px-10 sm:py-24">
      <Reveal>
        <p className="text-xs tracking-[0.2em] text-accent uppercase">
          Methodology
        </p>
        <h1 className="mt-6 font-serif text-3xl text-ink sm:text-4xl">
          How the model was built.
        </h1>
        <p className="mt-6 max-w-2xl text-base leading-relaxed text-ink-soft">
          A gradient-boosted regression model (XGBoost), trained on 50,000
          Dubai secondary-market listings (2020&ndash;2026). Every figure below
          is taken directly from the executed modeling notebook — none are
          estimated for this page.
        </p>
      </Reveal>

      <Reveal delay={0.08}>
        <section className="mt-20">
          <h2 className="font-serif text-2xl text-ink">Pipeline</h2>
          <ol className="mt-8 space-y-8">
            {[
              {
                title: "Cleaning",
                body: "The raw dataset was already unusually clean — zero duplicate rows, zero formatting issues. Floor/total_floors missingness (35.6% of rows) was confirmed structural — villas have no floor number — and preserved as null rather than imputed.",
              },
              {
                title: "Feature engineering",
                body: "Row-level features (size category, property age, log area, luxury indicator) were built from structural attributes only. Location tier — zones grouped by historical median price — is fit inside the modeling pipeline on the training fold only, so it can never leak validation-fold prices across cross-validation splits.",
              },
              {
                title: "Baseline and model comparison",
                body: "A mean-prediction baseline set the floor every model had to beat. Linear Regression, Random Forest, and XGBoost were then trained on identical preprocessing and an identical 80/20 split, each via a TransformedTargetRegressor on log1p(price), evaluated back on the original USD scale.",
              },
              {
                title: "Tuning and evaluation",
                body: "Only XGBoost — the model with the best test R² and the smallest train/test gap — was tuned, via RandomizedSearchCV across 30 parameter combinations on 5-fold cross-validation.",
              },
            ].map((step, i) => (
              <li key={step.title} className="grid gap-2 sm:grid-cols-[10rem_1fr] sm:gap-8">
                <p className="font-serif text-lg text-accent">
                  {String(i + 1).padStart(2, "0")} {step.title}
                </p>
                <p className="text-sm leading-relaxed text-ink-soft">
                  {step.body}
                </p>
              </li>
            ))}
          </ol>
        </section>
      </Reveal>

      <Reveal delay={0.1}>
        <section className="mt-20">
          <h2 className="font-serif text-2xl text-ink">Model comparison</h2>
          <div className="mt-8 overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead>
                <tr className="border-b border-line text-xs tracking-wide text-muted uppercase">
                  <th className="py-3 pr-4 font-normal">Model</th>
                  <th className="py-3 pr-4 font-normal">Train R&sup2;</th>
                  <th className="py-3 pr-4 font-normal">Test R&sup2;</th>
                  <th className="py-3 pr-4 font-normal">Test RMSE</th>
                  <th className="py-3 font-normal">Test MAE</th>
                </tr>
              </thead>
              <tbody>
                {MODEL_COMPARISON.map((row) => (
                  <tr
                    key={row.model}
                    className={`border-b border-line ${
                      "isBest" in row && row.isBest ? "text-ink" : "text-ink-soft"
                    }`}
                  >
                    <td className="py-3 pr-4">
                      {row.model}
                      {"isBest" in row && row.isBest && (
                        <span className="ml-2 text-xs text-accent">
                          selected
                        </span>
                      )}
                    </td>
                    <td className="py-3 pr-4 font-serif">{row.trainR2}</td>
                    <td className="py-3 pr-4 font-serif">{row.testR2}</td>
                    <td className="py-3 pr-4 font-serif">{row.testRmse}</td>
                    <td className="py-3 font-serif">{row.testMae}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs leading-relaxed text-muted">
            Random Forest shows the largest train/test gap (0.971 &rarr;
            0.849), a sign of overfitting. XGBoost generalizes better and was
            selected as the final model.
          </p>
        </section>
      </Reveal>

      <Reveal delay={0.12}>
        <section className="mt-20">
          <h2 className="font-serif text-2xl text-ink">Feature importance</h2>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-ink-soft">
            Gain-based importance from the tuned XGBoost model. A separate
            SHAP analysis ranks <code className="text-ink">log_area_sqft</code>{" "}
            above <code className="text-ink">bedrooms</code> — the two measures
            disagree on the top feature, which is expected: gain reflects
            average loss reduction per split, SHAP reflects typical
            per-prediction impact.
          </p>
          <div className="mt-8 space-y-6">
            {FEATURE_IMPORTANCE.map((f) => (
              <div key={f.feature}>
                <div className="flex items-baseline justify-between text-sm">
                  <span className="text-ink">{f.feature}</span>
                  <span className="font-serif text-ink-soft">
                    {f.importance.toFixed(3)}
                  </span>
                </div>
                <div className="mt-2 h-1 bg-paper-deep">
                  <div
                    className="h-1 bg-accent"
                    style={{
                      width: `${(f.importance / maxImportance) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
      </Reveal>

      <Reveal delay={0.14}>
        <section className="mt-20 grid gap-16 sm:grid-cols-2">
          <figure>
            <div className="border border-line">
              <Image
                src="/images/actual_vs_predicted.png"
                alt="Actual vs predicted price scatter plot on the held-out test set"
                width={700}
                height={700}
                className="w-full"
              />
            </div>
            <figcaption className="mt-3 text-xs leading-relaxed text-muted">
              Actual vs. predicted price, test set. Predictions cluster tightly
              for typical-range properties, with growing spread at the
              ultra-luxury end.
            </figcaption>
          </figure>
          <figure>
            <div className="border border-line">
              <Image
                src="/images/shap_summary.png"
                alt="SHAP summary beeswarm plot showing per-feature contribution to predictions"
                width={700}
                height={700}
                className="w-full"
              />
            </div>
            <figcaption className="mt-3 text-xs leading-relaxed text-muted">
              SHAP summary — each point is one prediction. Values are in
              log-price units, not dollars.
            </figcaption>
          </figure>
        </section>
      </Reveal>

      <Reveal delay={0.16}>
        <section className="mt-20">
          <h2 className="font-serif text-2xl text-ink">Key findings</h2>
          <ul className="mt-8 space-y-4">
            {KEY_FINDINGS.map((finding) => (
              <li
                key={finding}
                className="border-t border-line pt-4 text-sm leading-relaxed text-ink-soft"
              >
                {finding}
              </li>
            ))}
          </ul>
        </section>
      </Reveal>

      <Reveal delay={0.18}>
        <section className="mt-20 mb-8">
          <h2 className="font-serif text-2xl text-ink">
            Limitations and assumptions
          </h2>
          <ul className="mt-8 space-y-4">
            {LIMITATIONS.map((limitation) => (
              <li
                key={limitation}
                className="border-t border-line pt-4 text-sm leading-relaxed text-ink-soft"
              >
                {limitation}
              </li>
            ))}
          </ul>
        </section>
      </Reveal>
    </div>
  );
}
