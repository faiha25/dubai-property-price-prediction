export default function SiteFooter() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto max-w-6xl px-6 py-10 sm:px-10">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <p className="max-w-md text-sm leading-relaxed text-muted">
            Portfolio and educational project. Trained on a Kaggle dataset that
            shows signs of being synthetically generated. Estimates illustrate
            a machine learning model, not a professional property valuation.
          </p>
          <div className="flex gap-10 text-sm text-ink-soft">
            <a
              href="https://github.com/faiha25/dubai-property-price-prediction"
              className="hover:text-accent transition-colors"
            >
              Source
            </a>
            <a
              href="/methodology"
              className="hover:text-accent transition-colors"
            >
              Model card
            </a>
          </div>
        </div>
        <p className="mt-8 text-xs tracking-wide text-muted">
          XGBoost regression &middot; Test R&sup2; 0.888 &middot; Dubai secondary-market listings, 2020&ndash;2026
        </p>
      </div>
    </footer>
  );
}
