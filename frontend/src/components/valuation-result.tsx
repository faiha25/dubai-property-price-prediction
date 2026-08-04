"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import type { PredictionResponse } from "@/lib/api";

function formatPrice(amount: number): string {
  const rounded = Math.round(amount / 1000) * 1000;
  return rounded.toLocaleString("en-US");
}

function useCountUp(target: number, duration = 900) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let raf: number;
    const start = performance.now();

    function tick(now: number) {
      const elapsed = now - start;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(target * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    }

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);

  return value;
}

export default function ValuationResult({
  result,
}: {
  result: PredictionResponse;
}) {
  const animatedAed = useCountUp(result.predicted_price_aed);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="border-t border-line pt-10"
    >
      <p className="text-xs tracking-[0.2em] text-accent uppercase">
        Estimated value
      </p>
      <p className="mt-4 font-serif text-4xl text-ink sm:text-5xl">
        AED {formatPrice(animatedAed)}
      </p>
      <p className="mt-3 text-sm text-ink-soft">
        Equivalent to USD {formatPrice(result.predicted_price_usd)} — the
        model&apos;s native currency, shown here at the fixed AED/USD peg for
        display only.
      </p>
      <p className="mt-6 max-w-md text-xs leading-relaxed text-muted">
        Typical model error on held-out test data: &plusmn; USD{" "}
        {result.test_mae_usd.toLocaleString("en-US")} (MAE). Treat this as an
        estimate range, not an exact figure — and not a substitute for a
        professional valuation.
      </p>
    </motion.div>
  );
}
