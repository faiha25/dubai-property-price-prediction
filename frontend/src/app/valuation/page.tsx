import type { Metadata } from "next";
import Reveal from "@/components/reveal";
import ValuationForm from "@/components/valuation-form";

export const metadata: Metadata = {
  title: "Get a Valuation: Dubai Property Valuation",
};

export default function ValuationPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-16 sm:px-10 sm:py-24">
      <Reveal>
        <p className="text-xs tracking-[0.2em] text-accent uppercase">
          Valuation
        </p>
        <h1 className="mt-6 max-w-2xl font-serif text-3xl text-ink sm:text-4xl">
          Estimate a property&apos;s value.
        </h1>
        <p className="mt-6 max-w-xl text-base leading-relaxed text-ink-soft">
          Describe the property below. The form mirrors the exact fields the
          model was trained on, nothing more is asked, and nothing here is
          used beyond producing the estimate.
        </p>
      </Reveal>

      <div className="mt-16">
        <ValuationForm />
      </div>
    </div>
  );
}
