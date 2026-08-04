import Link from "next/link";
import Reveal from "@/components/reveal";
import { HEADLINE_STATS } from "@/lib/model-results";

const STEPS = [
  {
    number: "01",
    title: "Describe the property",
    body: "Zone, size, type, condition, and a handful of listing attributes — the same fields the model was trained on.",
  },
  {
    number: "02",
    title: "The trained pipeline runs",
    body: "The exact leakage-safe preprocessing and tuned XGBoost model used throughout this project. No shortcuts at inference time.",
  },
  {
    number: "03",
    title: "Receive an estimate",
    body: "A price in AED and USD, shown with the model's typical error margin — never a false-precision single number.",
  },
];

export default function Home() {
  return (
    <div>
      <section className="mx-auto max-w-6xl px-6 pt-20 pb-16 sm:px-10 sm:pt-28 sm:pb-24">
        <Reveal>
          <p className="text-xs tracking-[0.2em] text-accent uppercase">
            XGBoost regression &middot; trained on 50,000 Dubai listings
          </p>
        </Reveal>
        <Reveal delay={0.08}>
          <h1 className="mt-6 max-w-3xl font-serif text-4xl leading-[1.15] text-ink sm:text-5xl md:text-6xl">
            A considered read on{" "}
            <span className="italic text-accent">Dubai property value.</span>
          </h1>
        </Reveal>
        <Reveal delay={0.16}>
          <p className="mt-8 max-w-xl text-base leading-relaxed text-ink-soft sm:text-lg">
            A gradient-boosted regression model trained on secondary-market
            listings across Dubai, explaining 88.8% of price variance on
            held-out data. Built as a portfolio project — transparent about
            its data, its methodology, and its limits.
          </p>
        </Reveal>
        <Reveal delay={0.24}>
          <div className="mt-10 flex flex-wrap items-center gap-6">
            <Link
              href="/valuation"
              className="rounded-none border border-ink bg-ink px-7 py-3 text-sm tracking-wide text-paper transition-colors hover:bg-transparent hover:text-ink"
            >
              Get a Valuation
            </Link>
            <Link
              href="/methodology"
              className="group flex items-center gap-2 text-sm tracking-wide text-ink-soft transition-colors hover:text-ink"
            >
              Read the Methodology
              <span className="transition-transform group-hover:translate-x-1">
                &rarr;
              </span>
            </Link>
          </div>
        </Reveal>
      </section>

      <section className="border-y border-line">
        <div className="mx-auto grid max-w-6xl grid-cols-2 gap-y-10 px-6 py-14 sm:px-10 md:grid-cols-4 md:gap-y-0">
          {HEADLINE_STATS.map((stat, i) => (
            <Reveal key={stat.label} delay={i * 0.08}>
              <div
                className={`px-1 ${
                  i > 0 ? "md:border-l md:border-line md:pl-8" : ""
                }`}
              >
                <p className="font-serif text-3xl text-ink sm:text-4xl">
                  {stat.value}
                </p>
                <p className="mt-2 text-xs tracking-wide text-muted uppercase">
                  {stat.label}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-20 sm:px-10 sm:py-28">
        <Reveal>
          <p className="text-xs tracking-[0.2em] text-accent uppercase">
            How it works
          </p>
        </Reveal>
        <div className="mt-10 grid gap-12 md:grid-cols-3 md:gap-10">
          {STEPS.map((step, i) => (
            <Reveal key={step.number} delay={i * 0.1}>
              <div>
                <p className="font-serif text-2xl text-accent">
                  {step.number}
                </p>
                <h3 className="mt-4 font-serif text-xl text-ink">
                  {step.title}
                </h3>
                <p className="mt-3 text-sm leading-relaxed text-ink-soft">
                  {step.body}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="bg-paper-deep">
        <div className="mx-auto max-w-6xl px-6 py-20 text-center sm:px-10 sm:py-28">
          <Reveal>
            <h2 className="font-serif text-3xl text-ink sm:text-4xl">
              See what your property might be worth.
            </h2>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="mt-8">
              <Link
                href="/valuation"
                className="inline-block border border-ink bg-ink px-8 py-3 text-sm tracking-wide text-paper transition-colors hover:bg-transparent hover:text-ink"
              >
                Get a Valuation
              </Link>
            </div>
          </Reveal>
        </div>
      </section>
    </div>
  );
}
