"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="block text-xs tracking-wide text-muted uppercase">
        {label}
      </span>
      <div className="mt-2">{children}</div>
    </label>
  );
}

const inputClass =
  "w-full border-b border-line bg-transparent pb-2 pt-1 text-sm text-ink transition-colors focus:border-accent focus:outline-none";

export function SelectField({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={inputClass}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

export function NumberField({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      className={inputClass}
    />
  );
}

export function RangeField({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
      <p className="mt-2 text-sm text-ink-soft">{value.toFixed(2)}%</p>
    </div>
  );
}

export function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  layoutId,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
  layoutId: string;
}) {
  return (
    <div className="flex gap-8 border-b border-line pb-2">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`relative pb-2 text-sm transition-colors ${
            value === opt.value ? "text-ink" : "text-ink-soft hover:text-ink"
          }`}
        >
          {opt.label}
          {value === opt.value && (
            <motion.span
              layoutId={layoutId}
              className="absolute -bottom-[9px] left-0 h-px w-full bg-accent"
              transition={{ type: "spring", stiffness: 400, damping: 32 }}
            />
          )}
        </button>
      ))}
    </div>
  );
}

export function ToggleField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className="flex w-full items-center justify-between border-b border-line pb-2 pt-1 text-left"
    >
      <span className="text-sm text-ink">{label}</span>
      <span
        className={`relative h-4 w-8 shrink-0 rounded-full transition-colors ${
          value ? "bg-accent" : "bg-line"
        }`}
      >
        <motion.span
          className="absolute top-0.5 h-3 w-3 rounded-full bg-paper"
          animate={{ left: value ? "18px" : "2px" }}
          transition={{ type: "spring", stiffness: 500, damping: 34 }}
        />
      </span>
    </button>
  );
}
