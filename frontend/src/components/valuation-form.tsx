"use client";

import { useState, type FormEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Field,
  SelectField,
  NumberField,
  RangeField,
  SegmentedControl,
  ToggleField,
} from "@/components/form-fields";
import ValuationResult from "@/components/valuation-result";
import {
  ZONES,
  PROPERTY_TYPE_OPTIONS,
  PROPERTY_TYPE_LABELS,
  VIEW_OPTIONS,
  VIEW_LABELS,
  FURNISHING_OPTIONS,
  FURNISHING_LABELS,
  CONDITION_OPTIONS,
  CONDITION_LABELS,
  YEAR_BUILT_OPTIONS,
} from "@/lib/constants";
import { predictPrice, ApiError, type PredictionResponse } from "@/lib/api";

type PropertyCategory = "apartment" | "villa";

const SORTED_ZONES = [...ZONES].sort();

function toOptions<T extends string | number>(
  values: readonly T[],
  labels?: Record<string, string>
) {
  return values.map((v) => ({
    value: String(v),
    label: labels ? labels[String(v)] : String(v),
  }));
}

export default function ValuationForm() {
  const [category, setCategory] = useState<PropertyCategory>("apartment");
  const [propertyType, setPropertyType] = useState<string>("2BR");
  const [zone, setZone] = useState<string>("Dubai Marina");
  const [areaSqft, setAreaSqft] = useState<number>(1200);
  const [totalFloors, setTotalFloors] = useState<number>(30);
  const [floor, setFloor] = useState<number>(10);
  const [yearBuilt, setYearBuilt] = useState<number>(2018);
  const [view, setView] = useState<string>(VIEW_OPTIONS[0]);
  const [furnishing, setFurnishing] = useState<string>(FURNISHING_OPTIONS[0]);
  const [condition, setCondition] = useState<string>(CONDITION_OPTIONS[0]);
  const [parkingSpaces, setParkingSpaces] = useState<number>(1);
  const [isFreehold, setIsFreehold] = useState<boolean>(true);
  const [chillerIncluded, setChillerIncluded] = useState<boolean>(true);
  const [mortgageRate, setMortgageRate] = useState<number>(5.5);

  const [status, setStatus] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  function handleCategoryChange(next: PropertyCategory) {
    setCategory(next);
    setPropertyType(PROPERTY_TYPE_OPTIONS[next][0]);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setStatus("loading");
    setErrorMessage("");

    try {
      const response = await predictPrice({
        zone,
        property_category: category,
        property_type: propertyType,
        area_sqft: areaSqft,
        floor: category === "apartment" ? floor : null,
        total_floors: category === "apartment" ? totalFloors : null,
        year_built: yearBuilt,
        view,
        furnishing,
        condition,
        parking_spaces: parkingSpaces,
        is_freehold: isFreehold,
        chiller_included: chillerIncluded,
        mortgage_rate_at_listing: mortgageRate,
      });
      setResult(response);
      setStatus("success");
    } catch (err) {
      setErrorMessage(
        err instanceof ApiError
          ? err.message
          : "Something went wrong. Please try again."
      );
      setStatus("error");
    }
  }

  return (
    <div className="grid gap-16 lg:grid-cols-2 lg:gap-24">
      <form onSubmit={handleSubmit} className="space-y-12">
        <div>
          <p className="mb-6 text-xs tracking-[0.2em] text-accent uppercase">
            Category
          </p>
          <SegmentedControl
            layoutId="category-toggle"
            value={category}
            onChange={handleCategoryChange}
            options={[
              { value: "apartment", label: "Apartment" },
              { value: "villa", label: "Villa" },
            ]}
          />
        </div>

        <div className="grid grid-cols-2 gap-x-8 gap-y-8">
          <Field label="Zone">
            <SelectField
              value={zone}
              onChange={setZone}
              options={toOptions(SORTED_ZONES)}
            />
          </Field>

          <Field label="Property type">
            <SelectField
              value={propertyType}
              onChange={setPropertyType}
              options={toOptions(
                PROPERTY_TYPE_OPTIONS[category],
                PROPERTY_TYPE_LABELS
              )}
            />
          </Field>

          <Field label="Area (sqft)">
            <NumberField
              value={areaSqft}
              onChange={setAreaSqft}
              min={300}
              max={15000}
              step={50}
            />
          </Field>

          <Field label="Year built">
            <SelectField
              value={String(yearBuilt)}
              onChange={(v) => setYearBuilt(Number(v))}
              options={toOptions(YEAR_BUILT_OPTIONS)}
            />
          </Field>

          {category === "apartment" && (
            <>
              <Field label="Building height (total floors)">
                <NumberField
                  value={totalFloors}
                  onChange={(v) => {
                    setTotalFloors(v);
                    if (floor > v) setFloor(v);
                  }}
                  min={4}
                  max={90}
                />
              </Field>
              <Field label="Floor number">
                <NumberField
                  value={floor}
                  onChange={setFloor}
                  min={1}
                  max={totalFloors}
                />
              </Field>
            </>
          )}

          <Field label="View">
            <SelectField
              value={view}
              onChange={setView}
              options={toOptions(VIEW_OPTIONS, VIEW_LABELS)}
            />
          </Field>

          <Field label="Furnishing">
            <SelectField
              value={furnishing}
              onChange={setFurnishing}
              options={toOptions(FURNISHING_OPTIONS, FURNISHING_LABELS)}
            />
          </Field>

          <Field label="Condition">
            <SelectField
              value={condition}
              onChange={setCondition}
              options={toOptions(CONDITION_OPTIONS, CONDITION_LABELS)}
            />
          </Field>

          <Field label="Parking spaces">
            <SelectField
              value={String(parkingSpaces)}
              onChange={(v) => setParkingSpaces(Number(v))}
              options={toOptions([1, 2, 3, 4])}
            />
          </Field>

          <Field label="Prevailing mortgage rate">
            <RangeField
              value={mortgageRate}
              onChange={setMortgageRate}
              min={1.9}
              max={6.9}
              step={0.05}
            />
          </Field>
        </div>

        <div className="space-y-4">
          <ToggleField
            label="Freehold ownership"
            value={isFreehold}
            onChange={setIsFreehold}
          />
          <ToggleField
            label="AC chiller included"
            value={chillerIncluded}
            onChange={setChillerIncluded}
          />
        </div>

        <button
          type="submit"
          disabled={status === "loading"}
          className="w-full border border-ink bg-ink px-7 py-3.5 text-sm tracking-wide text-paper transition-colors hover:bg-transparent hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
        >
          {status === "loading" ? "Estimating…" : "Estimate Price"}
        </button>

        {status === "error" && (
          <p className="text-sm text-red-800">{errorMessage}</p>
        )}
      </form>

      <div>
        <AnimatePresence mode="wait">
          {result && status === "success" ? (
            <ValuationResult key="result" result={result} />
          ) : (
            <motion.div
              key="placeholder"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="border-t border-line pt-10 text-sm leading-relaxed text-muted"
            >
              Fill in a property&apos;s details and submit the form, and the
              trained pipeline will return an estimate here, alongside its
              typical error margin.
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
