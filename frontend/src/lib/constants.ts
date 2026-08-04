/**
 * Valid input options for the valuation form, mirroring backend/model.py
 * (which itself mirrors app.py and notebooks/03_modeling.ipynb Section 4).
 *
 * Duplicated here rather than fetched from the API at runtime, following
 * the same pattern already used between app.py and backend/model.py in
 * this project: the trained pipeline doesn't expose its valid input
 * space itself, so every consumer keeps its own copy in sync manually.
 * If these three copies ever drift, the backend's Pydantic validation
 * (backend/schemas.py) is the source of truth and will reject anything
 * this list gets wrong.
 */

export const ZONES = [
  "Al Barsha",
  "Al Furjan",
  "Al Mamzar",
  "Al Nahda",
  "Al Qusais",
  "Al Warqa",
  "Al Wasl",
  "Arabian Ranches",
  "Bukadra",
  "Bur Dubai",
  "Business Bay",
  "Creek",
  "DIFC",
  "DIP",
  "Damac",
  "Damac Hills",
  "Deira",
  "Deira Islands",
  "Discovery Gardens",
  "Downtown",
  "Dubai Marina",
  "Dubai Silicon Oasis",
  "Dubai South",
  "Dubai Sports City",
  "Dubai Studio City",
  "Dubai-Al Ain Road",
  "Dubailand",
  "Emirates Hills",
  "Emirates Living",
  "Festival City",
  "IMPZ",
  "International City",
  "JGE",
  "JLT",
  "JVC",
  "JVT",
  "Jebel Ali",
  "Jumeirah",
  "Jumeirah Bay",
  "MBR City",
  "Maritime City",
  "Meydan",
  "Mirdif",
  "Mohammed Bin Rashid",
  "Motor City",
  "Nshama",
  "Palm Jumeirah",
  "Reem",
  "Sustainable City",
  "The Greens",
  "Tilal Al Ghaf",
  "World Islands",
] as const;

export const PROPERTY_TYPE_OPTIONS = {
  apartment: ["studio", "1BR", "2BR", "3BR", "4BR_penthouse"],
  villa: ["3BR_villa", "4BR_villa", "5BR_villa", "6BR_villa"],
} as const;

export const PROPERTY_TYPE_LABELS: Record<string, string> = {
  studio: "Studio",
  "1BR": "1 Bedroom",
  "2BR": "2 Bedroom",
  "3BR": "3 Bedroom",
  "4BR_penthouse": "4 Bedroom Penthouse",
  "3BR_villa": "3 Bedroom Villa",
  "4BR_villa": "4 Bedroom Villa",
  "5BR_villa": "5 Bedroom Villa",
  "6BR_villa": "6 Bedroom Villa",
};

export const VIEW_OPTIONS = [
  "community",
  "park",
  "city",
  "pool",
  "sea",
  "marina",
  "golf_course",
  "burj_khalifa",
] as const;

export const VIEW_LABELS: Record<string, string> = {
  community: "Community",
  park: "Park",
  city: "City",
  pool: "Pool",
  sea: "Sea",
  marina: "Marina",
  golf_course: "Golf Course",
  burj_khalifa: "Burj Khalifa",
};

export const FURNISHING_OPTIONS = [
  "unfurnished",
  "semi_furnished",
  "fully_furnished",
] as const;

export const FURNISHING_LABELS: Record<string, string> = {
  unfurnished: "Unfurnished",
  semi_furnished: "Semi-furnished",
  fully_furnished: "Fully furnished",
};

export const CONDITION_OPTIONS = [
  "vacant_on_transfer",
  "tenanted",
  "off_plan_resale",
] as const;

export const CONDITION_LABELS: Record<string, string> = {
  vacant_on_transfer: "Vacant on transfer",
  tenanted: "Tenanted",
  off_plan_resale: "Off-plan resale",
};

export const YEAR_BUILT_OPTIONS = [2008, 2012, 2015, 2018, 2021, 2023] as const;

export const AED_PER_USD = 3.6725;
export const TEST_MAE_USD = 162_554;
