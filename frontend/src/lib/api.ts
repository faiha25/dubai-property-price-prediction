/**
 * Client for the FastAPI backend (backend/main.py). Talks to /predict
 * only — no other backend behavior is assumed here.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface PredictionRequest {
  zone: string;
  property_category: "apartment" | "villa";
  property_type: string;
  area_sqft: number;
  floor: number | null;
  total_floors: number | null;
  year_built: number;
  view: string;
  furnishing: string;
  condition: string;
  parking_spaces: number;
  is_freehold: boolean;
  chiller_included: boolean;
  mortgage_rate_at_listing: number;
}

export interface PredictionResponse {
  predicted_price_usd: number;
  predicted_price_aed: number;
  test_mae_usd: number;
}

export class ApiError extends Error {}

/**
 * Extracts a readable message from a FastAPI error body. Validation
 * errors (422) come back as {"detail": [{"msg": ..., "loc": [...]}]};
 * application errors (400/503) come back as {"detail": "some string"}.
 */
function extractErrorMessage(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((entry) =>
          entry && typeof entry === "object" && "msg" in entry
            ? String((entry as { msg: unknown }).msg)
            : JSON.stringify(entry)
        )
        .join("; ");
    }
  }
  return `Request failed with status ${status}`;
}

export async function predictPrice(
  payload: PredictionRequest
): Promise<PredictionResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(
      `Could not reach the prediction service at ${API_URL}. Is the FastAPI backend running?`
    );
  }

  const body = await res.json().catch(() => null);

  if (!res.ok) {
    throw new ApiError(extractErrorMessage(body, res.status));
  }

  return body as PredictionResponse;
}
