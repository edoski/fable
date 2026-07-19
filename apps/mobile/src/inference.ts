declare const process: {
  env: {
    EXPO_PUBLIC_FABLE_BACKEND_URL?: string;
  };
};

export const CHAINS = [
  "ethereum",
  "polygon",
  "avalanche",
] as const;

export type Chain = (typeof CHAINS)[number];

export const HORIZONS = [2, 3, 4, 5] as const;

export type Horizon = (typeof HORIZONS)[number];

export type InferenceRequest = {
  chain: Chain;
  K: Horizon;
};

export type InferenceResponse = {
  head_block: number;
  selected_action_k: number;
  target_block: number;
};

export async function requestInference(
  request: InferenceRequest,
): Promise<InferenceResponse> {
  const backendUrl = process.env.EXPO_PUBLIC_FABLE_BACKEND_URL;
  if (!backendUrl) {
    throw new Error("EXPO_PUBLIC_FABLE_BACKEND_URL is required");
  }

  const body = JSON.stringify(request);
  let response: Response;
  try {
    response = await fetch(backendUrl + "/inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error("Network error: " + message);
  }

  if (!response.ok) {
    const body = (await response.text()).trim();
    throw new Error("HTTP " + response.status + ": " + body);
  }

  return (await response.json()) as InferenceResponse;
}
