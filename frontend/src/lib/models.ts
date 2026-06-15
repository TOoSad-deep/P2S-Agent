// Model configuration: preset models (from backend /api/models) and
// user-defined custom models (persisted in localStorage). The selected model
// is sent to the backend inside input_spec.model on each run.

export interface ModelPreset {
  id: string;
  label: string;
  model: string;
  supports_image: boolean;
  configured: boolean; // false → placeholder (no API key), shown disabled
  default: boolean;
}

export interface CustomModel {
  id: string; // local-only id, e.g. "custom:<uuid>"
  label: string;
  base_url: string;
  api_key: string;
  model: string;
  supports_image: boolean;
}

// Selection payload merged into input_spec.model and sent to the backend.
export type ModelSelection =
  | { preset_id: string }
  | {
      label: string;
      base_url: string;
      api_key: string;
      model: string;
      supports_image: boolean;
    };

const STORAGE_KEY = "p2s.customModels";

export function loadCustomModels(): CustomModel[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (m): m is CustomModel =>
        m && typeof m.id === "string" && typeof m.model === "string" && typeof m.base_url === "string"
    );
  } catch {
    return [];
  }
}

export function saveCustomModels(models: CustomModel[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(models));
  } catch {
    // ignore quota / serialization errors — selection still works in-session
  }
}

export function newCustomModelId(): string {
  const uuid =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
  return `custom:${uuid}`;
}

// Build the input_spec.model payload for a given selection. Returns null when
// no usable model is selected (backend falls back to its default).
export function toModelSelection(
  selectedId: string | null,
  presets: ModelPreset[],
  customModels: CustomModel[]
): ModelSelection | null {
  if (!selectedId) return null;

  const preset = presets.find((p) => p.id === selectedId);
  if (preset) {
    return preset.configured ? { preset_id: preset.id } : null;
  }

  const custom = customModels.find((c) => c.id === selectedId);
  if (custom) {
    return {
      label: custom.label,
      base_url: custom.base_url,
      api_key: custom.api_key,
      model: custom.model,
      supports_image: custom.supports_image,
    };
  }

  return null;
}
