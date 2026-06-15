import { useCallback, useEffect, useMemo, useState } from "react";
import {
  loadCustomModels,
  saveCustomModels,
  newCustomModelId,
  toModelSelection,
  type CustomModel,
  type ModelPreset,
  type ModelSelection,
} from "../lib/models";

const API_BASE = import.meta.env.VITE_API_BASE || "";

export interface ModelControls {
  presets: ModelPreset[];
  customModels: CustomModel[];
  selectedId: string | null;
  loading: boolean;
  error: string | null;
  setSelectedId: (id: string) => void;
  addCustomModel: (model: Omit<CustomModel, "id">) => void;
  removeCustomModel: (id: string) => void;
  selection: ModelSelection | null;
}

export function useModels(): ModelControls {
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [customModels, setCustomModels] = useState<CustomModel[]>(() => loadCustomModels());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch presets once; default the selection to the backend-marked default
  // (first configured preset).
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/models`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: { presets: ModelPreset[] }) => {
        if (cancelled) return;
        const list = data.presets ?? [];
        setPresets(list);
        setSelectedId((prev) => {
          if (prev) return prev;
          const def = list.find((p) => p.default && p.configured) ?? list.find((p) => p.configured);
          return def?.id ?? null;
        });
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const addCustomModel = useCallback((model: Omit<CustomModel, "id">) => {
    const created: CustomModel = { ...model, id: newCustomModelId() };
    setCustomModels((prev) => {
      const next = [...prev, created];
      saveCustomModels(next);
      return next;
    });
    setSelectedId(created.id);
  }, []);

  const removeCustomModel = useCallback((id: string) => {
    setCustomModels((prev) => {
      const next = prev.filter((c) => c.id !== id);
      saveCustomModels(next);
      return next;
    });
    setSelectedId((prev) => (prev === id ? null : prev));
  }, []);

  const selection = useMemo(
    () => toModelSelection(selectedId, presets, customModels),
    [selectedId, presets, customModels]
  );

  return {
    presets,
    customModels,
    selectedId,
    loading,
    error,
    setSelectedId,
    addCustomModel,
    removeCustomModel,
    selection,
  };
}
