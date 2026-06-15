// ModelSelectorPanel.tsx
// Pick the LLM model used for this run: a preset (from backend /api/models) or
// a user-defined custom model (persisted in localStorage). Unconfigured presets
// (no API key in the backend .env) are shown disabled.
import { useState, useCallback } from "react";
import { Boxes, Plus, Trash2 } from "lucide-react";
import type { ModelControls } from "../hooks/useModels";

interface Props {
  controls: ModelControls;
  loading: boolean;
}

const emptyForm = {
  label: "",
  base_url: "",
  api_key: "",
  model: "",
  supports_image: true,
};

export default function ModelSelectorPanel({ controls, loading }: Props) {
  const { presets, customModels, selectedId, setSelectedId, addCustomModel, removeCustomModel, error } =
    controls;
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptyForm);

  const canSubmit =
    form.base_url.trim() !== "" && form.model.trim() !== "" && form.api_key.trim() !== "";

  const handleAdd = useCallback(() => {
    if (!canSubmit) return;
    addCustomModel({
      label: form.label.trim() || form.model.trim(),
      base_url: form.base_url.trim(),
      api_key: form.api_key.trim(),
      model: form.model.trim(),
      supports_image: form.supports_image,
    });
    setForm(emptyForm);
    setShowForm(false);
  }, [canSubmit, form, addCustomModel]);

  const inputCls =
    "w-full text-xs px-2 py-1.5 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-emerald-400/60";

  return (
    <div className="flex flex-col gap-2 px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg min-w-0">
      <div className="flex items-center gap-3 min-w-0">
        <Boxes className="w-4 h-4 flex-shrink-0 text-[var(--accent-primary)]" />
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-[var(--text-primary)] leading-tight whitespace-nowrap">
            LLM 模型
            <span className="ml-2 text-[var(--text-muted)] font-normal">Model</span>
          </p>
          <p className="text-[11px] text-[var(--text-muted)] leading-tight truncate">
            选择运行所用模型，或自定义新增
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2 min-w-0">
        <select
          value={selectedId ?? ""}
          onChange={(e) => setSelectedId(e.target.value)}
          disabled={loading}
          className="flex-1 min-w-0 text-xs px-2 py-1.5 rounded-md bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] disabled:opacity-40 focus:outline-none focus:border-emerald-400/60"
        >
          {presets.length === 0 && customModels.length === 0 && (
            <option value="">（无可用模型）</option>
          )}
          {presets.length > 0 && (
            <optgroup label="预设 Presets">
              {presets.map((p) => (
                <option key={p.id} value={p.id} disabled={!p.configured}>
                  {p.label}
                  {p.configured ? "" : "（未配置）"}
                </option>
              ))}
            </optgroup>
          )}
          {customModels.length > 0 && (
            <optgroup label="自定义 Custom">
              {customModels.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <button
          onClick={() => setShowForm((v) => !v)}
          disabled={loading}
          title="新增自定义模型"
          className="flex-shrink-0 flex items-center gap-1 px-2 py-1.5 text-xs rounded-md bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all disabled:opacity-40"
        >
          <Plus className="w-3.5 h-3.5" /> 新增
        </button>
      </div>

      {/* Selected custom model: allow removal */}
      {selectedId && customModels.some((c) => c.id === selectedId) && (
        <div className="flex items-center justify-end">
          <button
            onClick={() => removeCustomModel(selectedId)}
            disabled={loading}
            className="flex items-center gap-1 text-[11px] text-red-400/80 hover:text-red-400 transition-all disabled:opacity-40"
          >
            <Trash2 className="w-3 h-3" /> 删除此自定义模型
          </button>
        </div>
      )}

      {error && <p className="text-[11px] text-red-400">加载模型列表失败：{error}</p>}

      {showForm && (
        <div className="flex flex-col gap-2 pt-2 border-t border-[var(--border-color)]">
          <div className="grid grid-cols-2 gap-2">
            <input
              className={inputCls}
              placeholder="显示名 Label（可选）"
              value={form.label}
              onChange={(e) => setForm((f) => ({ ...f, label: e.target.value }))}
              disabled={loading}
            />
            <input
              className={inputCls}
              placeholder="模型ID Model *（如 qwen-3.7-plus）"
              value={form.model}
              onChange={(e) => setForm((f) => ({ ...f, model: e.target.value }))}
              disabled={loading}
            />
          </div>
          <input
            className={inputCls}
            placeholder="Base URL *（OpenAI 兼容 /v1）"
            value={form.base_url}
            onChange={(e) => setForm((f) => ({ ...f, base_url: e.target.value }))}
            disabled={loading}
          />
          <input
            className={inputCls}
            type="password"
            placeholder="API Key *"
            value={form.api_key}
            onChange={(e) => setForm((f) => ({ ...f, api_key: e.target.value }))}
            disabled={loading}
          />
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={form.supports_image}
                onChange={(e) => setForm((f) => ({ ...f, supports_image: e.target.checked }))}
                disabled={loading}
                className="accent-emerald-500"
              />
              <span className="text-xs text-[var(--text-secondary)]">支持图片输入</span>
            </label>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setShowForm(false);
                  setForm(emptyForm);
                }}
                disabled={loading}
                className="px-3 py-1.5 text-xs rounded-md text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all disabled:opacity-40"
              >
                取消
              </button>
              <button
                onClick={handleAdd}
                disabled={loading || !canSubmit}
                className="px-3 py-1.5 text-xs rounded-md bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium shadow-sm shadow-emerald-500/25 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
              >
                添加
              </button>
            </div>
          </div>
          <p className="text-[11px] text-[var(--text-muted)]">
            自定义模型（含 API Key）仅保存在本地浏览器，并随请求发送给后端。
          </p>
        </div>
      )}
    </div>
  );
}
