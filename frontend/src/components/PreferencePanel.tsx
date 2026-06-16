// PreferencePanel.tsx — V4.3 Preference profile viewer + editor.
// Props-driven; owns its own loaded profile state. No any; strict tsc.
import { useState, useEffect, useCallback, useRef } from "react";
import { RefreshCw, X, Plus, Trash2, RotateCcw } from "lucide-react";
import type { PreferenceProfile } from "../hooks/usePngShader";

// ─── Types ────────────────────────────────────────────────────────────────────

type EditablePatch = Partial<
  Pick<
    PreferenceProfile,
    | "enabled"
    | "default_locks"
    | "positive_preferences"
    | "negative_preferences"
    | "score_drop_tolerance_hint"
  >
>;

interface PreferencePanelProps {
  fetchPreferenceProfile: () => Promise<PreferenceProfile>;
  patchPreferenceProfile: (patch: EditablePatch) => Promise<PreferenceProfile | null>;
  rebuildPreferences: () => Promise<PreferenceProfile | null>;
  clearPreferences: () => Promise<void>;
  onClose?: () => void;
}

// ─── Helper ───────────────────────────────────────────────────────────────────

function relativeTime(epoch: number): string {
  const diffMs = Date.now() - epoch * 1000;
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `${diffH}h ago`;
  return new Date(epoch * 1000).toLocaleDateString();
}

// ─── Chip list component (positive / negative preferences) ───────────────────

interface ChipListProps {
  label: string;
  sublabel: string;
  items: string[];
  disabled: boolean;
  onUpdate: (next: string[]) => void;
  chipClass: string;
}

function ChipList({ label, sublabel, items, disabled, onUpdate, chipClass }: ChipListProps) {
  const [inputVal, setInputVal] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleAdd = useCallback(() => {
    const trimmed = inputVal.trim();
    if (!trimmed || items.includes(trimmed)) return;
    onUpdate([...items, trimmed]);
    setInputVal("");
    inputRef.current?.focus();
  }, [inputVal, items, onUpdate]);

  const handleRemove = useCallback(
    (item: string) => {
      onUpdate(items.filter((i) => i !== item));
    },
    [items, onUpdate],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleAdd();
      }
    },
    [handleAdd],
  );

  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-xs font-medium text-[var(--text-primary)]">
        {label}
        <span className="ml-2 text-[var(--text-muted)] font-normal">{sublabel}</span>
      </p>
      <div className="flex flex-wrap gap-1 min-h-[28px]">
        {items.map((item) => (
          <span
            key={item}
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${chipClass}`}
          >
            {item}
            {!disabled && (
              <button
                onClick={() => handleRemove(item)}
                className="opacity-60 hover:opacity-100 transition-opacity ml-0.5"
                title="Remove"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </span>
        ))}
        {items.length === 0 && (
          <span className="text-[11px] text-[var(--text-muted)] italic">None</span>
        )}
      </div>
      {!disabled && (
        <div className="flex gap-1">
          <input
            ref={inputRef}
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Add item…"
            className="flex-1 text-xs px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-primary)]"
          />
          <button
            onClick={handleAdd}
            disabled={!inputVal.trim()}
            className="p-1.5 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] disabled:opacity-40 transition-all"
            title="Add"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}

// ─── PreferencePanel ──────────────────────────────────────────────────────────

export default function PreferencePanel({
  fetchPreferenceProfile,
  patchPreferenceProfile,
  rebuildPreferences,
  clearPreferences,
  onClose,
}: PreferencePanelProps) {
  const [profile, setProfile] = useState<PreferenceProfile | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [mutating, setMutating] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  // ── Load ────────────────────────────────────────────────────────────────────

  const loadProfile = useCallback(async () => {
    setLoadingProfile(true);
    setLoadError(null);
    try {
      const p = await fetchPreferenceProfile();
      setProfile(p);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingProfile(false);
    }
  }, [fetchPreferenceProfile]);

  useEffect(() => {
    void loadProfile();
  }, [loadProfile]);

  // ── Mutation helpers ────────────────────────────────────────────────────────

  const applyPatch = useCallback(
    async (patch: EditablePatch) => {
      if (mutating) return;
      setMutating(true);
      const merged = await patchPreferenceProfile(patch);
      if (merged) setProfile(merged);
      setMutating(false);
    },
    [mutating, patchPreferenceProfile],
  );

  const handleToggleEnabled = useCallback(() => {
    if (!profile) return;
    void applyPatch({ enabled: !profile.enabled });
  }, [profile, applyPatch]);

  const handlePositiveUpdate = useCallback(
    (next: string[]) => {
      void applyPatch({ positive_preferences: next });
    },
    [applyPatch],
  );

  const handleNegativeUpdate = useCallback(
    (next: string[]) => {
      void applyPatch({ negative_preferences: next });
    },
    [applyPatch],
  );

  const handleRebuild = useCallback(async () => {
    if (mutating) return;
    setMutating(true);
    const rebuilt = await rebuildPreferences();
    if (rebuilt) setProfile(rebuilt);
    setMutating(false);
  }, [mutating, rebuildPreferences]);

  const handleClear = useCallback(async () => {
    if (mutating) return;
    setConfirmClear(false);
    setMutating(true);
    await clearPreferences();
    setMutating(false);
    void loadProfile();
  }, [mutating, clearPreferences, loadProfile]);

  // ── Derived ─────────────────────────────────────────────────────────────────

  const isDisabled = loadingProfile || mutating;
  const activeLockKeys = profile
    ? Object.entries(profile.default_locks)
        .filter(([, v]) => v)
        .map(([k]) => k)
    : [];

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-3 p-3 bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-[var(--text-primary)]">
          偏好配置
          <span className="ml-2 text-[var(--text-muted)] font-normal">Preference Profile</span>
        </p>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => void loadProfile()}
            disabled={isDisabled}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] disabled:opacity-40 transition-all"
            title="Refresh / 刷新"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingProfile ? "animate-spin" : ""}`} />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all"
              title="Close / 关闭"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Load error */}
      {loadError && (
        <div className="px-2 py-1.5 bg-red-500/10 border border-red-500/30 rounded-lg">
          <p className="text-[11px] text-red-400">{loadError}</p>
        </div>
      )}

      {/* Loading skeleton */}
      {loadingProfile && !profile && (
        <div className="flex items-center gap-2 py-2">
          <RefreshCw className="w-3.5 h-3.5 text-[var(--text-muted)] animate-spin" />
          <span className="text-xs text-[var(--text-muted)]">Loading…</span>
        </div>
      )}

      {profile && (
        <>
          {/* Enabled toggle */}
          <div className="flex items-center justify-between px-2 py-1.5 bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-color)]">
            <div>
              <p className="text-xs font-medium text-[var(--text-primary)]">
                启用偏好
                <span className="ml-2 text-[var(--text-muted)] font-normal">Enable Preferences</span>
              </p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                {profile.enabled ? "活跃 — 影响新运行的生成" : "已禁用 — 不影响生成"}
                <span className="ml-1.5">{profile.enabled ? "Active" : "Disabled"}</span>
              </p>
            </div>
            <button
              onClick={handleToggleEnabled}
              disabled={isDisabled}
              className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none disabled:opacity-40 ${
                profile.enabled ? "bg-emerald-500" : "bg-[var(--bg-tertiary)]"
              }`}
              role="switch"
              aria-checked={profile.enabled}
            >
              <span
                className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                  profile.enabled ? "translate-x-4" : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {/* Positive preferences */}
          <ChipList
            label="正向偏好"
            sublabel="Positive Preferences"
            items={profile.positive_preferences}
            disabled={isDisabled}
            onUpdate={handlePositiveUpdate}
            chipClass="bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
          />

          {/* Negative preferences */}
          <ChipList
            label="负向偏好"
            sublabel="Negative Preferences"
            items={profile.negative_preferences}
            disabled={isDisabled}
            onUpdate={handleNegativeUpdate}
            chipClass="bg-red-500/15 text-red-400 border border-red-500/30"
          />

          {/* Read-only fields */}
          <div className="flex flex-col gap-1.5 px-2 py-2 bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-color)]">
            <p className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wide">
              只读信息 Read-only
            </p>

            {/* preferred_variant_labels */}
            <div>
              <p className="text-[11px] font-medium text-[var(--text-primary)]">
                偏好变体标签 <span className="text-[var(--text-muted)] font-normal">Preferred Variant Labels</span>
              </p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5 break-words">
                {profile.preferred_variant_labels.length > 0
                  ? profile.preferred_variant_labels.join(", ")
                  : "—"}
              </p>
            </div>

            {/* active locks */}
            <div>
              <p className="text-[11px] font-medium text-[var(--text-primary)]">
                默认锁定 <span className="text-[var(--text-muted)] font-normal">Active Default Locks</span>
              </p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5 break-words">
                {activeLockKeys.length > 0 ? activeLockKeys.join(", ") : "—"}
              </p>
            </div>

            {/* score_drop_tolerance_hint */}
            <div>
              <p className="text-[11px] font-medium text-[var(--text-primary)]">
                分数容差提示 <span className="text-[var(--text-muted)] font-normal">Score Drop Tolerance Hint</span>
              </p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                {profile.score_drop_tolerance_hint}
              </p>
            </div>

            {/* summary_source_event_count */}
            <div>
              <p className="text-[11px] font-medium text-[var(--text-primary)]">
                事件数 <span className="text-[var(--text-muted)] font-normal">Source Event Count</span>
              </p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                {profile.summary_source_event_count}
              </p>
            </div>

            {/* updated_at */}
            <div>
              <p className="text-[11px] font-medium text-[var(--text-primary)]">
                更新时间 <span className="text-[var(--text-muted)] font-normal">Updated At</span>
              </p>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                {relativeTime(profile.updated_at)}{" "}
                <span className="opacity-60">({new Date(profile.updated_at * 1000).toISOString()})</span>
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            {/* Rebuild */}
            <button
              onClick={() => void handleRebuild()}
              disabled={isDisabled}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--border-hover)] disabled:opacity-40 transition-all"
              title="Rebuild profile from recent events / 从最近事件重建配置"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              重建 Rebuild
            </button>

            {/* Clear (with confirm step) */}
            {!confirmClear ? (
              <button
                onClick={() => setConfirmClear(true)}
                disabled={isDisabled}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-muted)] hover:text-red-400 hover:border-red-500/40 disabled:opacity-40 transition-all"
                title="Clear all preference data / 清除所有偏好数据"
              >
                <Trash2 className="w-3.5 h-3.5" />
                清除 Clear
              </button>
            ) : (
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-red-400">确认清除？</span>
                <button
                  onClick={() => void handleClear()}
                  disabled={isDisabled}
                  className="px-2.5 py-1 text-xs rounded-lg bg-red-500/20 border border-red-500/40 text-red-400 hover:bg-red-500/30 disabled:opacity-40 transition-all"
                >
                  确认
                </button>
                <button
                  onClick={() => setConfirmClear(false)}
                  className="px-2.5 py-1 text-xs rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all"
                >
                  取消
                </button>
              </div>
            )}

            {/* Mutating indicator */}
            {mutating && (
              <span className="ml-auto text-[11px] text-[var(--text-muted)] flex items-center gap-1">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Saving…
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
