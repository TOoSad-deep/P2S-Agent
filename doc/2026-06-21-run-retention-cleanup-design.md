# Run 保留与清理 + 数据根 env 覆盖 — 设计文档

- 日期:2026-06-21
- 状态:已批准设计,待实现
- 关联记忆:[[macos-tcc-eperm-data-root]](EPERM/TCC 坑)、存储架构(`backend/test_results/`)

## 1. 背景与问题

后端每次 pipeline 运行都会在 `backend/test_results/<YYYY-MM-DD>_png-shader_<label>_<run_id>/`
下写一个**自包含**的 run 目录(输入图、`candidates/*_render.png`/`*_webgl.png`、各阶段
渲染、judge、metrics、`selected_shader.glsl` 等)。当前**没有任何自动清理 / TTL /
容量上限**,run 目录无限堆积。

实测(2026-06-21):`test_results` 占用 **1.5 GB**、**5914 个** run 目录、**131473** 张 PNG,
约 10 天数据 → 日增 ~150MB / ~600 dirs。

同时数据根硬编码在 `~/Documents` 下,踩 macOS TCC 坑(`uvicorn --reload` worker 丢
`~/Documents` 访问权 → EPERM 500)。

本设计交付三件事:
1. 数据根支持环境变量覆盖(可移出 `~/Documents`);
2. 可配置的保留策略(数量 / 天数 / 容量,可组合);
3. 后端启动时自动清理一次 + 一个默认 dry-run 的 CLI 清理工具。

## 2. 关键决策(已与用户确认)

| 维度 | 决策 |
|------|------|
| 保留策略 | 三类(数量/天数/容量)全做成可配,**默认只启用"保留最近 N 个"** |
| 触发时机 | **后端启动时执行一次**(无后台调度器) |
| 范围 | **仅 run 目录**;绝不动 draw_sessions / variant_groups / fusions / preferences 子系统 |
| 数据根 | 新增 `P2S_RESULTS_ROOT` env 覆盖,**不设时行为零变化** |
| 默认值 | `ENABLED=true`、`MAX_RUNS=1000`、age/bytes 关闭 |
| run_index | 删 run 目录时**同步剔除**其 `run_index.jsonl` 记录(血缘树不留幽灵节点) |

> ⚠️ 首次影响:现有 5914 个 run,默认 `MAX_RUNS=1000` 且启动即清理 →
> **首次重启后端会删掉约 4900 个最旧的"可删类" run**(用户已知晓并批准)。

## 3. 架构与组件

### 3.1 数据根 env 覆盖

唯一改动点 `backend/p2s_agent/core/pipeline/artifacts.py`:

```python
import os
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("P2S_RESULTS_ROOT") or
    (Path(__file__).resolve().parents[3] / "test_results")
)
```

所有派生常量(`variant_groups.py` 的 `_DEFAULT_GROUPS_DIR`、`draw_sessions.py`、
`preferences.py`、`fusion_plans.py`、`run_index.py` 的 `_DEFAULT_INDEX_PATH` 等)
都从 `DEFAULT_RESULTS_ROOT` 在 import 期推导 → 改一处即全覆盖。env 在进程启动前设置,
import 期即生效;不引入运行时动态切换(YAGNI)。

### 3.2 保留核心模块 `p2s_agent/orchestration/retention.py`(纯函数为主)

- `@dataclass RetentionPolicy`:`enabled: bool`、`max_runs: int | None`、
  `max_age_days: int | None`、`max_bytes: int | None`(None/<=0 表示该项关闭)。
- `policy_from_env(env=os.environ) -> RetentionPolicy`:解析
  `P2S_RETENTION_ENABLED`(默认 true)、`P2S_RETENTION_MAX_RUNS`(默认 1000)、
  `P2S_RETENTION_MAX_AGE_DAYS`(默认 0=关)、`P2S_RETENTION_MAX_BYTES`(默认空=关;
  接受 `2GB`/`500MB`/纯字节数,`parse_size()` 辅助函数)。
- `@dataclass CleanupPlan`:`delete: list[DeletionTarget]`、`freed_bytes: int`、
  `kept_count: int`、`skipped_no_record: list[str]`、`protected: list[str]`。
- `@dataclass DeletionTarget`:`run_id`、`run_dir: Path`、`size_bytes: int | None`、
  `reason: str`(如 `"beyond max_runs"`/`"older than 30d"`/`"over size cap"`)。
- `plan_cleanup(root, policy, *, now, index_records, compute_sizes=False) -> CleanupPlan`
  **纯函数**,不触盘删除(只在 `compute_sizes` 或 size 策略时 `du` 目录)。
- `apply_cleanup(plan, *, index_path) -> CleanupResult`:对每个 target `shutil.rmtree`,
  成功后从 `run_index.jsonl` 剔除该记录(复用 run_index 的 compaction 重写);
  单目录失败只记日志、继续。

### 3.3 启动钩子(`backend/app/...` 应用工厂 / lifespan)

```python
policy = policy_from_env()
if policy.enabled:
    try:
        plan = plan_cleanup(results_root, policy, now=utcnow(),
                            index_records=load_run_index())
        result = apply_cleanup(plan, index_path=run_index_path)
        log.info("retention: deleted %d runs, freed %.1f MB",
                 len(result.deleted), result.freed_bytes / 1e6)
    except Exception:
        log.exception("retention cleanup failed; continuing startup")
```

清理失败**绝不阻断**后端启动。

### 3.4 CLI `python -m p2s_agent.tools.cleanup_runs`

- **默认 dry-run**:打印将删 run 表(run_id / 日期 / 大小 / 原因)+ 合计释放字节,不动盘。
- `--apply`:真正执行。
- `--max-runs N` / `--max-age-days D` / `--max-bytes SIZE` / `--root PATH`:临时覆盖策略。
- `--include-orphans`:额外删除"磁盘上有目录但 run_index 无记录"的孤儿 run(按 mtime + age 判定);
  默认**不**删孤儿(安全)。

## 4. 安全不变量(可删判定:全部满足才删)

一个 run 目录仅当**同时满足**以下条件才进入删除候选:

1. **是 run 目录**:匹配 `*_png-shader_*` 且位于数据根下;
2. **有终态 index 记录**:`run_index` 中存在该 run 且 `status` 为终态
   (非 `running`/`queued`/`acquired`/任何活跃态);**无 index 记录的目录默认跳过**
   (记入 `skipped_no_record`,仅 CLI `--include-orphans` 可删);
3. **独立 run**:`variant_group_id`/`draw_session_id`/`fusion_id`/`base_run_id` 均为空,
   且不被任一 variant_group / draw_session / fusion 记录引用 → 不碰会话子系统成员;
4. **无存活子代**:没有任一**存活**(未被本次删除)的 run 把它列为
   `parent_run_id`/`root_run_id`/`base_run_id`/`source_run_ids`。
   实现为"从叶子向上删":先按血缘建反向引用集,只有引用计数归零的 run 才可删;
5. **落在保留窗口外**(见 §5)。

> 不变量 4 的依据:分支 run 在创建时读父 run 的 `reference_input.png` 作种子
> ([sessions.py:340]);虽然每个 run 之后会写自己的 `reference_input.png`、serving 也
> 只读自身目录,但血缘树展示仍会引用父节点。保留祖先避免画布树出现指向缺失目录的节点。

## 5. 组合策略语义

按"并集 + 容量兜底"计算可删集(仅在满足 §4 不变量的候选内):

1. **max_runs**:候选按 `created_at` 降序,保留最新 N 个,其余标记删除(`reason="beyond max_runs"`);
2. **max_age_days**:`created_at` 早于 `now - D 天` 的候选标记删除(`reason="older than Dd"`);
   与 1 取并集(满足任一即删);
3. **max_bytes**(兜底):若开启,在 1+2 删除后估算存活总字节;若仍超 cap,
   从最旧存活候选继续往下删,直到达标(`reason="over size cap"`)。仅此步需 `du` 目录大小。

默认仅 max_runs 开启 → 启动清理**无需遍历目录大小**,快。

## 6. 测试策略(TDD)

`backend/tests/unit/test_retention.py` — `plan_cleanup` 纯函数全分支,用 `tmp_path`
造 run 目录 + 构造 `RunLineageRecord` 列表:

- max_runs:保留最新 N、删其余;N >= 总数时零删除;
- max_age_days:按 `now` 注入,边界(恰好 D 天)不删;
- max_bytes:估算后兜底删到达标;
- 不变量 2:非终态/活跃 run 不删;无 index 记录目录进 `skipped_no_record` 不删;
- 不变量 3:有 `variant_group_id`/`fusion_id`/被会话引用的 run 不删;
- 不变量 4:父 run 有存活子代时受保护;子代也被删时父可删(叶子向上);
- `policy_from_env`:各默认值、`ENABLED=false`、`parse_size("2GB")` 等;
- `apply_cleanup`(集成,tmp_path):真删目录 + run_index 记录被剔除。

门禁:`python3 -m pytest -q`(backend/)全绿;CLI dry-run 手验输出。
前端不涉及(无 UI 改动)。

## 7. 显式不做(YAGNI)

- 后台周期性清理 / 调度器(只在启动跑一次);
- 运行时动态切换数据根(env 仅 import 期生效);
- 删除 draw_sessions/variant_groups/fusions/preferences 子系统数据;
- 软删除 / 回收站 / 撤销(删即 rmtree;CLI dry-run 作为预览闸门)。

## 8. 实测发现(2026-06-21,在真实 `backend/test_results` 上 dry-run)

实现完成后用 CLI dry-run 跑真实数据,结论与 §2 的"首次删约 4900 个"**预测不符**,记录如下:

| 项 | 实测 |
|----|------|
| 磁盘 run 目录 | 5914 |
| `run_index.jsonl` 有记录的 run | **仅 148**(另 27 条记录无 run_dir) |
| 无记录(orphan)目录 | **5766** |
| 其中被会话子系统 JSON 引用 | 58 |
| 既不在 index 也不被会话引用 | **5667** |

**根因**:`run_index` 是项目中途(M3-1)才引入,且大量 orphan 目录其实是**测试污染** ——
`tests/unit/test_graph.py` 多处 `run_png_shader_pipeline(..., run_id="my_custom_run" /
"force_fft_run" / "seedtest" / "strategy_reader_*")` **未把 root 指向 tmp_path**,
直接写进了真实数据根;每天跑 pytest 就沉淀一批,跨 10 天累积约 5600 个。

**对本特性的影响(均为安全表现,符合不变量设计)**:
1. **默认 `MAX_RUNS=1000` 的启动清理会删 ~0 个** —— 148 个有记录的 run 远少于 1000,
   全部保留;orphan 因"无终态 index 记录"被不变量 2 跳过。即**首次重启不会误删**,
   比预测更安全,但也**不会自动释放那 1.5GB**。
2. 真正占空间的是 5667 个无引用 orphan(测试污染),只能经 CLI
   `--include-orphans --max-age-days N` 清理。实测
   `--max-runs 1000 --include-orphans --max-age-days 3` dry-run → 4246 dir / **722.5MB**,
   4 个被血缘保护、会话引用的 58 个被 `session_referenced_run_ids` 保护(已加固)。

**加固(本次新增,超出原 §4)**:`orphan_targets` 增加 `protected_run_ids` 参数 +
`session_referenced_run_ids(root)`(扫描 variant_groups/draw_sessions/fusions JSON 的
`run_` 引用),CLI `--include-orphans` 自动排除会话引用的 orphan,避免误删 record-less
的会话子代。

**注意(语义耦合)**:`--max-age-days` **同时**作用于"有记录的 run"和 orphan ——
若只想清 orphan 而不动有记录的 run,需保持 `--max-runs` 足够大并接受 age 也会删旧的有记录 run;
如需解耦可后续加独立的 `--orphan-age-days`(当前 YAGNI 未做)。

**后续(独立于本特性)**:测试隔离 bug(测试写真实数据根)应单独修复 —— 已作为后台任务标记。
建议把 `test_graph.py` 里相关调用改为 `root=tmp_path` / monkeypatch `DEFAULT_RESULTS_ROOT`。
