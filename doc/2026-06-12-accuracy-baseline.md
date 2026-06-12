# 准确率基线（优化前，2026-06-12）

- 样本数: 4
- avg_final_score: 0.8242
- PASS(≥0.85): 2 / ACCEPTABLE(≥0.55): 2 / FAIL: 0

## 每样本分数

| sample | run_id | status | final_score | selected_source |
|--------|--------|--------|-------------|-----------------|
| box-gradient | run_09ac289a | completed | 0.9870 | decompose |
| circle | run_ca8629f0 | completed | 0.9209 | decompose |
| glow-ring | run_6b72a795 | completed | 0.6365 | baseline |
| roundedbox-vignette | run_b9182702 | completed | 0.7525 | decompose |

## 备注

- decompose 候选在 3/4 样本上被选中，表现优异
- glow-ring 样本 decompose 候选得分仅 0.2546（poor），回退到 baseline
- 残差增层在 glow-ring 上尝试但被拒绝（score 从 0.6365 降到 0.3181）
- 合成样本分数偏乐观，需补充真实 PNG 样本
