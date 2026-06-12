# 准确率基线（优化前，2026-06-12）

## 合成样本基线（4 samples）

- avg_final_score: 0.8242
- PASS(≥0.85): 2 / ACCEPTABLE(≥0.55): 2 / FAIL: 0

## 真实样本基线（10 samples，含 6 个真实 PNG）

- avg_final_score: 0.6757
- PASS(≥0.85): 2 / ACCEPTABLE(≥0.55): 4 / FAIL: 4

## 每样本分数

| sample | final_score | selected_source | colors | band | 备注 |
|--------|-------------|-----------------|--------|------|------|
| box-gradient | 0.9870 | decompose | 109 | PASS | 合成 |
| circle | 0.9209 | decompose | 32 | PASS | 合成 |
| 下载 | 0.7335 | cv | 1240 | ACCEPTABLE | 残差+2 |
| roundedbox-vignette | 0.7525 | decompose | 40 | ACCEPTABLE | 合成 |
| 81ebfa1e... | 0.7957 | decompose | 26 | ACCEPTABLE | |
| glow-ring | 0.6365 | baseline | 32 | ACCEPTABLE | 合成 |
| images (1) | 0.5159 | cv | 790 | FAIL | 照片类 |
| 下载 (1) | 0.5163 | cv | 731 | FAIL | 残差+4 |
| 下载 (2) | 0.5135 | cv | 856 | FAIL | 照片类 |
| images | 0.3849 | cv | 2545 | FAIL | 照片类 |

## 分析

### 候选分布
- decompose 被选中: 4/10 (合成样本 + 81ebfa1e)
- cv 被选中: 5/10 (真实照片类样本)
- baseline 被选中: 1/10 (glow-ring)

### 问题样本特征
- 高颜色数 (colors > 700): decompose 被跳过 (photo_like_score > 0.7)
- 照片类样本 cv 候选得分普遍在 0.38-0.52 之间
- 残差增层对部分样本有效 (下载: 0.71→0.73, 下载(1): 0.45→0.52)

### 待优化方向
1. 照片类样本的 cv 候选质量
2. photo_like 阈值调优 (当前 0.7 可能过低)
3. 残差增层的接受门槛 (ACCEPT_MIN_DELTA=0.003)
