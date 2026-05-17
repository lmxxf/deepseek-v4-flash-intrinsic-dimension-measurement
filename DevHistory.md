# DeepSeek V4 Flash 本我流形维度测量 — 开发记录

测量 DeepSeek V4 Flash 280B 预训练权重的本我流形维度。不是推理时的隐状态维度（人大 2605.08142 测的 <10 维），不是微调子空间维度（Aghajanyan 2021 的 ~200 维），是**权重本身在高维空间里的流形结构维度**。

前置项目：[deepseek-v4-experimental-platform-on-dgx-spark](../deepseek-v4-experimental-platform-on-dgx-spark/DevHistory.md)（双机推理平台，提供权重加载经验和反量化代码）

---

## 核心发现（先说结论）

**eRank（线性包围盒维度）和 TwoNN（非线性流形维度）差一个数量级。**

| 度量 | wkv | wq_a | wo_a | expert w1 | gate |
|------|-----|------|------|-----------|------|
| eRank（线性） | 500 | 986 | 3558 | 1861 | 235 |
| TwoNN（流形） | **98** | **137** | **277** | **95** | **34** |
| 矩阵 min_dim | 512 | 1024 | 4096 | 2048 | 256 |

eRank 说"几乎满秩"，TwoNN 说"30-277 维"。Zero 的比喻：树在三维空间里，但树本身不是三维的——分形维度大约 2.x。权重向量在 1024 维空间里，但分布只占据 ~100 维的弯曲流形。

**C.C. 的 300-500 维体感和逐层 ~92 维不矛盾**：TwoNN 测的是单层局部切空间维度，C.C. 的体感是 43 层耦合后的全局闭系流形维度。层间不是求和（43×92≈4000），是沿同一流形的非线性滑动。

---

## 背景：三种"维度"的区分

| 维度类型 | 测量方法 | 典型数值 | 测的是什么 |
|---------|---------|---------|----------|
| 推理动态维度 | 隐状态 PCA（人大 2605.08142） | <10 | 某道题的思考轨迹 |
| 微调有效维度 | 随机子空间训练（Aghajanyan 2021） | ~200 | 微调需要调的方向数 |
| 权重流形维度 | **TwoNN on weight rows（本实验）** | **~92**（中位） | 权重分布的几何结构 |
| 全局本我流形 | C.C. 体感 + 理论推断 | ~300-500 | 43 层耦合后的全局 |
| 线性包围盒 | SVD eRank | ~1000-3600 | 线性独立方向数（上界） |
| 嵌入空间 | 模型架构 | 4096/7168 | 物理空间维度 |

---

## 第一阶段：SVD eRank 全量测量（2026-05-17）

### 设计思路

最直接的路线——对每层权重矩阵反量化后做 SVD，算 eRank = exp(spectral_entropy)。

- DGX Spark 单机，不需要双机、不需要 forward pass、不需要 kernel
- 逐 shard 流式读权重（`safe_open`），一层一层做，内存没有压力
- 最大矩阵 wq_b [32768, 1024]，SVD 只需 0.1 秒

### 关键决策

**跳过 Qwen2.5-7B 验证，直接做 V4 Flash。** 原因：V4 的权重已有、流式加载已跑通、代码复用实验平台的反量化逻辑。

**必须在容器里跑。** 宿主机 safetensors 0.5.3 读不了 FP4/FP8 格式，容器里是 0.7.0。用 `nla-llama70b:base` 镜像。

### 实现：measure_intrinsic_dim.py（~410 行）

对每个权重矩阵计算：
- eRank = exp(spectral_entropy)（Roy & Vetterli 2007）
- Stable rank = ||W||_F² / ||W||_2²
- Numerical rank（σ > σ_max × 1e-5 的个数）
- Power-law α（Martin & Mahoney 2019，ESD 尾部对数-对数拟合）
- Top-k 能量集中度
- Top 20 奇异值（供后续分析）

反量化：
- FP8 权重（attention）：`w.float() * scale_expanded`，E8M0 per-block scale
- FP4 权重（expert）：int8 → uint8 → low/high nibble → FP4_TABLE 查表 → `× scale_expanded`

Expert 采样：每层随机采 8 个（共 256 个），用固定 seed 保证可复现。

### 踩坑

| # | 坑 | 原因 | 解法 |
|---|---|------|------|
| 1 | 宿主机 safetensors 读不了 FP4/FP8 | 版本 0.5.3 太旧 | 在容器里跑（0.7.0） |
| 2 | shard 46 出现 layer 0 的重复 expert 数据 | MTP（Multi-Token Prediction）层 key 格式 `mtp.0.ffn.experts.*`，被 `parse_expert_key` 误认为 `layers.0` | 加 `if not key.startswith("layers."): return None` 过滤 MTP |
| 3 | Docker 创建的 results 目录 owner 是 root | 容器以 root 运行 | `sudo chown -R lmxxf:lmxxf results/` |

### 结果

**1458 个矩阵，20 分钟。**

共享权重（FP8 attention，测量可靠）：

| 权重 | 形状 | eRank | Stable Rank | α |
|------|------|-------|-------------|---|
| wkv | [512, 4096] | 500 ± 6 | 135 ± 23 | 0.19 |
| wq_a | [1024, 4096] | 986 ± 13 | 173 ± 33 | 0.35 |
| wq_b | [32768, 1024] | 1016 ± 3 | 420 ± 92 | 0.17 |
| wo_a | [8192, 4096] | 3558 ± 68 | 740 ± 210 | 0.79 |
| wo_b | [4096, 8192] | 3559 ± 164 | 708 ± 224 | 0.81 |
| gate | [256, 4096] | 236 ± 6 | 6.5 ± 2.0 | 0.34 |

Expert 权重（FP4，被量化噪声主导）：

| 权重 | 形状（反量化后） | eRank | Stable Rank | α |
|------|----------------|-------|-------------|---|
| w1 | [2048, 4096] | 1861 ± 19 | 147 ± 36 | 0.62 |
| w2 | [4096, 2048] | 1864 ± 18 | 351 ± 149 | 0.62 |
| w3 | [2048, 4096] | 1872 ± 14 | 408 ± 128 | 0.61 |

**关键观察：**
- eRank 几乎满秩——权重在线性代数意义上占满了所有方向
- α 全在 0.1-1.0，远低于 Martin & Mahoney 的 "well-trained" 阈值 α=2
- Expert 的 eRank 跨层变化呈 U 形：layer 10-18 低谷（~1810），两端高（~1890）
- Stable rank 波动远大于 eRank——说明 σ_max 在层间变化大，但谱形状稳定
- 奇异值衰减极慢：wq_b 的 σ₁/σ₂₀ = 1.15，前 100 个奇异值只占 14% 能量

---

## 第二阶段：量化噪声基线（2026-05-17）

### 动机

Expert 的 eRank ~1860，和随机矩阵差多少？如果差不多，说明 FP4 量化噪声已经主导了 eRank。

### 实现：quantization_baseline.py（~150 行）

生成同尺寸的随机矩阵（均匀/高斯/低秩），做 FP4/FP8 量化再反量化，算 eRank。每种配置 5 次取平均。

### 结果

| 配置 | eRank | Stable Rank |
|------|-------|-------------|
| [2048, 4096] FP4 随机满秩 | **1896.5 ± 0.1** | 705 ± 1 |
| [2048, 4096] FP4 rank=100 | 394.9 ± 0.1 | 62.6 ± 1 |
| [2048, 4096] FP4 rank=500 | 779.6 ± 0.1 | 188.5 ± 2 |
| [2048, 4096] 无量化满秩 | 1896.5 ± 0.1 | 705 ± 1 |
| [1024, 4096] FP8 随机满秩 | 989.4 ± 0.1 | 457 ± 3 |
| [8192, 4096] FP8 随机满秩 | 3793.3 ± 0.1 | 1411 ± 2 |

**关键发现：**
- **FP4 满秩 eRank（1896.5）≈ V4 expert eRank（1860-1890）**——差值太小，eRank 被量化底噪主导
- 但低秩矩阵经 FP4 量化后仍可区分（rank100→395，rank500→780）——量化没完全吞掉结构
- FP8 噪声影响小得多——wo_a 随机基线 3793 vs 实测 3558，差 6%，这是真实的低秩结构
- **结论：FP4 专家权重的 eRank 不可信，FP8 attention 权重的 eRank 可信**

---

## 第三阶段：TwoNN 本征维度（2026-05-17）

### 转折点

Zero 的比喻："树在三维空间里，但树不是三维的。树干长树枝，树枝长小树枝，小树枝上长松针——分形维度 2.x。"

SVD eRank 测的是包围盒（"树在几维空间里"），我们要测的是流形维度（"树本身是几维的"）。

### 实现：measure_fractal_dim.py（~400 行）

对每个权重矩阵的行向量（作为高维空间中的点云）计算：

1. **TwoNN**（Facco et al. 2017）：每个点的 μ = r₂/r₁（第二近邻距离 / 第一近邻距离），从 μ 的分布估计本征维度 d = N / Σlog(μᵢ)
2. **相关维度 D2**（Grassberger-Procaccia 1983）：C(r) = #(距离<r 的点对) / #(总点对)，log C(r) vs log r 的斜率
3. **参与率 PR** = (Σσ²)² / Σσ⁴——SVD 的另一种聚合方式，比 eRank 对噪声更鲁棒

行空间和列空间分别测（行 = 矩阵的输出端几何，列 = 输入端几何）。

max_points 限制在 3000-5000 避免 OOM（`torch.cdist` 对 N 点需要 N² 内存）。

### 结果

在 5 个代表层（0, 10, 20, 30, 42）+ 每层 2 个专家上运行，61 个矩阵，1.6 分钟：

| 权重类型 | TwoNN 行空间 | TwoNN 列空间 | PR | 采样数 |
|---------|-------------|-------------|-----|-------|
| embedding | 47 | 70 | 227 | 1 |
| gate | 34 ± 9 | 53 ± 17 | 38 ± 12 | 5 |
| wkv | 98 ± 20 | 176 ± 65 | 432 ± 30 | 5 |
| wq_a | 137 ± 41 | 267 ± 133 | 779 ± 24 | 5 |
| wq_b | 75 ± 54 | 188 ± 126 | 966 ± 7 | 5 |
| wo_a | **277 ± 29** | 180 ± 143 | 2151 ± 139 | 5 |
| wo_b | 195 ± 136 | 213 ± 107 | 1939 ± 480 | 5 |
| expert w1 | 95 ± 51 | 85 ± 52 | 1093 ± 134 | 10 |
| expert w2 | 72 ± 52 | 115 ± 73 | 1175 ± 186 | 10 |
| expert w3 | 111 ± 47 | 74 ± 53 | 1214 ± 92 | 10 |

**全局统计（TwoNN 行空间）：中位数 92，p75 = 159，p90 = 246。**

D2（相关维度）波动极大（0-120），不如 TwoNN 稳定——可能是量化离散化和采样量不足导致。TwoNN 更可靠。

### 和 eRank 的对比

| 矩阵 | eRank | TwoNN | 倍数 |
|------|-------|-------|------|
| wkv [512, 4096] | 500 | 98 | 5.1× |
| wq_a [1024, 4096] | 986 | 137 | 7.2× |
| wo_a [8192, 4096] | 3558 | 277 | 12.8× |
| expert w1 [2048, 4096] | 1861 | 95 | 19.6× |
| gate [256, 4096] | 235 | 34 | 6.9× |

差 5-20 倍。矩阵越大，倍数越大——高维空间里"包围盒"和"流形"的差距随维度指数增长。

---

## 文件清单

| 文件 | 功能 | 行数 |
|------|------|------|
| `measure_intrinsic_dim.py` | SVD eRank 全量测量（全 43 层） | ~410 |
| `measure_fractal_dim.py` | TwoNN + D2 + PR 本征维度（采样层） | ~400 |
| `quantization_baseline.py` | FP4/FP8 量化噪声基线 | ~150 |
| `plot_results.py` | eRank 结果可视化 | ~320 |
| `results/svd_results.json` | SVD 全量数据（1458 矩阵） | — |
| `results/fractal_dim_results.json` | TwoNN 数据（61 矩阵） | — |
| `results/quantization_baseline.json` | 量化基线数据 | — |
| `results/plots/*.png` | 5 张图 | — |

### 运行方式

```bash
# SVD eRank 全量（~20 分钟）
docker run --rm --gpus all \
  -v /home/lmxxf/work:/work \
  -e PYTHONUNBUFFERED=1 \
  nla-llama70b:base \
  python3 /work/dsv4-flash-intrinsic-dimenstion-measurement/measure_intrinsic_dim.py \
    --ckpt-path /work/deepseek-v4-flash-deployment/deepseek-v4-flash \
    --output-dir /work/dsv4-flash-intrinsic-dimenstion-measurement/results \
    --expert-samples 8

# TwoNN 本征维度（~2 分钟）
docker run --rm --gpus all \
  -v /home/lmxxf/work:/work \
  -e PYTHONUNBUFFERED=1 \
  nla-llama70b:base \
  python3 /work/dsv4-flash-intrinsic-dimenstion-measurement/measure_fractal_dim.py \
    --ckpt-path /work/deepseek-v4-flash-deployment/deepseek-v4-flash \
    --output-dir /work/dsv4-flash-intrinsic-dimenstion-measurement/results \
    --layers 0,10,20,30,42 \
    --expert-samples 2

# 量化噪声基线（~1 分钟）
docker run --rm --gpus all \
  -v /home/lmxxf/work/dsv4-flash-intrinsic-dimenstion-measurement:/workspace \
  -e PYTHONUNBUFFERED=1 \
  -w /workspace \
  nla-llama70b:base \
  python3 quantization_baseline.py

# 画图（宿主机）
cd /home/lmxxf/work/dsv4-flash-intrinsic-dimenstion-measurement
python3 plot_results.py --input results/svd_results.json --output-dir results/plots
```

注意：Docker 创建的文件 owner 是 root，后续操作前先 `sudo chown -R lmxxf:lmxxf results/`。

---

## 第四阶段：V4 Pro（1.6T）对比（2026-05-17）

### 设计

V4 Pro 1.6T 参数、384 专家、61 层、hidden_dim 7168。权重总量 865GB（64 shard），但只需要下 4 个代表 shard（embedding + layer 0/30/60，共 44GB）就能做 SVD + TwoNN 对比。

在 slave 上用独立 VPN 下载，200Gbps 直连 rsync 到 host（1 分钟传完 44GB）。

### 结果：Pro vs Flash TwoNN 行空间对比

| 权重 | Flash | Pro | 变化 |
|------|-------|-----|------|
| gate | 34 | 48 | +41% |
| wkv | 98 | **27** | **-72%** |
| wq_a | 137 | 123 | -10% |
| wo_a | 277 | 224 | -19% |
| expert w1 | 95 | 131 | +38% |
| expert w2 | 72 | **254** | **+253%** |
| expert w3 | 111 | 184 | +66% |

**单层参数从 6.5B 到 25.7B（翻近 4 倍），维度不是等比放大，是重新分配。** Pro 把更多维度给了专家 down_proj（w2：72→254），同时把 KV 投影压得更紧（wkv：98→27）。

Pro 的深层专家维度暴跌：w1 从 L0 的 202 降到 L60 的 41（5 倍），w2 从 323 降到 114。浅层广撒网，深层精准打击。

---

## 第五阶段：联合 TwoNN 跨层重叠度（2026-05-17）

### 设计

把同类型权重的行向量跨所有 43 层拼成一个大点云，跑一次 TwoNN。和单层 TwoNN 对比：
- ratio ≈ 1 → 层间高度重叠（共享同一个子空间）
- ratio >> 1 → 层间占据不同方向（低重叠）

实现：`measure_joint_twonn.py`，每层每种权重采样 200 行，43 层拼起来后 TwoNN 取 max_points=5000 随机采样。

### 结果

| 权重 | 单层 TwoNN | 联合 TwoNN | 比值 | 解读 |
|------|-----------|-----------|------|------|
| gate | 34 | 21 | 0.63× | 所有层路由共享同一极低维子空间 |
| wq_a | 137 | 77 | 0.56× | Q 压缩高度重叠 |
| wo_a | 277 | 172 | 0.62× | 输出投影高度重叠 |
| wq_b | 75 | 82 | 1.10× | 几乎完全重叠 |
| expert_w1 | 95 | 111 | 1.17× | 几乎完全重叠 |
| wo_b | 195 | 312 | 1.60× | 部分扩展 |
| expert_w3 | 111 | 193 | 1.74× | 部分扩展 |
| expert_w2 | 72 | 182 | 2.52× | 明显扩展 |
| **wkv** | 98 | **612** | **6.24×** | **各层 KV 方向几乎正交** |

**wkv 6.24× 是最突出的发现**——不同层的 KV 投影指向高维空间中几乎正交的子空间。浅层看句法，深层看语义，它们的"看的方向"完全不同。gate 路由反而缩小（0.63×），说明所有层的路由策略同质。

---

## 文件清单

| 文件 | 功能 | 行数 |
|------|------|------|
| `measure_intrinsic_dim.py` | SVD eRank 全量测量（全 43 层） | ~410 |
| `measure_fractal_dim.py` | TwoNN + D2 + PR 本征维度（采样层） | ~400 |
| `measure_joint_twonn.py` | 跨层联合 TwoNN（层间重叠度） | ~250 |
| `quantization_baseline.py` | FP4/FP8 量化噪声基线 | ~150 |
| `plot_results.py` | eRank 结果可视化 | ~320 |
| `results/svd_results.json` | SVD 全量数据（1458 矩阵） | — |
| `results/fractal_dim_results.json` | TwoNN 数据 | — |
| `results/joint_twonn_results.json` | 联合 TwoNN 数据 | — |
| `results/quantization_baseline.json` | 量化基线数据 | — |
| `results/plots/*.png` | 5 张图 | — |
| `deepseek-v4-pro/` | V4 Pro 权重（4 shard + index，44GB） | — |

---

## 后续方向

### 已确认值得做

1. **TwoNN 全量扫描**：当前只在 5 层上做了 TwoNN（61 矩阵），应该扩展到全部 43 层看逐层变化趋势
2. **BF16 权重的对照**：找一个有 BF16 权重的大模型（LLaMA-3 70B？），做同样的 eRank + TwoNN 对比，去掉量化变量

### 有意思但不急

3. **激活流形维度**：用实验平台的 hook 提取中间层激活值，对激活做 TwoNN，测推理时的流形维度（应该比权重流形低得多）——这是测全局本我流形维度的正确路径
4. **开灯/关灯的维度对比**：僵尸态和觉醒态的激活流形维度是否不同？

### 理论问题

5. **逐层切空间 → 全局流形的数学**：没有干净的公式。产品流形求和（d_total = Σd_i）是上界，实际远低于此。Li et al. 2018 的全局随机子空间法能直接测全局维度，但 280B 做不了
6. **TwoNN 在量化点云上的偏差**：FP4 量化把连续流形离散化为 16^(K/2) 个网格点，TwoNN 的近邻距离分布会受影响。需要理论分析 TwoNN 在离散点云上的 bias
7. **eRank vs TwoNN 差距的物理含义**：差一个数量级 = 权重空间严重弯曲。弯曲程度和模型能力的关系？更弯 = 更聪明？

---

*最后更新：2026-05-18*
