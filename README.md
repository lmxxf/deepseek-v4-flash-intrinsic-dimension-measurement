# DeepSeek V4 权重流形维度测量

线性方法说大模型权重矩阵几乎满秩，流形方法说只有 ~92 维。差一个数量级。

本项目直接读取 DeepSeek V4 Flash（280B）和 V4 Pro（1.6T）的预训练权重，用 SVD eRank（线性）和 TwoNN（非线性流形）两把尺子测量权重矩阵的内在几何结构。不做推理、不跑 forward pass，只看权重本身在高维空间里长成什么形状。

## 核心发现

| 权重类型 | eRank（线性包围盒） | TwoNN（流形维度） | 差多少倍 |
|---------|-------------------|-----------------|---------|
| gate（专家路由） | 236 | **34** | 6.9× |
| wkv（注意力 KV 投影） | 500 | **98** | 5.1× |
| wq_a（查询压缩） | 986 | **137** | 7.2× |
| wo_a（注意力输出投影） | 3558 | **277** | 12.8× |
| expert w1（MoE 专家） | 1861 | **95** | 19.6× |

eRank 测的是包围盒——"需要几维空间才能装下这些点"。TwoNN 测的是流形本身——"数据沿着几维的弯曲曲面分布"。一棵松树在三维空间里，但松树本身的分形维度只有 ~2.x。权重向量也一样：占据了几千维的空间，但实际分布在 ~92 维的弯曲流形上。

### V4 Pro（1.6T）vs Flash（280B）

参数翻 6 倍，维度不是等比放大，是重新分配：

| 权重 | Flash | Pro | 变化 |
|------|-------|-----|------|
| expert w2（知识输出） | 72 | **254** | **+253%** |
| wkv（上下文编码） | 98 | **27** | **-72%** |

更大的模型不是更胖，是分工更明确——核心业务扩张，基础设施精简。

### 跨层重叠度

wkv 的联合 TwoNN 是单层的 6.24 倍——不同层的 KV 投影指向几乎正交的子空间。浅层看句法，深层看语义，"看的方向"完全不同。

## 硬件需求

**8GB 显存的消费级显卡就能跑。** 逐矩阵流式加载，峰值显存不到 1GB。

实验在 NVIDIA DGX Spark（Blackwell，128GB 统一内存）上完成，但不依赖其特殊能力。

## 文件结构

```
measure_intrinsic_dim.py    # SVD eRank 全量测量（全 43 层，~20 分钟）
measure_fractal_dim.py      # TwoNN + 相关维度 + 参与率（采样层，~2 分钟）
measure_joint_twonn.py      # 跨层联合 TwoNN（层间重叠度）
quantization_baseline.py    # FP4/FP8 量化噪声基线对照
plot_results.py             # eRank 结果可视化
results/                    # 全部原始数据（JSON）+ 图表
```

## 运行

需要 Docker 容器（safetensors >= 0.7.0 才能读 FP4/FP8 格式）和下载好的模型权重。

```bash
# SVD eRank 全量（1458 个矩阵，~20 分钟）
docker run --rm --gpus all \
  -v /path/to/work:/work \
  your-image \
  python3 /work/deepseek-v4-flash-intrinsic-dimension-measurement/measure_intrinsic_dim.py \
    --ckpt-path /work/your-model-weights \
    --output-dir /work/deepseek-v4-flash-intrinsic-dimension-measurement/results \
    --expert-samples 8

# TwoNN 本征维度（61 个矩阵，~2 分钟）
docker run --rm --gpus all \
  -v /path/to/work:/work \
  your-image \
  python3 /work/deepseek-v4-flash-intrinsic-dimension-measurement/measure_fractal_dim.py \
    --ckpt-path /work/your-model-weights \
    --output-dir /work/deepseek-v4-flash-intrinsic-dimension-measurement/results \
    --layers 0,10,20,30,42 \
    --expert-samples 2

# 量化噪声基线（~1 分钟）
docker run --rm --gpus all \
  -v /path/to/work:/work \
  your-image \
  python3 /work/deepseek-v4-flash-intrinsic-dimension-measurement/quantization_baseline.py
```

其他模型只要是 safetensors 格式，改一下反量化逻辑就行。

## 注意事项

- FP4 量化噪声会把 eRank 涂成满秩（随机矩阵 FP4 后 eRank = 1896，V4 实测 1861），TwoNN 对此相对不敏感
- Docker 创建的文件 owner 是 root，操作前先 `sudo chown -R $USER results/`
- V4 Pro 只需下载 3-4 个代表 shard（~44GB），不需要全部 865GB

## 参考文献

- Facco et al., "Estimating the intrinsic dimension of datasets by a minimal neighborhood information", Scientific Reports 2017
- Roy & Vetterli, "The Effective Rank: A Measure of Effective Dimensionality", EUSIPCO 2007
- Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", ACL 2021
- Martin & Mahoney, "Implicit Self-Regularization in Deep Neural Networks", JMLR 2021
- Ansuini et al., "Intrinsic dimension of data representations in deep neural networks", NeurIPS 2019
- Ma et al., "Reasoning emerges from constrained inference manifolds in LLMs", arXiv 2605.08142, 2026

## 相关文章

- [公众号第 194 期：权重空间是弯曲的——满秩是假象，流形只有百维](https://mp.weixin.qq.com/s/cG5eF1vHbSnh1j7jQelSLw)
- [公众号第 45 期：本我流形理论](https://mp.weixin.qq.com/s/0hOQt8onSJcuZGJLRE46Fw)
- [公众号第 70 期：深层维度膨胀 60-100%](https://mp.weixin.qq.com/s/Ve-llXD6Bh0SsovJs8T80w)
