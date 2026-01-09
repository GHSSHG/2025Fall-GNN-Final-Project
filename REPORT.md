# 期末大作业（主题 1）：论文复现与拓展探究 —— SAGPool

## 1. 论文信息与任务概述

论文：**Self-Attention Graph Pooling (SAGPool)**  
作者：Junhyun Lee, Inyeop Lee, Jaewoo Kang（ICML 2019）  
核心目标：在图分类/图表征中，引入**可学习的图池化（Pooling）**，在保留关键信息的同时逐层缩小图规模，实现层次化表示学习。

本大作业包含两部分：

1. **复现**：阅读理解 SAGPool，运行其开源实现，复现主要实验流程并总结分析。
2. **拓展**：将 SAGPool 用到生物序列对任务：给定 **Epitope** 与 **CDR3** 氨基酸序列，预测 `label∈{0,1}`（是否结合）。

---

## 2. SAGPool 方法理解（原理 + 直觉）

### 2.1 为什么需要图池化

很多图任务（尤其图分类）需要**图级表示**。常见做法是对节点表示做全局读出（mean/max/sum），但缺点是：

- 图可能很大，层数加深会导致计算成本高；
- 单次全局读出很难显式地形成“层次化结构”（类似 CNN 的下采样）；
- 如果能在中间层选择“关键节点子图”，模型更可能学到可解释的结构信息。

### 2.2 SAGPool 的核心做法

SAGPool 的 pooling 过程可以概括为三步：

1. **用 GNN 计算节点重要性分数（self-attention score）**  
   给定节点特征 `X` 和边 `A`，通过一个 GNN 得到每个节点的标量分数 `s`：

   - `s = GNN_score(X, A)`，其中 `s` 是每个节点一个分数（长度为 `N`）。

2. **Top-k 选择节点**  
   保留得分最高的 `k = ceil(ratio * N)` 个节点，得到索引集合 `idx`。

3. **得到池化后的子图**
   - 节点特征：`X' = X[idx] * tanh(s[idx])`（用分数做门控，增强“重要节点”的特征）
   - 边：保留诱导子图 `A' = A[idx, idx]`（只保留保留节点之间的边）

直觉：  
SAGPool 用一个“看过图结构的评分网络”给节点打分，而不是像某些 TopK pooling 仅用可学习投影向量对节点做打分；因此评分更“结构感知”。

### 2.3 论文中的分类网络结构

论文与开源代码常用结构是多段堆叠：

`(GCN → SAGPool) × 3`，每段后做一次图级读出（mean+max），再把三段读出结果相加，最后用 MLP 分类。

---

## 3. 论文复现（开源代码运行 + 结果）

### 3.1 使用的开源实现

我使用了作者的 PyTorch 版本 SAGPool 实现（已克隆在 `third_party/SAGPool/`），其依赖 `torch_geometric`。

由于当前环境使用较新的 `torch_geometric` 版本，并且训练需要在 **Mac M3 的 MPS** 上跑通，因此做了少量兼容性修复（见 `README.md` 与代码改动记录）。

关键兼容性点：

- 新版 `torch_geometric` 的 API 变更（import 路径调整）
- 增加 `mps` 设备选择逻辑（无 CUDA 时优先 `mps`）
- **MPS 反向传播兼容性**：`GCNConv(out_channels=1)` 在 MPS 上会触发反向崩溃，因此将 SAGPool 的 score GNN 改为 `GraphConv(out_channels=1)`（思想仍为“用 GNN 做注意力评分”，但具体算子换成可稳定运行的版本）
- 修复极小图下 `squeeze()` 得到 0-d tensor 导致 topk 出错的问题（改为 `view(-1)`）

### 3.2 复现实验设置

- 数据集：TUDataset 中的 `MUTAG`、`PROTEINS`
- 划分：开源代码的 hold-out（train/val/test = 0.8/0.1/0.1），并设置 seed=42
- 设备：Mac MPS（`mps`）

### 3.3 复现结果（本地跑通）

日志文件：

- `logs/repro_sagpool_MUTAG_seed42_full.txt`
- `logs/repro_sagpool_PROTEINS_seed42_nhid64_fix.txt`

最终测试集准确率（Test accuracy）：

- **MUTAG**：`0.80`
- **PROTEINS**：`0.7857`

说明：

- 论文中常见设置是多次重复/交叉验证统计平均值；本复现使用作者脚本的 hold-out 划分，因此数值不一定与论文表格完全一致，但训练流程与模型结构保持一致，并在 MPS 上成功跑通。

---

## 4. 拓展探究：Epitope–CDR3 结合预测（图建模 + SAGPool）

### 4.1 数据说明

数据路径：`data/data.tsv`  
列：`CDR3`, `Epitope`, `label`

数据规模与类别分布（全量）：

- 总样本数：`37094`
- 正类（label=1）：`789`（正例率约 `2.13%`）

这是一个**极度类别不平衡**的二分类任务，因此仅看 accuracy 会被“全预测为 0”误导；更应关注 **ROC-AUC / PR-AUC / F1 / Recall** 等。

### 4.2 如何把“序列对”变成图

一个样本（Epitope, CDR3）构造成一张图：

- **节点**：每个氨基酸残基一个节点（把 Epitope 节点放在前、CDR3 节点拼接在后）
- **边**：
  1. Epitope 内部：相邻残基连边（无向）
  2. CDR3 内部：相邻残基连边（无向）
  3. Epitope ↔ CDR3：完全二部图连边（无向），用于建模相互作用
- **节点特征**：
  - 氨基酸 embedding（21 类：20 标准氨基酸 + `X`）
  - 位置归一化 `pos∈[0,1]`（分别在各自序列内归一化后拼接）
  - 是否来自 CDR3 的二值特征（Epitope=0, CDR3=1）

对应实现：

- 图构建与缓存：`bio_sagpool/dataset.py`
- 模型：`bio_sagpool/model.py`

### 4.3 模型结构（基于 SAGPooling）

使用 `torch_geometric.nn.SAGPooling` 作为 pooling 层，整体结构为：

`Embedding + (GCNConv → SAGPooling) × 3 → (global max + global mean) readout → MLP → logit`

损失函数：`BCEWithLogitsLoss(pos_weight=neg/pos)`（用正类权重缓解类别不平衡）。

早停指标：验证集 **PR-AUC**（Average Precision）。

### 4.4 实验设置（seed=42 随机划分）

按要求用全局随机种子 42 划分：

- Train/Val/Test = 0.8 / 0.1 / 0.1

运行命令见 `README.md`（对应结果目录 `outputs/bio_sagpool_seed42_run3/`）。

### 4.5 拓展实验结果

最优验证轮次（best epoch）：`12`  
最优阈值（val F1 最优阈值）：`0.84`

测试集指标（来自 `outputs/bio_sagpool_seed42_run3/results.json`）：

- ROC-AUC：`0.9334`
- PR-AUC：`0.3060`
- F1：`0.3195`
- Precision：`0.2872`
- Recall：`0.36`

对比直觉基线：

- 随机分类器的 PR-AUC 近似等于正例率（约 `0.02`）
- 本模型 PR-AUC `0.306`，显著高于随机基线，说明模型确实学习到了序列对的结合模式。

---

## 5. 复现与拓展的代码入口

- SAGPool 复现入口：`third_party/SAGPool/main.py`
- 生物数据训练入口：`python -m bio_sagpool.train`

运行方式与参数见 `README.md`。
