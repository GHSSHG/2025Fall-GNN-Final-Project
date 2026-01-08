# GNN 期末大作业：SAGPool 复现与生物数据拓展

本仓库包含两部分：

1) **SAGPool 论文复现**（基于作者开源实现做了兼容性修复，并在 Mac MPS 上跑通）  
2) **拓展任务：Epitope–CDR3 结合预测**（把“序列对”构造成图，用 SAGPooling 做二分类）

## 0. 环境准备（Mac M 系列 + MPS）

建议使用 Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# 安装 PyTorch（macOS wheel 自带 MPS）
pip install --upgrade pip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# 其余依赖
pip install -r requirements.txt
```

验证 MPS：

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

## 1. SAGPool 复现（TUDataset）

作者原仓库位于 `third_party/SAGPool/`，我做了以下必要改动：

- 适配新版 `torch_geometric` 的 import 路径
- 增加 `mps` 设备选择逻辑（无 CUDA 时优先用 MPS）
- 为兼容 MPS：将 SAGPool 的 score GNN 从 `GCNConv(out=1)` 改为 `GraphConv(out=1)`（避免 MPS 反向传播崩溃）
- 修复极小图下 `squeeze()` 变成 0-d tensor 的问题（改为 `view(-1)`）

运行示例（输出建议重定向到 log 文件）：

```bash
python -u third_party/SAGPool/main.py --dataset MUTAG --epochs 300 --patience 30 --batch_size 64 --seed 42 > logs/repro_MUTAG.txt 2>&1
python -u third_party/SAGPool/main.py --dataset PROTEINS --epochs 40 --patience 10 --batch_size 256 --nhid 64 --seed 42 > logs/repro_PROTEINS.txt 2>&1
```

## 2. 生物数据拓展：Epitope–CDR3 结合预测

数据文件：`data/data.tsv`（列：`CDR3`, `Epitope`, `label`）。

### 2.1 图构建（一个样本一张图）

- **节点**：氨基酸残基（Epitope + CDR3 拼接成一个图）
- **边**：
  - Epitope 序列内部：相邻残基连边（无向）
  - CDR3 序列内部：相邻残基连边（无向）
  - Epitope ↔ CDR3：完全二部图连边（无向），用于建模相互作用
- **节点特征**：
  - 氨基酸 embedding
  - 位置归一化 `pos∈[0,1]`
  - 是否来自 CDR3 的二值特征

### 2.2 训练（seed=42 全局随机划分）

脚本会按 `train/val/test = 0.8/0.1/0.1` 用随机种子 42 划分，并使用 `BCEWithLogitsLoss(pos_weight=neg/pos)` 处理类别极不平衡。

```bash
python -u -m bio_sagpool.train \
  --data_path data/data.tsv \
  --seed 42 \
  --epochs 15 \
  --patience 5 \
  --batch_size 1024 \
  --hidden_dim 64 \
  --aa_emb_dim 16 \
  --out_dir outputs/bio_sagpool_seed42_run3 \
  --no_progress
```

输出：

- `outputs/bio_sagpool_seed42_run3/results.json`
- `outputs/bio_sagpool_seed42_run3/history.json`

更详细的论文理解、复现结果与拓展分析见 `REPORT.md`。

