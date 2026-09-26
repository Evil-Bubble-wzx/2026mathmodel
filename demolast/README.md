# demolast — 问题二最终归档（程序 / 文件 / 数据集 / 论文）

本目录把 `paper/问题二.docx`（5.2 节）对应的**全部程序、文件、数据集与结果**规整到一起，
作为问题二《模态局部缺失条件下的鲁棒情感预测》的最终可复现归档。

## 溯源结论（三层面）

`paper/问题二.docx` 正文没有写代码溯源，但它的**数字指纹**把它钉死了：

> valid Acc 0.6374 / Macro-F1 0.6081 / MAE 0.6029 / Pearson 0.6517，样本量 3395/728/727，
> aligned_50 接口 → 精确等于 `results/label_sm015.json` 的 validation 字段。
> 配置：`label_smoothing=0.15`、`focal γ=0`、`neutral_scale=1.0`、`consistency=0.1`、
> 训练增强 `mask_probability=0.45`、`seed=2026`、`best_epoch=4`。

### ① 程序（`code/`，7 个文件）

| 文件 | 作用 |
|---|---|
| `robust_model.py` | 模型结构 `RobustFusionModel`（双掩码 / missing token / 单层 4 头 Transformer / 时序注意力 / 可靠度门控 / 双头） |
| `problem2_train.py` | 数据构造、连续块增强、训练、预测、指标 |
| `experiment_loss_sweep.py` | 损失消融（label smoothing / focal / neutral weight） |
| `q2_trackA_delivery.py` | 最终交付（45 缺失场景 + 7 消融 + 附件3 推理） |
| `q2_trackA_cv_roc.py` | 五折交叉验证 + ROC |
| `common.py` | 路径 / 缩放 / 冻结 BERT 缓存 |
| `run_mask_sweep.py` | 训练掩码概率扫描（证伪"调掩码救中性"） |

### ② 文件（`data/` + `results/`）

**数据**：
- `data/aligned_50.pkl`（948M，附件2 对齐特征，Track A 主模型输入）
- `data/label.xlsx`（附件2 标签）
- `data/attachment3/`（附件3 对齐版本 30 个 pkl，无标签最终推理）

**结果**：
- `results/label_sm015.json`（主结果：test acc 0.674 / macro-F1 0.6238 / 中性 F1 0.415）
- `results/trackA_delivery/`（缺失网格、消融、valid/test/附件3 预测、检查点、标准化器、错误审计）

### ③ 数据集

- **附件2 aligned_50**：train/valid/test = 3395/728/727，三模态各 50 对齐位置（text 50×768、audio 50×74、vision 50×35）
- **附件3**：30 条无标签对齐样本，仅最终推理

## 未包含（说明）

- `unaligned_50.pkl`（2.7G）：Track B（M0/M1 传统对照）的数据，`问题二.docx` 主线（Track A）用不到，未复制。
- 附件3 未对齐版本：同理未复制。

## 复现

```bash
conda activate mathModel
python3 code/run_mask_sweep.py --mask-probabilities 0.45 --epochs 40 --device cpu
```

`common.py` 中 `PROJECT_ROOT = Path(__file__).resolve().parents[2]`、`DATA_ROOT = PROJECT_ROOT / "E题数据"`。
代码放在 `demolast/code/` 下时 `parents[2]` 仍解析到项目根，故需把 `data/` 下的数据放回
`E题数据/附件2-数据集特征文件/` 与 `E题数据/附件3-模态缺失特征样本/对齐版本/`，或改 `common.py` 的路径常量。

## 论文版本说明

`paper/问题二.docx` 是 5.2 节章节稿，正文到 5.2.9 消融为止，**缺"代码对应关系"与"附件3 推理/结论"两节**。
本 README 即补足其溯源；更完整版本（含 6.11 代码对应关系）见
`paper/02/问题二_最终重写版_16页_36公式_20图.docx`。
