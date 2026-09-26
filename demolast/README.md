# demolast — 问题二最终归档（Track A + Track B，程序/文件/数据集/结果/论文）

本目录把问题二《模态局部缺失条件下的鲁棒情感预测》的**两条路线**——
主线 **Track A**（深度学习）与传统对照 **Track B**（M0/M1 机器学习）——对应的
程序、文件、数据集与结果规整到一起，作为最终可复现归档。

## 路线总览

| 路线 | 接口 | 技术 | 代码 | 结果 |
|---|---|---|---|---|
| Track A | aligned_50（50 步对齐） | BERT 缓存 + 双掩码 Transformer + 可靠度门控 + 双头 | `code/` 7 个主文件 | `results/label_sm015.json`、`results/trackA_delivery/` |
| Track B | unaligned_50（500 步未对齐） | M0：TF-IDF + 音视统计 + Logistic/Ridge；M1：文本 Logistic/Ridge + 音视 ExtraTrees 专家晚期融合 | `code/q2_trackB_cv_roc.py` 等 4 个 | `results/trackB/` |

## 溯源结论

`paper/问题二.docx`（5.2 节）正文只写了 Track A 主线，其数字指纹：

> valid Acc 0.6374 / Macro-F1 0.6081 / MAE 0.6029 / Pearson 0.6517，样本量 3395/728/727，
> aligned_50 接口 → 精确等于 `results/label_sm015.json` 的 validation 字段。
> 配置：`label_smoothing=0.15`、`focal γ=0`、`neutral_scale=1.0`、`consistency=0.1`、
> 训练增强 `mask_probability=0.45`、`seed=2026`、`best_epoch=4`。

### ① 程序（`code/`）

**Track A（主模型）**：

| 文件 | 作用 |
|---|---|
| `robust_model.py` | 模型结构 `RobustFusionModel`（双掩码 / missing token / 单层 4 头 Transformer / 时序注意力 / 可靠度门控 / 双头） |
| `problem2_train.py` | 数据构造、连续块增强、训练、预测、指标 |
| `experiment_loss_sweep.py` | 损失消融（label smoothing / focal / neutral weight） |
| `q2_trackA_delivery.py` | 最终交付（45 缺失场景 + 7 消融 + 附件3 推理） |
| `q2_trackA_cv_roc.py` | 五折交叉验证 + ROC |
| `common.py` | 路径 / 缩放 / 冻结 BERT 缓存 |
| `run_mask_sweep.py` | 训练掩码概率扫描（证伪"调掩码救中性"） |

**Track B（传统对照，M0/M1）**：

| 文件 | 作用 |
|---|---|
| `q2_trackB_cv_roc.py` | Track B 五折 + ROC（M0/M1 的 OOF 与 test 点估计） |
| `run_q2_round1.py` | M0/M1 核心实现（TF-IDF / ExtraTrees / Logistic / Ridge） |
| `q2_adjust_neutral.py` | 中性类分数调整（`scale_neutral`） |
| `q2_execute.py` | Track B 执行入口 |

M0（usable_baseline）= 文本 TF-IDF + 音视时序统计摘要 → 早期拼接 → 一对多 Logistic（分类）+ Ridge（回归）。
M1（main_candidate）= 文本 Logistic/Ridge + 音频/视觉 ExtraTrees 专家 → 固定权重 0.50/0.25/0.25 晚期融合。
二者均走 unaligned_50 接口，与 Track A 的 aligned_50 不是同一特征版本，**不存在组件消融关系**。

### ② 文件（`data/` + `results/`）

**数据**：
- `data/aligned_50.pkl`（948M，附件2 对齐特征，Track A 输入）
- `data/unaligned_50.pkl`（2.7G，附件2 未对齐特征，Track B 输入）
- `data/label.xlsx`（附件2 标签）
- `data/attachment3/`（附件3 对齐版本 30 个 pkl，无标签最终推理）

**结果**：
- `results/label_sm015.json`（Track A 主结果：test acc 0.674 / macro-F1 0.6238 / 中性 F1 0.415）
- `results/trackA_delivery/`（Track A 缺失网格、消融、valid/test/附件3 预测、检查点、标准化器）
- `results/trackB/`（Track B 的 M0/M1 OOF 预测与 `summary.json`）

### ③ 数据集

- **附件2**：aligned_50（3395/728/727，三模态各 50 位置）用于 Track A；unaligned_50（500 位置）用于 Track B。
- **附件3**：30 条无标签对齐样本，仅最终推理。

## 复现

```bash
conda activate mathModel
# Track A 主结果（= label_sm015）
python3 code/run_mask_sweep.py --mask-probabilities 0.45 --epochs 40 --device cpu
```

`common.py` 中 `PROJECT_ROOT = Path(__file__).resolve().parents[2]`、`DATA_ROOT = PROJECT_ROOT / "E题数据"`。
代码放在 `demolast/code/` 下时 `parents[2]` 仍解析到项目根，故需把 `data/` 下的数据放回
`E题数据/附件2-数据集特征文件/` 与 `E题数据/附件3-模态缺失特征样本/对齐版本/`，或改 `common.py` 的路径常量。

## 论文版本说明

`paper/问题二.docx` 是 5.2 节章节稿，正文到 5.2.9 消融为止，只覆盖 Track A 主线，
**缺"代码对应关系"、"Track B 对照"与"附件3 推理/结论"**。本 README 即补足其溯源与 Track B 对照；
更完整版本（含 6.11 代码对应关系）见 `paper/02/问题二_最终重写版_16页_36公式_20图.docx`。
