# demo8 — 问题二 最终交付（论文 + 支撑代码）

问题二《模态局部缺失条件下的鲁棒情感预测》的最终交付：**最终论文 + 支撑论文结论的全部核心代码**。

## 内容

### `paper/`
- `问题二_学术排版终稿_审稿修订.docx` — 最终论文。相对"学术排版终稿"的改动仅一处：
  6.6.2 补充验证集三类真实样本量（负向 206 / 中性 184 / 正向 338，约 28:25:46），
  以显式说明中性类并非数量级稀少的极端少数类。
  （PDF 未随包，如需请在 Word 中另存导出，以保留原始版式与字体嵌入。）

### `code/`（论文 6.9 节"数值来源与复核路径"所列代码的依赖闭包）

| 文件 | 来源 | 作用 |
|---|---|---|
| `common.py` | demo1/code/ | 数据路径、特征缩放、冻结 BERT 缓存等公共模块 |
| `robust_model.py` | demo1/code/ | 双掩码缺失感知时序模型（分类+回归双头） |
| `problem2_train.py` | demo1/code/ | 训练/早停/验证/预测流程 |
| `experiment_loss_sweep.py` | demo1/code/ | 损失函数变体消融（label smoothing / focal / neutral weight） |
| `q2_trackA_delivery.py` | demo3/code/Q2/ | 最终 Track A 交付（valid/test/附件3 推理） |
| `q2_trackA_cv_roc.py` | demo3/code/Q2/ | 五折验证与 ROC（`q2_trackA_delivery` 依赖） |
| `run_mask_sweep.py` | demo7/code/ | 训练掩码概率扫描（证伪"调掩码比例救中性"） |

七个文件互相平级 `import`，与原始目录深度一致，故 `common.py` 的
`PROJECT_ROOT = Path(__file__).resolve().parents[2]` 在本目录下仍解析到项目根。

## 复现

环境：`conda activate mathModel`（Python 3.13，torch 2.10）。

数据依赖（未随包，体积大）：`E题数据/附件2-数据集特征文件/aligned_50.pkl` 与 `label.xlsx`、
附件3 对齐版本 30 个 pkl。保持数据目录名为 `E题数据/` 并置于项目根即可直接运行。

```bash
conda activate mathModel
# 复现主结果（mask_probability=0.45，等价于 loss_sweep 的 label_sm015）
python3 code/run_mask_sweep.py --mask-probabilities 0.45 --epochs 40 --device cpu
```

## 关键数字（论文结论，可溯源）

| 项 | 值 | 来源 |
|---|---|---|
| valid Acc / Macro-F1 / MAE / Pearson | 0.6374 / 0.6081 / 0.6029 / 0.6517 | `results/loss_sweep/label_sm015.json` |
| test Acc / Macro-F1 / 中性 F1 | 0.6740 / 0.6238 / 0.415 | 同上 |
| 掩码概率 0.45 | 缺失鲁棒性最优（text-mid-30% 相对降幅 2.5%） | `demo7/results/mask_sweep/` |

掩码概率 0.45 是**缺失鲁棒性**设置，不是中性类的可调旋钮；中性召回随掩码比例为噪声主导
（CV≈24%）。详见 `demo7/results/comparison.md`。

## 审稿结论

MiniMax-M3 外部审稿：**7/10，almost**。三件重点核对（中性 F1 如实披露 / 掩码结论未写反 /
数字可溯源）全部通过；审稿人三条意见经核对为"已披露 / 受数据约束 / 不准确 / 超范围"，
无必须修复的硬伤。
