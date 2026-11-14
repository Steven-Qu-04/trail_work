# 预测报告

- 输入：`test.csv`（20 行）
- 模型：`hgb` + 特征集 `desc+morgan`（训练时测试集 RMSE 0.807）
- 输出：`predictions.csv`

## 1. 处理统计

| 项目 | 行数 |
| --- | --- |
| 输入总行数 | 20 |
| 成功预测 | 16 |
| 结构无效（未预测） | 4 |
| 参与打分的行 | 15 |

## 2. 指标

| 指标 | 值 |
| --- | --- |
| RMSE | **0.510** |
| MAE | 0.347 |
| R² | 0.875 |
| 参与打分的行数 | 15 |

> 打分只包含“预测成功且标签是合法数字（|logD| ≤ 10）”的行；结构无效的行照样写在 predictions.csv 里，状态标记为 failed。

## 3. 失败与跳过

| CMPD_CHEMBLID | 问题 |
| --- | --- |
| `DEMO_BLANK_SMILES` | SMILES 为空 |
| `DEMO_BAD_SMILES` | SMILES 无法解析 |
| `DEMO_BAD_PAREN` | SMILES 无法解析 |
| `DEMO_TOO_LONG` | SMILES 过长（400 字符） |
| `DEMO_BAD_EXP` | 标签不是数字（`definitely_wrong`），已预测但不打分 |

## 4. 误差最大的 5 条

| CMPD_CHEMBLID | 真实 | 预测 | 绝对误差 |
| --- | --- | --- | --- |
| `CHEMBL440335` | 4.50 | 2.974 | 1.526 |
| `CHEMBL292376` | 3.00 | 2.290 | 0.710 |
| `CHEMBL1270773` | 2.50 | 3.208 | 0.708 |
| `CHEMBL1350196` | -0.70 | -0.312 | 0.388 |
| `CHEMBL1645501` | 1.41 | 1.787 | 0.377 |

## 5. 复现命令

```bash
python src/07_predict_test.py --input test.csv --output predictions.csv
```
