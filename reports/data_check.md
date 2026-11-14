# 数据体检报告

- 数据文件：`Lipophilicity.csv`
- 行数 **4200**，列数 **3**
- 列名：`CMPD_CHEMBLID`、`exp`、`smiles`

> 读入时全部按字符串处理：`exp` 列里存在 `.05` 这种无前导零的写法，直接让 pandas 推断类型会把整列变成 object。

## 1. 缺失值

| 列 | 空值行数 |
| --- | --- |
| `CMPD_CHEMBLID` | 0 |
| `exp` | 1 |
| `smiles` | 1 |

## 2. 标签列 `exp`

- 转不成数字的行：**2**
- 数值超出合理范围（|logD| > 10）的行：**1**

| 行号 | CMPD_CHEMBLID | exp 原值 |
| --- | --- | --- |
| 3792 | `CHEMBL_BLANK_EXP` | `` |
| 4048 | `CHEMBL_BAD_EXP_EXTREME` | `999999.0` |
| 4085 | `CHEMBL_BAD_EXP_TEXT` | `definitely_wrong` |

剔除异常值后的标签分布：

| 指标 | 值 |
| --- | --- |
| 样本数 | 4197 |
| 最小值 | -1.500 |
| 25% 分位 | 1.410 |
| 中位数 | 2.360 |
| 75% 分位 | 3.100 |
| 最大值 | 4.500 |
| 均值 | 2.188 |
| 标准差 | 1.204 |

## 3. ID 列 `CMPD_CHEMBLID`

- 重复的 ID 行数：**410**
- 不符合 `CHEMBL+数字` 格式的 ID：**9**

  `CHEMBL_BLANK_EXP`、`CHEMBL_HUGE_2`、`CHEMBL_HUGE_1`、`CHEMBL_RDKit_FAIL_2`、`CHEMBL_BAD_EXP_EXTREME`、`CHEMBL_BLANK_SMILES`、`CHEMBL_BAD_EXP_TEXT`、`ljiohohoihoa`、`CHEMBL_RDKit_FAIL_1`

> 其中 `ljiohohoihoa` 的分子结构和标签都正常，只是 ID 本身是乱码，不影响训练，不必剔除。

> ID 不能当唯一键用，去重必须按分子结构（canonical SMILES）来判。

## 4. 结构列 `smiles`

- 空 SMILES：**1**
- RDKit 无法解析：**2**
- 长度超过 300 字符：**2**

SMILES 长度分布（字符数）：

| 指标 | 值 |
| --- | --- |
| 最短 | 0 |
| 中位数 | 48 |
| 最长 | 6000 |

需要剔除的异常行：

| 行号 | CMPD_CHEMBLID | exp | SMILES（截断） | 原因 |
| --- | --- | --- | --- | --- |
| 3792 | `CHEMBL_BLANK_EXP` | `（空）` | `Cc1sc2c(C(=N[C@@H](CC(=O)OC(C)(C)C)c3nnc` | 标签为空或非数值 |
| 3858 | `CHEMBL_HUGE_2` | `1.55` | `CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC` | SMILES 过长（4612 字符） |
| 3932 | `CHEMBL_HUGE_1` | `2.3` | `CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC` | SMILES 过长（6000 字符） |
| 3961 | `CHEMBL_RDKit_FAIL_2` | `2.87` | `not_a_smiles` | SMILES 无法解析 |
| 4048 | `CHEMBL_BAD_EXP_EXTREME` | `999999.0` | `Oc1c2ncc3ccccc3c2nn1c4ccc(Cl)cc4` | 标签异常（999999.0） |
| 4050 | `CHEMBL_BLANK_SMILES` | `2.35` | `` | SMILES 为空 |
| 4085 | `CHEMBL_BAD_EXP_TEXT` | `definitely_wrong` | `O[C@@H](CNCCOc1cccc(CNCCc2ccccc2F)c1)c3c` | 标签为空或非数值 |
| 4124 | `CHEMBL_RDKit_FAIL_1` | `2.55` | `C1(CC` | SMILES 无法解析 |

## 5. 重复情况

- 三列完全相同的重复行：**410**
- 按 canonical SMILES 重复的行：**415**

- 同一结构但标签互相矛盾的结构数：**3**

> 这些重复必须在划分数据集之前处理掉，否则同一分子会同时出现在训练集和测试集，RMSE 会被人为压低。

## 6. 清洗后样本量预估

| 阶段 | 行数 |
| --- | --- |
| 原始 | 4200 |
| 剔除异常行后 | 4192（剔除 8） |
| 按 canonical SMILES 去重后 | 3780 |

## 7. 结论

1. 数据不能直接拿来训练：共 **8** 行存在标签或结构问题，必须先剔除。
2. 有 **410** 行完全重复、**415** 行结构重复，必须去重，否则验证结果不可信。
3. 清洗去重后可用样本约 **3780** 条。
4. 标签量级：中位数 2.36，标准差 1.20，后续 RMSE 应与这个量级对比才有意义。
