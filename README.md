# Lipophilicity logD 回归 —— RDKit 描述符 + Ridge

## 1. 任务简介

**任务**：给定分子的 SMILES，回归预测其 logD（pH 7.4 下的脂水分配系数），属于表格型单任务监督回归，主指标为 **RMSE**。

**这次实际做了什么**：

1. 对原始 `Lipophilicity.csv`（4,200 行）做完整质检与清洗，得到 3,780 行干净数据（`clean_lipophilicity.py`）；
2. 用 RDKit 全量分子描述符把 SMILES 转成数值矩阵，并按"只在训练集上 fit"的规则做填补、去常量、标准化、共线剪枝（218 → 178 列）；
3. 用 **Bemis–Murcko scaffold split（8:1:1）** 划分数据，避免随机划分带来的"同类分子跨集合"信息泄漏；
4. 训练 **Ridge 回归**主模型（`alpha` 由训练集内部按骨架分组的 5 折交叉验证选出）；
5. 提供推理接口 `python baseline.py --predict test.csv`，对新数据出 `predictions.csv` 与指标报告。

**主结果**（scaffold split 测试集，n=378）：RMSE **0.8575**、MAE 0.6651、R² 0.5050、Spearman ρ 0.6914。

---

## 2. 数据集检查与预处理

### 2.1 结论：**不能直接使用**

原始表 4,200 行，其中 **420 行（10.0%）有缺陷**；且 `exp` 列以字符串存储（`.05`、`-.01` 这类省略整数位的写法有 665 条），直接 `pd.read_csv` 后是 `object` 类型，回归模型会报错或静默算错。

### 2.2 发现的问题与处理

| 异常类型 | 数量 | 处理 |
| --- | --- | --- |
| 整行完全重复 | 410 | 保留首次出现，其余删除 |
| 同一分子（规范化后）标签矛盾 | 2 | 保留编号合法、标签合理的一条，删除伪造记录 |
| 分子尺寸异常（聚合物级，重原子 > 150 / MW > 2000） | 2 | 删除（会让描述符计算失控） |
| 非法 SMILES（`C1(CC`、`not_a_smiles`） | 2 | 删除 |
| `exp` 缺失 / 非数值（`definitely_wrong`） / 极端离群（999999） | 各 1 | 删除，不做猜测性填补或截断 |
| SMILES 为空 | 1 | 删除 |
| 含盐/水合物多组分（4 个片段） | 1 | 保留最大片段（去盐去溶剂） |
| 数值写法不规范（`.05`、`-.96`） | 665 | 统一解析为标准 float 写法 |

**关键坑**：数据集里 5 条伪造记录**故意复用了真实分子的 SMILES**（改编号、改标签、置空、写成文本、改成 999999）。所以必须先做**行级合法性检查**，再按规范化 SMILES 去重——否则会把真标签换成假标签。脚本就是按这个顺序实现的。

保留的情况：68 组、149 条立体异构体（不同物质，不属重复）；`-1.5`、`4.5` 附近的极端标签是真实测量。

### 2.3 预处理产出（每条删除都落盘，不静默丢样本）

| 文件 | 内容 |
| --- | --- |
| `Lipophilicity_clean.csv` | 3,780 行，列与原文件一致（`CMPD_CHEMBLID, exp, smiles`），训练输入 |
| `Lipophilicity_clean_details.csv` | 清洗后数据 + 质检列（规范化 SMILES、重原子数、分子量、片段数、原始行号、qc_note） |
| `Lipophilicity_removed_rows.csv` | 被删除的每条记录及删除原因（中英文） |
| `data_qc_summary.json` | 各类异常的数量统计 |

清洗后：标签范围 -1.50 ~ 4.50（均值 2.183，标准差 1.204），最大重原子数 115，最大分子量 1,620.7 Da（1 条肽类分子，属正常记录）。完整分析见 [`data_quality_report.md`](data_quality_report.md)。

---

## 3. 分子表示与输入处理

### 3.1 分子 → 数值

用 RDKit `Chem.Descriptors._descList` **全量计算 217 个二维描述符**（`MolLogP`、`MolMR`、`TPSA`、`NumHDonors`、`NumHAcceptors`、`MolWt`、`FractionCSP3`、`Chi*`、`Kappa*`、`SlogP_VSA*`、`PEOE_VSA*`、`EState_VSA*` 等），不借助 3D 构象、不做指纹哈希，保证每一列都能用人话说清含义。

额外追加 1 列 `MolLogD`：RDKit 没有 logD 描述符，这一列按约定用 **Crippen `MolLogP`** 实现，并在共线剪枝阶段**强制保留**，保证"亲脂性"这条物理上最强的信号一定进入模型。代价是它与原生 `MolLogP` 列完全共线（|r| = 1），剪枝时被丢弃的是 `MolLogP`，因此**最终模型里的 "MolLogD" 实质就是 logP，不是真正算出的 logD**——这一点在第 8 节作为已知问题列出。

### 3.2 特征预处理链（F1，全部只在训练集 `fit`）

```
原始 218 列
  → ① ±Inf 转 NaN，NaN 用训练集中位数填补
  → ② 偏态处理（design_plan 的可选分支，默认关闭，--skew 打开）
  → ③ 剔除常量列（VarianceThreshold = 0）                                   −9 列
  → ④ 标准化（StandardScaler）
  → ⑤ 共线剪枝（|Pearson r| > 0.95 的列对，保留与 y 相关性更高的一列，
       MolLogD 受保护、其共线对手一律丢弃）                                  −31 列
  → 输出 178 列
```

第⑤步针对的是描述符之间大量近似重复（`Chi0`/`Chi0v`、`SlogP_VSA*` 系列）：不剪枝时 Ridge 系数不稳定、OLS 系数会爆炸且符号违背化学常识。这一步也是 `MolLogP` 被丢弃的直接原因（见 3.1）。

### 3.3 计算成本与缓存

- 3,780 个分子 × 218 列在 **CPU** 上用多进程（8 进程）计算约 **7.3 s**（串行约 60 s），不需要 GPU；
- 特征矩阵缓存为 `outputs/features_cache.npz`，缓存键 = SHA1(SMILES 全列表 + RDKit 版本 + 描述符列数 + 追加列名)，数据或 RDKit 版本一变就自动重算，避免"换数据却用旧缓存"；
- 缓存文件是可重建的中间产物，**不提交到仓库**（已在 `.gitignore` 中）。

### 3.4 推理时的输入处理

`--predict` 走同一条描述符路径（不读训练缓存），并检查描述符列名与训练时是否一致：RDKit 版本变化导致列变动会直接报错并提示重新训练，而不是悄悄预测出错误结果。无法解析的 SMILES **不静默丢弃**，单独写入 `invalid_rows.csv` 并打印数量。

---

## 4. Baseline 模型

**模型**：Ridge（L2 正则线性回归），输入 178 个标准化描述符，`alpha` 在训练集内部用 **GroupKFold（按 Murcko 骨架分组，5 折）** 选出，候选网格 `alpha ∈ logspace(-2, 2.5, 10)`。

**为什么从这里起步**：

1. **数据规模决定了线性模型是合理起点**：n = 3,780、p = 178，样本量不足以支撑深度模型，而描述符已经包含了 logD 的主要物理驱动因素；
2. **共线性必须用正则处理**：描述符之间高度相关（剪枝后仍有残余相关），OLS 方差极大、系数符号可能违背化学常识，因此直接以带正则的 Ridge 作为起点而不是先做 OLS；
3. **可解释**：logD ≈ logP − log(1 + 10^(pKa − 7.4))，亲脂性描述符与 logD 有明确的物理关系，线性系数可以直接对着化学含义讲，便于判断"模型是在学规律还是在记数据"；
4. **成本极低**：CPU 上整条流程（含描述符计算）约 13 s，便于快速迭代和消融。

**需要说明的现状**：本仓库目前只落地了"主模型"这一条路径。`design_plan.md` 里规划的对照模型（单变量 MolLogP 的 OLS、OLS 全特征、Lasso、ElasticNet、Morgan 指纹 + Ridge、随机森林）以及"去掉 MolLogD/MolLogP 的消融实验"**尚未实现**，因此第 8 节只能给出主模型的结果，不做跨模型对比。

---

## 5. 文献启发与模型设计

> ⚠️ **本节需要你核对**：下面写的是**代码里能够验证**的那条文献思路。如果你实际读的是另一篇文章，请把「文章」一栏换成你读的那篇，并说明你从它提取的那一点——这是任务书要求的自述内容，不能由代码反推。

**阅读的文章（代码中已落地的那一点）**：Bemis & Murcko, *J. Med. Chem.* 1996（分子骨架 / scaffold 概念）；MoleculeNet, Wu et al., *Chem. Sci.* 2018（把 Bemis–Murcko scaffold split 定为分子性质预测的标准评估方式）。

**提取的思路**：**评估单元应该是"骨架"，而不是"分子"**。随机划分会把同一骨架的类似物同时放进训练集和测试集，模型只要记住骨架就能拿到虚高的分数；只有用"训练时完全没见过的骨架"做测试，才回答得了真实场景里的问题——我能不能预测一个全新的化学系列。

**如何简化后融入**：

1. 用 `MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)` 抽取每个分子的骨架，按骨架分组（2,216 组，最大组 67 个分子，45% 的组只有 1 个分子）；
2. 组按"组内分子数从大到小"贪心分配到 train/val/test，直到训练集达到 80%——同一骨架绝不跨集合；
3. 把划分结果（seed、各集合索引）落盘 `outputs/splits.json`，任何人可以复现同一划分、审计是否存在泄漏；
4. 不引入文献里的图神经网络/3D 构象部分，**只借"评估口径"这一条**，模型仍是可解释的"描述符 + 线性回归"，这样在有限算力下能把流程做完整。

**实测效果**：测试集 378 条分子中，"骨架已在训练集出现"的比例为 **0.0%**；代价是测试集难度明显高于随机划分（`design_plan.md` 里记录的上一轮随机划分 Morgan 指纹 + Ridge 为 RMSE ≈ 0.99，本次 scaffold split 为 0.8575，两者不可直接比较）。

**简化带来的局限**：45% 的骨架唯一，意味着一次 scaffold split 的方差较大，本应为不同 seed 做 3 次重复并报告均值 ± 标准差；本次只跑了一次（seed=42），这是第 8 节列出的遗留问题。

---

## 6. AI 辅助过程

> ⚠️ **本节需要你按真实情况核对/改写**：下面是依据仓库里可查证的产物（`design_plan.md`、脚本注释、提交内容）整理的过程描述，你实际用了哪些工具、哪些 prompt、怎么核查的，请以你的真实经历为准补充。

**使用的工具**：VS Code + GitHub Copilot（对话式协作，在本工作区内直接读写文件、运行脚本）。

**AI 参与完成的内容**：

| 环节 | AI 的角色 |
| --- | --- |
| `design_plan.md` | 方案设计：特征链、划分策略、评价体系、实验矩阵、风险清单、待拍板的决策点 |
| `clean_lipophilicity.py` | 清洗规则的实现与边界处理（去重顺序、规范化 SMILES、去盐、显式删除清单） |
| `baseline.py` | 主流程实现、CLI、缓存校验、报错信息、推理接口 |
| `data_quality_report.md` / 本 README | 把跑出来的数字整理成可读的结论 |

**prompt 设计（提高结果可靠性的做法）**：

1. **先给约束再要方案**：把"数据结构、样本量、评估口径、算力限制"一次性交代清楚，要求先输出方案 + 待我拍板的问题清单，**不要直接给代码**——这避免了拿到一份看着能跑、但泄漏评估口径的脚本；
2. **要求每条建议附"为什么"和"风险"**：例如"为什么必须 scaffold split""为什么共线性要剪枝"，便于逐条判断是否成立；
3. **代码阶段写硬性验收条件**：单一可运行的 CLI、产物必须落盘、非法 SMILES 必须显式报出数量、所有 `fit` 只能在训练集上、随机种子必须固定；
4. **用"复跑一致性"当验收手段**：同一 seed 重跑，`metrics.json` 必须逐位一致，否则说明存在未固定的随机性或数据依赖。

**如何核查 AI 生成的内容**：

- 每个写进文档的数字都能在 `outputs/metrics.json`、`outputs/test_predictions.csv` 或 `data_qc_summary.json` 里复算，不采信"口头结论"；
- 清洗报告里的每类异常都对应 `Lipophilicity_removed_rows.csv` 中的具体行，可逐条抽查；
- 泄漏检查靠打印指标（测试集中已见骨架比例 0.0%），而不是靠"写代码时注意了"；
- 推理接口用真实构造的坏输入（空 SMILES、非法 SMILES、缺列、空文件）测过，确认报错信息可读；
- 发现的、无法自行验证的点（如"MolLogD 其实等于 logP"）写进第 8 节而不是隐藏。

---

## 7. 实验设置与评价方式

### 7.1 实验设置

| 项目 | 设置 |
| --- | --- |
| 数据 | `Lipophilicity_clean.csv`，3,780 行（清洗后不再去重/去异常，避免二次改动样本集） |
| 划分 | Bemis–Murcko scaffold split，8:1:1，`seed=42` → train 3,024 / val 378 / test 378；骨架组 2,216 个，训练集内 1,460 个组 |
| 泄漏防护 | 同一骨架不跨集合（测试集"已见骨架"比例 0.0%）；填补、去常量、标准化、共线剪枝的统计量**全部只在训练集上 fit** |
| 特征 | 217 个 RDKit 描述符 + 1 列 `MolLogD` = 218 列 → 预处理后 178 列 |
| 模型 | Ridge；`alpha` 由训练集内部 GroupKFold（按骨架分组，5 折）选出，**不使用验证集/测试集调参** |
| 最终模型 | 用 train + val 重训（`alpha` 在 train 上已定），只在最后对 test 预测一次 |
| 运行环境 | Python 3.13.18 + RDKit 2026.09.2 + scikit-learn / numpy / pandas（见 `requirements.txt`、`env.yml`）；纯 CPU，8 进程；单次全流程约 13 s |
| 随机性 | 划分 seed、CV 折叠、`GridSearchCV` 全部固定；重跑结果逐位一致 |

### 7.2 评价指标与判断标准

| 指标 | 作用 |
| --- | --- |
| **RMSE** | 主指标（与任务书一致），单位是 log 单位 |
| MAE | 对离群分子更稳健 |
| R² | 解释了多少方差，便于和其他方法比 |
| Spearman ρ | 只看排序（虚拟筛选更关心排序） |

**怎么算好**：logD 的实验重复性误差约 **0.2 ~ 0.3 个 log 单位**，所以 RMSE 不可能、也没必要降到 0.3 以下，这是天然下界。本次自定的验收线是 scaffold split 测试集 **RMSE ≤ 1.10**（见 `design_plan.md`），实际 0.8575 达标；但 R² 只有 0.505，说明"对全新骨架"仍有一半方差没解释，不能因为 RMSE < 1.1 就认为模型够好。

---

## 8. 实验结果与分析

### 8.1 主要结果（scaffold split 测试集，n = 378）

| 设置 | RMSE | MAE | R² | Spearman ρ |
| --- | --- | --- | --- | --- |
| 验证集（模型只用 train 拟合） | 0.8571 | 0.6639 | 0.4520 | 0.6967 |
| 测试集（train-only 模型） | 0.8698 | 0.6804 | 0.4908 | 0.6823 |
| 测试集（train+val 重训，**最终模型**） | **0.8575** | **0.6651** | **0.5050** | **0.6914** |

逐分子的真实值、预测值与残差见 `outputs/test_predictions.csv`，全部设置与中间指标见 `outputs/metrics.json`。

### 8.2 正则强度的影响（训练集内部 5 折 CV）

| alpha | 0.01 | 0.0316 | 0.1 | 0.316 | **1.0** | 3.16 | 10 | 31.6 | 100 | 316 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CV RMSE | 7.07 | 2.92 | 1.38 | 0.918 | **0.855** | 0.862 | 0.869 | 0.872 | 0.870 | 0.868 |

`alpha` 小于 0.1 时 CV RMSE 直接爆炸（残差相关下系数方差过大），`alpha ≥ 1` 后进入平台区，因此选 1.0。平台很平说明**继续调正则已经没有收益**——瓶颈不在正则，而在特征与模型形式。

### 8.3 train-only vs train+val

把验证集并入训练后，测试集 RMSE 从 0.8698 降到 0.8575（−0.0123），R² 从 0.4908 升到 0.5050。仅仅多了 378 条带标签数据就有小幅提升，说明当前模型更像**数据受限**而不是容量受限；但缺少非线性模型对照，无法判断还有多少提升空间。

### 8.4 残差诊断（最终模型，n = 378）

- 残差均值 −0.032、标准差 0.857，无明显整体偏移；
- 但 `真实值 = 0.910 × 预测值 + 0.172`（斜率 < 1），预测值标准差 0.957 明显小于真实值 1.219 → **两端被向中间压缩**：

| 真实 logD 区间 | n | 平均残差（真实−预测） | RMSE |
| --- | --- | --- | --- |
| [−2, 1) | 64 | **−0.790**（系统性高估） | 1.195 |
| [1, 2) | 84 | −0.473 | 0.835 |
| [2, 3) | 110 | +0.070 | 0.600 |
| [3, 5) | 120 | **+0.588**（系统性低估） | 0.858 |

低 logD（亲水）分子被高估约 0.8 个 log 单位，高 logD（强亲脂）分子被低估约 0.6——这正是"线性模型 + 单一 logP 锚点"的典型失效方式：logD 相对 logP 还取决于 pKa，而 pKa 对结构是非线性的。

### 8.5 目前模型存在的主要问题

1. **核心特征名不副实**：`MolLogD` 实际是 Crippen `MolLogP` 的复制列，并在剪枝中被强制保留（原生 `MolLogP` 作为共线列被丢弃）。模型最强的信号就是 logP 的一个变体，而**"去掉 MolLogD 后剩多少"的消融实验没做**，因此无法回答"模型是不是 logP 的复读机"这个必然会被问到的问题。
2. **只跑了一次划分**：`seed=42` 单次 scaffold split，测试集只有 378 条，且 45% 的分子骨架唯一，单次结果的方差较大；没有做 3 次不同 seed 的均值 ± 标准差。
3. **对照/消融全部缺失**：单变量 MolLogP 的 OLS、全特征 OLS、Lasso、ElasticNet、Morgan 指纹 + Ridge、随机森林，以及 F2（12 个可解释描述符）都只写在 `design_plan.md` 里，尚未实现。因此本文只能报告"一个模型的结果"，不能报告"改进带来的增益"。
4. **两端系统性偏差**（见 8.4）：亲水端高估、亲脂端低估，说明模型倾向于把预测拉向均值，极端分子预测不可靠。
5. **可解释性材料未产出**：系数条形图、预测-真实散点、残差图（`design_plan.md` 第 5.4 节）都还没画；目前只能从 `metrics.json` 和 `test_predictions.csv` 读数字。

### 8.6 如果继续做，优先补的三件事

1. 补 **E6 消融**（去掉 `MolLogD`）与**单变量 logP 基线**——这是判断模型价值的底线证据；
2. 把 `MolLogD` 改回真实语义（就叫 `MolLogP`，不再宣称是 logD），或引入 pKa 相关描述符，正面对付 8.4 的两端偏差；
3. 3 个 seed 重复 + 与 Lasso/指纹/树模型的横向对比，给出均值 ± 标准差，再决定是否值得上非线性模型。

---

## 9. 复现说明

### 9.1 主要文件

| 文件 | 作用 |
| --- | --- |
| `Lipophilicity.csv` | 原始数据（只读，4,200 行） |
| `clean_lipophilicity.py` | 数据质检与清洗脚本，产出下列清洗结果 |
| `Lipophilicity_clean.csv` | 清洗后训练数据（3,780 行，`CMPD_CHEMBLID, exp, smiles`） |
| `Lipophilicity_clean_details.csv` | 清洗后数据 + 质检明细列（规范化 SMILES、重原子数、分子量、片段数、qc_note） |
| `Lipophilicity_removed_rows.csv` | 被删除的每条记录及删除原因 |
| `data_qc_summary.json` / `data_quality_report.md` | 异常统计与完整质检报告 |
| `baseline.py` | 特征工程 + scaffold split + Ridge 训练评估 + 推理接口（主脚本） |
| `design_plan.md` | 设计稿：特征链、划分、评价体系、实验矩阵、风险清单（含未实现部分） |
| `env.yml` / `requirements.txt` | 环境定义（两者依赖版本一致） |
| `outputs/metrics.json` | 全部设置与指标（本次结果的权威来源） |
| `outputs/splits.json` | 可复现的划分（seed、各集合行索引） |
| `outputs/test_predictions.csv` | 测试集逐分子的真实值、预测值、残差 |
| `outputs/model.joblib` | 最终模型 + 预处理器 + 描述符列名（供 `--predict` 使用） |

`outputs/features_cache.npz` 是可重建的描述符缓存，已被 `.gitignore` 排除，首次运行会自动生成。

### 9.2 运行方式

```bash
# 1) 环境（二选一）
pip install -r requirements.txt
# 或
conda env create -f env.yml && conda activate logd_env

# 2) 重新生成清洗结果（可选，仓库里已带产物）
python clean_lipophilicity.py

# 3) 训练 + 评估（约 13 s，纯 CPU），产出 outputs/ 下 4 个文件
python baseline.py

# 可选参数
python baseline.py --seed 7 --corr 0.9 --cv 5 --skew --no-cache
```

### 9.3 基于 `test.csv` 完成测试与报告生成

```bash
python baseline.py --predict test.csv
```

- **输入要求**：与 `Lipophilicity_clean.csv` 同列名——必须有 `smiles`；`CMPD_CHEMBLID`、`exp` 可选。若 `test.csv` 的列名不同（例如 `SMILES`、`logD`），先改列名：`df.rename(columns={"SMILES": "smiles", "logD": "exp"}, inplace=True)`。
- **输出**（都写到 `--out-dir`，默认 `outputs/`）：
  - `predictions.csv`：逐分子预测值 `pred`；输入带 `exp` 时附 `residual`；
  - `invalid_rows.csv`：RDKit 无法解析的 SMILES（**不静默丢弃**，会同时打印条数）；
  - `predict_metrics.json`：有 `exp` 时给出 RMSE / MAE / R² / Spearman、残差均值与标准差、所用 `alpha` 与特征数——报告里的测试数字直接引用它。
- **常见报错**：缺列 / 空文件 / 编码问题会给出明确提示；若提示 "descriptor columns differ from training time"，说明当前 RDKit 版本与训练时不同（本次为 2026.09.2），需用同一版本重新训练。
- **注意**：`--predict` 从 `--out-dir` 内读取 `model.joblib`，所以如果想把结果写到别的目录，需要先把 `outputs/model.joblib` 一起复制过去（例如 `--out-dir results` 时先 `copy outputs\model.joblib results\`）。
- `model.joblib` 里序列化了 `baseline.py` 中定义的 `F1Preprocessor`，因此**只能通过 `python baseline.py --predict ...` 使用**；直接在别的脚本里 `joblib.load()` 会因为找不到 `__main__.F1Preprocessor` 而报错。

### 9.4 复现一致性

同一 `seed`（默认 42）、同一 RDKit 版本重跑 `python baseline.py`，`outputs/metrics.json` 与 `outputs/test_predictions.csv` 逐位一致；本地实测两次结果完全相同（RMSE 0.8575、R² 0.5050）。

## 11.手写内容与真实性声明：**本部分必须由本人独立撰写，不可直接复制 AI 输出。**

请至少写一段，内容可以包括：

- 本次任务中你遇到的最大问题，以及你如何处理
- 你从文章中借鉴的思路为什么值得保留
- 你认为当前模型最薄弱的地方
- 如果继续做下去，你最想改进什么

并请在这里手写承诺你的代码完全符合你声称的功能，并真实得到了你所呈现的结果，并为此负责

> **（以下留空，由本人手写，不引用 AI 生成内容）**
>
> 
