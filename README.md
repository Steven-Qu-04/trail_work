# Lipophilicity 分子亲脂性预测

## 1. 任务简介

本项目使用老师提供的 `data/raw/Lipophilicity.csv`，根据分子的 SMILES 结构预测
pH 7.4 条件下的 logD（数据列 `exp`）。当前已完成数据审计、清洗、结构
异常筛查、固定 Scaffold Split、Morgan 指纹与RDKit描述符提取、多模型比较、
XGBoost优化以及XGBoost-SVR加权融合。默认交付模型为最终融合模型。

---

## 2. 数据集检查与预处理

原始数据共 4200 条。检查发现缺失或无法转换的标签、缺失 SMILES、
异常占位标签、完全重复记录、无效 SMILES、两组重复分子的冲突标签，
以及两条分别包含 6000 和 4612 个重原子的极端超长碳链。

原始文件保持不变。冲突标签暂以平均值合并；超长碳链使用“重原子数大于
200”规则排除，并保存在 `data/audit/structural_outliers.csv`。最终建模数据
`data/processed/Lipophilicity_model_filtered.csv` 包含 3780 个分子。

---

## 3. 分子表示与输入处理

使用 RDKit 将 SMILES 解析为 Mol，再生成 `radius=2`、`fpSize=2048` 的
Morgan位指纹。最终XGBoost另外加入分子量、TPSA、氢键供体/受体数、可旋转
键数、环数、芳香环数、FractionCSP3、重原子数和形式电荷。最终优化模型使用
4096位Morgan指纹，共形成4106维
特征。没有加入与目标过于接近的RDKit MolLogP。训练和预测共同调用
`src/features.py`，预测时从模型文件读取完整特征配置。全部处理在CPU上完成。

---

## 4. Baseline 模型

使用 `RandomForestRegressor` 作为第一个机器学习基线，同时使用训练标签均值
作为最低参照。在验证集上比较 11 组参数后，最终参数为 300 棵树、
`max_features=0.1`、`min_samples_leaf=1`、不限制树深度、
`random_state=42`。完整搜索结果保存在 `reports/baselines/rf_tuning_results.csv`。

---

## 5. 文献启发与模型设计

本项目的方法设计主要参考以下三项工作：

1. Wu等人的MoleculeNet论文（DOI：
   [10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A)）将Lipophilicity列为
   分子性质回归任务，并说明Scaffold Split能够把不同二维骨架分配到不同集合，
   比随机划分更严格地检验结构外推能力。本项目据此使用固定80%/10%/10%的
   Bemis-Murcko Scaffold Split，并在训练集内部继续使用Scaffold GroupKFold选参。
2. Rogers和Hahn提出的扩展连接指纹ECFP（DOI：
   [10.1021/ci100050t](https://doi.org/10.1021/ci100050t)）使用局部原子环境表示分子
   子结构。本项目使用RDKit Morgan指纹作为相应的圆形指纹实现，并通过消融实验
   比较半径、位数和计数形式；最终采用`radius=2`、4096位指纹。
3. Chen和Guestrin的XGBoost论文（DOI：
   [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)）给出了带正则化的
   梯度提升树系统。本项目使用XGBoost学习稀疏指纹和理化描述符之间的非线性关系，
   并利用子采样、列采样、树深和正则项控制过拟合。

为保持CPU实验规模可控，本项目没有直接复现复杂图神经网络，而是将上述思想简化为
“Morgan指纹 + 10个RDKit理化描述符 + XGBoost”。最后只根据训练集的Scaffold OOF
预测确定XGBoost 0.85、RBF-SVR 0.15的融合权重。

---

## 6. AI 辅助过程

本项目使用OpenAI Codex辅助完成代码检查、数据处理流程整理、模型训练脚本编写、
参数实验组织、错误排查和README整理。向AI提出的任务尽量包含明确的数据文件、目标列、
划分要求、评价指标和“测试集不得参与选参”等限制，避免只要求笼统地“提高准确率”。

AI辅助生成的代码通过以下方式核查：

- 对数据清洗前后的行数、排除记录和Scaffold交叉情况进行程序化检查；
- 实际运行每个训练脚本，并将RMSE、MAE和R²写入JSON/CSV报告；
- 分离训练、验证和内部测试用途，参数与融合权重不根据内部测试集选择；
- 使用模型文件中的特征配置进行推理，避免训练和预测特征不一致；
- 使用带标签、原始三字段格式的临时CSV回归测试检验接口；
- 使用语法编译检查和默认模型回归测试确认目录整理没有改变预测结果。


---

## 7. 实验设置与评价方式

采用固定 Bemis-Murcko Scaffold Split：训练集 3024 条、验证集 378 条、
内部测试集 378 条，比例为 80%/10%/10%，三个集合之间没有 scaffold 交叉。
训练和验证完成后，使用二者重新拟合最终模型，内部测试集只用于最终评估。
评价指标为 RMSE、MAE 和 R²，并与均值预测基线比较。

---

## 8. 实验结果与分析

当前 Random Forest 基线的实际结果：

| 数据集 | RMSE | MAE | R² |
|---|---:|---:|---:|
| 训练集（重新拟合前） | 0.3105 | 0.2337 | 0.9360 |
| 验证集（重新拟合前） | 0.8351 | 0.6520 | 0.3688 |
| 内部 Scaffold 测试集（最终模型） | 0.8880 | 0.7181 | 0.4054 |
| 内部测试集均值基线 | 1.1584 | 0.9388 | -0.0119 |

原始随机森林的验证集 RMSE 为 0.8491、内部测试集 RMSE 为 0.9039。调参后
分别下降到 0.8351 和 0.8880。模型明显优于均值基线，但训练集与验证集之间
仍存在较大差距，说明当前模型有一定过拟合。

进一步使用相同数据划分和 Morgan 指纹训练 Extra Trees，并只根据验证集
RMSE 从 11 组参数中选择最终配置。结果对比如下：

| 模型 | 验证 RMSE | 验证 R² | 内部测试 RMSE | 内部测试 MAE | 内部测试 R² |
|---|---:|---:|---:|---:|---:|
| 优化后 Random Forest | 0.8351 | 0.3688 | 0.8880 | 0.7181 | 0.4054 |
| Extra Trees | 0.8173 | 0.3955 | 0.8681 | 0.6976 | 0.4317 |
| RBF-SVR | 0.7540 | 0.4854 | 0.8549 | 0.6535 | 0.4488 |
| XGBoost | 0.7921 | 0.4322 | 0.8204 | 0.6454 | 0.4924 |
| XGBoost + RDKit描述符 | **0.6798** | **0.5817** | **0.7471** | **0.5729** | **0.5791** |

Extra Trees 的最终参数为 500 棵树、`max_features=0.1`、
`min_samples_leaf=1`、不限制树深度且不使用 bootstrap。它在验证集和内部
测试集上均优于随机森林，但训练集 R² 为 0.9997，显示出较强的过拟合倾向。

SVR比较了18组 `C`、`gamma`、`epsilon` 和核函数组合。最优模型使用RBF核、
`C=30`、`gamma=0.01`、`epsilon=0.1`，验证集表现为四种模型中最好。其训练集
R² 为0.9933，仍有过拟合风险。

XGBoost比较了12组参数并使用验证集早停。最优配置为1232棵树、
`max_depth=5`、`learning_rate=0.05`、`subsample=0.8`、
`colsample_bytree=0.6`、`reg_lambda=5.0`。XGBoost的内部测试结果最好，且训练
与验证之间的差距小于其他模型，因此该阶段选择XGBoost继续优化；后续再通过
特征消融和OOF融合形成最终交付模型。最终泛化能力仍需由尚未使用的外部
`test.csv`独立确认。

进一步的特征消融使用训练集内部5折Scaffold GroupKFold比较10组表示。最优表示为
`radius=2`、4096位Morgan指纹加10个RDKit理化描述符，共4106维；随后比较7组
XGBoost参数。最优模型使用1369棵树、`max_depth=5`、`learning_rate=0.05`、
`min_child_weight=3`、`reg_lambda=5.0`和`reg_alpha=0.5`。其Scaffold CV RMSE为
0.7144±0.0323，独立验证RMSE为0.6527，内部测试RMSE为0.7489。

最后使用相同Scaffold折生成OOF预测，只根据训练集OOF RMSE选择融合权重。最优权重为
XGBoost 0.85、RBF-SVR 0.15；融合后的独立验证RMSE为0.6490，内部测试RMSE为
0.7401、MAE为0.5735、R²为0.5869。测试集未参与特征、参数或融合权重选择。

---

## 9. 复现说明

所有命令均在项目根目录运行。首次使用时根据仓库中的`env.yml`创建统一环境：

```bash
conda env create -f env.yml
conda activate logd
```

如果本机已存在`logd`环境，可执行：

```bash
conda env update -n logd -f env.yml --prune
conda activate logd
```

### 9.1 从原始数据复现训练结果

依次执行：

```bash
python scripts/preprocessing/clean_data.py
python scripts/preprocessing/check_smiles.py
python scripts/audit/audit_duplicates.py
python scripts/preprocessing/prepare_model_data.py
python scripts/preprocessing/filter_structural_outliers.py
python scripts/preprocessing/split_data.py
python src/train.py
python src/train_extra_trees.py
python src/train_svr.py
python src/train_xgboost.py
python src/train_xgboost_descriptors.py
python src/optimize_xgboost_features.py
python src/train_ensemble.py
```

如只需检查仓库中已经训练好的最终模型，可执行`python src/predict.py`。该命令默认
使用`data/processed/scaffold_test.csv`，并把结果写入`reports/production/`。

### 9.2 老师提供的带标签test.csv检验接口

外部测试文件应命名为`test.csv`并放在项目根目录，至少包含以下三列：

```csv
CMPD_CHEMBLID,exp,smiles
CHEMBL_EXAMPLE,2.31,CCOc1ccccc1
```

- `CMPD_CHEMBLID`：分子编号，可以是任意字符串；
- `exp`：真实logD标签，必须能够转换为数值；
- `smiles`：用于生成模型特征的SMILES结构。

正式检验命令为：

```bash
python src/predict.py --input test.csv --require-labels \
  --model models/production/xgboost_svr_ensemble.joblib \
  --output reports/production/external_test_predictions.csv \
  --report reports/production/external_test_report.json
```

Windows PowerShell也可以写成一行：

```powershell
python src/predict.py --input test.csv --require-labels --model models/production/xgboost_svr_ensemble.joblib --output reports/production/external_test_predictions.csv --report reports/production/external_test_report.json
```

`--require-labels`用于正式带标签评估：如果缺少`exp`列，或者有效标签不足2条，脚本会
直接报错，避免误把纯预测当成模型检验。无效SMILES不会使整批任务失败，而会在结果中标记。

脚本生成两个文件：

1. `external_test_predictions.csv`：保留输入字段，并增加`predicted_exp`、
   `prediction_status`、`residual`和`absolute_error`；
2. `external_test_report.json`：记录输入和模型路径、总行数、有效预测数、无效SMILES
   的CSV行号、实际参与评估的行数、RMSE、MAE、R²和不同误差阈值的样本数。

其中`residual = exp - predicted_exp`；RMSE和MAE越低越好，R²越接近1越好。

随机森林、Extra Trees、SVR、基础XGBoost、增强XGBoost和融合模型分别保存在
`models/baselines/`、`models/components/`和`models/production/`。对应的内部评估报告
按相同用途保存在`reports/baselines/`、`reports/xgboost/`和`reports/production/`。
预测接口默认使用最终选定的XGBoost+SVR融合模型；如需复核其他模型，
可通过 `--model` 显式指定。

## 10. 项目目录

- `data/raw/`：原始数据，只读保留
- `data/interim/`：清洗、校验和合并后的中间数据
- `data/processed/`：最终建模数据及固定Scaffold划分
- `data/audit/`：排除记录、冲突标签和结构异常审计
- `scripts/preprocessing/`：数据处理主流程
- `scripts/audit/`：探索和质量检查脚本
- `src/`：特征、训练、优化、融合与预测代码
- `models/production/`：默认交付模型
- `models/components/`：融合模型的组成模型
- `models/baselines/`：历史对比模型
- `reports/`：按production、xgboost和baselines分类的实验结果

## 11. 手写内容与真实性声明：**本部分必须由本人独立撰写，不可直接复制 AI 输出。**


本次实验中，我遇到的最大问题是模型过拟合。最初训练随机森林、Extra Trees和SVR时，模型在训练集上的R²很高，但验证集和Scaffold测试集上的表现明显下降。例如，Extra Trees的训练集R²接近1，而验证集R²只有约0.40，说明模型更倾向于记住训练分子的结构特征，无法稳定预测具有新骨架的分子。为缓解这一问题，我没有使用容易产生数据泄漏的随机划分，而是采用Bemis–Murcko Scaffold Split，将不同分子骨架分配到训练集、验证集和测试集。同时，我在XGBoost中调整树深、子采样、列采样和正则化参数，并使用训练集内部的Scaffold交叉验证选择参数。之后加入RDKit理化描述符，并通过OOF预测确定XGBoost与SVR的融合权重。最终融合模型在内部Scaffold测试集上的R²约为0.59、RMSE约为0.74，泛化能力比早期模型有所改善，但过拟合问题并没有完全消失。对于训练集中较少出现的新骨架、带电分子和复杂结构，模型仍可能产生较大误差。因此我认为该模型更适合用于初步筛选，而不能替代真实的logD实验。
如果继续改进，我希望加入适用域判断和预测不确定性，使模型在给出预测结果的同时，也能说明该结果是否可信。然后可以扩大训练数据，调整模型使用图神经网络等观察是否可以提高预测准确度。
我承诺已检查主要代码、确认README与真实运行过程一致、确认实验指标来自实际报告没有伪造数据或结果，愿意对提交内容负责。