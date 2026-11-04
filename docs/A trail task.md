# A Trail Task

欢迎来到真实的药物筛选建模任务。从这份指南开始，你将尝试第一次去预测分子的性质。

## 总则

1. 不要急于求成。对于初学者而言，这是一个很有挑战性的项目，而且我在里面埋了一点“小坑”。不过只要认真按照指南推进，你大概率能发现它们。所以没有必要急着在一周内完成，更重要的是慢慢探索、尝试和复盘。
2. 善用 AI 工具。我会在教程中专门介绍如何通过 API 配置 Codex，并在 VS Code 中使用。你完全可以把它当作高效工具，而不是把时间都花在手写样板代码上。更值得投入精力的是，如何准确地向 Codex 提需求，也就是 prompt engineering。这本身就是一项很有技术含量的工作。与此同时，请始终记住，**你是代码的唯一责任主体**，AI 只是你的工具。因此我会要求你在提交的 `README.md` 中手写一部分内容，并对代码的真实性和可靠性作出承诺。在真实科研中，放任 AI 生成内容是很容易导致学术不端的，所以请从第一次使用开始就保持警惕。面对大量代码时，如何快速检查代码真实性，也希望听到你的方案。
3. 充分沟通交流。至少到目前为止，我们已经发现了几个很值得讨论的话题：我到底埋了什么坑、怎样做好 prompt engineering、如何快速准确地核查代码。对于研究者或开发者来说，分享与沟通的方式是多元的。你可以在组会中分享自己的实践和收获，也可以在群里、私聊里讨论，还可以去我最近发的项目相关朋友圈留言。你也可以把自己觉得不错的实践发到自己的 GitHub，或者去我的 GitHub 仓库 <https://github.com/Steven-Qu-04/> 看看有没有你感兴趣的项目，提一个 issue 或 comment，顺手点个 star，我会非常感谢。

## 准备你的工作环境

这永远是开始任何工作前的第一步。这里默认你已经掌握了基于 conda 的 PyTorch 基本使用方式；如果还有疑问，请先查看相关教程或直接提出来。这里我会重点介绍一种新的协作方式：Git 和 GitHub。

### 版本管理与 Git/GitHub

你在写代码时，大概率已经遇到过这样的问题：写完一部分代码后，发现里面有错误，或者又有了新的想法，于是开始修改；但修改过程中，可能又引入了更多 bug，或者让问题变得更复杂，最后回退困难、整理痛苦。

一个很直观但有限的做法，是依赖 `Ctrl+C` / `Ctrl+V` / `Ctrl+Z`，或者复制一个新副本继续改。这个方法看起来简单，但一旦你同时有多条思路，或者你和同学在同一份代码上做不同修改，最后还需要合并，就会变得很麻烦。

于是，一些聪明的开发者发明了 Git。简单来说，Git 是一个版本控制工具。只要在项目目录中初始化 Git，它就会跟踪当前目录中文件的状态。当你执行一次 `commit` 并写下对应的 commit message，Git 就会把当时的文件内容和版本信息记录下来。这样你就可以把任意文件回滚到过去某次提交时的状态，只要记得及时提交，就不需要不断复制副本，可以更安心地编辑代码。

GitHub 则是把带有 Git 历史的项目托管到云端的平台。这样你和你的同伴就不需要手动拷贝代码文件，可以直接同步彼此的修改，也方便协作和备份。

参考视频：

- [VScode 与 GitHub：20 分钟学会用 Git 管理自己的代码](https://www.bilibili.com/video/BV1dK411p7RF?vd_source=cf2f65058ff6f4c195b663ba65c0586c)

### 拉取你的作业仓库

这次的作业仓库在：

- <https://github.com/Steven-Qu-04/trail_work>

理论上我已经把大家都加入 collaborators 了。先点击我发在群里的邀请链接，然后在 VS Code 里把仓库 clone 下来即可。

![Clone repository step 1](./A%20trail%20task.assets/image-20260321144942432.png)

![Clone repository step 2](./A%20trail%20task.assets/image-20260321145046031.png)

### 创建属于你的分支

**这很重要。如果你直接在 `main` 上提交代码，请准备承接我的怒火（开玩笑的）。**

![Create a new branch](./A%20trail%20task.assets/image-20260321145246307.png)

先点击左下角的分支图标，再在上方弹出的列表里选择 `Create New Branch`。建议使用自己姓名的拼音作为分支名。

### 配置 Codex

我给每个人都分配了一个密钥，请务必只使用你自己的，不要提供给其他人使用。我可以在后台看到每个人的使用情况。

- 先编辑下面的配置文件；如果不存在就新建，已存在就覆盖。
- macOS / Linux：
  - 配置目录在 `~/.codex/`
  - 创建或覆盖 `config.toml` 和 `auth.json`
- 如果没有 `.codex` 目录，也可以用命令创建：

```bash
rm -rf ~/.codex
mkdir ~/.codex
```

- Windows：
  - 配置目录在 `C:\Users\你的用户名\.codex\`
  - 创建或覆盖 `config.toml` 和 `auth.json`
- 注意：如果你之前登录过官方或其他平台，请先退出登录，再进行下面的配置。

`config.toml` 内容如下：

```toml
model = "gpt-5.4"
model_reasoning_effort = "xhigh"
disable_response_storage = true
sandbox_mode = "danger-full-access"
windows_wsl_setup_acknowledged = true
approval_policy = "never"
profile = "auto-max"
file_opener = "vscode"
model_provider = "codex"
web_search = "cached"
suppress_unstable_features_warning = true

[history]
persistence = "save-all"

[tui]
notifications = true

[shell_environment_policy]
inherit = "all"
ignore_default_excludes = false

[sandbox_workspace_write]
network_access = true

[features]
plan_tool = true
apply_patch_freeform = true
view_image_tool = true
unified_exec = false
streamable_shell = false
rmcp_client = true
elevated_windows_sandbox = true

[profiles.auto-max]
approval_policy = "never"
sandbox_mode = "workspace-write"

[profiles.review]
approval_policy = "on-request"
sandbox_mode = "workspace-write"

[notice]
hide_gpt5_1_migration_prompt = true

[model_providers.codex]
name = "codex"
base_url = "https://gmncode.cn"
wire_api = "responses"
requires_openai_auth = true
```

`auth.json` 内容如下：

```json
{
  "OPENAI_API_KEY": "这里填入你自己生成的、以 sk"
}
```

### 在 Visual Studio Code 中安装 Codex 插件

适合通过 VS Code 图形界面安装 Codex 插件的同学。

- 打开 VS Code 后，先任选一个项目文件夹进入工作区，否则可能会反复弹窗要求你先打开项目目录。
- 打开左侧插件扩展面板。

![Open extensions panel](./A%20trail%20task.assets/clip_image002.jpg)

1. 搜索 `Codex`，看到结果后点击安装。

![Install Codex extension](./A%20trail%20task.assets/clip_image004.jpg)

2. 安装完成后，侧栏会出现 Codex 入口。

![Codex sidebar entry](./A%20trail%20task.assets/clip_image006.jpg)

3. 如果之前登录过官方或其他平台，请先按图示退出登录，再重新配置。

![Sign out before reconfiguring](./A%20trail%20task.assets/clip_image008.jpg)

4. 配置完成后，界面会出现当前卡片，点击 `Continue` 完成登录流程。

![Continue sign-in flow](./A%20trail%20task.assets/clip_image010.jpg)

5. 注意：`.3` 版本的模型可能不会直接显示在下拉列表里，把鼠标移动到自定义项附近时，可以看到当前正在使用的模型。

![Model selection hint 1](./A%20trail%20task.assets/clip_image012.gif)

![Model selection hint 2](./A%20trail%20task.assets/clip_image014.gif)

## 开始任务

### 提示

1. 下面列出的任务要点顺序只是我个人的思路。你不一定要严格按这个顺序推进，也可以从你觉得最容易、最有信心，或者最感兴趣的部分开始。
2. 下面很多问题其实都是真实研究中的难点。所以你的目标并不是在每一个点上都做出完美方案，而是尽量应用你已经学到的知识，体验完整流程，感受研究本身的乐趣。
3. 请在任意步骤中积极使用 AI 来提升效率，但也要始终保持独立思考、主动捕捉灵感，并对 AI 生成内容保持审慎与负责的态度。你是 `AI pilot`，不是 `driven by AI`。
4. 再次欢迎任何友善交流，内容不限于任务本身。比如中午吃什么能不能带来科研灵感，这种话题我也很需要。
5. 这个任务会是一次很有意义的入门尝试。完成之后，你可以考虑把它 fork 到自己的个人仓库里，以免我误删。但在完成过程中，请把你的修改提交到我创建的这个仓库、你自己的分支里，这样我才能看到并和你交流。最后，再求一个 star。

---

## 任务内容

你现在可以看到，这个项目中已经准备好了 `Lipophilicity.csv`。这个数据集的背景如下：

```text
Lipophilicity 是 MoleculeNet 中一个经典的分子性质预测数据集。任务目标是根据分子的结构信息，预测其在 pH 7.4 条件下辛醇/水分配系数的对数值（logD）。这个指标常用于描述分子的亲脂性，数值越大，通常表示分子越倾向进入脂相，因此它在药物发现中经常被用来评估分子的吸收、分布和成药性。

对于初学者来说，可以把它理解成一个单任务回归问题：输入通常是分子的 SMILES 表示，输出是对应的 logD 数值标签。该数据集约包含 4200 个样本，推荐采用 scaffold split（骨架划分）来切分训练集、验证集和测试集，并使用 RMSE 作为主要评估指标。这样更能检验模型对“新骨架分子”的泛化能力，而不是只记住训练集中相似分子。

刚开始使用时，建议先完成 SMILES 规范化、无效分子过滤和基础特征化（例如分子指纹或图表示），再从线性回归、随机森林或简单图神经网络开始，逐步理解“分子结构如何影响脂溶性”的建模思路。
```

为了让这个任务更真实、也更可检验，我对仓库中的数据集做过抽取和处理。**请务必只使用这里提供的数据集完成任务，不要再另外获取或下载别的数据集。**

所以你的任务就是基于这个数据集，构建一个能够从 SMILES 预测 `logD(exp)` 的模型。你可以围绕下面这些问题开展研究和尝试：

1. 这个数据集是否可以直接使用？在正式开始之前，是否需要做必要的预处理和统计分析？注意不要忽略任何一列。
2. SMILES 能否直接输入模型？是否需要对数据结构做转换、补充额外信息，或者进行别的处理？如果需要，可能会用到什么工具？这是一个更偏 CPU 密集还是 GPU 密集的任务？怎样优化运行时间？
3. 如果暂时没有方向，或者希望先跑通一个 baseline，可以考虑 `RDKit + 分子描述符 + 线性回归 / 广义回归 / 随机森林` 这条思路。
4. 除了分子输入形式，模型架构本身也会影响预测准确性。什么样的架构更适合这个任务？
5. 针对上面第 2 和第 4 个问题，你读到的文献里有没有可借鉴的点？不需要完全复现，你可以只抽取一个输入方式上的小技巧，或者一个架构上的启发，构建一个简化的小模型。
6. 模型训练本身也是一门技术。你是如何选择和调整参数的？你的模型是否收敛到了稳定、可用的状态？
7. 你如何验证模型性能？

---

## 交付要求

1. 你的分支中需要包含完整代码，并保证仓库里已经包含运行模型和复现结果所需的必要文件。
2. 需要提供一个检验接口。我会额外给你一个带标签的 `test.csv`，字段与你当前手里的数据集完全一致。请设计一个脚本，让你的模型在拿到这个数据集后可以完成推理，对比预测结果和真实标签，并输出你认为必要的检验报告。请在 `README.md` 中写清楚如何完成这一步。
3. 仓库中的 `README.md` 已经预置了一个模板，其中多数内容你可以借助 AI 帮你整理，但请务必明确两点：
   - 你需要承诺：AI 填写的内容与你实际代码执行过程和实现功能完全一致。
   - `README.md` 中有一个强制手写板块，其中包含承诺声明。请你自己录入这部分内容；如果条件允许，请在仓库中上传一份带手写签名的真实性承诺照片。
4. 请在整个任务中使用一个统一的 conda 环境，并在仓库中提供一份 `env.yml`，记录环境配置，方便我复现你的代码。
