# A Trail Task

Welcome to a real-world drug screening modeling task. Starting from this guide, you will make your first attempt at predicting molecular properties.

## General Principles

1. Do not rush to finish. For beginners, this is a fairly challenging project, and I have intentionally left a few small "traps" in it. Still, if you work through the guide carefully, you should be able to spot them. So there is no need to force yourself to finish everything within a single week. It is more important to explore, experiment, and reflect step by step.
2. Make good use of AI tools. I will specifically explain how to configure Codex through the API and use it inside VS Code. You are absolutely encouraged to use it as a productivity tool instead of spending all your time hand-writing boilerplate code. What matters more is how accurately you communicate with Codex, which is essentially prompt engineering. That is itself a highly technical skill. At the same time, please always remember that **you are the sole person responsible for the code**. AI is only your tool. That is why I will require you to hand-write part of the submitted `README.md` and make an explicit statement about the authenticity and reliability of your code. In real research, blindly accepting AI-generated content can easily lead to serious academic misconduct, so please stay cautious from the very beginning. When dealing with a large amount of code, I would also like to hear your thoughts on how to quickly verify that the code is genuine and trustworthy.
3. Communicate actively. So far, we have already found several topics worth discussing: what kinds of "traps" I planted, how to do prompt engineering well, and how to verify code quickly and accurately. For a researcher or developer, there are many ways to share and communicate. You can talk about your practice and what you learned during group meetings, discuss ideas in group chats or private messages, comment on my recent project-related posts, or share your own good practice on GitHub. You are also welcome to visit my GitHub at [https://github.com/Steven-Qu-04/](https://github.com/Steven-Qu-04/), look for projects that interest you, open an issue or leave a comment, and give the repository a star if you find it useful. I would really appreciate it.

## Prepare Your Working Environment

This is always the first step before starting any real work. Here I assume that you are already familiar with basic PyTorch usage in a conda environment. If not, please review relevant tutorials first or ask directly. In this section, I will mainly introduce a new way of collaborating: Git and GitHub.

### Version Control with Git and GitHub

When writing code, you have probably run into this situation before: you finish part of the code, then notice bugs or come up with a new idea, so you begin editing it again. But during that process, you may introduce even more bugs or make the overall structure harder to manage, and rolling back becomes painful.

One intuitive but limited approach is to rely on `Ctrl+C`, `Ctrl+V`, and `Ctrl+Z`, or to make a full copy of the project and continue editing there. This can seem fine at first, but once you have multiple ideas in parallel, or you and your teammate are modifying the same code in different ways and later need to merge those changes, things quickly become messy.

That is why Git exists. In simple terms, Git is a version control tool. Once Git is initialized in a project directory, it keeps track of file states in that directory. When you create a `commit` and write a commit message, Git records the contents of the files and the corresponding version state at that moment. This allows you to roll any file back to the state of a previous commit. As long as you commit in time, you do not need to keep making copies of your project and can edit with much more confidence.

GitHub is a cloud platform for hosting Git-based projects. It allows you and your collaborators to synchronize code changes directly, without manually sending files back and forth, and it makes collaboration and backup much easier.

Reference video:

- [VS Code and GitHub: Learn Git in 20 Minutes](https://www.bilibili.com/video/BV1dK411p7RF?vd_source=cf2f65058ff6f4c195b663ba65c0586c)

### Clone Your Assignment Repository

The assignment repository for this task is:

- [https://github.com/Steven-Qu-04/trail_work](https://github.com/Steven-Qu-04/trail_work)

In principle, I have already added everyone as collaborators. First, click the invitation link I shared in the group chat, then clone the repository in VS Code.

![Clone repository step 1](./A%20trail%20task.assets/image-20260321144942432.png)

![Clone repository step 2](./A%20trail%20task.assets/image-20260321145046031.png)

### Create Your Own Branch

**This is very important. If you commit directly to `main`, be prepared for my anger. Just kidding, but please do not do that.**

![Create a new branch](./A%20trail%20task.assets/image-20260321145246307.png)

Click the branch icon in the lower-left corner of VS Code, then choose `Create New Branch` from the menu. I recommend using the pinyin of your own name as the branch name.

### Configure Codex

I assigned a separate key to each person. Please use only your own key and do not share it with anyone else. I can see usage records on the backend.

- First, edit the configuration files below. If they do not exist, create them. If they already exist, overwrite them.
- On macOS / Linux:
  - The configuration directory is `~/.codex/`
  - Create or overwrite `config.toml` and `auth.json`
- If the `.codex` directory does not exist, you can also create it with:

```bash
rm -rf ~/.codex
mkdir ~/.codex
```

- On Windows:
  - The configuration directory is `C:\Users\YourUserName\.codex\`
  - Create or overwrite `config.toml` and `auth.json`
- Important: if you previously signed in through the official service or another platform, please sign out first before applying the following configuration.

Contents of `config.toml`:

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

Contents of `auth.json`:

```json
{
  "OPENAI_API_KEY": "Paste your own generated key here, starting with sk-"
}
```

### Install the Codex Extension in Visual Studio Code

This section is for users who prefer installing the Codex extension through the VS Code graphical interface.

- After opening VS Code, first open any project folder as your workspace. Otherwise, VS Code may repeatedly prompt you to open a folder first.
- Open the Extensions panel on the left sidebar.

![Open extensions panel](./A%20trail%20task.assets/clip_image002.jpg)

1. Search for `Codex`, and click install when it appears.

![Install Codex extension](./A%20trail%20task.assets/clip_image004.jpg)

2. After installation, the Codex entry should appear in the sidebar.

![Codex sidebar entry](./A%20trail%20task.assets/clip_image006.jpg)

3. If you previously signed in through the official service or another platform, please sign out first as shown below, then reconfigure.

![Sign out before reconfiguring](./A%20trail%20task.assets/clip_image008.jpg)

4. After configuration, a card like the one below should appear. Click `Continue` to complete the sign-in flow.

![Continue sign-in flow](./A%20trail%20task.assets/clip_image010.jpg)

5. Note: models with a `.3` version suffix may not appear directly in the dropdown list. Move the cursor near the custom item to check which model is currently in use.

![Model selection hint 1](./A%20trail%20task.assets/clip_image012.gif)

![Model selection hint 2](./A%20trail%20task.assets/clip_image014.gif)

## Start the Task

### Notes

1. The order of the task points listed below simply reflects my personal thinking. You do not have to follow that exact order. Feel free to begin with whatever looks easiest, most interesting, or most manageable to you.
2. Many of the points below are real research challenges. So your goal is not to produce a perfect solution for every single one of them, but rather to apply what you have learned, experience the workflow, and get a sense of the fun in research.
3. Please make active use of AI at any stage to improve your efficiency, but always maintain independent thinking, stay alert to new ideas, and take responsibility for AI-generated content. You are an `AI pilot`, not someone who is simply `driven by AI`.
4. Friendly communication of any kind is always welcome, not just task-related discussion. Even questions like what to eat for lunch if it helps spark scientific inspiration are acceptable.
5. This should be a meaningful first attempt at a real research-oriented task. After finishing, you may consider forking it into your own repository in case I accidentally delete something later. But during the task itself, please commit your changes to your own branch in the repository I created, so that I can review your work and communicate with you. And once again, I would appreciate a star.

---

## Task Description

As you can see, this project already contains `Lipophilicity.csv`. The dataset can be described as follows:

```text
Lipophilicity is a classic molecular property prediction dataset from MoleculeNet. The task is to predict the logarithm of the octanol/water distribution coefficient (logD) of a molecule at pH 7.4 based on its structural information. This metric is commonly used to describe molecular lipophilicity. Larger values usually indicate that a molecule tends to partition into the lipid phase, so it is often used in drug discovery to assess absorption, distribution, and drug-likeness.

For beginners, you can think of this as a single-task regression problem: the input is usually the SMILES representation of a molecule, and the output is the corresponding logD value. The dataset contains about 4,200 samples. A scaffold split is recommended for dividing the training, validation, and test sets, and RMSE is recommended as the main evaluation metric. This better measures how well a model generalizes to molecules with new scaffolds, rather than merely memorizing molecules similar to those seen during training.

When getting started, it is recommended that you first perform SMILES normalization, invalid molecule filtering, and basic feature extraction, such as molecular fingerprints or graph representations. Then you can begin with linear regression, random forests, or a simple graph neural network, and gradually develop an intuition for how molecular structure affects lipophilicity.
```

To make the task more realistic and more verifiable, I have already performed some extraction and preprocessing on the dataset in this repository. **Please use only the dataset provided here to complete the task. Do not obtain or download any external dataset instead.**

Your task is therefore to build a model that predicts `logD(exp)` from SMILES based on this dataset. You may explore the task from the following directions:

1. Is this dataset directly usable? Before formally starting, do you need any necessary preprocessing or statistical analysis? Do not ignore any column.
2. Can SMILES be fed directly into a model? Do you need to transform the data structure, add additional information, or perform some other processing? If yes, what tools might be needed? Is this mainly a CPU-intensive task or a GPU-intensive one? How can you optimize runtime?
3. If you do not yet have a clear direction, or if you want to get a baseline running quickly, you may consider a route like `RDKit + molecular descriptors + linear regression / generalized regression / random forest`.
4. In addition to the molecular input format, the model architecture itself will also affect predictive accuracy. What kind of architecture is better suited to this task?
5. Regarding points 2 and 4 above, is there anything in the literature that you can borrow as inspiration? You do not need to reproduce a paper completely. You can simply take one small idea about the input representation or one architectural insight, and build a simplified model from it.
6. Model training is also a technical skill. How did you choose and tune your parameters? Did your model converge to a stable and usable state?
7. How will you verify the performance of your model?

---

## Deliverables

1. Your branch must contain the full code required for the task, and the repository should include all necessary files needed to run your model and reproduce the results you report.
2. You must provide an evaluation interface. I will reserve an additional labeled dataset called `test.csv`, whose columns will be fully consistent with the dataset you currently have. Please design a script so that, once this dataset is provided, your model can perform inference on it, compare predictions with the true labels, and output whatever validation report you consider necessary. Please clearly explain in `README.md` how this should be done.
3. The repository already contains a `README.md` template. You may use AI to help draft most sections, but you must make two things explicit:
   - You must promise that all AI-written content is fully consistent with what your code actually does and with the results you present.
   - There is a mandatory handwritten section in `README.md`, including a declaration of authenticity. You must write that part yourself. If possible, please also upload a photo of a handwritten and signed authenticity statement to the repository.
4. Please use one consistent conda environment throughout the task, and include an `env.yml` file in the repository so that I can reproduce your environment and run your code.
