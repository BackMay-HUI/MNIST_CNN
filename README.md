# PyTorch 实战：MNIST 手写数字识别

一本**面向初学者的参考书** + 一整套**可以直接跑的代码**。
用 PyTorch 的 CNN 跑通 MNIST 手写数字识别，覆盖入门任务要求的 7 个知识点：
数据加载、数据增强、模型初始化、模型构建、超参数调节、训练和测试、模型性能测试。

**不是零散脚本的堆砌** —— 是可以从头读到尾的一本书，配 9 个互相独立、都能单独运行的脚本。

---

## 这本书适合谁

刚配好 Anaconda + PyTorch + PyCharm 环境、要完成"跑通第一个神经网络"任务的初学者。

**深度契约**（本书刻意划的边界）：

- ✅ 讲清**每个 API 的参数语义** —— 因为调超参依赖这个
- ❌ **不推导公式**、不手算卷积输出尺寸、不讲感受野和反向传播数学
- 🎯 目标是"**知道它在干嘛、会调 API**"，不是"会证明它"

想看数学推导的话，附录 C 推荐了《动手学深度学习》——
你在本书里见过的每个概念，那里都有理论版。

---

## 目录

| 章 | 标题 | 讲什么 | 脚本 |
|---|---|---|---|
| 0 | 环境：先把地基打对 | conda 环境、`is_available()` 的陷阱、随机种子 | `code/ch00_env_check.py` |
| 1 | 数据加载：把原材料搬进流水线 | `Dataset` / `DataLoader` / `ToTensor` / `Normalize` | `code/ch01_data.py` |
| 2 | 全连接网络与训练循环 | `nn.Module`、**训练循环五步**、`train()`/`eval()` | `code/ch02_mlp.py` |
| 3 | 卷积神经网络 | 卷积在干嘛、四个旋钮、池化、**权重初始化四档实测** | `code/ch03_cnn.py` |
| 4 | 数据增强 | 为什么只能对训练集做、MNIST 上翻转为什么有害 | `code/ch04_augment.py` |
| 5 | 超参数调节 | 学习率 / batch size / 优化器 / 调度器、early stopping | `code/ch05_hyperparams.py` |
| 6 | 正则化与结构改进 | BatchNorm、Dropout、加深一层 | `code/ch06_regularize.py` |
| 7 | 模型性能测试与误差分析 | 准确率不够用、混淆矩阵、逐类指标、错例、t-SNE | `code/ch07_evaluate.py` |
| 8 | 从实验代码到工程结构 | 把 700 行脚本拆成 5 个文件、依赖方向、`Config` | `code/ch08_project/train.py` |
| 附录 A | 知识点索引 | 任务书那 7 个知识点 ↔ 本书章节 的反查表 | — |
| 附录 B | 常见报错速查 | **22 条实际撞到过的报错**：现象 → 原因 → 解决 | — |
| 附录 C | 延伸资源 | 想补理论 / 查 API / 做下一个任务分别去哪 | — |

正文在 `book/`，代码在 `code/`，两者一一对应。

**只有半天时间？** 看附录 A 的 §A.3「如果只有半天时间」。

---

## 快速开始

### 1. 环境准备

**先读第 0 章**（`book/00_环境.md`）。它讲的三个坑会直接影响你能不能跑起来：

1. **必须用 conda 的 `pytorch` 环境**，不是 PATH 上的 `python`
2. **`torch.cuda.is_available()` 返回 `True` 也不能信** —— 本项目就是从"打印 `True` 但卷积全炸"的环境起步的（换 cu128 才修好）；所以每个脚本都用 `get_device()` 跑一次真实卷积来探设备，而不是只问 `is_available()`
3. Windows 上乱码、文件占用、下载卡住怎么办

验证环境（在**仓库根目录**下运行）：

```bash
C:/Users/ASUS/.conda/envs/pytorch/python.exe code/ch00_env_check.py
```

### 2. 怎么跑代码

<blockquote class="tip">

**💡 每个脚本都能单独跑，互不依赖**

唯一的例外是第 7 章要加载第 5 章存下的 `results/best_cnn.pt`。
没有的话它会现场补训，并**打印醒目警告**告诉你"这一份不是书里的模型、数字对不上"。

</blockquote>

#### 方式一：命令行

```bash
# 用 conda 那个环境的解释器（路径按你自己的装法改）
C:/Users/ASUS/.conda/envs/pytorch/python.exe code/ch02_mlp.py --epochs 10

# 快速试一下（1 个 epoch + 512 张样本，几十秒）
C:/Users/ASUS/.conda/envs/pytorch/python.exe code/ch02_mlp.py --smoke
```

每个脚本都支持 `--help`：

```bash
C:/Users/ASUS/.conda/envs/pytorch/python.exe code/ch03_cnn.py --help
```

<blockquote class="warn">

**⚠️ 别用裸 `python`**

本机 PATH 上的 `python` 是 Anaconda **base** 环境，里面**没有** torch，
跑起来会报 `ModuleNotFoundError: No module named 'torch'`。
**这不是 torch 装坏了，是跑错了解释器**（附录 B 第 1 条）。

</blockquote>

#### 方式二：PyCharm

1. 用 PyCharm **打开文件夹** `MNIST_CNN`
2. `File → Settings → Project → Python Interpreter`
3. 选 **conda 环境 `pytorch`**（不是 base）
4. 右键脚本 → `Run`。想传参数就在 `Run → Edit Configurations` 里填 `Parameters`

<blockquote class="warn">

**⚠️ `--smoke` 的产物不能用来下结论**

冒烟模式只跑 1 个 epoch、512 张样本，**它的准确率没有任何参考价值**。
它的产物一律带 `_smoke` 后缀（如 `ch02_curve_smoke.png`），
和书里的正式图物理隔离，不会互相覆盖。

</blockquote>

### 3. 怎么导出 PDF

两条路，产物不一样。

**A. 整本书 —— 一份带封面、目录、页码的 PDF（推荐）**

```bash
bash tests/build_book_pdf.sh
# -> book/PyTorch_MNIST_全书.pdf   （约 160 页）
```

依赖 `pandoc` + Node 22+ + Edge（Windows 11 自带）。**不需要 VSCode**，
也不需要装任何 Python 包。管线是：

```
book/*.md --拼接--> pandoc --> HTML --> Edge(DevTools 协议) --> PDF
```

想顺带截几张图核对排版：

```bash
bash tests/build_book_pdf.sh book/out.pdf _shots sel:img sel:table sel:blockquote.warn
```

`sel:` 后面写 CSS 选择器，脚本会滚到第一个匹配的元素再截图
（也可以在 `sel:` 的位置写 `y:8000` 直接给纵坐标）。

排版文件都在 `styles/`：

| 文件 | 作用 |
|---|---|
| `book.css` | 全部样式。**两条导出路径共用这一份** |
| `pdf-template.html` | 合并导出的页面骨架（为什么不用 pandoc 默认模板，文件里有注释） |
| `cover.html` | 封面 |
| `print_pdf.js` | 驱动 Edge 出 PDF，负责页码 |

<blockquote class="warn">

**改 `book.css` 要重新导一次才看得到** —— 它是运行时读的，不参与构建缓存。

另外：pandoc 会把正文里的直引号 `"` 转成中文弯引号 `“ ”`（本书正文有 1372 处），
中文排版这样才对。VSCode 那条路（markdown-it）不做这个转换，所以两条路的
引号长相会有差别，属于预期。

</blockquote>

**B. 单章，在 VSCode 里导 —— 只想单独看某一章时用**

需要 VSCode + **Markdown PDF** 插件（`yzane.markdown-pdf`）。

1. 用 VSCode **打开文件夹** `MNIST_CNN`
   （**必须打开文件夹**，不能只打开单个 `.md`，也**不能打开它的上级目录** ——
   工作区设置只认被打开的那一层。`MNIST_CNN/.vscode/settings.json` 里已经配好了，
   打开对文件夹就自动生效，不用再手工合并任何东西）
2. 打开 `book/` 下任意一章 `.md`
3. `Ctrl+Shift+P` → `Markdown PDF: Export (pdf)`

那份工作区配置做了四件事（文件里有逐条注释）：指定 `styles/book.css`、关掉插件自带样式、
指定用 Edge 出 PDF（不指的话插件会去下载约 150MB 的 Chromium）、设好 A4 和页边距。
另外它还给 `markdown.styles` 接了同一份 CSS，所以 `Ctrl+Shift+V` 的**预览**也是排好版的。

---

## 建议阅读顺序

**完整读**（正文约 2 小时，加上跑代码约半天）：

```
第 0 章 → 第 1 章 → 第 2 章 → 第 3 章 → 第 4 章
        → 第 5 章 → 第 6 章 → 第 7 章 → 第 8 章 → 附录 A / B / C
```

**只想跑通任务**（跳过第 6、8 章，它们不影响"跑通"）：

```
第 0 章 → 第 1 章 → 第 2 章 → 第 3 章 → 第 5 章 → 第 7 章
```

**第 2 章是重点** —— 训练循环的骨架后面 6 章都不会再变，
后头换的只是"网络长什么样"和"旋钮拧到多少"。

---

## 目录结构

```
MNIST_CNN/
├── book/          正文（Markdown，9 章 + 3 个附录）
├── code/          配套脚本（每章一个，可单独运行）
│   └── ch08_project/    第 8 章的工程结构版（5 个文件）
├── styles/        PDF 排版用的 CSS + VSCode 配置模板
├── tests/         自动检查（见下）
├── figures/       运行产物：所有插图   ┐
├── results/       运行产物：日志与权重  ├ 不进 git，跑一遍就有
└── data/          运行产物：MNIST 数据 ┘
```

`figures/`、`results/`、`data/` 都是**可再生产物**，没有提交到仓库。
刚克隆下来时它们不存在 —— 跑一遍脚本就有了。

<blockquote class="tip">

**💡 想留一份自己的日志**

书里各章提到的 `results/xxx.txt` 都是这么来的（`tee` 把输出同时打到屏幕和文件）：

```bash
python code/ch02_mlp.py --epochs 10 | tee results/ch02_full.txt
```

</blockquote>

---

## 自动检查

```bash
PY="/c/Users/ASUS/.conda/envs/pytorch/python.exe"

# 书稿的提示框写法是否合法（12 个文件）
$PY tests/check_book.py

# 守卫回归：除了书里那条命令，任何跑法都不许写正式文件名
$PY tests/test_canonical_guard.py

# 全部脚本的冒烟测试（每个跑 1 个 epoch，确认没坏、也没碰坏书里的图）
$PY tests/smoke_test.py

# 用书里写的那条命令重新生成全部插图和日志
bash tests/regen_all.sh
```

`regen_all.sh` 是给导出 PDF 之前用的：
开发期间跑过的探索性命令可能把某些图换成了非正式版本，
这个脚本用正式命令把它们全部重新生成一遍。

<blockquote class="tip">

**💡 这三条检查在防同一件事**

书里引用的每一张图、每一个日志、每一个 checkpoint，都只许由**书里写的那条命令**产出。
其余任何跑法（`--smoke`、改轮数、换种子……）一律用带后缀的文件名，物理隔离。

这条规矩很容易被破坏 —— 比如"新增了一个参数，忘了把它加进判断条件"。
所以它有三个层次的保险：正文里讲清楚（第 2 章常见坑 5）、
`is_canonical` 写成一张表、`tests/test_canonical_guard.py` 把每个开关都列成一个用例。
`smoke_test.py` 再在每次冒烟前后比一遍正文引用到的产物的 md5。

**这些覆盖是静默的**：不报错、不提示，等你发现时原图已经没了。所以必须自动查。

</blockquote>

---

## 关于书里的数字

**所有数字都是实测的，没有一个是编的。** 每条结论后面都标了实验条件
（轮数、种子、数据量）。

有两点读的时候要注意：

1. **看噪声带**：同一个模型换个种子，准确率会波动 **0.43 个百分点**。
   小于这个数的差异**不能当结论**。逐类指标（比如"6 的准确率"）
   因为样本少，噪声带更宽，**约 1 个百分点**。
2. **时长不可复现**：准确率是算出来的、每次都一样；
   但"每个 epoch 跑了几秒"取决于你机器的负载，**只有这一列会变**。

跑出来的数字和书里对不上时，先看附录 B 第 21 条。

---

## 下一步

实验室的后续任务是 **PoolNet**（池化方式的研究）：<https://github.com/backseason/PoolNet>

第 8 章 §8.7 专门写了"怎么把本书的结构用到 PoolNet 上"——
哪些文件要改、哪些一个字都不用动。详见附录 C §C.3。
