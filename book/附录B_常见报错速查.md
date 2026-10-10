# 附录 B　常见报错速查

这份表里的**每一条都是本书实际撞到过的**（写在各章"常见坑"小节里，
或是在搭建这本书的过程中踩出来的）。没有一条是"网上常见"式地凑数的 ——
编出来的报错最坑人：你真遇到时按它对不上，反而更迷糊。

**怎么用**：按报错信息里的关键词在本文页内搜索（Ctrl+F）。分不清属于哪类就先看 B.4 的排查套路。

| 你的症状 | 去哪一节 |
|---|---|
| 装不上 / 跑不起来 / 找不到模块 | **B.1 环境与安装**（第 1~7 条） |
| 程序抛异常、Traceback 一长串 | **B.2 运行时报错**（第 8~13 条） |
| **能跑完，但结果明显不对** | **B.3 不报错但结果是错的**（第 14~22 条） |

<blockquote class="warn">

**⚠️ 先记住这一条：跑错解释器是最常见的原因**

本书实测的第一坑：报 `ModuleNotFoundError: No module named 'torch'`，
九成不是 torch 装坏了，而是**用错了 Python**。详见 B.1 第 1 条。

</blockquote>

---

## B.1 环境与安装

### 1. `ModuleNotFoundError: No module named 'torch'`

**原因**：跑脚本用的不是装 torch 的那个环境。本机 PATH 上的 `python` 是 Anaconda **base** 环境，
里面**没有** torch；torch 装在 `pytorch` 这个 conda 环境里。

**解决**：

- 命令行：显式用那个环境的解释器，别用裸 `python`
  ```bash
  C:/Users/ASUS/.conda/envs/pytorch/python.exe code/ch02_mlp.py --smoke
  ```
- PyCharm：`File → Settings → Project → Python Interpreter`，选 `pytorch` 环境
- 验证：`python -c "import torch; print(torch.__version__)"` 打出 `2.8.0+cu128` 才对

---

### 2. `error: no kernel image is available for execution on the device`

**原因**：**装的那个 torch，在编译时没有把你显卡的算力编进去。**
本机是 RTX 5060（算力 sm_120），而某些 cu126 的 wheel 只编到 sm_90。

于是出现了这个分裂现象：CUDA 仍然能直通 cuBLAS（`matmul`、`nn.Linear` 走的就是 cuBLAS，照跑），
但 PyTorch 自己编的算子（`ReLU` / `Conv2d` / `MaxPool2d` / `BatchNorm2d` / `CrossEntropyLoss`）
找不到对应的 kernel，全部报这个错。**这是本书最隐蔽的坑**：

```
torch.cuda.is_available()  ->  True      ← 骗你的
ReLU / Conv2d / ...        ->  no kernel image
```

所以**不能只信 `is_available()`**：它返回 `True` 只说明驱动能看到卡、装的是 CUDA 版，
**不代表算法能跑**。一路要等到训练时才炸，前面几十分钟全白费。

**解决**：**换成编进了你这张卡算力的 wheel。** 本书就是换到 `cu128` 才修好的
（第 0 章 §0.3 有完整步骤）—— 注意这不是"GPU 坏了"，换 wheel 就好，**别因此去强制用 CPU**。

在那之前，程序里也要有兜底：用 `get_device()` 探测 —— 真跑一个小的卷积前向，跑通才算能用：

```python
conv = nn.Conv2d(1, 4, 3).cuda()
y = torch.relu(conv(torch.randn(2, 1, 28, 28, device="cuda")))   # 跑得通吗？
```

本书所有脚本都带这个探测，失败自动降级 CPU。

**想强制用 CPU**：`CUDA_VISIBLE_DEVICES=-1`（不是在代码里改）。
**注意 `CUDA_VISIBLE_DEVICES=""`（空字符串）在 Windows 上无效** —— 空串被当成"没设置"，
`is_available()` 依然是 `True`。实测：

```
CUDA_VISIBLE_DEVICES=""      -> is_available = True     ← 没屏蔽掉
CUDA_VISIBLE_DEVICES="-1"    -> is_available = False    ← 正确
CUDA_VISIBLE_DEVICES="none"  -> is_available = False    ← 也可以
```

---

### 3. MNIST 下载卡住不动（或最终只下到一半）

**原因**：torchvision 会依次尝试几个下载源，其中 `download.pytorch.org` 在国内**直接返回 403**，
而它的失败重试有**很长的超时**。本书实测：卡了整整 10 分钟没有任何输出。

**解决**：手动下载那 4 个文件，跳过 `download` 那一步。用 `curl` 从 Stanford 镜像拿，
实测 14 秒下完（torchvision 自己 10 分钟都没搞定）：

```bash
cd "D:/Learning/Pytorch/MNIST_CNN/data/MNIST/raw"
BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do
  curl -L -O "$BASE/$f"
done
```

**注意：不要解压**，torchvision 要的就是 `.gz` 原文件。下好之后再跑脚本，它不会再下载。

---

### 4. `OSError: [WinError 126] ... caffe2_nvrtc.dll`（安装过程中）

**原因**：**这不是坏了。** pip 装大包时会先卸载旧的、再装新的，中间有个短暂窗口
`torch` 目录不完整。

**解决**：**等 pip 跑完再试**。不要以为装坏了就去重装 —— 那只会把窗口拉得更长。

---

### 5. `RuntimeError: DataLoader worker (pid xxx) is killed by signal`

**原因**：`num_workers > 0` 在 Windows 上要 spawn 子进程，教程环境里容易出各种怪问题。

**解决**：`num_workers=0`（在主进程里读数据）。MNIST 小到不需要多进程 ——
生产环境可以调到 4 或 8，但学习阶段没必要。

---

### 6. 输出的中文变成乱码（`CNN ǰ��+����`）

**原因**：Windows 上 Python 的 stdout 默认用 **gbk** 编码，管道重定向时中文就花了。

**解决**：脚本头部加这三行，运行时再设个环境变量：

```python
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
```

```bash
PYTHONIOENCODING=utf-8 python code/ch01_data.py | tee results/ch01_stats_full.txt
```

（重定向到文件时尤其要注意 —— 不设的话存下来的日志本身就是乱码。）

---

### 7. `UserWarning: Glyph 38598 ... missing from font(s)`

**原因**：matplotlib 默认字体 DejaVu Sans **没有中文字形**，图上的中文全变成方块（□□□）。

**解决**：在想画图之前设好字体（本书每个出图脚本都带这几行）：

```python
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False    # 否则负号也是方块
```

---

## B.2 运行时报错

### 8. `AttributeError: cannot assign module before Module.__init__() call`

**原因**：在 `super().__init__()` **之前**就给子模块赋值了。`nn.Module` 的魔法都在
`__init__` 里准备好，没调它之前，`self.xxx = nn.Linear(...)` 无处安放。

**解决**：把 `super().__init__()` 放在**第一行**：

```python
def __init__(self):
    super().__init__()          # ← 必须第一行
    self.fc1 = nn.Linear(784, 128)
```

---

### 9. `TypeError: Invalid shape (1, 28, 28) for image data`

**原因**：`imshow` 要的是 `(高, 宽)` 或 `(高, 宽, 通道)`，而张量是 `(通道, 高, 宽)`。
多出来的那个通道维它不认。

**解决**：`squeeze` 掉通道维：`ax.imshow(img.squeeze(0), cmap="gray")`。

<blockquote class="tip">

**💡 顺带一个不报错但图是错的陷阱**

标准化之后的像素值**不再是 0~1**（有负数）。这时 `imshow` 不会报错，
但画出来**一片灰**。要显示原图就得反标准化：

```python
img * 0.3081 + 0.1307        # 还原回 0~1
```

</blockquote>

---

### 10. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x2304 and 1600x128)`

**原因**：**改了卷积层，忘了同步改 `fc1` 的输入维度。** 卷积输出被拉直之后是 2304 个数，
而 `fc1` 还按老尺寸 1600 建着。

（`128x…` 里那个数字是**新的真实长度**，它随你改了什么而变。实测：给其中一层卷积加
`padding=1` 是 2304，给两层都加是 3136，把 `conv1` 的 `kernel_size` 改成 7 是 1024。
所以你看到的很可能不是 2304 —— **那不重要，把它读出来填给 `fc1` 就行**。）

**解决**：**别手算。** 把卷积部分的输出形状打印出来，照着改 `fc1`：

```python
x = torch.randn(1, 1, 28, 28)
x = self.features(x)                 # 只跑到卷积部分
print(x.shape)                       # 看这里，再决定 fc1 的第一个参数
```

本书第 3 章 §3.5 专门讲这个：**形状靠打印，不靠手算。**
`figures/ch03_shape_trace.txt` 就是每层的真实形状记录。

---

### 11. `RuntimeError: Given groups=1, weight of size [64, 32, 3, 3], expected input[128, 16, 13, 13] to have 3...`

**原因**：**通道数接不上。** 这一层声明"我要吃 32 通道"，但上一层只吐了 16 通道。
读法：`weight of size [64, 32, 3, 3]` = `[out_channels=64, in_channels=32, 3, 3]`，
所以 `in_channels=32`；而 `input[128, 16, 13, 13]` 的第二个数 16 才是实际通道数。**32 ≠ 16。**

**解决**：改这一层的 `in_channels`，**或者**改上一层的 `out_channels`，让它们对上。
改的时候记得连带检查后面的 `fc1`（见第 10 条）—— 它们是一条链，动一个要顺下去看。

---

### 12. `RuntimeError: Error(s) in loading state_dict for SimpleCNN: size mismatch for ...`

**原因**：**拿旧 checkpoint 去加载改过的模型。** `load_state_dict()` 会逐键核对名字和形状，
对不上就报错 —— 这本身是好事，但报错信息是键名列表，第一次见容易发懵。

**解决**：**checkpoint 和模型结构是绑死的。** 改了 `SimpleCNN`（加层、改宽度），
旧 checkpoint 就作废，得重新训练。想留住旧模型，就在代码里保留旧的结构定义。

---

### 13. `OSError: [Errno 22] Invalid argument`（保存图片时）

**原因**：Windows 上文件被短暂独占 —— 杀毒软件正在扫描新文件、资源管理器预览窗格开着、
或者你刚用看图软件打开过它。这时 `savefig` 会抛这个错。

**这个错特别气人**：前面几分钟的训练全白跑了，却倒在最后一步，而报错信息
**完全看不出是文件占用**。

**解决**：重试几次就行，真正的原因通常在一两秒内消失。本书的 `_savefig()` 实现了有界重试
（5 次 / 每次等 1 秒）。

---

## B.3 不报错，但结果是错的

<blockquote class="warn">

**⚠️ 这一类最危险 —— 程序安安静静跑完，还打印了个漂亮的数字**

前面两类至少会给你一个 Traceback。这一类**什么都不说**，你看半天看不出问题，
甚至会拿错误的结果去写报告。下面每一条都是本书实测撞到的。

</blockquote>

### 14. loss 一直卡在 `2.3015` 不动

**这个数就是 10 类问题"纯猜"的损失值** —— `ln(10) ≈ 2.3026`（实测打印出的 2.3015
是在某一个 batch 上的取值，和理论值只差千分之一）。它不动说明**模型根本没在学**。

**可能原因**（按可能性排序）：

1. **权重初始化成全 0** —— 本书实测：全 0 初始化时 10000 张测试图**全部判成同一个类**，
   准确率 11.35%，正好等于测试集里「1」的占比（1135 / 10000）。它没看图，只是输出训练集里最多的类。
2. **学习率太大**，一上来就炸进死区（和全 0 一样的死状态）。
3. 梯度没传到 —— 检查有没有 `loss.backward()`、有没有 `optimizer.step()`。

### 15. loss 变成 `nan` 或炸到 `1e15` 量级

**原因**：初始化方差过大（本书实测：正态 `std=1.0` 时第 1 轮 loss 就炸到 1.5×10¹⁵），
或者学习率太大。

**解决**：用 `nn.init.kaiming_normal_` 初始化；学习率回到 0.01 量级再试。

### 16. 准确率恰好是 11.35%（或某个"很整"的小数字）

**原因**：模型**在输出同一个类**。11.35% 就是"全判成 1"的得分（测试集里 1 占 11.35%）。
**准确率不是 10% 而是 11.35%，说明它连随机都没做到。**

**解决**：看输出的**预测分布**（每个类各判了多少张）—— 一眼就能确认。回到第 14 条排查。

### 17. 加了 BN / Dropout 之后，"过拟合 gap"变成负数

**症状**：第 6 章那组 `+BN+Dropout` 实测训练集 **93.83%**、验证集 **97.76%**，
减出来 gap = **−3.93%**，看起来"泛化极好"。

**这是假的。** 原因：训练集准确率在 `train()` 模式下量的（Dropout 开着、BN 用 batch 统计量），
验证集在 `eval()` 模式下量的 —— **两边考的难度不同**，减出来的数没有意义。

**解决**：**量训练集准确率时也要 `model.eval()`。** 本书实测：`eval()` 模式下四组 gap 是
`+1.14 / +0.80 / +0.64 / +0.54`，全为正且单调递减 —— 那才是正则化在压 gap 的真实形状。
详见第 6 章 §6.4。

### 18. 同一个种子，跑两遍结果不一样

**原因**：cuDNN 的卷积反向默认会挑"最快"算法，其中一些用原子加法累加梯度 ——
**累加顺序不同 → 浮点结果有微小差异 → 10 个 epoch 后放大成可观差别。**
本书实测：同一份代码同一种子跑出过 98.39% 和 98.69%。

**解决**：每个训练脚本的 `set_seed()` 里都加这两行：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**代价**：**基本不花什么时间。** 本书实测（第 3 章那个 CNN，batch=128，GPU，
关/开各跑 3 个 epoch 取中位数）：关 **3.35 秒** → 开 **3.43 秒**，**慢 2%**。

这个代价只在**卷积**上才存在 —— 关掉的是 cuDNN 挑卷积算法的自由。第 2 章那个 MLP
一层卷积都没有，所以这两行对它不起作用（第 2 章正式日志 10 轮共 32.0 秒，
就是开着它们跑出来的）。

换来的是结果可复现 —— 不加的话，书里"对照结果确认一致"的练习全是空话。

### 19. 加了正则化，结果反而明显变差

**原因**：**正则化加太狠，训练集都拟合不了了。** Dropout 的 `p` 太大（比如 0.9）时，
每次前向只有一小部分神经元在工作，模型**根本学不动**。（这本书没有实测过 `p=0.9` 那个点 ——
第 6 章练习 4 就是让你亲手把它拉上去看训练集准确率塌下来，读那里时别把它当成已知数字。）

**判断方法**：看**训练集**准确率。它要是也掉下来了，说明不是"泛化更好"，是"学不动了"。

### 20. 书里引用的图，被我跑一次就换掉了

**原因**：探索性运行（`--smoke`、`--epochs 1`、`--limit 512`……）写出的文件名
和正式产物一模一样，**静默覆盖**，不报错、不提示。

**解决**：本书的判断标准是「**这次运行是不是书上写的那条命令**」（`is_canonical`），
正式运行写原名、其余一律加后缀（`_smoke` / `_e1` / `_e3`）。
不小心已经覆盖了的话，跑 `tests/regen_all.sh` 用正式命令重新生成全部产物。

### 21. 我跑出来的数字和书里不一样

**先别怀疑代码。** 按这个顺序查：

1. **种子对不对** —— 书里默认 `--seed 42`
2. **轮数对不对** —— 默认 10 epoch，`--smoke` 是 1 epoch
3. **数据是不是全量** —— `--limit 0` 才是全部；有 `--limit` 时准确率会明显不同
4. **设备对不对** —— 书里的数字**全部来自 GPU**：每份正式日志的开头都有一行
   `[设备] CUDA 可用且真实算子通过，输出形状 (2, 4, 26, 26)`。
   要是你按第 0 章 §0.4 用 `CUDA_VISIBLE_DEVICES=-1` 强制走了 CPU，数字会有出入
   （第 2 章的练习里就有 CPU / GPU 的对照）
5. **差多少** —— 小于 **0.43pp** 就是种子噪声，不用管；**逐类指标**的噪声带要放宽到 **~1pp**

### 22. 换了执行顺序，整条曲线都偏了

**症状**：同一份模型、同一个种子，第 1 轮 val 从 96.68% 变成 96.76%，最终测试集 98.53%
而不是 98.62%。差 0.09pp 看着像噪声，**但整条曲线从第 1 轮起就不一样** —— 那是初始化变了。

**原因**：`get_device()` 为了探测 GPU 会跑一次 `torch.randn(...)`，**这消耗全局随机数流**。
先 `set_seed` 再探测设备，种子的"起点"就被这次探测挪走了，
后面 `kaiming_normal_` 拿到的是一段**偏移过的**随机数。

**解决**：**先探测设备，再固定随机源。**

```python
device = get_device()
set_seed(cfg.seed)          # ← 顺序不能换
```

**通用规矩**：凡是在"该确定的流程"之前动过随机数流的东西，
要么放到 `set_seed` 前面，要么自己在内部用独立的 `generator`。

---

## B.4 排查套路

按这四步走，能解决绝大多数问题：

1. **先确认解释器** —— 是不是装 torch 的那个环境？`python -c "import torch; print(torch.__version__)"`
2. **再确认设备** —— 别信 `is_available()`，跑一次小的卷积前向（第 2 条）
3. **把形状打出来** —— 形状类的报错（第 10、11 条）一律靠 `print(x.shape)`，别手算
4. **用 `--smoke` 快速试** —— 冒烟模式 1 个 epoch + 512 张样本，几十秒出结果。
   **但记住：`--smoke` 的产物带 `_smoke` 后缀，它不能用来下任何结论**（第 20 条）

<blockquote class="done">

**📌 最后一招：看预测分布**

如果"能跑完但结果不对"（B.3 全部条目），**第一个该看的不是准确率，是预测分布** ——
每个类各被判了多少张。模型要是把 10000 张全判成一个类，或者某一类一张都没判，
问题立刻现形。本书第 3 章 §3.8 就是这么做诊断的。

</blockquote>
