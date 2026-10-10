"""第 3 章：卷积神经网络

第 2 章的全连接网络把 28x28 的图拉直成 784 个数的向量，**丢掉了像素之间的
空间关系** —— 它眼里的输入是一维的 784 个数，不知道哪个数和哪个数挨着。

本章换成卷积神经网络（CNN）：用小的卷积核在图上滑动，提取"局部特征"。
本章要做三件事：

  1. **打印每一层的输出形状** —— 形状靠观察，不靠背公式（本书的深度契约）
  2. 看清卷积 / 池化各自在干什么（正文细讲）
  3. 亲手看到"权重全初始化为 0"会让网络**完全学不动**，一步都不动

用法：
    python ch03_cnn.py                     # 默认 kaiming 初始化，10 epoch
    python ch03_cnn.py --init default      # 用 PyTorch 默认初始化
    python ch03_cnn.py --init zeros        # 全 0 初始化（应该学不动）
    python ch03_cnn.py --compare-inits     # 三种初始化各跑一遍，画在同一张图上
    python ch03_cnn.py --smoke             # 冒烟：1 epoch + 512 张
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
N_VAL = 10000

INIT_MODES = ("default", "kaiming", "zeros", "large")


# ---------------------------------------------------------------------------
# 全书统一的三个工具函数（与第 0、1、2 章一致，各脚本自带一份以保证可单独运行）
# ---------------------------------------------------------------------------
def parse_args(desc: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--smoke", action="store_true", help="快速冒烟：1 epoch + 少量样本")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    # 和上一章同一个默认值：SGD 配 lr=0.01。理由见第 2 章脚本里的对照表。
    p.add_argument("--lr", type=float, default=0.01, help="学习率（第 5 章细讲）")
    p.add_argument(
        "--init", choices=INIT_MODES, default="kaiming",
        help="权重初始化：default=PyTorch 默认 / kaiming=He / zeros=全 0 / large=方差过大",
    )
    p.add_argument(
        "--compare-inits", action="store_true",
        help="三种初始化各跑一遍 --epochs 轮，把 loss/acc 画在同一张图上",
    )
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.limit = 1, 512
    return a


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # GPU 上的卷积反向默认会挑"最快"的算法，其中有些用原子加法累加梯度，
    # 每次累加顺序不同 -> 结果每次都不一样（实测同一份代码同一种子跑出过
    # 98.39% 和 98.69% 两个测试集准确率）。下面两行强制用确定性算法：
    #   deterministic=True  只选结果可复现的算法
    #   benchmark=False     不要每次重新搜索最快算法
    # 代价是可能慢一点 —— 但对"照着书复现结果"来说，可复现比快重要。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """探测设备。不能只信 is_available()，必须真跑一个卷积前向（见第 0 章）。"""
    if torch.cuda.is_available():
        try:
            conv = nn.Conv2d(1, 4, 3).cuda()
            y = torch.relu(conv(torch.randn(2, 1, 28, 28, device="cuda")))
            torch.cuda.synchronize()
            print(f"[设备] CUDA 可用且真实算子通过，输出形状 {tuple(y.shape)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"[设备] 警告：is_available() 为 True 但真实算子失败，降级 CPU：{e}")
    else:
        print("[设备] CUDA 不可用，使用 CPU。")
    return torch.device("cpu")


def _savefig(fig, path: Path, attempts: int = 5) -> None:
    """存图，遇到 Windows 的文件短占就重试（详见第 1 章常见坑 4）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(attempts):
        try:
            fig.savefig(path, dpi=130, bbox_inches="tight")
            return
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"  [警告] 保存 {path.name} 失败（{e}），1 秒后重试 {i + 1}/{attempts - 1}")
            time.sleep(1.0)


# 书里两条正式命令各自用的轮数。
CANONICAL_EPOCHS = 10          # `python ch03_cnn.py --epochs 10`          -> 图 3-1、图 3-2
CANONICAL_COMPARE_EPOCHS = 5   # `python ch03_cnn.py --compare-inits --epochs 5` -> 图 3-3

# 两条正式命令的完整取值 —— **每个能改变产物的开关都要在这里判一遍**。
#
# 写成一张表、而不是串成一行 and，是因为"漏一个"这件事在本项目里复发过四次：
# 第 2 章漏过 `--epochs`/`--limit`，第 4 章漏过 `--level`，第 6 章漏过 `--depth`，
# 验收时又发现**每一章都漏了 `--seed`**。而本章练习 7 就叫读者跑 `--epochs 3`
# 去量耗时 —— 照着练习做一遍，书里的图就没了。
#
# 以后再加参数，往这两张表里补一行即可。可执行的版本见
# `tests/test_canonical_guard.py`：每个开关都列成了一个用例，漏了就会红。
_COMMON = dict(limit=0, seed=42, batch_size=128, lr=0.01)
CANONICAL_CURVE = dict(_COMMON, epochs=CANONICAL_EPOCHS, init="kaiming")
CANONICAL_COMPARE = dict(_COMMON, epochs=CANONICAL_COMPARE_EPOCHS)


def _matches(a: argparse.Namespace, ref: dict) -> bool:
    """表里每一项的取值都对得上吗 —— 这就是"是不是书上那条命令"的判断。"""
    return all(getattr(a, k) == v for k, v in ref.items())


def is_canonical_curve(a: argparse.Namespace) -> bool:
    """这次运行是不是产出**图 3-1（卷积核响应）和图 3-2（训练曲线）**那条命令。

    和上一章同一个坑，而且在这里更容易踩：书里**练习 7** 让读者跑
    `--epochs 3`（为了量耗时）。本章初版只要不是冒烟就写正式文件名，
    于是照练习做一遍，`ch03_curve_kaiming.png` 和 `ch03_feature_maps.png`
    就被 3 轮的结果**静默换掉了**（实测 `51b4bad1…` → `5fffcc91…`）。
    """
    return (not a.smoke) and (not a.compare_inits) and _matches(a, CANONICAL_CURVE)


def is_canonical_compare(a: argparse.Namespace) -> bool:
    """这次运行是不是产出**图 3-3（四档初始化对比）**那条命令。

    单独判一次，是因为它的轮数（5）和上面那条（10）不同 ——
    用一个统一的 `is_canonical` 会把图 3-3 也误伤成非正式产物。
    """
    return (not a.smoke) and a.compare_inits and _matches(a, CANONICAL_COMPARE)


def out_figure(base: str, a: argparse.Namespace, canonical: bool) -> Path:
    """正式运行用原名，其余一律加后缀（`_smoke` 或 `_e{N}`）。

    哪些产物需要保护、由哪条命令产出，见上面两个 `is_canonical_*`。
    """
    stem, ext = base.rsplit(".", 1)
    if canonical:
        return FIGURES / base
    tag = "smoke" if a.smoke else f"e{a.epochs}"
    return FIGURES / f"{stem}_{tag}.{ext}"


# ---------------------------------------------------------------------------
# 数据（与第 1、2 章同一套，签名不变）
# ---------------------------------------------------------------------------
def get_loaders(batch_size: int, limit: int = 0, seed: int = 42):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    full_train = datasets.MNIST(str(DATA), train=True, download=True, transform=tf)
    test_set = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [len(full_train) - N_VAL, N_VAL], generator=g
    )
    if limit > 0:
        train_set = torch.utils.data.Subset(train_set, range(min(limit, len(train_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """两层卷积 + 两层全连接的 CNN。

    结构（箭头表示数据的流动方向）：
        输入 1x28x28
        -> Conv2d(1 -> 32, 3x3)  -> ReLU -> MaxPool2d(2)
        -> Conv2d(32 -> 64, 3x3) -> ReLU -> MaxPool2d(2)
        -> Flatten
        -> Linear(1600 -> 128) -> ReLU -> Linear(128 -> 10)

    每一层的输出形状不靠背公式算，脚本会用一个假 batch 走一遍前向、
    把真实形状打出来（见 trace_shapes）。**以打印结果为准。**

    关于 init_mode（本章的核心演示之一）：
        "default" : 完全不动，用 PyTorch 建层时自带的随机初始化
        "kaiming" : He 初始化，专门为 ReLU 网络设计的方差
        "zeros"   : 权重和偏置全部置 0 —— 会**彻底学不动**
        "large"   : 方差过大的随机初始化（标准差 1.0）—— 信号会逐层放大到炸
    这三档一起说明了一件事：初始化不是"要不要随机"的问题，而是**方差要多大**
    的问题 —— 太小（0）学不动，太大（large）训不起来，中间一段才work。
    """

    def __init__(self, init_mode: str = "kaiming"):
        super().__init__()          # 必须先调用父类构造，否则 PyTorch 认不出你的层
        if init_mode not in INIT_MODES:
            raise ValueError(f"init_mode 只能是 {INIT_MODES}，收到 {init_mode!r}")
        self.init_mode = init_mode

        # --- 特征提取部分：卷积层，负责"看图" ---
        # Conv2d(in_channels, out_channels, kernel_size)
        #   in_channels=1  因为 MNIST 是灰度图，只有 1 个通道（彩色图是 3）
        #   out_channels=32 表示"用 32 个不同的 3x3 卷积核去看这张图"，
        #                   得到 32 张特征图（feature map）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # --- 分类部分：全连接层，负责"下结论" ---
        # 这里写 1600 是**按打印出来的形状填的**，不是算出来的。
        # 想知道它为什么是 1600：看脚本开头的形状追踪输出。
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

        self._init_weights()

    def _init_weights(self) -> None:
        """按 init_mode 初始化权重。"""
        if self.init_mode == "default":
            return  # 什么都不做 —— PyTorch 建层时已经给了默认初始化

        if self.init_mode == "zeros":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
            return

        if self.init_mode == "large":
            # 故意用一个大得离谱的标准差。正态分布本身没问题，
            # 问题在于**尺度**：每过一层信号都会被放大，很快就溢出成一个大数。
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(m.weight, mean=0.0, std=1.0)
                    nn.init.zeros_(m.bias)
            return

        # kaiming：He 初始化。给 ReLU 网络用的方差。
        #   mode="fan_out" 让方差按"这一层输出的神经元个数"缩放
        #   nonlinearity="relu" 把 ReLU 砍掉一半负值这件事补偿回来
        # 偏置一律置 0（卷积/全连接的偏置不需要随机）
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 卷积部分：形状一直是 4 维 (N, C, H, W)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        # Flatten 把 (N, 64, 5, 5) 压成 (N, 1600) —— 卷积部分到此结束
        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        # 和上一章一样：这里**不接 softmax**，输出的 10 个数是 logits，
        # 直接交给 CrossEntropyLoss。
        return x


class SimpleMLP(nn.Module):
    """第 2 章那个全连接网络，原样搬过来一份，只为了和 CNN 比参数量。

    （本书要求每章脚本自包含、不跨章 import，所以是复制而非导入。）
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.flatten(x))))


# ---------------------------------------------------------------------------
# 形状追踪 —— 本章第一个核心产出
# ---------------------------------------------------------------------------
def trace_shapes(model: nn.Module, path: Path, device: torch.device) -> str:
    """用一个假 batch 走一遍前向，把每一层输出的真实形状打出来并存档。

    实现上用的是 PyTorch 的 forward hook：给每个子层挂一个回调，
    该层一算完就把它的输出形状记下来。好处是**不用手写公式**，
    网络改成什么样，打印出来的就是什么样。

    返回写进文件的那段文本（方便同时打印到屏幕上）。
    """
    # 假 batch：2 张 1x28x28 的图。不需要是真的 MNIST 图，形状对就行。
    fake = torch.randn(2, 1, 28, 28, device=device)

    rows: list[tuple[str, str, int]] = []
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            # output.shape 是 (N, C, H, W) 或 (N, F)。第 0 维是 batch，
            # 对"这一层长什么样"没意义，去掉它只留后面几维。
            shape = "×".join(str(s) for s in output.shape[1:])
            n = sum(p.numel() for p in module.parameters())
            rows.append((name, shape, n))
        return hook

    for name, child in model.named_children():
        handles.append(child.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(fake)
    model.train(was_training)
    for h in handles:
        h.remove()

    lines = [
        "第 3 章 SimpleCNN 逐层形状追踪",
        "=" * 46,
        f"假 batch 输入形状：(2, 1, 28, 28)  =  N×C×H×W",
        "",
        f"{'层名':<10}{'输出形状':<14}{'参数量':>12}",
        "-" * 46,
        f"{'输入':<10}{'1×28×28':<14}{0:>12,}",
    ]
    for name, shape, n in rows:
        lines.append(f"{name:<10}{shape:<14}{n:>12,}")
    lines.append("-" * 46)
    total = sum(p.numel() for p in model.parameters())
    lines.append(f"{'合计':<10}{'':<14}{total:>12,}")
    lines.append("")
    lines.append("形状变化（去掉 batch 维，连续相同的形状已折叠 ——")
    lines.append("ReLU / 池化以外的激活都不改变形状，重复写反而看不清关键的那几次变化）：")
    chain = ["1×28×28"]
    for _, shape, _ in rows:
        if shape != chain[-1]:      # 和上一个不同才记，相邻重复的折叠掉
            chain.append(shape)
    lines.append("  " + " → ".join(chain))
    lines.append("")
    lines.append("（以上都是打印出来的真实形状，不是手算的。）")

    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return text


def conv_param_probe(path: Path, device: torch.device) -> str:
    """把常见参数组合下卷积层的输出形状打出来，存成一张对照表。

    为什么不讲公式：本书的深度契约是"知道卷积在干嘛、会调 API"。
    手推 (H - K + 2P)/S + 1 属于数学推导，不在范围内。
    想知道某个组合输出多大 —— **改了参数重跑这张表**，比记公式可靠。
    """
    combos = [
        # (kernel_size, stride, padding)
        (3, 1, 0), (3, 1, 1), (5, 1, 0), (5, 1, 2),
        (3, 2, 0), (3, 2, 1), (2, 2, 0), (7, 1, 3),
    ]
    x = torch.randn(1, 1, 28, 28, device=device)
    rows = []
    for k, s, p in combos:
        conv = nn.Conv2d(1, 32, kernel_size=k, stride=s, padding=p).to(device)
        with torch.no_grad():
            y = conv(x)
        n = sum(pp.numel() for pp in conv.parameters())
        rows.append(((k, s, p), tuple(y.shape[1:]), n))

    lines = [
        "卷积层参数对照表（输入固定为 1×28×28，输出通道固定为 32）",
        "=" * 68,
        "kernel_size  stride  padding   ->  输出形状        参数量",
        "-" * 68,
    ]
    for (k, s, p), shape, n in rows:
        shape_s = "×".join(str(v) for v in shape)
        lines.append(f"{k:^11}  {s:^6}  {p:^7}   ->  {shape_s:<14}  {n:>6,}")
    lines += [
        "-" * 68,
        "读法：",
        "  kernel_size 变大 -> 每次看的范围更大，输出变小",
        "  padding     变大 -> 输出变大（=1 时输出和输入一样大，叫 same padding）",
        "  stride      变大 -> 跳着滑，输出成倍变小",
        "  out_channels 变大 -> 输出形状多一维，参数量跟着涨",
        "",
        "（改完参数重跑本脚本即可，不需要手算。本书不教手算输出尺寸。）",
    ]
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return text


def param_breakdown(model: nn.Module) -> list[tuple[str, int]]:
    """按层列出参数量。参数"堆在哪一层"比总数更能说明问题。"""
    return [
        (name, sum(p.numel() for p in m.parameters()))
        for name, m in model.named_children()
        if sum(p.numel() for p in m.parameters()) > 0
    ]


# ---------------------------------------------------------------------------
# 训练与评估 —— 与第 2 章完全一致的签名
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    """跑完一个 epoch，返回 (平均损失, 准确率)。训练循环的五个动作，顺序不能乱。"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()               # ① 清空上一轮残留的梯度（梯度是累加的！）
        logits = model(x)                   # ② 前向
        loss = criterion(logits, y)         # ③ 算损失
        loss.backward()                     # ④ 反向：自动求梯度
        optimizer.step()                    # ⑤ 更新参数

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)

    return total_loss / n, correct / n


def prediction_histogram(model, loader, device) -> list[int]:
    """统计模型在测试集上把多少个样本判成了每个类别（10 个数，下标即类别）。

    全 0 初始化的模型会一直输出同一个类别 —— 用这个函数就能看出是哪一个。
    """
    model.eval()
    counts = [0] * 10
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            for c in pred.tolist():
                counts[c] += 1
    return counts


def evaluate(model, loader, device) -> tuple[float, float]:
    """在验证集/测试集上评估，返回 (平均损失, 准确率)。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)

    return total_loss / n, correct / n


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def plot_curve(history: dict, path: Path, title: str) -> None:
    """左右双图：左边 loss，右边 accuracy，各画 train / val 两条线。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="训练集")
    ax1.plot(epochs, history["val_loss"], "s-", label="验证集")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("损失 loss")
    ax1.set_title("损失曲线")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", label="训练集")
    ax2.plot(epochs, history["val_acc"], "s-", label="验证集")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("准确率")
    ax2.set_title("准确率曲线")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def plot_init_compare(histories: dict, path: Path) -> None:
    """四种初始化的曲线画在同一张图上 —— 肉眼直接看谁在学、谁没学。

    画法上有个坑：zeros 和 large 最终会落到**完全相同的数值**上（见正文），
    如果两条都画成带 marker 的实线，后画的那条会把先画的完全盖住 ——
    图例里有 4 项，图上只看得见 3 条线。所以 zeros 画成粗实线打底，
    large 只画 marker 不画线，叠在粗线上就能同时看见。
    """
    styles = {
        "default": dict(marker="o", ls="-", color="tab:blue", label="PyTorch 默认"),
        "kaiming": dict(marker="s", ls="-", color="tab:orange", label="Kaiming (He)"),
        "zeros": dict(marker="^", ls="-", color="tab:green", lw=3.5,
                      label="全 0（方差太小）"),
        "large": dict(marker="d", ls="none", color="tab:red", markersize=7,
                      label="std=1.0（方差太大，与全 0 重合）"),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    n_epochs = 1
    for mode in INIT_MODES:
        h = histories[mode]
        n_epochs = len(h["val_loss"])
        epochs = range(1, n_epochs + 1)
        # 画验证集：zeros / large 是水平直线，default / kaiming 一路向下
        ax1.plot(epochs, h["val_loss"], **styles[mode])
        ax2.plot(epochs, h["val_acc"], **styles[mode])

    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.set_xticks(range(1, n_epochs + 1))
        ax.grid(alpha=0.3)

    ax1.set_ylabel("验证集损失")
    ax1.set_title("验证集 loss：两条平线 = 两种「没在学」")
    ax1.legend(fontsize=9)

    ax2.set_ylabel("验证集准确率")
    ax2.set_title("验证集准确率：卡在随机水平上不去")
    ax2.axhline(0.10, color="gray", ls=":", lw=1, label="随机猜（10%）")
    ax2.legend(loc="center right", fontsize=9)

    fig.suptitle("第 3 章 不同权重初始化的对比", fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_once(init_mode: str, args, device: torch.device):
    """完整跑一次训练，返回 (history, 测试集 loss, 测试集 acc, 训练好的模型)。"""
    """完整跑一次训练，返回 (history, 测试集 loss, 测试集 acc)。"""
    # 每次重设种子 + 重建 loader：让三种初始化面对**完全相同**的数据顺序，
    # 差异才只可能来自初始化方式。
    set_seed(args.seed)
    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size, limit=args.limit, seed=args.seed
    )

    model = SimpleCNN(init_mode=init_mode).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print(f"[init={init_mode}] 参数量 {count_params(model):,}  "
          f"epochs={args.epochs}  lr={args.lr}")
    print(f"{'epoch':>5} | {'train_loss':>10} {'train_acc':>9} | "
          f"{'val_loss':>9} {'val_acc':>8} | {'秒':>5}")
    print("-" * 62)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(1, args.epochs + 1):
        te = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"{ep:>5} | {tr_loss:>10.4f} {tr_acc:>8.2%} | "
              f"{va_loss:>9.4f} {va_acc:>7.2%} | {time.time() - te:>5.1f}")

    print("-" * 62)
    te_loss, te_acc = evaluate(model, test_loader, device)
    print(f"[init={init_mode}] 测试集 loss={te_loss:.4f}  acc={te_acc:.2%}   "
          f"（{int(round(te_acc * len(test_loader.dataset)))}/{len(test_loader.dataset)} 张正确）")

    if init_mode == "zeros":
        # 全 0 初始化时多看一眼：它到底把样本判成了哪一类？
        counts = prediction_histogram(model, test_loader, device)
        top = max(range(10), key=lambda c: counts[c])
        print(f"[init=zeros] 测试集 10000 张的预测分布（下标=类别）:")
        print("             " + "  ".join(f"{c}:{n}" for c, n in enumerate(counts)))
        print(f"             全部 10000 张都被判成了「{top}」—— 它没有在看图，"
              f"只是在输出训练集里最多的那个类。")
    print()
    return history, te_loss, te_acc, model


def plot_feature_maps(model, loader, path: Path, device: torch.device) -> None:
    """拿一张真实的测试图，看 conv1 的 32 个卷积核各自"看到"了什么。

    这是"卷积在干什么"最直白的证据 —— 不用画示意图，直接把中间结果摊开看。
    """
    model.eval()
    x, y = next(iter(loader))
    img = x[0:1].to(device)                    # 取第一张，形状 (1,1,28,28)
    with torch.no_grad():
        feats = model.conv1(img)[0].cpu()      # (32, 26, 26)

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(4, 9, width_ratios=[1.35] + [1] * 8)

    ax_in = fig.add_subplot(gs[1:3, 0])
    ax_in.imshow(img[0, 0].cpu(), cmap="gray")
    ax_in.set_title(f"输入图（标签 {int(y[0])}）", fontsize=9)
    ax_in.set_xticks([])
    ax_in.set_yticks([])

    # 关键：32 张图**共用同一个颜色范围**，围绕 0 对称。
    # 如果每张图各用各的范围（imshow 的默认行为），每张都会被拉伸成"满屏色彩"，
    # 看上去 32 个核几乎一模一样 —— 那是画法造成的假象，不是真实情况。
    # 卷积输出有正有负，用红蓝发散色表：红=强正响应，蓝=强负响应，白=接近 0。
    vmax = float(feats.abs().max())

    n_show = min(32, feats.shape[0])
    for i in range(n_show):
        r, c = divmod(i, 8)
        ax = fig.add_subplot(gs[r, c + 1])
        ax.imshow(feats[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"核{i}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("conv1 的 32 个卷积核各自提取到的特征（同一张输入图）", fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def main() -> None:
    args = parse_args("第 3 章：卷积神经网络")
    set_seed(args.seed)
    device = get_device()

    # --- 1. 形状追踪：先把网络长什么样看清楚，再谈训练 ---
    probe = SimpleCNN(init_mode=args.init).to(device)
    trace = trace_shapes(probe, FIGURES / "ch03_shape_trace.txt", device)
    print()
    print(trace)
    print(f"[产出] 形状追踪已保存 {FIGURES / 'ch03_shape_trace.txt'}")
    print()

    # --- 2. 卷积参数对照表：改参数前先查表，别手算 ---
    probe_txt = conv_param_probe(FIGURES / "ch03_conv_params.txt", device)
    print(probe_txt)
    print(f"[产出] 参数对照表已保存 {FIGURES / 'ch03_conv_params.txt'}")
    print()

    # --- 3. MLP vs CNN 参数量对比（真实数字，脚本算出来的）---
    mlp = SimpleMLP()
    cnn = SimpleCNN(init_mode=args.init)
    print("参数量对比")
    print("-" * 46)
    print(f"{'模型':<10}{'总参数量':>12}   参数都在哪")
    for name, m in (("SimpleMLP", mlp), ("SimpleCNN", cnn)):
        rows = param_breakdown(m)
        detail = "  ".join(f"{k}={v:,}" for k, v in rows)
        print(f"{name:<10}{count_params(m):>12,}   {detail}")
    print("-" * 46)
    print()

    # --- 4. 训练 ---
    if args.compare_inits:
        histories = {}
        for mode in INIT_MODES:
            print("=" * 62)
            histories[mode], _, _, _ = run_once(mode, args, device)

        out = out_figure("ch03_init_compare.png", args, is_canonical_compare(args))
        plot_init_compare(histories, out)
        print("=" * 62)
        print(f"[产出] 已保存 {out}")
        print()
        print("  看这张图：")
        print("  - default 和 kaiming 都在往下走，两条线基本重合 —— 说明 PyTorch")
        print("    的默认初始化**本来就落在合理区间**，kaiming 是锦上添花，不是救命稻草。")
        print("  - 全 0 那条是**水平直线**，准确率贴着随机猜不动 —— 它根本没在学。")
        print("  - std=1.0 那条也起不来 —— 随机了，但**尺度错了**。")
        print("    所以初始化要解决的不是「要不要随机」，而是「方差该多大」。")
        print("    原因见第 3 章正文。")
        return

    history, _, _, model = run_once(args.init, args, device)

    canonical = is_canonical_curve(args)
    out = out_figure(f"ch03_curve_{args.init}.png", args, canonical)
    plot_curve(history, out, f"第 3 章 SimpleCNN 训练曲线（init={args.init}）")
    print(f"[产出] 已保存 {out}")
    if not canonical and args.init == "kaiming":
        print(f"       （非正式运行，产物带后缀，不会覆盖书里的 {FIGURES / 'ch03_curve_kaiming.png'}）")

    # 特征图：只有单次训练时画（对比模式下有 4 个模型，画哪个都不合适）
    _, _, test_loader = get_loaders(args.batch_size, limit=args.limit, seed=args.seed)
    fm_out = out_figure("ch03_feature_maps.png", args, canonical)
    plot_feature_maps(model, test_loader, fm_out, device)
    print(f"[产出] 已保存 {fm_out}（conv1 的 32 个卷积核各看到了什么）")
    if args.smoke:
        print("       （冒烟模式产物，带 _smoke 后缀，不会覆盖正式训练曲线）")
    print()
    print("  看这张图：")
    print("  - 训练集的 loss 应当一路下降；验证集的 loss 若开始上升，就是过拟合了。")
    print("  - CNN 的验证集准确率应当明显高于第 2 章的全连接网络（同一批数据、同样 10 轮）。")


if __name__ == "__main__":
    main()
