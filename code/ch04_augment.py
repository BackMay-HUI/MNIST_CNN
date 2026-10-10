"""第 4 章：数据增强

前三章的训练集一直是原样的 60000 张图。数据增强的思路是：**把每张训练图
随机变一点点**（转个角度、挪一挪位置、缩放一下），让模型每次看到的都不太一样，
从而不容易死记硬背。

本章要做四件事：

  1. 看清 `transforms` 家族各是干什么的，怎么串成一条流水线
  2. **验证集和测试集永远不加增强** —— 这一点在代码里是硬性保证的
  3. 三档增强强度（none / light / strong）各跑一遍，看真实差别
  4. 讲清楚一个反直觉的结论：**MNIST 上手写数字本身变化就小，增强的收益有限**

用法：
    python ch04_augment.py                    # 三档全跑，产出对比图 + CSV
    python ch04_augment.py --level strong     # 只跑 strong 这一档
    python ch04_augment.py --smoke            # 冒烟：1 epoch + 512 张
"""
import argparse
import csv
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
RESULTS = ROOT / "results"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
N_VAL = 10000

# 书里正式引用的对比图/对比表，只由"书上写的那条命令"产生：
#     python code/ch04_augment.py --epochs 10
# 其它任何跑法（--smoke、改轮数、只跑单档）都属于探索性运行，产物一律另起名字。
# 教训：曾经用 `--epochs 1` 做确定性测试，把书里的 10 轮对比图静默覆盖成 1 轮版本。
# 光靠 `--smoke` 后缀不够 —— 判断标准是"这是不是书上那条命令"，不是"是不是冒烟"。
CANONICAL_EPOCHS = 10

LEVELS = ("none", "light", "strong")
LEVEL_LABEL = {"none": "不增强", "light": "轻增强", "strong": "强增强"}


# ---------------------------------------------------------------------------
# 全书统一的工具函数（各章自带一份，保证可单独运行）
# ---------------------------------------------------------------------------
def parse_args(desc: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--smoke", action="store_true", help="快速冒烟：1 epoch + 少量样本")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01, help="配合 SGD 用（第 5 章细讲）")
    p.add_argument(
        "--level", choices=LEVELS, default=None,
        help="只跑某一档增强；不给则三档全跑并产出对比图",
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
    # 让 GPU 训练结果可复现（原因见第 3 章常见坑 2）。
    # 本章尤其需要：增强本身是随机的，如果训练还不可复现，结果就没法归因了。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """探测设备，不能只信 is_available()，必须真跑一个卷积前向（见第 0 章）。"""
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


def is_canonical(args: argparse.Namespace) -> bool:
    """这次运行是不是"书上写的那条命令"（三档全跑 + 默认轮数 + 非冒烟）。

    只有它才允许写 `ch04_compare.png` / `ch04_compare.csv` 这两个被书引用的产物。
    """
    return (not args.smoke) and args.epochs == CANONICAL_EPOCHS and args.level is None


def tagged(base: str, args: argparse.Namespace) -> Path:
    """非正式运行统一加后缀，避免覆盖书里的正式图（详见第 2 章常见坑 5）。

    注意：**只对「内容跟训练轮数有关」的图这样做**。像本章的增强示意图，
    冒烟跑和完整跑出来的图是一模一样的（它不训练），就不需要后缀。
    """
    stem, ext = base.rsplit(".", 1)
    if is_canonical(args):
        return FIGURES / base
    tag = "smoke" if args.smoke else f"e{args.epochs}"
    return FIGURES / f"{stem}_{tag}.{ext}"


# ---------------------------------------------------------------------------
# 数据增强 —— 本章的核心
# ---------------------------------------------------------------------------
def get_aug_transform(level: str) -> transforms.Compose:
    """按增强强度返回一条 transform 流水线。

    流水线的**顺序很重要**，不是随便排的：

        PIL 图 ──> 几何变换（旋转/平移/缩放）──> ToTensor ──> Normalize ──> RandomErasing

    原因：
      - `RandomRotation` / `RandomAffine` 是在 **PIL 图像**上操作的，
        所以要放在 `ToTensor` 前面（那时数据还是 PIL 图）。
      - `RandomErasing`（随机擦除）要抹掉一块区域，需要在 **张量**上做，
        所以必须放在 `ToTensor` 后面。
      - `Normalize` 放最后，让送进网络的数值是标准化的。
    """
    if level == "none":
        # 原样：只做"变成张量"和"标准化"这两步必需的转换
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])

    if level == "light":
        return transforms.Compose([
            # 随机旋转 ±10 度
            transforms.RandomRotation(degrees=10),
            # 随机平移，最多挪动 10% 的宽/高（degrees=0 表示不旋转）
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])

    if level == "strong":
        return transforms.Compose([
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0, translate=(0.2, 0.2), scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            # 随机擦除：以 0.5 的概率挖掉一小块（模拟"笔画缺了一截"）
            transforms.RandomErasing(p=0.5),
        ])

    raise ValueError(f"level 只能是 {LEVELS}，收到 {level!r}")


def get_loaders(batch_size: int, limit: int = 0, seed: int = 42,
                train_tf=None):
    """建三个 DataLoader。

    **注意这个函数的签名**：它只接受一个 `train_tf`（训练集的 transform），
    验证集和测试集的 transform 在函数内部**写死**成不增强的那一档。

    这是刻意的设计：让"给验证集也加了增强"这个错误**在结构上不可能发生**，
    而不是靠写注释提醒你别犯。原因见本章 4.3 节。
    """
    # 验证/测试永远用最朴素的 transform —— 没有旋转、没有平移、没有擦除。
    # 评估时我们要看的是"模型在真实数据上表现如何"，把图变来变去只会让
    # 测出来的分数失去意义（而且每次评估结果都不一样，没法比较）。
    eval_tf = get_aug_transform("none")

    if train_tf is None:
        train_tf = eval_tf

    full_train = datasets.MNIST(str(DATA), train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST(str(DATA), train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [len(full_train) - N_VAL, N_VAL], generator=g
    )
    # 注意：random_split 出来的 val_set 继承的是 full_train 的 transform，
    # 也就是**带增强的**那个！必须显式改掉，否则验证集也被增强了。
    # （这是本章最容易踩的坑，详见常见坑 1。）
    val_set.dataset = datasets.MNIST(str(DATA), train=True, download=True, transform=eval_tf)

    if limit > 0:
        train_set = torch.utils.data.Subset(train_set, range(min(limit, len(train_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 模型（与第 3 章同一个 SimpleCNN，方便对照）
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """第 3 章那个 CNN：Conv(1->32) -> Conv(32->64) -> FC(128) -> FC(10)。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# 训练与评估（与第 2、3 章一致）
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def evaluate(model, loader, device) -> tuple[float, float]:
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
def _denorm(t: torch.Tensor) -> np.ndarray:
    """把标准化过的图还原回 [0,1]，否则显示出来是一片灰。"""
    return (t * MNIST_STD + MNIST_MEAN).clamp(0, 1).squeeze().numpy()


def plot_aug_grid(path: Path, seed: int) -> None:
    """同一张原图，经过 8 次不同的随机增强 —— 一眼看清增强在干什么。

    用 "strong" 那一档来画，效果最明显。
    """
    set_seed(seed)
    raw = datasets.MNIST(str(DATA), train=True, download=True, transform=None)
    img, label = raw[0]           # PIL 图，不加任何变换

    tf = get_aug_transform("strong")
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 8))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title(f"原图（标签 {label}）", fontsize=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    slots = [(r, c) for r in range(3) for c in range(3)][1:]   # 剩下 8 个格子
    for i, (r, c) in enumerate(slots, start=1):
        aug = tf(img)              # 每次调用都重新随机一次
        axes[r, c].imshow(_denorm(aug), cmap="gray")
        axes[r, c].set_title(f"增强 #{i}", fontsize=10)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

    fig.suptitle("同一张图，8 次强增强（strong）：旋转 ±30°、平移 ±20%、缩放 0.9~1.1、随机擦除", fontsize=11)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def plot_flip_trap(path: Path, digits=(2, 5, 6, 7)) -> None:
    """证明「翻转类增强在 MNIST 上有害」—— 直接看图，不靠断言。

    书上会说"水平翻转把 2 变成不存在的形状"，这句话得有图撑着。
    这张图把原图、水平翻转、垂直翻转并排放在一起。
    """
    raw = datasets.MNIST(str(DATA), train=False, download=True, transform=None)
    picked = {}
    for img, label in raw:                    # 每个数字取第一张
        if label in digits and label not in picked:
            picked[label] = img
        if len(picked) == len(digits):
            break

    from torchvision.transforms import functional as TF

    fig, axes = plt.subplots(len(digits), 3, figsize=(6.8, 2.15 * len(digits)))
    for r, d in enumerate(digits):
        img = picked[d]
        panels = [
            (img, f"原图（标签 {d}）"),
            (TF.hflip(img), "水平翻转 —— 还是个数字吗？"),
            (TF.vflip(img), "垂直翻转 —— 还是个数字吗？"),
        ]
        for c, (im, title) in enumerate(panels):
            axes[r, c].imshow(im, cmap="gray")
            axes[r, c].set_title(title, fontsize=9.5)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("MNIST 上为什么不能用翻转类增强：翻转后标签就错了", fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def plot_compare(histories: dict, path: Path) -> None:
    """三档增强的验证集曲线画在一起。"""
    styles = {
        "none": dict(marker="o", ls="-", color="tab:gray", label="不增强"),
        "light": dict(marker="s", ls="-", color="tab:blue", label="轻增强"),
        "strong": dict(marker="^", ls="-", color="tab:red", label="强增强"),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    n_epochs = 1
    for lv in LEVELS:
        h = histories[lv]
        n_epochs = len(h["val_acc"])
        epochs = range(1, n_epochs + 1)
        ax1.plot(epochs, h["val_loss"], **styles[lv])
        ax2.plot(epochs, h["val_acc"], **styles[lv])

    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.set_xticks(range(1, n_epochs + 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    ax1.set_ylabel("验证集损失")
    ax1.set_title("验证集 loss")
    ax2.set_ylabel("验证集准确率")
    ax2.set_title("验证集准确率")

    fig.suptitle("第 4 章 三档数据增强的对比（验证集从不加增强）", fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_level(level: str, args, device: torch.device):
    """跑一档增强，返回 (history, 测试集 loss, 测试集 acc, 参数量)。"""
    set_seed(args.seed)
    train_tf = get_aug_transform(level)
    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size, limit=args.limit, seed=args.seed, train_tf=train_tf
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print(f"[level={level}] {LEVEL_LABEL[level]}  参数量 {count_params(model):,}  "
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
    print(f"[level={level}] 测试集 loss={te_loss:.4f}  acc={te_acc:.2%}   "
          f"（{int(round(te_acc * len(test_loader.dataset)))}/{len(test_loader.dataset)} 张正确）")

    # 看一眼训练集里"训练集准确率"和"测试集准确率"的差 ——
    # 增强的核心作用就是压住过拟合，这个差值是最直接的证据。
    final_train_acc = history["train_acc"][-1]
    print(f"[level={level}] 最后一轮 训练集 {final_train_acc:.2%} vs "
          f"验证集 {history['val_acc'][-1]:.2%}  →  过拟合 gap "
          f"{final_train_acc - history['val_acc'][-1]:.2%}")
    print()
    return history, te_loss, te_acc, count_params(model)


def main() -> None:
    args = parse_args("第 4 章：数据增强")
    set_seed(args.seed)
    device = get_device()

    # --- 1. 两张与训练无关的示意图（冒烟也写正式文件名，因为它们不依赖轮数）---
    plot_aug_grid(FIGURES / "ch04_aug_grid.png", args.seed)
    print(f"[产出] 已保存 {FIGURES / 'ch04_aug_grid.png'}（同一张图的 8 次强增强）")
    plot_flip_trap(FIGURES / "ch04_flip_trap.png")
    print(f"[产出] 已保存 {FIGURES / 'ch04_flip_trap.png'}（翻转为什么有害）")
    print()

    levels = [args.level] if args.level else list(LEVELS)
    histories, rows = {}, []
    for lv in levels:
        print("=" * 62)
        h, te_loss, te_acc, n_params = run_level(lv, args, device)
        histories[lv] = h
        rows.append({
            "level": lv,
            "label": LEVEL_LABEL[lv],
            "params": n_params,
            "final_train_acc": f"{h['train_acc'][-1]:.4f}",
            "final_val_acc": f"{h['val_acc'][-1]:.4f}",
            "test_loss": f"{te_loss:.4f}",
            "test_acc": f"{te_acc:.4f}",
            "overfit_gap": f"{h['train_acc'][-1] - h['val_acc'][-1]:.4f}",
        })

    print("=" * 62)
    print("汇总")
    print("-" * 62)
    print(f"{'增强':<8}{'训练集':>9}{'验证集':>9}{'测试集':>9}{'过拟合gap':>11}")
    for r in rows:
        print(f"{r['label']:<8}{float(r['final_train_acc']):>9.2%}"
              f"{float(r['final_val_acc']):>9.2%}{float(r['test_acc']):>9.2%}"
              f"{float(r['overfit_gap']):>11.2%}")
    print("-" * 62)

    # --- 结果落盘 CSV ---
    # 只有"书上那条命令"才写正式文件名（见 is_canonical 的说明）。
    # 探索性运行（冒烟 / 改轮数 / 只跑单档）一律另起名字，绝不覆盖书里引用的那份表。
    if is_canonical(args):
        csv_path = RESULTS / "ch04_compare.csv"
    elif len(levels) == len(LEVELS):
        tag = "smoke" if args.smoke else f"e{args.epochs}"
        csv_path = RESULTS / f"ch04_compare_{tag}.csv"
    else:
        csv_path = RESULTS / f"ch04_compare_{levels[0]}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[产出] 已保存 {csv_path}")

    # --- 对比图（只在三档都跑过时画）---
    if len(levels) == len(LEVELS):
        out = tagged("ch04_compare.png", args)
        plot_compare(histories, out)
        print(f"[产出] 已保存 {out}")
    print()
    print("  看这张图，重点看两件事：")
    print("  1. 三条验证集准确率曲线。差别大不大？")
    print("     —— MNIST 上手写数字本来就写得很规范，增强的收益可能很小。")
    print("  2. 过拟合 gap（训练集准确率 - 验证集准确率）。")
    print("     —— 增强的主要作用是把这个 gap 压小，不一定是把准确率抬高。")


if __name__ == "__main__":
    main()
