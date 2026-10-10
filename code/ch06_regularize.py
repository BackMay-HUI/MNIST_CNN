"""第 6 章：正则化与结构改进（轻量版）

第 4 章给训练集"加难度"，第 5 章把训练过程调顺。这一章换个方向：
**给模型本身加约束**，看能不能压住过拟合。

本章只看两件事：

  1. **正则化**：BatchNorm 和 Dropout 加上去，train/val 的 gap 会怎么变？
  2. **结构**：把网络加深一层，又怎么变？

"轻量版"的意思是：跑一次、看曲线、给结论，**不展开原理推导**。
但你得知道这两个东西在干什么、参数怎么填、加在哪一行 —— 这些 API 语义要讲透。

用法：
    python ch06_regularize.py                      # 四组对照（本章正式命令）
    python ch06_regularize.py --bn                 # 只跑 BN 一组
    python ch06_regularize.py --dropout            # 只跑 Dropout 一组
    python ch06_regularize.py --bn --dropout       # 只跑 BN+Dropout 一组
    python ch06_regularize.py --depth 3            # 四组对照，但网络加深一层
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
RESULTS = ROOT / "results"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
N_VAL = 10000

# 正式命令：不带 --bn/--dropout、不带 --smoke、轮数为默认值 —— 也就是"四组全跑"那次。
# 只有它允许写 figures/ch06_gap.png。理由见第 4 章 4.5 节 + 第 5 章 5.7 节。
CANONICAL_EPOCHS = 10

# Dropout 的丢弃概率。全连接层用 0.5 是经典默认值（见 6.3 节）。
DROPOUT_P = 0.5


# ---------------------------------------------------------------------------
# 全书统一的工具函数（各章自带一份，保证可单独运行）
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第 6 章：正则化与结构改进")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：1 epoch + 少量样本")
    p.add_argument("--epochs", type=int, default=CANONICAL_EPOCHS)
    p.add_argument("--limit", type=int, default=0, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--bn", action="store_true", help="单独跑「带 BatchNorm」的那一组")
    p.add_argument("--dropout", action="store_true", help="单独跑「带 Dropout」的那一组")
    p.add_argument("--depth", type=int, choices=[2, 3], default=2,
                   help="卷积层的层数：2（默认）或 3（加深一层）")
    p.add_argument("--dropout-p", type=float, default=DROPOUT_P,
                   help="Dropout 的丢弃概率（默认 0.5）")
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.limit = 1, 512
    return a


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 GPU 训练结果可复现（原因见第 3 章常见坑 2）。本章要对比 gap，
    # 不可复现的话"这组的 gap 比那组小"就可能是噪声。
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
    """这次运行是不是「书上写的那条命令」。

    正式命令 = 四组全跑（没给 --bn/--dropout）+ 默认轮数 + 默认层数 + 非冒烟。
    只跑一组（--bn 之类）或换层数（--depth 3）都是探索，产物另起名字，
    免得盖掉书里那张对照图。
    """
    if args.smoke or args.epochs != CANONICAL_EPOCHS or args.depth != 2:
        return False
    return not (args.bn or args.dropout)


def tagged(base: str, args: argparse.Namespace) -> Path:
    """非正式运行统一加后缀，避免覆盖书里的正式产物。"""
    stem, ext = base.rsplit(".", 1)
    if is_canonical(args):
        return FIGURES / base
    if args.smoke:
        tag = "smoke"
    elif args.depth != 2:
        tag = f"depth{args.depth}"
    elif args.bn or args.dropout:
        tag = ("bn" if args.bn else "") + ("dropout" if args.dropout else "")
    else:
        tag = f"e{args.epochs}"
    return FIGURES / f"{stem}_{tag}.{ext}"


# ---------------------------------------------------------------------------
# 数据（与第 5 章一致；本章不做数据增强）
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
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_set


# ---------------------------------------------------------------------------
# 模型：在 SimpleCNN 上加两个开关和一个层数参数
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """第 3 章那个 CNN，加上 use_bn / use_dropout / depth 三个开关。

    depth=2: conv1(1->32) -> conv2(32->64)            -> fc1(1600->128) -> fc2(128->10)
    depth=3: conv1(1->32) -> conv2(32->64) -> conv3(64->128)
                                                      -> fc1(1152->128) -> fc2(128->10)

    加深那层**不接池化**，否则 28x28 走三次池化只剩 1x1，特征图太小。
    """

    def __init__(self, use_bn: bool = False, use_dropout: bool = False,
                 depth: int = 2, dropout_p: float = DROPOUT_P):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.depth = depth

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else None
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64) if use_bn else None

        if depth == 3:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
            self.bn3 = nn.BatchNorm2d(128) if use_bn else None
            flat = 128 * 3 * 3
        else:
            self.conv3 = None
            self.bn3 = None
            flat = 64 * 5 * 5

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat, 128)
        self.drop = nn.Dropout(dropout_p) if use_dropout else None
        self.fc2 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 卷积块的顺序：Conv -> (BN) -> ReLU -> Pool
        # BN 放在 ReLU **之前**，原因见 6.2 节。
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.pool(torch.relu(x))

        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.pool(torch.relu(x))

        if self.conv3 is not None:
            x = self.conv3(x)
            if self.bn3 is not None:
                x = self.bn3(x)
            x = torch.relu(x)          # 最后一个卷积块不池化

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        if self.drop is not None:
            x = self.drop(x)           # Dropout 只加在全连接层（见 6.3 节）
        return self.fc2(x)


def count_params(model) -> int:
    return sum(int(np.prod(p.shape)) for p in model.parameters())


# ---------------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()                      # BN 用 batch 统计量、Dropout 生效
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
    """评估。**必须先 eval()** —— 否则 BN 用 batch 统计量、Dropout 还在丢，
    量出来的"验证集准确率"是错的（见常见坑 1）。"""
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


def per_class_stats(model, loader, device, n_classes: int = 10):
    """统计每一类的 正确数 / 总数。**同样必须在 eval() 模式下跑**。

    为什么要分类看：过拟合不是均匀的。模型往往只把某几类背得特别死，
    整体 gap 会把这个信息平均掉。
    """
    model.eval()
    correct = [0] * n_classes
    total = [0] * n_classes
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
                total[t] += 1
                if t == p:
                    correct[t] += 1
    return correct, total


# ---------------------------------------------------------------------------
# 四组对照
# ---------------------------------------------------------------------------
# (显示名, use_bn, use_dropout)
ALL_CONFIGS = [
    ("baseline", False, False),
    ("+BN", True, False),
    ("+Dropout", False, True),
    ("+BN+Dropout", True, True),
]
SINGLE_CONFIGS = {
    "bn": ("+BN", True, False),
    "dropout": ("+Dropout", False, True),
    "bn+dropout": ("+BN+Dropout", True, True),
}


def pick_configs(args) -> list[tuple[str, bool, bool]]:
    """默认四组全跑；给了 --bn/--dropout 就只跑那一组。"""
    if not (args.bn or args.dropout):
        return ALL_CONFIGS
    if args.bn and args.dropout:
        return [SINGLE_CONFIGS["bn+dropout"]]
    return [SINGLE_CONFIGS["bn"] if args.bn else SINGLE_CONFIGS["dropout"]]


def run_config(name: str, use_bn: bool, use_dropout: bool, args,
               device) -> dict:
    """跑一组配置，返回这一组的成绩。"""
    set_seed(args.seed)          # 每组都从同一个随机状态出发，才谈得上"只改了一个变量"
    train_loader, val_loader, test_set = get_loaders(args.batch_size, args.limit, args.seed)

    model = SimpleCNN(use_bn=use_bn, use_dropout=use_dropout,
                      depth=args.depth, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    t0 = time.time()
    for _ in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        best_val_acc = max(best_val_acc, va_acc)
    elapsed = time.time() - t0

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    te_loss, te_acc = evaluate(model, test_loader, device)

    # --- 末尾单独做一次"每类 gap"统计（eval 模式下，train 和 val 各过一遍）---
    tr_c, tr_t = per_class_stats(model, train_loader, device)
    va_c, va_t = per_class_stats(model, val_loader, device)

    # 两套 gap，本章的核心就在这个对比上：
    #   gap_train_mode —— 训练集准确率取"训练过程中、train() 模式下"的平均值
    #   gap_eval_mode  —— 训练集准确率改用"训练结束后、eval() 模式下"重算一遍
    # 带 Dropout/BN 的模型下这两者差很多，原因见 6.5 节。
    eval_train_acc = sum(tr_c) / sum(tr_t)
    eval_val_acc = sum(va_c) / sum(va_t)
    gap_train_mode = history["train_acc"][-1] - history["val_acc"][-1]
    gap_eval_mode = eval_train_acc - eval_val_acc

    print(f"  {name:<14} 参数量 {count_params(model):>7,} | "
          f"train {history['train_acc'][-1]:>7.2%}  val {history['val_acc'][-1]:>7.2%}  "
          f"gap(train模式) {gap_train_mode:>6.2%} | "
          f"gap(eval模式) {gap_eval_mode:>6.2%} | "
          f"test {te_acc:>7.2%} | {elapsed:>5.1f}s")

    return {
        "name": name,
        "use_bn": use_bn,
        "use_dropout": use_dropout,
        "depth": args.depth,
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "gap": gap_train_mode,
        "eval_train_acc": eval_train_acc,
        "eval_val_acc": eval_val_acc,
        "gap_eval_mode": gap_eval_mode,
        "best_val_acc": best_val_acc,
        "test_acc": te_acc,
        "params": count_params(model),
        "elapsed": elapsed,
        "per_class": [
            {
                "cls": c,
                "train_acc": tr_c[c] / tr_t[c] if tr_t[c] else 0.0,
                "val_acc": va_c[c] / va_t[c] if va_t[c] else 0.0,
                "train_n": tr_t[c],
                "val_n": va_t[c],
            }
            for c in range(10)
        ],
    }


# ---------------------------------------------------------------------------
# 出图与报表
# ---------------------------------------------------------------------------
def plot_gap(rows: list[dict], args, path: Path) -> None:
    """左图：每组的 train/val 曲线（train 虚线、val 实线，同色配对）。
    右图：两套 gap 并排 —— 训练模式下算的 vs eval 模式下算的。

    右图刻意画成并排，因为**只看左边那根柱子会得出错误结论**：
    带 Dropout 的模型左边会变成负数，看起来"泛化得极好"，
    其实那只是因为量训练集准确率时 Dropout 还开着（见 6.5 节）。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for r, c in zip(rows, colors):
        ep = range(1, len(r["history"]["val_acc"]) + 1)
        ax1.plot(ep, r["history"]["train_acc"], linestyle="--", marker="^", ms=4,
                 color=c, alpha=0.75, label=f"{r['name']} train")
        ax1.plot(ep, r["history"]["val_acc"], linestyle="-", marker="s", ms=4,
                 color=c, label=f"{r['name']} val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("准确率")
    ax1.set_title("训练集（虚线）与验证集（实线）")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5, ncol=2)

    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    w = 0.36
    raw = [r["gap"] * 100 for r in rows]
    fixed = [r["gap_eval_mode"] * 100 for r in rows]
    b1 = ax2.bar(x - w / 2, raw, w, label="训练模式下算的 gap", color="#c44e52")
    b2 = ax2.bar(x + w / 2, fixed, w, label="eval 模式下算的 gap", color="#4c72b0")
    ax2.bar_label(b1, fmt="%.2f", fontsize=8.5, padding=1)
    ax2.bar_label(b2, fmt="%.2f", fontsize=8.5, padding=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("最后一轮的 gap（百分点）")
    ax2.set_title("同一个模型，两套 gap")
    ax2.grid(alpha=0.3, axis="y")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.legend(fontsize=8.5)

    depth_note = "" if args.depth == 2 else f"（{args.depth} 层卷积）"
    fig.suptitle(f"正则化对过拟合 gap 的影响{depth_note}", fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def print_class_table(rows: list[dict]) -> None:
    """逐类打印 train/val 准确率与差值 —— 过拟合在哪些类上最严重。"""
    print()
    print("=" * 70)
    print("每类的 train / val 准确率（eval 模式，各组最后一轮的模型）")
    print("=" * 70)
    for r in rows:
        worst = max(r["per_class"], key=lambda d: d["train_acc"] - d["val_acc"])
        print(f"\n[{r['name']}]  最难过拟合的是「{worst['cls']}」")
        print(f"{'类':>4} | {'train_acc':>10} {'val_acc':>9} | {'gap':>8} | "
              f"{'train_n':>8} {'val_n':>7}")
        print("-" * 62)
        for d in r["per_class"]:
            gap = d["train_acc"] - d["val_acc"]
            print(f"{d['cls']:>4} | {d['train_acc']:>9.2%} {d['val_acc']:>8.2%} | "
                  f"{gap:>7.2%} | {d['train_n']:>8,} {d['val_n']:>7,}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def print_dropout_demo(p: float = DROPOUT_P, n: int = 10, rows: int = 3) -> None:
    """把 Dropout 干的事直接打出来：丢掉一半，活下来的放大成 2.0。

    输入全是 1.0，所以打印出来的数字本身就是"放大倍数"。
    用 RNG 状态保存/恢复把这一段的随机消耗隔离掉 —— 否则这个演示会挪动
    随机数流，后面训练的每一组结果都跟着变，书里的数字就对不上了。
    """
    print()
    print("=" * 70)
    print(f"Dropout(p={p:g}) 在干什么：输入全是 1.0，看输出")
    print("=" * 70)

    rng_state = torch.get_rng_state()
    try:
        drop = nn.Dropout(p)
        x = torch.ones(n)
        for _ in range(rows):
            print(f"  {drop(x).tolist()}")
        drop.eval()
        print(f"  eval 模式下（原样通过）: {drop(x).tolist()}")
    finally:
        torch.set_rng_state(rng_state)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    configs = pick_configs(args)
    depth_note = "" if args.depth == 2 else f"，卷积 {args.depth} 层"
    print()
    print("=" * 70)
    print(f"跑 {len(configs)} 组配置（{args.epochs} 轮，种子 {args.seed}，"
          f"SGD+Momentum lr={args.lr:g}{depth_note}）")
    print("=" * 70)

    print_dropout_demo(args.dropout_p)

    rows = [run_config(name, bn, do, args, device) for name, bn, do in configs]

    print()
    if len(rows) > 1:
        plot_gap(rows, args, tagged("ch06_gap.png", args))
        print(f"[产出] 已保存 {tagged('ch06_gap.png', args)}")
    print_class_table(rows)

    print()
    print("  读法：gap = train acc − val acc，**只看最后一轮**。")
    print("  gap 大 = 模型把训练集背下来了；gap 小 = 泛化得好。")
    print("  但注意：gap 小不等于 test acc 高 —— 两者要一起看。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
