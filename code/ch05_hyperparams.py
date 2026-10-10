"""第 5 章：超参数调节

前四章的配置基本是写死的：lr=0.01、batch_size=128、SGD+momentum。
这一章把这些"旋钮"一个个拧一遍，看它们各自到底影响什么。

本章要回答四个问题：

  1. **学习率**：为什么说它是最重要的一个超参数？（1e-1 / 1e-2 / 1e-3 / 1e-4）
  2. **batch size**：调大调小分别有什么代价？（32 / 128 / 512）
  3. **优化器**：SGD、SGD+Momentum、Adam、AdamW，**都用同一个学习率**会怎样？
  4. **学习率调度器**：训练过程中把 lr 逐渐调小，有用吗？

外加两件工程上必备的事：

  5. **early stopping**：验证集不再进步就提前停，别白跑
  6. **checkpoint**：把最好的模型存到硬盘，之后能原样加载回来（第 7 章要用）

用法：
    python ch05_hyperparams.py                      # 单次训练（存的模型就是给第 7 章用的）
    python ch05_hyperparams.py --sweep lr           # 只扫学习率
    python ch05_hyperparams.py --sweep all          # 四个子实验全扫
    python ch05_hyperparams.py --epochs 3 --lr 1e-1 # 亲眼看它发散
    python ch05_hyperparams.py --verify-checkpoint  # 把存下来的模型加载回来验一遍
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Optional

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

# 书里正式引用的产物（图 / CSV / checkpoint），只由"书上那条命令"产生：
#     python code/ch05_hyperparams.py
# 其它任何跑法（--smoke、改轮数、--sweep ...）都是探索性的，产物一律另起名字。
# 这条规矩来自第 4 章的一次事故：用 `--epochs 1` 做测试，把书里的 10 轮对比图
# 静默覆盖成了 1 轮版本。判断标准是"这是不是书上那条命令"，不是"是不是冒烟"。
CANONICAL_EPOCHS = 10
CANONICAL_LR = 0.01
# 扫参子实验书里跑的是 8 轮（16 组配置 × 8 轮，全跑一遍要十几分钟）
SWEEP_EPOCHS = 8

# 书上两条正式命令的完整取值 —— **每个能改变产物的开关都要在这里判一遍**。
#
# 本项目因为"漏一个开关"翻过四次车：第 2 章漏 `--epochs`/`--limit`，
# 第 4 章漏 `--level`，第 6 章漏 `--depth`，验收时又发现**每一章都漏了 `--seed`**，
# 而本章练习 3 就叫读者换种子跑。本章还多一种漏法：
#
#   旧版只要"轮数对"就算正式，于是 `--sweep lr` 这种**部分扫参**也会写正式的
#   `results/ch05_sweep.csv` —— 书里那张表是 16 行（四个子实验全扫），
#   读者照练习 4 跑一次 `--sweep optimizer`，表就只剩 3 行了。
#
# 所以每个开关都进表。可执行的版本见 `tests/test_canonical_guard.py`。
_SINGLE = dict(epochs=CANONICAL_EPOCHS, sweep="none", limit=0, seed=42,
               batch_size=128, lr=CANONICAL_LR, momentum=0.9, patience=3)
_SWEEP = dict(_SINGLE, epochs=SWEEP_EPOCHS, sweep="all")
CANONICAL = (_SINGLE, _SWEEP)

# 第 7 章要加载这个文件，所以路径固定（见 ledger 的 Ruling 5）
CHECKPOINT = RESULTS / "best_cnn.pt"


# ---------------------------------------------------------------------------
# 全书统一的工具函数（各章自带一份，保证可单独运行）
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第 5 章：超参数调节")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：1 epoch + 少量样本")
    p.add_argument("--epochs", type=int, default=CANONICAL_EPOCHS)
    p.add_argument("--limit", type=int, default=0, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=CANONICAL_LR)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=3,
                   help="early stopping：验证集连续多少轮没进步就停（0 = 关闭）")
    p.add_argument("--sweep", choices=["none", "lr", "batch", "optimizer", "scheduler", "all"],
                   default="none", help="跑哪个子实验；默认 none = 单次训练")
    p.add_argument("--verify-checkpoint", action="store_true",
                   help="只做一件事：把 checkpoint 加载回来，在测试集上验一遍")
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
    # 本章尤其需要：扫参就是拿一堆数字互相比，不可复现的话比较就没有意义。
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
    """这次运行是不是"书上写的那条命令"。

    书上有两条正式命令，各自对应一组产物：
        单次训练（产出 best_cnn.pt）        --epochs 10
        扫参（产出 ch05_sweep.csv 等）      --epochs 8 --sweep all

    两条都逐项对上面那两张表。特别地：

      - **部分扫参不算正式。** `--sweep lr` 也会走到写 CSV 的那段代码，
        但书里那张 16 行表是四个子实验全扫出来的，少扫几个等于把表换了。
      - `--seed` / `--batch-size` / `--momentum` / `--patience` 每一个都能
        改变曲线和 checkpoint，全都要对。

    **`--verify-checkpoint` 故意不进这两张表。** 它只做一件事：把 checkpoint
    读回来验一遍，不写任何训练产物；而它要读的路径正是 `checkpoint_path(args)`
    给出来的。把它算成"非正式"会让书上那条验证命令去找一个不存在的文件。
    """
    return (not args.smoke) and any(
        all(getattr(args, k) == v for k, v in ref.items()) for ref in CANONICAL
    )


def tagged(base: str, args: argparse.Namespace) -> Path:
    """非正式运行统一加后缀，避免覆盖书里的正式产物（详见第 4 章 4.5 节）。"""
    stem, ext = base.rsplit(".", 1)
    if is_canonical(args):
        return FIGURES / base
    tag = "smoke" if args.smoke else f"e{args.epochs}"
    return FIGURES / f"{stem}_{tag}.{ext}"


def tagged_path(base: str, args: argparse.Namespace, folder: Path) -> Path:
    """同 tagged()，但产物放在 figures/ 之外的目录（CSV、checkpoint 用）。"""
    stem, ext = base.rsplit(".", 1)
    if is_canonical(args):
        return folder / base
    tag = "smoke" if args.smoke else f"e{args.epochs}"
    return folder / f"{stem}_{tag}.{ext}"


# ---------------------------------------------------------------------------
# 数据（与第 1 章一致）
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
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 模型（与第 3 章同一个 SimpleCNN）
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
# 优化器与调度器 —— 本章的两个主角
# ---------------------------------------------------------------------------
def build_optimizer(model, name: str, lr: float, momentum: float = 0.9):
    """按名字建优化器。名字就是书上要打印出来的标签。"""
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if name == "SGD+Momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"未知优化器 {name!r}")


def build_scheduler(name: str, optimizer, epochs: int):
    """按名字建学习率调度器。None 表示不调度（lr 恒定）。"""
    if name == "固定 lr":
        return None
    if name == "StepLR":
        # 每 3 个 epoch 把 lr 乘 0.5
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    if name == "CosineAnnealingLR":
        # 沿着余弦曲线从 lr 平滑降到 0
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"未知调度器 {name!r}")


# ---------------------------------------------------------------------------
# 训练与评估
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


def prediction_histogram(model, loader, device) -> list[int]:
    """统计模型把这批图都判成了哪些类 —— 用来诊断"它到底有没有在看图"。"""
    model.eval()
    counts = [0] * 10
    with torch.no_grad():
        for x, _ in loader:
            pred = model(x.to(device)).argmax(dim=1)
            for c in pred.cpu().tolist():
                counts[c] += 1
    return counts


def run_trial(lr: float, batch_size: int, optimizer_name: str, epochs: int,
              args, device, scheduler_name: str = "固定 lr",
              patience: int = 0, save_to: Optional[Path] = None,
              verbose: bool = True, print_epochs: bool = False) -> dict:
    """跑一组超参配置，返回这一组的成绩。

    返回的 dict 至少包含 brief 约定的五个键：
    lr / batch_size / optimizer / best_val_acc / final_train_loss。
    另外还带上 history、epochs_run、epoch_times 供画图用。
    """
    set_seed(args.seed)          # 每组都从同一个随机状态出发，才谈得上"只改了一个变量"
    train_loader, val_loader, test_loader = get_loaders(batch_size, args.limit, args.seed)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr, args.momentum)
    scheduler = build_scheduler(scheduler_name, optimizer, epochs)

    # 调度器不同、其他全同的三组配置，只有把调度器名字带进 tag 才区分得开，
    # 否则屏幕上会打出三行一模一样的 "SGD+Momentum(lr=0.01,bs=128)"。
    tag = f"{optimizer_name}(lr={lr:g},bs={batch_size})"
    if scheduler_name != "固定 lr":
        tag += f",{scheduler_name}"
    if print_epochs:
        print(f"[{tag}]")
        print(f"{'epoch':>5} | {'train_loss':>12} {'train_acc':>9} | "
              f"{'val_loss':>10} {'val_acc':>8} | {'lr':>8} | {'秒':>5}")
        print("-" * 74)
    elif verbose:
        print(f"  {tag:<46} ", end="", flush=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    epoch_times = []
    best_val_acc, best_epoch = -1.0, 0
    no_improve = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        epoch_times.append(time.time() - t0)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            scheduler.step()

        # --- checkpoint：只在验证集变好的时候存 ---
        # 存的是 state_dict（只有权重和偏置），不是整个模型对象。原因见 5.5 节。
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, ep
            no_improve = 0
            if save_to is not None:
                save_to.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_to)
        else:
            no_improve += 1

        if print_epochs:
            print(f"{ep:>5} | {tr_loss:>12.4f} {tr_acc:>8.2%} | "
                  f"{va_loss:>10.4f} {va_acc:>7.2%} | "
                  f"{optimizer.param_groups[0]['lr']:>8.2e} | {epoch_times[-1]:>5.1f}")

        # --- early stopping：连续 patience 轮没进步就停 ---
        if patience > 0 and no_improve >= patience:
            if verbose:
                print(f"early stop @ep{ep}（best ep{best_epoch} {best_val_acc:.2%}）", end=" ")
            break

    te_loss, te_acc = evaluate(model, test_loader, device)

    # 准确率贴地（<30%，也就是比"全猜最多的那个类"好不了多少）时自动多打一段诊断，
    # 看看它到底把图判成了什么。lr 太大、初始化坏掉，都会掉进这个区间。
    if print_epochs and te_acc < 0.30:
        counts = prediction_histogram(model, test_loader, device)
        print()
        print(f"[诊断] 测试集预测分布（准确率只有 {te_acc:.2%}，说明模型基本没学会）：")
        print("        " + "  ".join(f"{i}:{c}" for i, c in enumerate(counts)))
        top = max(range(10), key=lambda i: counts[i])
        if counts[top] == sum(counts):
            print(f"        10000 张全部判成「{top}」—— 输出是个常数，它根本没看图。")
        elif max(counts) - min(counts) < 0.05 * sum(counts):
            print("        十类基本均匀 —— 输出接近随机，等于在瞎猜。")
        else:
            print(f"        分布有明显偏向（最多的是「{top}」），模型在押注某一类。")

    if verbose:
        print(f"best_val={best_val_acc:.2%} test={te_acc:.2%} "
              f"跑完 {len(history['val_acc'])}/{epochs} 轮")

    return {
        "lr": lr,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "best_val_acc": best_val_acc,
        "final_train_loss": history["train_loss"][-1],
        "test_acc": te_acc,
        "epochs_run": len(history["val_acc"]),
        "best_epoch": best_epoch,
        "epoch_times": epoch_times,
        "history": history,
    }


# ---------------------------------------------------------------------------
# 四个子实验
# ---------------------------------------------------------------------------
# (优化器名, 学习率, 是否画进图里)
# 前四条用**同一个 lr=0.01** —— 这是 brief 要求的对照方式，也正是本章最重要的一课：
# 拿 SGD 的学习率直接去开 Adam，结果会很难看。
# 后两条是"Adam 该用的学习率"，只记录不画图，用来量化这个差别有多大。
OPTIMIZER_TRIALS = [
    ("SGD", 0.01, True),
    ("SGD+Momentum", 0.01, True),
    ("Adam", 0.01, True),
    ("AdamW", 0.01, True),
    ("Adam", 0.001, False),
    ("AdamW", 0.001, False),
]

BATCH_TRIALS = [32, 128, 512]
LR_TRIALS = [1e-1, 1e-2, 1e-3, 1e-4]
SCHEDULER_TRIALS = ["固定 lr", "StepLR", "CosineAnnealingLR"]


def sweep_lr(args, device) -> list[dict]:
    print("=" * 66)
    print("子实验 1：学习率（其余全同：SGD+Momentum，batch=128）")
    print("=" * 66)
    out = []
    for lr in LR_TRIALS:
        out.append(run_trial(lr, args.batch_size, "SGD+Momentum", args.epochs, args, device))
    plot_sweep(out, "lr", "学习率", tagged("ch05_lr_sweep.png", args))
    return out


def sweep_batch(args, device) -> list[dict]:
    print("=" * 66)
    print("子实验 2：batch size（其余全同：SGD+Momentum，lr=0.01）")
    print("=" * 66)
    out = []
    for bs in BATCH_TRIALS:
        out.append(run_trial(args.lr, bs, "SGD+Momentum", args.epochs, args, device))
    plot_batch(out, tagged("ch05_batch.png", args))
    return out


def sweep_optimizer(args, device) -> list[dict]:
    print("=" * 66)
    print("子实验 3：优化器（前四条**同一个 lr=0.01**；后两条是 Adam/AdamW 自己的常用 lr）")
    print("=" * 66)
    out = []
    for name, lr, _ in OPTIMIZER_TRIALS:
        out.append(run_trial(lr, args.batch_size, name, args.epochs, args, device))
    plotted = [r for r, (_, _, keep) in zip(out, OPTIMIZER_TRIALS) if keep]
    plot_sweep(plotted, "optimizer", "优化器（都在 lr=0.01）",
               tagged("ch05_optimizer.png", args))
    return out


def sweep_scheduler(args, device) -> list[dict]:
    print("=" * 66)
    print("子实验 4：学习率调度器（其余全同：SGD+Momentum，lr=0.01）")
    print("=" * 66)
    out = []
    for name in SCHEDULER_TRIALS:
        out.append(run_trial(args.lr, args.batch_size, "SGD+Momentum", args.epochs, args,
                             device, scheduler_name=name))
    plot_sweep(out, "scheduler", "调度器", tagged("ch05_scheduler.png", args),
               key=lambda r: r["scheduler"])
    return out


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def _label(row: dict, kind: str) -> str:
    if kind == "lr":
        return f"lr={row['lr']:g}"
    if kind == "optimizer":
        return f"{row['optimizer']} (lr={row['lr']:g})"
    if kind == "scheduler":
        return row["scheduler"]
    return str(row["batch_size"])


def plot_sweep(rows: list[dict], kind: str, title: str, path: Path, key=None) -> None:
    """通用的"多组配置画在一起"：左边 loss，右边准确率，都画验证集。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for r in rows:
        epochs = range(1, len(r["history"]["val_acc"]) + 1)
        lbl = key(r) if key else _label(r, kind)
        ax1.plot(epochs, r["history"]["val_loss"], marker="o", ms=4, label=lbl)
        ax2.plot(epochs, r["history"]["val_acc"], marker="s", ms=4, label=lbl)

    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    ax1.set_ylabel("验证集损失")
    ax1.set_title("验证集 loss")
    ax2.set_ylabel("验证集准确率")
    ax2.set_title("验证集准确率")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def plot_batch(rows: list[dict], path: Path) -> None:
    """batch size 特殊一点：除了准确率，还要看**每轮的耗时**。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))

    labels = []
    for r in rows:
        epochs = range(1, len(r["history"]["val_acc"]) + 1)
        lbl = f"batch={r['batch_size']}"
        labels.append(lbl)
        ax1.plot(epochs, r["history"]["val_acc"], marker="s", ms=4, label=lbl)

    means = [float(np.mean(r["epoch_times"])) for r in rows]
    bars = ax2.bar(labels, means, color=["tab:blue", "tab:orange", "tab:green"])
    ax2.bar_label(bars, fmt="%.1f s", fontsize=9)
    ax2.set_ylabel("每轮平均耗时（秒）")
    ax2.set_title("每轮耗时")
    ax2.grid(alpha=0.3, axis="y")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("验证集准确率")
    ax1.set_title("验证集准确率")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    fig.suptitle("batch size 的影响：准确率（左）与耗时（右）", fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# checkpoint 的存取
# ---------------------------------------------------------------------------
def checkpoint_path(args) -> Path:
    """正式命令跑出来的才叫 best_cnn.pt（第 7 章按这个路径加载）。"""
    if is_canonical(args):
        return CHECKPOINT
    tag = "smoke" if args.smoke else f"e{args.epochs}"
    return RESULTS / f"best_cnn_{tag}.pt"


def dump_state_dict_shape(model, path: Path) -> None:
    """把 state_dict 里的东西列出来 —— "存下来的到底是什么"最直观的答案。"""
    sd = model.state_dict()
    lines = []
    lines.append("model.state_dict() 里有什么")
    lines.append("=" * 66)
    lines.append(f"{'层名':<16}{'形状':<20}{'元素个数':>12}")
    lines.append("-" * 66)
    total = 0
    for k, v in sd.items():
        n = int(np.prod(v.shape))
        total += n
        lines.append(f"{k:<16}{str(tuple(v.shape)):<20}{n:>12,}")
    lines.append("-" * 66)
    lines.append(f"{'合计':<16}{'':<20}{total:>12,}")
    lines.append("")
    lines.append("注意：state_dict 里**只有权重和偏置**（张量），")
    lines.append("不包含网络结构、优化器状态、epoch 数。想要一次全存下来，")
    lines.append("得自己拼一个 dict（见 5.6 节的写法）。")
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def verify_checkpoint(args, device) -> int:
    """把 checkpoint 加载回来，在测试集上验一遍 —— 证明"存下来的确实能用"。"""
    ckpt = checkpoint_path(args)
    if not ckpt.exists():
        print(f"[错误] 找不到 {ckpt}")
        print("       先跑一次正式的 `python code/ch05_hyperparams.py` 生成它。")
        return 1

    print(f"[验证] 加载 {ckpt}")
    model = SimpleCNN().to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)          # 键名和形状必须完全对得上，否则这里就报错
    model.eval()

    _, _, test_loader = get_loaders(args.batch_size, 0, args.seed)
    te_loss, te_acc = evaluate(model, test_loader, device)
    print(f"[验证] 加载后 测试集 loss={te_loss:.4f}  acc={te_acc:.2%}")
    print("[验证] 能加载回来 = 键名和形状都对得上，模型确实被完整地存下来了。")
    print()

    dump_state_dict_shape(model, FIGURES / "ch05_state_dict.txt")
    return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print()

    if args.verify_checkpoint:
        return verify_checkpoint(args, device)

    rows: list[dict] = []
    todo = (["lr", "batch", "optimizer", "scheduler"] if args.sweep == "all"
            else ([] if args.sweep == "none" else [args.sweep]))
    for name in todo:
        part = {"lr": sweep_lr, "batch": sweep_batch,
                "optimizer": sweep_optimizer, "scheduler": sweep_scheduler}[name](args, device)
        for r in part:
            r["experiment"] = name          # CSV 里用它区分是哪个子实验
        rows += part
        print()

    # --- 单次正式训练：存下给第 7 章用的 checkpoint ---
    # 只在 --sweep none（也就是不带 --sweep 的默认调用）时跑。
    # 扫参的时候不跑它，是为了让 results/best_cnn.pt 永远只由书里写的那条
    # 命令（--epochs 10）产出：否则 --sweep all 会顺带用一个 8 轮的模型
    # 把它覆盖掉，两个产物谁写的就说不清了。
    if args.sweep == "none":
        save_to = checkpoint_path(args)
        print("=" * 66)
        print(f"正式训练（lr={args.lr:g}, batch={args.batch_size}, SGD+Momentum, "
              f"early stopping patience={args.patience}）")
        print("=" * 66)
        main_row = run_trial(args.lr, args.batch_size, "SGD+Momentum", args.epochs, args,
                             device, patience=args.patience, save_to=save_to,
                             print_epochs=True)
        if save_to.exists():
            print(f"[产出] 最好的模型已保存到 {save_to}"
                  f"（验证集最好的一轮，第 {main_row['best_epoch']} 轮）")
        else:
            print("[提示] 本次运行没有保存 checkpoint（未变好或已关闭）")
        print()

    # --- 结果落盘 CSV ---
    if rows:
        csv_path = tagged_path("ch05_sweep.csv", args, RESULTS)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["experiment", "config", "best_val_acc", "epochs_run",
                "test_acc", "final_train_loss", "best_epoch"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "experiment": r["experiment"],
                    "config": f"{r['optimizer']}|lr={r['lr']:g}|bs={r['batch_size']}"
                              f"|sched={r['scheduler']}",
                    "best_val_acc": f"{r['best_val_acc']:.4f}",
                    "epochs_run": r["epochs_run"],
                    "test_acc": f"{r['test_acc']:.4f}",
                    "final_train_loss": f"{r['final_train_loss']:.4f}",
                    "best_epoch": r["best_epoch"],
                })
        print(f"[产出] 已保存 {csv_path}")

    print()
    print("  调参的正确姿势：**一次只改一个变量，看曲线做决策**。")
    print("  不要背「最佳值」—— 换个数据集、换个网络，最优超参就不一样了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
