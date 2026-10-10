"""第 2 章：全连接网络与训练循环

本章是全书第一次真正训练模型。用的网络很朴素（两层全连接），
目的是把「训练循环」这件事彻底讲清楚 —— 因为后面所有章节，
不管你换卷积、换优化器、加数据增强，这个循环的骨架都不会变。

用法：
    python ch02_mlp.py                 # 完整：10 epoch
    python ch02_mlp.py --epochs 3      # 只跑 3 轮
    python ch02_mlp.py --smoke         # 冒烟：1 epoch + 512 张
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


# ---------------------------------------------------------------------------
# 全书统一的三个工具函数（与第 0、1 章一致，各脚本自带一份以保证可单独运行）
# ---------------------------------------------------------------------------
def parse_args(desc: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--smoke", action="store_true", help="快速冒烟：1 epoch + 少量样本")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    # 学习率的默认值 0.01 是**配合 SGD 用**的，不能随便改成 1e-3。
    # 1e-3 是 Adam 的经典默认值；SGD 只看当前 batch 的梯度、方向噪声大，
    # 需要大 10~50 倍的学习率才走得动。本书实测（10 epoch，其余相同）：
    #     SGD lr=1e-3  -> 测试集 94.51%   （明显没训够）
    #     SGD lr=0.01  -> 测试集 97.90%
    #     SGD lr=0.05  -> 测试集 98.19%
    #     Adam lr=1e-3 -> 测试集 97.95%
    # 第 5 章会把学习率系统扫一遍，这里先记住结论：**换优化器就要重新想学习率**。
    p.add_argument("--lr", type=float, default=0.01, help="学习率（第 5 章细讲）")
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.limit = 1, 512
    return a


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 GPU 训练结果可复现 —— 不设的话同一份代码、同一个种子，每次跑出来
    # 都不完全一样，书里"对照 results/ 里的日志确认结果一致"这句话就不成立了。
    # 实测与原因见第 3 章常见坑 2。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
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


def out_figure(base: str, smoke: bool) -> Path:
    """决定图存到哪个文件名。

    冒烟模式（1 个 epoch）写出来的曲线没有参考价值，如果直接用正式文件名，
    就会把书里要引用的完整训练曲线覆盖掉 —— 而且不容易察觉。所以冒烟产物
    一律加 `_smoke` 后缀，两者互不干扰。
    """
    stem, ext = base.rsplit(".", 1)
    return FIGURES / (f"{stem}_smoke.{ext}" if smoke else base)


# ---------------------------------------------------------------------------
# 数据（与第 1 章同一套，签名不变）
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
class SimpleMLP(nn.Module):
    """最简单的多层感知机：784 -> 256 -> ReLU -> 10。

    输入是一张 28x28 的图，把它拉直成 784 个数的向量，当成一维数据看 ——
    **完全不考虑像素之间的空间关系**。这是它不如卷积的地方（第 3 章对比）。

    nn.Module 的子类必须写两个东西：
      __init__ : 声明这一层有哪些"零件"（带参数的层、不带参数的算子）
      forward  : 声明数据怎么从输入流到输出
    剩下的（反向传播、参数更新）PyTorch 用 autograd 自动搞定，不用你写。
    """

    def __init__(self, hidden: int = 256):
        super().__init__()          # 必须先调用父类构造，否则 PyTorch 认不出你的层
        # Flatten 没有参数，只是把 (N,1,28,28) 变成 (N,784)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden)   # 784 -> 256
        self.relu = nn.ReLU()                    # 激活函数，给网络"非线性"
        self.fc2 = nn.Linear(hidden, 10)         # 256 -> 10，10 个类别各得一个分数

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 注意：这里**不接 softmax**。输出的 10 个数叫 logits（未归一化的分数），
        # 直接交给 CrossEntropyLoss —— 它内部会做 softmax。
        # 再自己 softmax 一遍是新手最常犯的错，会让训练变差（详见本章常见坑）。
        return x


# ---------------------------------------------------------------------------
# 训练与评估 —— 这两个函数的签名后面每一章都会复用
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    """跑完一个 epoch，返回 (平均损失, 准确率)。

    训练循环的五个动作，顺序不能乱：
    """
    model.train()                     # 切到训练模式（影响 Dropout / BatchNorm 的行为）
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)   # 数据搬到和模型同一个设备上

        optimizer.zero_grad()               # ① 清空上一轮残留的梯度（梯度是累加的！）
        logits = model(x)                   # ② 前向：算出预测分数
        loss = criterion(logits, y)         # ③ 算损失：预测和正确答案差多少
        loss.backward()                     # ④ 反向：自动求出每个参数的梯度
        optimizer.step()                    # ⑤ 更新：按梯度调整参数

        # 下面三行只用于统计显示，不参与训练。
        # 乘以 y.size(0) 是因为最后一个 batch 可能不满，要先还原成"总量"再平均。
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)

    return total_loss / n, correct / n


def evaluate(model, loader, device) -> tuple[float, float]:
    """在验证集/测试集上评估，返回 (平均损失, 准确率)。"""
    model.eval()                      # 切到评估模式
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, n = 0.0, 0, 0
    # no_grad 告诉 PyTorch："这一段不用记录计算图"。
    # 评估不需要反向传播，省掉记录能明显省显存、提速。
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


def plot_curve(history: dict, path: Path) -> None:
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

    fig.suptitle("第 2 章 SimpleMLP 训练曲线", fontsize=12)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def main() -> None:
    args = parse_args("第 2 章：全连接网络与训练循环")
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size, limit=args.limit, seed=args.seed
    )

    # 建模型 -> 搬到设备上
    model = SimpleMLP().to(device)
    # CrossEntropyLoss：内部 = softmax + 负对数似然，所以喂给它的必须是
    # **未过 softmax 的 logits**。它同时也能接受类别编号（不用 one-hot）。
    criterion = nn.CrossEntropyLoss()
    # 优化器负责"怎么根据梯度改参数"。SGD 是最基础的，第 5 章会换别的比较。
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print(f"[信息] 训练样本 {len(train_loader.dataset)}  "
          f"验证 {len(val_loader.dataset)}  测试 {len(test_loader.dataset)}")
    print(f"[信息] 模型参数量 {count_params(model):,}")
    print(f"[信息] epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print()
    print(f"{'epoch':>5} | {'train_loss':>10} {'train_acc':>9} | {'val_loss':>9} {'val_acc':>8} | {'秒':>5}")
    print("-" * 62)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
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
    print(f"总耗时 {time.time() - t0:.1f} 秒")

    # 测试集只在最后用一次（第 1 章讲的规矩）
    te_loss, te_acc = evaluate(model, test_loader, device)
    print()
    print(f"[测试集] loss={te_loss:.4f}  acc={te_acc:.2%}   "
          f"（{int(round(te_acc * len(test_loader.dataset)))}/{len(test_loader.dataset)} 张正确）")

    out = out_figure("ch02_curve.png", args.smoke)
    plot_curve(history, out)
    print(f"[产出] 已保存 {out}")
    if args.smoke:
        print("       （冒烟模式产物，带 _smoke 后缀，不会覆盖正式训练曲线）")
    print()
    print("  看这张图：")
    print("  - 训练集的 loss 应当一路下降；验证集的 loss 若开始上升，就是过拟合了。")
    print("  - 两条曲线之间的**间隙**说明模型对训练集的记忆程度 —— 间隙越大越危险。")


if __name__ == "__main__":
    main()
