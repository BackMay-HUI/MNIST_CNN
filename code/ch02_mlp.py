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
    p.add_argument("--demos", action="store_true",
                   help="只打印正文四处教学演示的数字（不训练、不写任何产物、不碰 GPU）")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    # 【优化器和学习率要配套】这是本章那个"常见坑"的开关，正文那张表就是用
    # 下面这几条命令量出来的（都是 10 epoch、seed 42、其余设置全同）：
    #     --optimizer sgd  --lr 1e-3   -> 测试集 94.51%   太小：SGD 走不动
    #     --optimizer sgd  --lr 0.01   -> 测试集 97.91%   本书采用的组合
    #     --optimizer sgd  --lr 0.05   -> 测试集 98.17%
    #     --optimizer adam --lr 1e-3   -> 测试集 97.99%   Adam 的经典默认值
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                   help="优化器：sgd（配 momentum=0.9）或 adam（第 5 章细讲）")
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


# 书里那条正式命令用的是这个轮数。改这里就等于改"什么算正式运行"。
CANONICAL_EPOCHS = 10

# 书里那条命令的完整取值 —— **每个能改变产物的开关都要在这里判一遍**。
#
# 写成一张表、而不是串成一行 and，是因为"漏一个"这件事在本项目里复发过四次：
# 本章漏过 `--epochs` / `--limit`，第 4 章漏过 `--level`，第 6 章漏过 `--depth`，
# 验收时又发现**每一章都漏了 `--seed`** —— 而第 4/5/6 章的练习 3 恰恰叫读者
# 换成种子 0/1/2 再跑一遍。也就是说：照着书上的练习做，就会把书里的图弄坏。
#
# 以后再加参数，往这张表里补一行即可。可执行的版本见
# `tests/test_canonical_guard.py` —— 每个开关都列成了一个用例，漏了就会红。
CANONICAL = dict(epochs=CANONICAL_EPOCHS, limit=0, seed=42, batch_size=128,
                 lr=0.01, optimizer="sgd")


def is_canonical(a: argparse.Namespace) -> bool:
    """这次运行是不是"书上写的那条命令"。

    **不能只看 `--smoke`。** 本章初版的判断条件是"冒烟就加后缀"，
    结果是：`--epochs 1` 这类探索性运行**不是冒烟**，于是照样写
    `ch02_curve.png`，把书里引用的完整 10 轮曲线**静默覆盖掉** —— 不报错、
    不提示，等你发现时原始曲线已经没了（实测：`b8c62033…` → `7772d565…`）。

    所以判断标准改成「**这次运行是不是书上那条命令**」：非冒烟，
    并且上面 `CANONICAL` 里每一项的取值都对得上。
    """
    return (not a.smoke) and all(getattr(a, k) == v for k, v in CANONICAL.items())


def out_figure(base: str, a: argparse.Namespace) -> Path:
    """决定图存到哪个文件名：正式运行用原名，其余一律加后缀。

    后缀规则：冒烟 `_smoke`，其余按轮数 `_e{N}`（比如 `--epochs 1`
    写 `ch02_curve_e1.png`）。这样任何探索性运行都不会碰到书里的图。
    """
    stem, ext = base.rsplit(".", 1)
    if is_canonical(a):
        return FIGURES / base
    tag = "smoke" if a.smoke else f"e{a.epochs}"
    return FIGURES / f"{stem}_{tag}.{ext}"


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


def print_demos() -> None:
    """打印 2.4~2.7 节那四处教学演示的数字 —— 正文里 "实测：" 后面的每一块。

    **为什么要有这个入口**：这四处是"讲原理"用的、不是训练结果，一度只以
    文本形式躺在正文里 —— 读者照书敲一遍，数字对不上时没法判断是书错了还是
    自己错了，因为**没有"书上那条命令"可跑**。第 6 章的 `print_dropout_demo`
    早就是正确做法，这里补齐。

    两条纪律：

    1. **不训练、不写产物、不碰 GPU** —— 在 `main()` 里 parse 完就短路返回，
       所以它既不会覆盖书里的图，也不需要进 `is_canonical` 的判断。
    2. **进出各存一次 RNG 状态**。第 ③ ④ 段会新建 `nn.Linear`（初始化要抽
       随机数），不隔离的话"跑没跑过 --demos"会让后面正式训练的每一组数字都
       跟着变 —— 本书第 2 章对随机数流敏感（见 `main()` 里 get_device/set_seed
       那段注释）。虽然这里提前 return 了、本来影响不到训练，但演示函数将来
       若被别处复用，隔离就是必须的，索性现在就做对。
    """
    rng_state = torch.get_rng_state()
    try:
        # ---------------- ① 梯度累加：不清零，连续 backward 三次 ----------------
        print("=" * 66)
        print("① 忘了 zero_grad() 会怎样：不清零，连续 backward 三次")
        print("=" * 66)
        torch.manual_seed(0)
        lin = nn.Linear(8, 1)
        x = torch.ones(1, 8)
        y = torch.zeros(1, 1)
        lin.zero_grad()
        for i in range(1, 4):
            loss = nn.functional.mse_loss(lin(x), y)
            loss.backward()          # 注意：中间**没有** zero_grad()
            print(f"  第 {i} 次: {lin.weight.grad[0, 0].item():.6f}")
        print()
        print("  三次的比值正是 1 : 2 : 3 —— 第二次拿到的梯度里，"
              "还留着第一次那一份。")

        # ---------------- ② no_grad 关掉的是"记账" ------------------------------
        print()
        print("=" * 66)
        print("② no_grad() 关掉的是计算图记录，不是梯度本身")
        print("=" * 66)
        probe = nn.Linear(4, 2)
        one = torch.ones(1, 4)
        out = probe(one)
        print(f"  默认      : requires_grad={out.requires_grad}  "
              f"grad_fn={type(out.grad_fn).__name__}")
        with torch.no_grad():
            quiet = probe(one)
        print(f"  no_grad 内: requires_grad={quiet.requires_grad}  "
              f"grad_fn={quiet.grad_fn}")

        # ---------------- ③ CrossEntropyLoss 吃 logits，不吃概率 ------------------
        print()
        print("=" * 66)
        print("③ CrossEntropyLoss 吃的是 logits（它内部自己做 softmax）")
        print("=" * 66)
        logits = torch.tensor([[2.0, 0.5, -1.0]])   # 三个类别的原始分数
        target = torch.tensor([0])                  # 正确答案是第 0 类
        ce = nn.CrossEntropyLoss()
        direct = ce(logits, target)
        pre_softmax = ce(torch.softmax(logits, dim=1), target)
        by_hand = nn.functional.nll_loss(
            nn.functional.log_softmax(logits, dim=1), target)
        print(f"  直接喂 logits        : 损失 = {direct.item():.6f}   <- 正确用法")
        print(f"  先 softmax 再喂      : 损失 = {pre_softmax.item():.6f}   <- 错误")
        print(f"  手算 log_softmax+NLL : 损失 = {by_hand.item():.6f}   <- 与第一行一致")
        print("  「先 softmax 再喂」之所以更大：CE 拿到什么都会先做一次 "
              "log_softmax，")
        print("  所以这一写法等于对概率又 softmax 了一遍 —— 概率都在 0~1 之间，"
              "再压一次更平，")
        print("  正确答案那一项的概率反而变小，损失就凭空变大了。")

        # ---------------- ④ nn.Linear 的默认初始化 ------------------------------
        print()
        print("=" * 66)
        print("④ nn.Linear 没写初始化时，权重是什么")
        print("=" * 66)
        torch.manual_seed(0)
        layer = nn.Linear(784, 10)          # 本章第一层的形状
        bound = 1 / 784 ** 0.5
        print(f"  weight 范围 : [{layer.weight.min().item():.6f}, "
              f"{layer.weight.max().item():.6f}]")
        print(f"  bias   范围 : [{layer.bias.min().item():.6f}, "
              f"{layer.bias.max().item():.6f}]")
        print(f"  理论界 ±1/sqrt(784) = ±{bound:.6f}")
        ok = bool((layer.weight.abs() <= bound).all()) and \
            bool((layer.bias.abs() <= bound).all())
        print(f"  weight / bias 的每一个值都落在界内: "
              f"{'全部落在界内' if ok else '有越界值（说明假设错了）'}")
        print("=" * 66)
    finally:
        torch.set_rng_state(rng_state)


def main() -> None:
    args = parse_args("第 2 章：全连接网络与训练循环")

    # `--demos` 到此为止：只打印演示，不碰 GPU、不写任何产物。
    if args.demos:
        print_demos()
        return


    # **顺序有讲究：先探测设备，再固定随机源。**
    # `get_device()` 里的 `nn.Conv2d(1, 4, 3)` 是在 CPU 上建好再 `.cuda()` 的，
    # 所以它消耗的是**全局 CPU 随机数流**。反过来的话（先 set_seed 再探测），
    # 权重初始化拿到的是一段**偏移过的**随机数 —— 同一份代码、同一个种子，
    # 结果就变了。本章实测（lr=0.01、10 epoch）：
    #     先 set_seed 再探测 -> 第 1 轮 val 93.74%，测试集 97.87%
    #     先探测再 set_seed  -> 第 1 轮 val 93.68%，测试集 97.91%
    # 差 0.04pp 看着像噪声，但整条曲线从第 1 轮起就不一样 —— 那是初始化变了。
    # 第 8 章 §8.5 和附录 B 第 22 条专门讲这个坑：**凡是会动随机数流的东西，
    # 要么放在 set_seed 前面，要么自己在内部用独立的 generator。**
    device = get_device()
    set_seed(args.seed)

    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size, limit=args.limit, seed=args.seed
    )

    # 建模型 -> 搬到设备上
    model = SimpleMLP().to(device)
    # CrossEntropyLoss：内部 = softmax + 负对数似然，所以喂给它的必须是
    # **未过 softmax 的 logits**。它同时也能接受类别编号（不用 one-hot）。
    criterion = nn.CrossEntropyLoss()
    # 优化器负责"怎么根据梯度改参数"。SGD 是最基础的，第 5 章会换别的比较。
    # `--optimizer adam` 换的就是这一行 —— 注意同时要换 lr（见本章常见坑）。
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        opt_desc = f"SGD(lr={args.lr:g}, momentum=0.9)"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        opt_desc = f"Adam(lr={args.lr:g})"

    print(f"[信息] 训练样本 {len(train_loader.dataset)}  "
          f"验证 {len(val_loader.dataset)}  测试 {len(test_loader.dataset)}")
    print(f"[信息] 模型参数量 {count_params(model):,}")
    print(f"[信息] epochs={args.epochs}  batch_size={args.batch_size}  {opt_desc}")
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

    out = out_figure("ch02_curve.png", args)
    plot_curve(history, out)
    print(f"[产出] 已保存 {out}")
    if not is_canonical(args):
        print(f"       （非正式运行，产物带后缀，不会覆盖书里的 {FIGURES / 'ch02_curve.png'}）")
    print()
    print("  看这张图：")
    print("  - 训练集的 loss 应当一路下降；验证集的 loss 若开始上升，就是过拟合了。")
    print("  - 两条曲线之间的**间隙**说明模型对训练集的记忆程度 —— 间隙越大越危险。")


if __name__ == "__main__":
    main()
