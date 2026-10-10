"""第 7 章：模型性能测试与误差分析

第 2~6 章都在回答"怎么把准确率做上去"。这一章回答另一个问题：
**这个 98.6% 是什么水平，它错在哪、错得有多自信？**

准确率是**一个数字**，它会藏起很多东西：

  - 10 个类里可能有一个类几乎全错，另外 9 个接近满分，平均下来照样 98%
  - 错误往往集中在某几对数字上（4↔9 之类），那说明错得**有结构**，不是随机噪声
  - 有些错例模型"错得非常自信"（softmax 给到 99%），这比"犹豫地错"危险得多

本章把这些问题一个个摊开：混淆矩阵 → 精确率/召回率/F1 → 每类准确率 →
错例逐张看 → 置信度分布 → 特征可视化（t-SNE）→ 第一层卷积核。

用法：
    python ch07_evaluate.py                       # 本章正式命令：六张图
    python ch07_evaluate.py --smoke               # 冒烟：小测试集 + 小 t-SNE
    python ch07_evaluate.py --tsne-samples 1000   # 换个 t-SNE 规模（探索）
"""
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

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

# 正式命令 = 默认规模那次。只有它允许写 figures/ch07_*.png。
# 理由见第 4 章 4.5 节 + 第 5 章 5.7 节：不是书里那条命令，就别用书里的文件名。
CANONICAL_TSNE_SAMPLES = 3000

# t-SNE 的三个参数（它们本身就是 t-SNE 最常见的调参点，见 7.7 节）：
#   perplexity  每个点"关心多少个邻居"。小 → 只看局部、易碎成一簇簇；
#               大 → 看全局、易糊成一团。5~50 是常用范围，30 是通用默认值。
#   init="pca"  先用 PCA 做一次线性初始化再迭代。比默认的 "random" 稳，
#               而且**结果可复现**（随机初始化每次跑出来的图形状会变）。
#   random_state 固定随机种子，同上。
TSNE_PERPLEXITY = 30
TSNE_INIT = "pca"
TSNE_SEED = 42


# ---------------------------------------------------------------------------
# 全书统一的工具函数（各章自带一份，保证可单独运行）
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第 7 章：模型性能测试与误差分析")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：小测试集 + 小 t-SNE")
    p.add_argument("--epochs", type=int, default=10,
                   help="仅当 checkpoint 缺失、需要现场补训时用到")
    p.add_argument("--limit", type=int, default=0, help="限制测试集样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ckpt", type=str, default=str(RESULTS / "best_cnn.pt"),
                   help="要评估的 checkpoint（默认第 5 章存下的那个）")
    p.add_argument("--tsne-samples", type=int, default=CANONICAL_TSNE_SAMPLES,
                   help=f"t-SNE 用多少个样本（默认 {CANONICAL_TSNE_SAMPLES}）")
    p.add_argument("--n-errors", type=int, default=20, help="错例网格里放几张图")
    a = p.parse_args()
    if a.smoke:
        a.limit, a.tsne_samples, a.n_errors = 1000, 300, 8
    return a


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 GPU 结果可复现（原因见第 3 章常见坑 2）。
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
    """存图，遇到 Windows 的文件占用就重试（详见第 1 章常见坑 4）。"""
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

    正式命令 = 非冒烟 + 默认 t-SNE 规模 + 不限制测试集 + 默认 checkpoint。

    **每个能改变产物的参数都要在这里判一遍** —— `--tsne-samples` 只影响六张图里
    的一张，但它一旦被改，这批产物就不是书里那一批了。按「一条命令一个出口」
    （第 5 章 5.7 节），整批一起加后缀，不做"某几张正式、某几张探索"的混合。

    本章初版漏了 `--seed` / `--epochs` / `--batch-size` 三个：它们平时看不出
    问题（只在 checkpoint 缺失、现场补训时才起作用，见 7.2 节），但只要补训
    发生，产出的就是另一个模型、另一批数字。`--seed` 尤其要防 —— 本项目
    每一章都漏过它，而第 4/5/6 章的练习 3 就叫读者换种子跑。
    """
    if args.smoke or args.limit != 0 or args.n_errors != 20:
        return False
    if args.tsne_samples != CANONICAL_TSNE_SAMPLES:
        return False
    if args.seed != 42 or args.epochs != 10 or args.batch_size != 256:
        return False
    return Path(args.ckpt).resolve() == (RESULTS / "best_cnn.pt").resolve()


def tagged(base: str, args: argparse.Namespace) -> Path:
    """非正式运行统一加后缀，避免覆盖书里的正式产物。"""
    stem, ext = base.rsplit(".", 1)
    if is_canonical(args):
        return FIGURES / base
    if args.smoke:
        tag = "smoke"
    elif args.tsne_samples != CANONICAL_TSNE_SAMPLES:
        tag = f"tsne{args.tsne_samples}"
    elif args.seed != 42:
        # 换种子的探索性运行（练习 3）单独起名，免得几次探索互相覆盖
        tag = f"seed{args.seed}"
    elif args.epochs != 10:
        tag = f"e{args.epochs}"
    else:
        tag = "alt"
    return FIGURES / f"{stem}_{tag}.{ext}"


# ---------------------------------------------------------------------------
# 模型（与第 5 章存档时同一个 SimpleCNN，逐层一致，否则 load_state_dict 会报错）
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

    def features(self, x):
        """fc2 **之前** 的 128 维表示 —— 也就是模型"最后想清楚的东西"。

        7.7 节的 t-SNE 要的就是它：把这一层压到二维平面上，
        看十个数字类在模型眼里到底分不分得开。
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.relu3(self.fc1(self.flatten(x)))

    def forward(self, x):
        return self.fc2(self.features(x))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def get_test_set(limit: int = 0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    test = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)
    if limit and limit < len(test):
        # 取前 limit 个：测试集本身是固定顺序的，所以这也**可复现**。
        test = torch.utils.data.Subset(test, list(range(limit)))
    return test


def get_test_loader(batch_size: int, limit: int = 0) -> DataLoader:
    return DataLoader(get_test_set(limit), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# 加载模型：书里的数字来自第 5 章的 checkpoint
# ---------------------------------------------------------------------------
def load_model(args, device) -> tuple:
    """返回 (model, 是否加载自 checkpoint)。"""
    model = SimpleCNN().to(device)
    ckpt = Path(args.ckpt)

    if ckpt.exists():
        # weights_only=True：这个文件里只有张量，没有需要执行的代码。
        # 第 5 章存的就是纯 state_dict（见 5.6 节）。
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"[模型] 已加载 {ckpt}")
        return model, True

    print(f"[模型] 警告：找不到 {ckpt}")
    print("        现场补训一个代替 —— 但**这一份不是书里的模型**，")
    print("        下面所有数字都会和书里对不上（第 5 章那份是 early stop 在第 9 轮的）。")
    print("        想复现书里的数字，先跑：python code/ch05_hyperparams.py --epochs 10")
    print("        （本章**不会**把补训的模型存回 best_cnn.pt —— 那个文件只由")
    print("          第 5 章的正式命令产出，多一个写入者就说不清它是谁写的。）")
    _train_in_memory(model, device, args)
    return model, False


def _train_in_memory(model, device, args) -> None:
    """checkpoint 缺失时的兜底：照第 5 章的配方训一次，只放在内存里。"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train = datasets.MNIST(str(DATA), train=True, download=True, transform=tf)
    loader = DataLoader(train, batch_size=128, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(args.epochs):
        total = correct = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
        print(f"        [补训] 第 {ep + 1}/{args.epochs} 轮  训练集 {correct / total:.2%}")


# ---------------------------------------------------------------------------
# 核心：把"准确率"拆成一张完整成绩单
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_detailed(model, loader, device) -> dict:
    """跑一遍测试集，返回那张成绩单。

    混淆矩阵**用手写 numpy 累积**，不调 sklearn.metrics：
    多分类的精确率/召回率/F1 是本章要讲清楚的东西，藏进一行函数调用里就没得讲了。

    返回的 dict：
      confusion_matrix : (10, 10) ndarray。**cm[t, p] = 真实是 t、被判成 p 的张数**
                         行 = 真实标签，列 = 预测标签。约定别记反。
      per_class_acc    : {类: 该类判对的比例}（分母是这一类的真实张数）
      precision/recall/f1 : {类: 值}。多分类下逐类算：**把这一类当正类、
                         其余九类一起当负类**，就退回成二分类，公式不用改。
      macro_*          : 十个类的算术平均（10 个类等权，不管样本多少）
      y_true/y_pred/confidence : 逐样本，给后面的图用
    """
    model.eval()                       # 必须！见第 6 章常见坑 1
    n_classes = 10
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    confs, y_true_all, y_pred_all = [], [], []

    for x, y in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1)      # 先转概率，好拿置信度
        conf, pred = prob.max(dim=1)

        # 这一行就是混淆矩阵的全部：真实标签 t、预测标签 p，在 [t, p] 上记一笔。
        # 10000 个样本循环 10000 次——慢，但**看得见**。换成 numpy 花活没必要。
        for t, p in zip(y.tolist(), pred.cpu().tolist()):
            cm[t, p] += 1

        confs.append(conf.cpu().numpy())
        y_true_all.append(y.numpy())
        y_pred_all.append(pred.cpu().numpy())

    confidence = np.concatenate(confs)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    def safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    per_class_acc, precision, recall, f1 = {}, {}, {}, {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        # 分母只用到"行和/列和"，所以矩阵里每一格都有用 —— 这是混淆矩阵比
        # 一个准确率数字信息量大的原因。
        row_sum = int(cm[c, :].sum())      # 真实是 c 的总数
        col_sum = int(cm[:, c].sum())      # 被判成 c 的总数
        fn, fp = row_sum - tp, col_sum - tp

        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        # F1 = 精确率和召回率的调和平均。用调和平均而不是算术平均，
        # 是因为它对"一个高一个低"特别敏感：P=1.0、R=0.0 时 F1=0，而算术平均是 0.5。
        per_class_acc[c] = safe_div(tp, row_sum)
        precision[c], recall[c] = p, r
        f1[c] = safe_div(2 * p * r, p + r)

    return {
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(np.mean(list(precision.values()))),
        "macro_recall": float(np.mean(list(recall.values()))),
        "macro_f1": float(np.mean(list(f1.values()))),
        "accuracy": safe_div(int(np.trace(cm)), int(cm.sum())),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence,
    }


def print_report(rep: dict, n_classes: int = 10) -> None:
    """把成绩单打成表 —— 屏幕上看一遍，正文里对一遍。"""
    cm = rep["confusion_matrix"]
    print()
    print("=" * 70)
    print("混淆矩阵（行 = 真实，列 = 预测；对角线是判对的）")
    print("=" * 70)
    # 注意：Python 3.9 的 f-string **表达式里不能有反斜杠**，
    # 所以"真\预"这个表头要先放进变量，不能直接写进 {}。
    corner = "真\\预"
    print(f"{corner:>5} | " + " ".join(f"{c:>5}" for c in range(n_classes)) + " |  行和")
    print("-" * 70)
    for t in range(n_classes):
        cells = " ".join(f"{cm[t, p]:>5}" for p in range(n_classes))
        print(f"{t:>5} | {cells} | {cm[t, :].sum():>5}")
    print("-" * 70)
    print(f"{'列和':>5} | " + " ".join(f"{cm[:, p].sum():>5}" for p in range(n_classes)))

    print()
    print("=" * 70)
    print("每类的精确率 / 召回率 / F1")
    print("=" * 70)
    print(f"{'类':>3} | {'测试集':>6} | {'准确率':>8} | {'精确率':>8} | {'召回率':>8} | {'F1':>8}")
    print("-" * 62)
    for c in range(n_classes):
        n = int(cm[c, :].sum())
        print(f"{c:>3} | {n:>6} | {rep['per_class_acc'][c]:>8.2%} | "
              f"{rep['precision'][c]:>8.2%} | {rep['recall'][c]:>8.2%} | {rep['f1'][c]:>8.2%}")
    print("-" * 62)
    print(f"{'宏平均':>3} | {'':>6} | {rep['accuracy']:>8.2%} | "
          f"{rep['macro_precision']:>8.2%} | {rep['macro_recall']:>8.2%} | "
          f"{rep['macro_f1']:>8.2%}")
    print()
    print("  最后一行的『准确率』是整体准确率（对角线之和 ÷ 总数），不是宏平均。")
    print("  宏平均 = 十个类的算术平均，**每类等权** —— 所以它和整体准确率")
    print("  不一定相等：类之间样本数不同（第 1 章看过，1 最多、5 最少），")
    print("  宏平均会把小类的表现放大。两者都报，才不会漏掉短板。")

    # 最弱类
    weakest = min(rep["per_class_acc"], key=rep["per_class_acc"].get)
    print()
    print(f"  最弱的类：「{weakest}」 准确率只有 {rep['per_class_acc'][weakest]:.2%}"
          f"（召回率 {rep['recall'][weakest]:.2%}）")

    # 最容易混的几对：只看对角线以外的格子
    pairs = []
    for t in range(n_classes):
        for p in range(n_classes):
            if t != p and cm[t, p] > 0:
                pairs.append((int(cm[t, p]), t, p))
    pairs.sort(reverse=True)
    print()
    print("  最容易混的几对（真实 → 误判成）：")
    for cnt, t, p in pairs[:5]:
        print(f"    {t} → {p} : {cnt} 张"
              f"（占真实「{t}」的 {cnt / cm[t, :].sum():.2%}）")

    # "错得自信"
    wrong = rep["y_pred"] != rep["y_true"]
    if wrong.any():
        print()
        print(f"  错例共 {int(wrong.sum())} 张，其中置信度 ≥ 0.99 的有 "
              f"{int((rep['confidence'][wrong] >= 0.99).sum())} 张 —— 这些是"
              f"「错得最自信」的（7.6 节）。")


# ---------------------------------------------------------------------------
# 六张图
# ---------------------------------------------------------------------------
def plot_confusion(rep: dict, args, n_classes: int = 10) -> None:
    """左：原始张数；右：按行归一化（= 每类的召回率分布）。"""
    cm = rep["confusion_matrix"]
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sum, 1)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("混淆矩阵：原始张数", fontsize=13)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    # 张数 0~1000 跨度大，全都标数字会糊成一片；只标非零且够大的。
    for t in range(n_classes):
        for p in range(n_classes):
            if cm[t, p] > 0:
                axes[0].text(p, t, str(cm[t, p]), ha="center", va="center",
                             fontsize=7.5,
                             color="white" if cm[t, p] > cm.max() * 0.55 else "black")

    im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("同一张矩阵，按行归一化（每格 = 占该真实类的比例）", fontsize=13)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for t in range(n_classes):
        for p in range(n_classes):
            v = cm_norm[t, p]
            if v >= 0.001:      # 只看千分之一以上，否则铺满噪声
                axes[1].text(p, t, f"{v:.1%}", ha="center", va="center", fontsize=7,
                             color="white" if v > 0.55 else "black")

    for ax in axes:
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xlabel("预测", fontsize=11)
        ax.set_ylabel("真实", fontsize=11)

    fig.suptitle(f"混淆矩阵（测试集 {int(cm.sum())} 张，整体准确率 {rep['accuracy']:.2%}）"
                 f" —— 对角线越亮越好", fontsize=14)
    fig.tight_layout()
    _savefig(fig, tagged("ch07_confusion.png", args))
    plt.close(fig)


def plot_misclassified(model, test_set, rep: dict, args, device) -> None:
    """错例网格：挑**错得最自信**的那几张，标注「真实→预测」和置信度。"""
    wrong_idx = np.where(rep["y_pred"] != rep["y_true"])[0]
    if wrong_idx.size == 0:
        print("  [提示] 没有错例，跳过错例网格。")
        return

    # 按置信度从高到低排，取前 n 张 —— 置信度越高的错例越值得看：
    # 模型"很确定却错了"，说明它学到了一条**错的规律**（7.6 节）。
    order = wrong_idx[np.argsort(-rep["confidence"][wrong_idx])]
    picked = order[:args.n_errors]

    n = len(picked)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 3.05 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, picked):
        img, _ = test_set[int(idx)]          # 取原图（已 Normalize，下面反归一化）
        arr = img.squeeze().numpy() * MNIST_STD + MNIST_MEAN
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"真 {rep['y_true'][idx]} → 预 {rep['y_pred'][idx]}\n"
                     f"置信度 {rep['confidence'][idx]:.1%}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # 错的那一格描红边，扫一眼就知道"这是错的"
        for s in ax.spines.values():
            s.set_edgecolor("#c44e52")
            s.set_linewidth(2.0)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"错得最自信的 {n} 张（共 {wrong_idx.size} 张错例）"
                 f" —— 不是「随机错」，是有规律的错", fontsize=13)
    fig.tight_layout()
    _savefig(fig, tagged("ch07_misclassified.png", args))
    plt.close(fig)


def plot_per_class(rep: dict, args, n_classes: int = 10) -> None:
    """每类准确率柱状图，最弱的类标红。"""
    accs = [rep["per_class_acc"][c] for c in range(n_classes)]
    weakest = int(np.argmin(accs))
    overall = rep["accuracy"]

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    colors = ["#c44e52" if c == weakest else "#4c72b0" for c in range(n_classes)]
    bars = ax.bar(range(n_classes), [a * 100 for a in accs], color=colors, width=0.68)

    for c, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                f"{accs[c]:.2%}", ha="center", fontsize=9.5)

    # 整体准确率画成横线：它和每类柱子的差距就是"平均掩盖了什么"
    ax.axhline(overall * 100, color="#55a868", linestyle="--", linewidth=1.8,
               label=f"整体准确率 {overall:.2%}")

    # 只放大顶部，否则 97%~100% 的差别在 0~100 的坐标里根本看不见
    lo = min(accs) * 100
    ax.set_ylim(max(0, lo - 1.2), 100.4)
    ax.set_xticks(range(n_classes))
    ax.set_xlabel("数字类别", fontsize=11)
    ax.set_ylabel("该类准确率（%）", fontsize=11)
    ax.set_title(f"每类准确率 —— 最弱是「{weakest}」{accs[weakest]:.2%}"
                 f"，整体 {overall:.2%}（纵轴已放大，注意起点不是 0）", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, tagged("ch07_per_class.png", args))
    plt.close(fig)


def plot_confidence(rep: dict, args) -> None:
    """左：正确/错误的置信度分布；右：每个置信度区间里的实际准确率。"""
    conf, correct = rep["confidence"], rep["y_pred"] == rep["y_true"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    bins = np.linspace(0, 1, 41)
    axes[0].hist(conf[correct] * 100, bins=bins * 100, color="#4c72b0", alpha=0.85,
                 label=f"判对（{int(correct.sum())} 张）")
    # 错例只有一百多张，用密度而不是张数，否则在这张图上根本看不见
    axes[0].hist(conf[~correct] * 100, bins=bins * 100, color="#c44e52", alpha=0.75,
                 density=False, label=f"判错（{int((~correct).sum())} 张）")
    axes[0].set_yscale("log")     # 数量差两个数量级，取对数才看得见尾巴
    axes[0].set_xlabel("模型给出的置信度（%）", fontsize=11)
    axes[0].set_ylabel("样本数（对数轴）", fontsize=11)
    axes[0].set_title("置信度分布：判对的挤在右边，判错的散在左边", fontsize=12.5)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # 右图：把样本按置信度分箱，看每箱里的实际准确率。
    # 如果一个模型"说 99% 时只有 90% 是对的"，它的置信度就**没校准**。
    edges = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001])
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() > 0:
            centers.append((lo + hi) / 2 * 100)
            accs.append((rep["y_pred"][m] == rep["y_true"][m]).mean() * 100)
            counts.append(int(m.sum()))

    axes[1].plot(centers, accs, "o-", color="#55a868", linewidth=2, markersize=7)
    for x, y, n in zip(centers, accs, counts):
        axes[1].annotate(f"n={n}", (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=9)
    # 完美校准 = 落在 y=x 这条线上
    axes[1].plot([0, 100], [0, 100], "--", color="gray", linewidth=1.4,
                 label="完美校准（说 x% 就真有 x% 对）")
    axes[1].set_xlabel("模型说的置信度（%）", fontsize=11)
    axes[1].set_ylabel("该区间实际准确率（%）", fontsize=11)
    axes[1].set_title("置信度可信吗：置信度区间 vs 实际准确率", fontsize=12.5)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.suptitle("置信度分析 —— 模型「有多确定」和「有多对」是两件事", fontsize=14)
    fig.tight_layout()
    _savefig(fig, tagged("ch07_confidence.png", args))
    plt.close(fig)


@torch.no_grad()
def extract_features(model, loader, device):
    """把整个 loader 的 128 维特征全取出来。

    10000 × 128 个 float32 只有 5 MB，**全取比"只取一部分"更省事**：
    只取一部分就得处理"取哪一部分"，而下面会看到，随手取前 N 个是有坑的。
    """
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(model.features(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def plot_tsne(model, loader, rep, args, device, n_classes: int = 10) -> None:
    """t-SNE：把 128 维特征压到 2 维，看十类分不分得开。"""
    feats_all, labels_all = extract_features(model, loader, device)
    total = len(feats_all)
    n = min(args.tsne_samples, total)

    if n < total:
        # **不能用「前 n 个」**：实测测试集前 3000 张的准确率是 97.80%，
        # 比整体 98.62% 低 0.8pp —— 那是个偏难的子集，画出来的图会
        # 夸大错误密度。随机抽 + 固定种子，既代表整体又可复现。
        rng = np.random.default_rng(args.seed)
        sel = np.sort(rng.choice(total, size=n, replace=False))
    else:
        sel = np.arange(total)

    feats, labels = feats_all[sel], labels_all[sel]
    # 下标对齐：rep 也是按同样的顺序、在同一个 loader 上算的。
    wrong = rep["y_pred"][sel] != rep["y_true"][sel]
    sub_acc = float((~wrong).mean())
    print(f"  [t-SNE] 子集自身的准确率 {sub_acc:.2%}"
          f"（全测试集 {rep['accuracy']:.2%}，两者接近才说明抽样没偏）")

    # perplexity 不能超过样本数太多。经验上要满足 n > 3 * perplexity。
    perplexity = TSNE_PERPLEXITY
    if n <= 3 * perplexity:
        perplexity = max(5, n // 4)
        print(f"  [提示] 样本只有 {n}，perplexity 从 {TSNE_PERPLEXITY} 降到 {perplexity}。")

    print(f"  [t-SNE] 正在降到 2 维（{n} 个样本，perplexity={perplexity}，"
          f"init={TSNE_INIT!r}）—— 要几十秒，稍等...")
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=perplexity, init=TSNE_INIT,
               random_state=TSNE_SEED, max_iter=1000).fit_transform(feats)
    print(f"  [t-SNE] 完成，用时 {time.time() - t0:.1f}s")

    try:
        cmap = plt.get_cmap("tab10", n_classes)
    except Exception:
        cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, title in ((axes[0], "按真实标签着色"), (axes[1], "判错的点标红")):
        for c in range(n_classes):
            m = labels == c
            ax.scatter(emb[m, 0], emb[m, 1], s=7, color=cmap(c), label=str(c),
                       alpha=0.75, linewidths=0)
        ax.set_title(title, fontsize=12.5)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].scatter(emb[wrong, 0], emb[wrong, 1], s=26, facecolors="none",
                    edgecolors="#c44e52", linewidths=1.3,
                    label=f"判错（{int(wrong.sum())} 张）")
    axes[1].legend(fontsize=9, markerscale=1.6, ncol=2)

    axes[0].legend(fontsize=9, markerscale=2.2, ncol=2, title="真实类别", title_fontsize=9)
    fig.suptitle(f"最后一层 128 维特征的 t-SNE（{n} 个测试样本）"
                 f" —— 同类聚在一起 = 模型把它们看成了一类", fontsize=13.5)
    fig.tight_layout()
    _savefig(fig, tagged("ch07_tsne.png", args))
    plt.close(fig)


def plot_kernels(model, args, max_k: int = 32) -> None:
    """第一层卷积核：32 个 3×3，看模型自己学出了什么。"""
    w = model.conv1.weight.detach().cpu().numpy()      # (32, 1, 3, 3)
    k = min(max_k, w.shape[0])
    ncols = 8
    nrows = (k + ncols - 1) // ncols

    # 32 张子图**共用**一套对称的发散色表：否则每张各自拉伸，
    # 会把不同的核画得看起来一样（第 3 章踩过这个坑）。
    vmax = float(np.abs(w[:k]).max())
    vmin = -vmax if vmax > 0 else -1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.65 * ncols, 1.85 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(k):
        axes[i].imshow(w[i, 0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"#{i}", fontsize=9)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for ax in axes[k:]:
        ax.axis("off")

    fig.suptitle(f"第一层 {k} 个 3×3 卷积核（红=正权重，蓝=负权重，同一套色标）\n"
                 f"它们不是人设计的，是训练出来的 —— 每个核是一种「小笔画探测器」",
                 fontsize=12.5)
    fig.tight_layout()
    _savefig(fig, tagged("ch07_kernels.png", args))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print()
    print("=" * 70)
    print("第 7 章：模型性能测试与误差分析")
    print("=" * 70)

    model, from_ckpt = load_model(args, device)
    test_set = get_test_set(args.limit)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(f"[数据] 测试集 {len(test_set)} 张")

    rep = evaluate_detailed(model, loader, device)
    print_report(rep)

    print()
    print("-" * 70)
    print("生成图...")
    plot_confusion(rep, args)
    plot_misclassified(model, test_set, rep, args, device)
    plot_per_class(rep, args)
    plot_confidence(rep, args)
    plot_tsne(model, loader, rep, args, device)
    plot_kernels(model, args)

    for name in ("ch07_confusion.png", "ch07_misclassified.png", "ch07_per_class.png",
                 "ch07_confidence.png", "ch07_tsne.png", "ch07_kernels.png"):
        p = tagged(name, args)
        print(f"  [产出] {p.relative_to(ROOT)}")

    print()
    print("=" * 70)
    print("  读法：准确率是一个数字，本章把它拆成矩阵、每类指标、错例和置信度。")
    print("  『最弱类』和『最容易混的对』才是下一步该动手的地方。")
    if not from_ckpt:
        print("  ⚠️ 本次没有用第 5 章的 checkpoint，数字与书中不一致。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
