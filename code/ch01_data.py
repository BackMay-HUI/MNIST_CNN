"""第 1 章：数据加载

本章不训练模型，只把「数据」这件事讲透：
  - MNIST 长什么样（多少张、多大、几个类别）
  - Dataset 和 DataLoader 各自解决什么问题
  - ToTensor() 和 Normalize() 到底对数据做了什么
  - 训练集 / 验证集 / 测试集为什么这么切

用法：
    python ch01_data.py            # 完整：准备数据 + 打印统计 + 出样本图
    python ch01_data.py --smoke    # 冒烟：只用 512 张训练样本，几秒跑完
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # 无窗口后端：不弹窗、可在没有显示器的环境里存图
import matplotlib.pyplot as plt

# matplotlib 默认字体不含中文字形，图里的中文会全变成"豆腐块"（□□□）。
# 这里按优先级挑一个本机存在的中文字体。最后留 DejaVu Sans 兜底。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans",
]
# 中文字体里的负号是"−"(U+2212) 而非 "-"，不关掉的话坐标轴负号也会变方块
matplotlib.rcParams["axes.unicode_minus"] = False

# Windows 上 Python 默认用 GBK 输出，中文会变乱码。强制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 本文件在 <ROOT>/code/ 下，所以上两级才是项目根目录。
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# MNIST 训练集的全局均值与标准差（下一节会讲这两个数怎么来的）
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# 训练集 60000 张里切出 10000 张做验证，剩下 50000 张训练
N_VAL = 10000


# ---------------------------------------------------------------------------
# 以下三个函数与第 0 章完全一致，每个章节脚本都自带一份，保证脚本可以单独运行
# ---------------------------------------------------------------------------
def parse_args(desc: str) -> argparse.Namespace:
    """全书统一的命令行参数约定。"""
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--smoke", action="store_true", help="快速冒烟：1 epoch + 少量样本")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.limit = 1, 512
    return a


def set_seed(seed: int) -> None:
    """固定四个随机源，保证同机同代码可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """选设备：is_available() 通过还不够，必须再跑一次真实算子（详见第 0 章）。"""
    if torch.cuda.is_available():
        try:
            conv = torch.nn.Conv2d(1, 4, 3).cuda()
            y = torch.relu(conv(torch.randn(2, 1, 28, 28, device="cuda")))
            torch.cuda.synchronize()
            print(f"[设备] CUDA 可用且真实算子通过，输出形状 {tuple(y.shape)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"[设备] 警告：is_available() 为 True 但真实算子失败，降级 CPU：{e}")
    else:
        print("[设备] CUDA 不可用，使用 CPU。")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 本章主角
# ---------------------------------------------------------------------------
def build_transform(normalize: bool = True) -> transforms.Compose:
    """把「PIL 图片」变成「模型能吃的张量」的处理流水线。

    两步：
      ToTensor()  : PIL 图片 (28x28, 取值 0~255) -> 张量 (1x28x28, 取值 0.0~1.0)
                    同时把通道维从最后挪到最前（HWC -> CHW）
      Normalize() : 逐通道做 (x - mean) / std，把数据拉成均值 0、标准差 1
                    MNIST 是灰度图只有 1 个通道，所以 mean/std 都是单个数

    normalize=False 时只做 ToTensor()。**画图时必须用这个版本**，
    因为归一化之后的像素值不再是 0~1，直接 imshow 会显示得不对。
    """
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return transforms.Compose(steps)


def _load_mnist(train: bool, transform) -> datasets.MNIST:
    """下载/读取 MNIST，失败时给出可照做的手动方案。

    torchvision 会去几个境外地址拉数据，国内网络经常超时。与其抛一个
    看不懂的 URLError，不如直接告诉用户「下哪 4 个文件、放到哪」。
    """
    try:
        return datasets.MNIST(str(DATA), train=train, download=True, transform=transform)
    except Exception as e:
        raw = DATA / "MNIST" / "raw"
        raise SystemExit(
            f"\nMNIST 下载失败：{type(e).__name__}: {e}\n\n"
            f"请手动下载下面 4 个文件，全部放进 {raw} ，然后重跑本脚本\n"
            f"（目录不存在就自己建一个）：\n"
            f"    train-images-idx3-ubyte.gz\n"
            f"    train-labels-idx1-ubyte.gz\n"
            f"    t10k-images-idx3-ubyte.gz\n"
            f"    t10k-labels-idx1-ubyte.gz\n\n"
            f"下载地址（Stanford 镜像，国内通常可达）：\n"
            f"    https://ossci-datasets.s3.amazonaws.com/mnist/\n\n"
            f"注意：不要解压，torchvision 要的就是 .gz 原文件。\n"
        ) from e


def get_loaders(
    batch_size: int, limit: int = 0, seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回 (train_loader, val_loader, test_loader)。**后续各章复用这个签名**。

    三个集合的分工：
      train : 用来更新权重
      val   : 训练过程中用来判断「有没有过拟合、该不该早停、选哪组超参」
              —— 它不参与梯度更新，但对我们是"看过"的，所以不能代表最终成绩
      test  : 全程不碰，只在最后用一次，代表模型面对全新数据的真实水平

    limit > 0 时只取训练集的前 limit 张（冒烟测试用），验证集和测试集不动。
    """
    tf = build_transform(normalize=True)
    full_train = _load_mnist(train=True, transform=tf)
    test_set = _load_mnist(train=False, transform=tf)

    # 从 60000 张里切 50000 训练 / 10000 验证。
    # 传一个固定种子的 generator，保证每次切出来的是同一批数据。
    g = torch.Generator().manual_seed(seed)
    n_train = len(full_train) - N_VAL
    train_set, val_set = random_split(full_train, [n_train, N_VAL], generator=g)

    if limit > 0:
        # 只截取前 limit 张。用 Subset 包一层，原数据集不动。
        train_set = torch.utils.data.Subset(train_set, range(min(limit, len(train_set))))

    # num_workers=0 表示在主进程里读数据。
    # Windows 下多进程要用 spawn 启动，教程环境里容易出各种怪问题，
    # 而 MNIST 小到根本不需要多进程 —— 所以这里固定 0。
    # 生产环境（数据大、预处理重）可以调到 4 或 8 来让读数据与算梯度并行。
    common = dict(num_workers=0)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,          # 训练集要打乱：否则模型会学到"样本顺序"这种无用规律
        generator=g,
        **common,
    )
    # 验证集/测试集不需要打乱：评估结果与顺序无关，打乱反而让复现变麻烦
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def plot_samples(path: Path, rows: int = 3, cols: int = 6, seed: int = 42) -> None:
    """画一张 rows×cols 的样本网格，每张图上方标出它的真实标签。

    注意这里用的是 **不归一化** 的数据集 —— 归一化后像素值不在 0~1 之间，
    直接画出来颜色是错的。这是新手画 MNIST 图最常见的翻车点。
    """
    raw = _load_mnist(train=True, transform=build_transform(normalize=False))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(raw), generator=g)[: rows * cols].tolist()

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.35))
    for k, ax in enumerate(axes.flat):
        img, label = raw[idx[k]]
        # img 形状是 (1, 28, 28) = (通道, 高, 宽)。
        # matplotlib 要的是 (高, 宽) 或 (高, 宽, 通道)，所以必须 squeeze 掉通道维。
        ax.imshow(img.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"标签 {label}", fontsize=9)
        ax.axis("off")

    fig.suptitle("MNIST 训练集随机样本", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    _savefig(fig, path)
    plt.close(fig)


def _savefig(fig, path: Path, attempts: int = 5) -> None:
    """存图，遇到 Windows 的文件占用就等一会儿重试。

    Windows 上偶尔会有别的进程短暂独占你要写的文件（杀毒软件扫描新文件、
    资源管理器预览窗格、你刚用看图软件打开过它），此时 open() 会抛
    `OSError: [Errno 22] Invalid argument`。这个错很烦人 —— 脚本前面几分钟
    的训练全白跑了，却倒在最后一步，而且报错信息完全看不出是文件占用。

    所以这里重试几次。真正的原因通常会在一两秒内消失。
    """
    for i in range(attempts):
        try:
            fig.savefig(path, dpi=130, bbox_inches="tight")
            return
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"  [警告] 保存 {path.name} 失败（{e}），1 秒后重试 {i + 1}/{attempts - 1}")
            import time
            time.sleep(1.0)


def report(limit: int, batch_size: int, seed: int) -> None:
    """打印本章要用的全部真实数字 —— 书里引用的就是这些值。"""
    def line(t=""):
        print(t)

    line("=" * 62)
    line(" 1. 三个集合的规模")
    line("=" * 62)

    tf_norm = build_transform(normalize=True)
    tf_raw = build_transform(normalize=False)
    full = _load_mnist(train=True, transform=tf_norm)
    full_raw = _load_mnist(train=True, transform=tf_raw)
    test_set = _load_mnist(train=False, transform=tf_norm)
    line(f"  MNIST 训练集原始总数 : {len(full)}")
    line(f"  MNIST 测试集总数     : {len(test_set)}")
    line(f"  切分后 训练 / 验证   : {len(full) - N_VAL} / {N_VAL}")
    if limit > 0:
        line(f"  本次 --limit 限制为  : {limit}（仅训练集，验证/测试不截断）")
    line()

    line("=" * 62)
    line(" 2. 单张图像的形状与取值范围")
    line("=" * 62)
    raw_img, label = full_raw[0]   # ToTensor 之后、Normalize 之前
    norm_img, _ = full[0]          # 两步都做过之后
    line(f"  Dataset 取出一张      : {type(raw_img).__name__}  shape={tuple(raw_img.shape)}  dtype={raw_img.dtype}")
    line("    ^ 注意是 (通道, 高, 宽) = (1, 28, 28)，通道在最前面（CHW）")
    line(f"  该样本标签           : {label}")
    line(f"  ToTensor 之后范围    : [{raw_img.min():.4f}, {raw_img.max():.4f}]")
    line(f"  Normalize 之后范围   : [{norm_img.min():.4f}, {norm_img.max():.4f}]")
    line(f"    公式 (x - {MNIST_MEAN}) / {MNIST_STD}")
    line(f"    0 -> {(0 - MNIST_MEAN) / MNIST_STD:.4f}   1 -> {(1 - MNIST_MEAN) / MNIST_STD:.4f}")
    line(f"  归一化后均值/标准差  : {norm_img.mean():.4f} / {norm_img.std():.4f}")
    line("    ^ 单张图的均值不会正好是 0，因为 0.1307 是整个训练集的均值")
    line()

    line("=" * 62)
    line(" 3. 一个 batch 的形状")
    line("=" * 62)
    train_loader, val_loader, test_loader = get_loaders(batch_size, limit=limit, seed=seed)
    xb, yb = next(iter(train_loader))
    line(f"  batch_size 设为      : {batch_size}")
    line(f"  训练集批次数         : {len(train_loader)}")
    line(f"  验证集批次数         : {len(val_loader)}")
    line(f"  测试集批次数         : {len(test_loader)}")
    line(f"  一个 batch 的 x      : shape={tuple(xb.shape)}  dtype={xb.dtype}")
    line(f"  一个 batch 的 y      : shape={tuple(yb.shape)}  dtype={yb.dtype}  取值示例 {yb[:10].tolist()}")
    line("    ^ x 的第 0 维是 batch，所以是 (N, 通道, 高, 宽)")
    line(f"  一个 batch x 的范围  : [{xb.min():.4f}, {xb.max():.4f}]")
    line()

    line("=" * 62)
    line(" 4. 类别分布（训练集前 20000 张）")
    line("=" * 62)
    counts = torch.bincount(full.targets[:20000], minlength=10)
    for d in range(10):
        bar = "█" * int(counts[d] / 40)
        line(f"  数字 {d}: {counts[d]:5d}  {bar}")


def main() -> None:
    args = parse_args("第 1 章：数据加载")
    set_seed(args.seed)
    device = get_device()
    print(f"[信息] 项目根目录 {ROOT}")
    print(f"[信息] 数据目录   {DATA}")
    print(f"[信息] 本次设备   {device}")
    print()

    report(limit=args.limit, batch_size=args.batch_size, seed=args.seed)

    print()
    print("=" * 62)
    print(" 5. 产出样本图")
    print("=" * 62)
    out = FIGURES / "ch01_samples.png"
    plot_samples(out)
    print(f"  已保存 {out}  ({out.stat().st_size} bytes)")
    print()
    print("  打开这张图看看：每张小图上方是它的真实标签，")
    print("  数字应当清晰可辨 —— 如果整张图是黑的或像底片，")
    print("  说明画图时误用了归一化后的数据，见本章正文「常见坑」。")


if __name__ == "__main__":
    main()
