"""第 8 章工程版：数据加载。

从第 1 章搬来。**改动只有一处**：`get_loaders` 的签名从
`(batch_size, limit, seed)` 三个散装参数，变成一个 `cfg` 对象。

这个改动看着小，但它决定了以后加参数的成本：

    第 1 章写法   get_loaders(128, 0, 42)
                  -> 想加个 num_workers？所有调用点都得改一遍
    工程版写法   get_loaders(cfg)
                  -> 想加？在 Config 里加一行，调用点一个字不用改

第 1 章那种写法在**单文件脚本**里是对的：参数就三个，一眼看得全，
不用为了读一个函数去翻 Config 的定义。**取舍点在于"这个函数会被几处调用、
以后还会不会长参数"** —— 本章 8.4 节有完整的对比表。
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import MNIST_MEAN, MNIST_STD, N_VAL


def build_transform(normalize: bool = True) -> transforms.Compose:
    """PIL 图片 -> 模型能吃的张量。

    ToTensor()  : 28x28 取值 0~255 -> 1x28x28 取值 0.0~1.0，通道维挪到最前（HWC -> CHW）
    Normalize() : 逐通道 (x - mean) / std，把数据拉成均值 0、标准差 1

    normalize=False 只做 ToTensor()。**画图时要用这个版本** ——
    归一化之后像素值不再是 0~1，直接 imshow 显示出来是错的。
    """
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return transforms.Compose(steps)


def _load_mnist(data_dir, train: bool, transform) -> datasets.MNIST:
    """下载/读取 MNIST，失败时给出可照做的手动方案（国内网络常超时）。"""
    try:
        return datasets.MNIST(str(data_dir), train=train, download=True, transform=transform)
    except Exception as e:
        raw = data_dir / "MNIST" / "raw"
        raise SystemExit(
            f"\nMNIST 下载失败：{type(e).__name__}: {e}\n\n"
            f"请手动下载下面 4 个文件，全部放进 {raw} ，然后重跑\n"
            f"（目录不存在就自己建一个）：\n"
            f"    train-images-idx3-ubyte.gz\n"
            f"    train-labels-idx1-ubyte.gz\n"
            f"    t10k-images-idx3-ubyte.gz\n"
            f"    t10k-labels-idx1-ubyte.gz\n\n"
            f"下载地址（Stanford 镜像，国内通常可达）：\n"
            f"    https://ossci-datasets.s3.amazonaws.com/mnist/\n\n"
            f"注意：不要解压，torchvision 要的就是 .gz 原文件。\n"
        ) from e


def get_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回 (train_loader, val_loader, test_loader)。

    三个集合的分工（第 1 章讲过，这里再钉一次，因为它最容易用错）：
      train : 更新权重
      val   : 训练中做**所有决定**（早停、选超参、挑模型）—— 我们"看过"它了
      test  : 全程不碰，只做最终汇报 —— 它代表模型面对全新数据的真实水平

    `cfg.limit > 0` 时只取训练集前 N 张（冒烟用），验证集和测试集不动。
    """
    tf = build_transform(normalize=True)
    full_train = _load_mnist(cfg.data_dir, train=True, transform=tf)
    test_set = _load_mnist(cfg.data_dir, train=False, transform=tf)

    # 60000 张切成 50000 训练 / 10000 验证。固定种子的 generator
    # 保证每次切出来的是同一批数据 —— 否则换了随机切分，成绩的变化
    # 就说不清是"改了模型"还是"换了验证集"。
    g = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(
        full_train, [len(full_train) - N_VAL, N_VAL], generator=g
    )

    if cfg.limit > 0:
        train_set = torch.utils.data.Subset(train_set, range(min(cfg.limit, len(train_set))))

    # num_workers=0：在主进程里读数据。Windows 下多进程要 spawn，教程环境里
    # 容易出怪问题，而 MNIST 小到不需要多进程。生产环境可以调到 4 或 8。
    common = dict(num_workers=0)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True,            # 训练集要打乱，否则模型会学到"样本顺序"
        generator=g, **common,
    )
    # 验证/测试不需要打乱：评估结果与顺序无关，打乱反而让复现变麻烦
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
