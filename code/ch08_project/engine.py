"""第 8 章工程版：训练/评估引擎 + 运行环境。

这是本章"最值得抄走"的一个文件。前七章每章的脚本里都有一份
`train_one_epoch` / `evaluate` 的复制品（这是"每章自包含"的代价，
第 6 章 6.6 节算过：同一个训练循环抄了 6 遍）。工程版把它收成一份，
所有实验共用 —— **改一处，全书生效**。

关于 `set_seed` / `get_device` 为什么放在这里：它们既不属于"配置"
（不是可调的项）、也不属于"模型"或"数据"，而是**跑起来的运行环境**。
严格说它们该单独一个 `utils.py`，但本章只有五个文件、
这两个函数加起来 20 行 —— **为了 20 行多开一个文件，是另一种形式的浪费。**
文件该怎么拆，看的是"职责是否混在一起"，不是"能不能拆得更细"。
"""
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 运行环境
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定所有随机源。少一行，书里"对照我的结果"就变成空话。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 GPU 结果可复现：不加这两行，卷积反向的浮点累加顺序不定，
    # 同一份代码同一种子会跑出不同准确率（实测差 0.30pp，见第 3 章常见坑 2）。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """探测设备。

    **不能只信 `torch.cuda.is_available()`。** 第 0 章踩过：
    某些环境下它返回 True，但 torch 自己的卷积 kernel 跑不了
    （报 `no kernel image`），一路要到训练时才炸。

    所以这里真跑一个小的卷积前向，跑通才算能用。
    """
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


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    """跑完一个 epoch，返回 (平均损失, 准确率)。

    训练循环的五个动作，**顺序不能乱**：

        ① zero_grad()  清空上一轮残留的梯度（梯度是累加的！）
        ② forward      前向
        ③ loss         算损失
        ④ backward     反向：自动求梯度
        ⑤ step()       更新参数
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()               # ①
        logits = model(x)                   # ②
        loss = criterion(logits, y)         # ③
        loss.backward()                     # ④
        optimizer.step()                    # ⑤

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)

    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """在验证集/测试集上评估，返回 (平均损失, 准确率)。

    和第 3 章的版本相比只改了一处：`criterion` 从**函数内部新建**
    改成**调用方传进来**。原因是工程版希望全流程只有一个损失对象 ——
    万一以后要换 `label_smoothing`，改一处就够了，
    不会出现"训练用的这个损失、评估用的那个损失"这种对不上的情况。
    """
    model.eval()                            # 必须！否则 Dropout/BN 还开着
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


# ---------------------------------------------------------------------------
# checkpoint 存取
# ---------------------------------------------------------------------------
def save_checkpoint(model, path: Path) -> None:
    """存 `state_dict`（只有权重和偏置），不是整个模型对象。

    存整个对象（`torch.save(model)`）会把类的**代码路径**一起腌进文件，
    换个目录、改个类名就读不出来了，而且反序列化时会执行 pickle 里的代码
    （有安全风险）。存 `state_dict` 只留张量，加载时要求你的 `SimpleCNN`
    和当初那份**逐层一致**，对不上立刻报错 —— 这正是我们想要的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: Path, device) -> None:
    """把权重装回模型。

    `weights_only=True` 告诉 PyTorch"这个文件里只有张量、没有要执行的代码"。
    对纯 `state_dict` 一定加上。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))


# ---------------------------------------------------------------------------
# 完整训练流程
# ---------------------------------------------------------------------------
def fit(model, loaders, optimizer, criterion, device, cfg,
        verbose: bool = True) -> tuple[dict, float, int]:
    """训练到早停或轮数用完，返回 (history, best_val_acc, best_epoch)。

    每轮：训练 -> 验证 -> 若验证准确率创新高就存 checkpoint -> 判断早停。
    """
    train_loader, val_loader, _ = loaders

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "sec": []}
    best_val_acc, best_epoch = -1.0, 0
    no_improve = 0

    if verbose:
        print(f"{'epoch':>5} | {'train_loss':>10} {'train_acc':>9} | "
              f"{'val_loss':>9} {'val_acc':>8} | {'秒':>5}")
        print("-" * 62)

    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        for k, v in (("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", va_loss), ("val_acc", va_acc), ("sec", elapsed)):
            history[k].append(v)

        # --- checkpoint：只在验证集变好的时候存 ---
        # 比的是**验证集**，不是训练集。训练集准确率一直在涨，
        # 拿它当标准等于永远存最后一个、早停永远不会触发。
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, ep
            no_improve = 0
            save_checkpoint(model, cfg.checkpoint)
        else:
            no_improve += 1

        if verbose:
            print(f"{ep:>5} | {tr_loss:>10.4f} {tr_acc:>8.2%} | "
                  f"{va_loss:>9.4f} {va_acc:>7.2%} | {elapsed:>5.1f}")

        # --- early stopping：连续 patience 轮没进步就停 ---
        if cfg.patience > 0 and no_improve >= cfg.patience:
            if verbose:
                print(f"early stop @ep{ep}（best 是 ep{best_epoch}，val {best_val_acc:.2%}）")
            break

    return history, best_val_acc, best_epoch
