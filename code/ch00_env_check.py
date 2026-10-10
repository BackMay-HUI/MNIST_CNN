"""第 0 章：环境自检

这一章不训练任何模型，只做一件事：**确认你的 GPU 是真的能用**。

为什么值得单独一章？因为 `torch.cuda.is_available()` 返回 True 并不代表
你的 GPU 真的能算。本书就是在一个这样的坏环境里开始的 —— 详见
`results/env_before.txt`。

用法：
    python ch00_env_check.py            # 完整自检
    python ch00_env_check.py --smoke    # 与完整模式相同（本章无需训练，保留参数是为了全书统一）
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Windows 上 Python 默认用 GBK 输出，中文会变成乱码（"CNN 前向+反向" 会显示成
# "CNN ǰ��+����"）。把标准输出强制成 UTF-8 就正常了。
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent


def parse_args(desc: str) -> argparse.Namespace:
    """全书统一的命令行参数约定。

    每个章节脚本都带这几个参数，其中 --smoke 用于快速冒烟测试：
    1 个 epoch + 少量样本，几秒钟跑完，用来确认脚本没坏。
    """
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
    """固定随机种子，让每次运行的结果可复现。

    四行各管一件事：
      random.seed      -> Python 标准库的 random
      np.random.seed   -> NumPy
      torch.manual_seed-> PyTorch 的 CPU 随机数
      cuda.manual_seed_all -> PyTorch 在所有 GPU 上的随机数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_really_works() -> tuple[bool, str]:
    """真正验证 GPU 能不能算，而不是只问一句 is_available()。

    这里故意**不用矩阵乘法**做探测 —— 有个真实的坑：cuBLAS 实现的 matmul
    往往有可用的备用路径，即使这张卡的算力不在 PyTorch 的编译覆盖范围内，
    matmul 也可能跑得通。而卷积、ReLU 这些走 PyTorch 自研 kernel 的算子会
    直接失败。用 matmul 探测会漏报。

    所以这里探测的是**本书真正要用的东西**：一个小卷积 + ReLU 的前向。
    """
    try:
        conv = nn.Conv2d(1, 4, 3).cuda()
        x = torch.randn(2, 1, 28, 28, device="cuda")
        y = conv(x)
        y = torch.relu(y)
        # synchronize 必不可少：CUDA 是异步执行的，不同步的话错误可能
        # 根本不会在这里抛出，而是被静默吞掉或推迟到后面某处才爆。
        torch.cuda.synchronize()
        return True, f"卷积前向成功，输出形状 {tuple(y.shape)}"
    except Exception as e:
        return False, str(e).splitlines()[0]


def get_device() -> torch.device:
    """选一个真正可用的设备。

    注意这里的逻辑：`is_available()` 说 True 还不够，必须再跑一次真实算子。
    两者都通过才用 GPU，否则降级到 CPU —— 降级要打印原因，不要静默降级，
    否则你会以为自己在用 GPU 实际却在用 CPU。
    """
    if torch.cuda.is_available():
        ok, msg = cuda_really_works()
        if ok:
            print(f"[设备] CUDA 可用且真实算子通过：{msg}")
            return torch.device("cuda")
        print("[设备] 警告：is_available() 返回 True，但真实算子失败，降级到 CPU。")
        print(f"       失败原因：{msg}")
    else:
        print("[设备] CUDA 不可用，使用 CPU。")
    return torch.device("cpu")


def main() -> None:
    args = parse_args("第 0 章：环境自检")
    set_seed(args.seed)

    print("=" * 62)
    print(" 1. 版本信息")
    print("=" * 62)
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  torchvision: {__import__('torchvision').__version__}")
    print(f"  项目根目录 : {ROOT}")
    print()

    print("=" * 62)
    print(" 2. GPU 可见性（注意：这一节全部通过也不代表 GPU 能用）")
    print("=" * 62)
    avail = torch.cuda.is_available()
    print(f"  cuda.is_available() : {avail}")
    if avail:
        print(f"  设备名              : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  算力 (compute cap)  : {cap[0]}.{cap[1]}  ->  sm_{cap[0]}{cap[1]}")
        print(f"  这套 torch 支持的算力: {torch.cuda.get_arch_list()}")
        print()
        print("  ^ 把上面两行对着看：如果本机算力（sm_xxx）不在支持列表里，")
        print("    那么 GPU 就是不能用的，哪怕 is_available() 是 True。")
    print()

    print("=" * 62)
    print(" 3. GPU 真实可用性（这一节才是结论）")
    print("=" * 62)
    device = get_device()
    print()

    print("=" * 62)
    print(" 4. 随机种子：设了到底有没有用？")
    print("=" * 62)
    set_seed(args.seed)
    a1 = torch.randn(3)
    b1 = np.random.rand(3)
    set_seed(args.seed)
    a2 = torch.randn(3)
    b2 = np.random.rand(3)
    print(f"  两次 set_seed({args.seed}) 后的 torch 随机数是否一致: {torch.equal(a1, a2)}")
    print(f"  两次 set_seed({args.seed}) 后的 numpy 随机数是否一致: {np.allclose(b1, b2)}")
    print()
    print("  种子能保证的是：同一台机器、同一份代码、同样的库版本下，")
    print("  结果可复现。换机器 / 换 PyTorch 版本 / 换了 cuDNN 算法，")
    print("  结果仍可能不同 —— 不要指望它能跨环境对齐。")
    print()

    print("=" * 62)
    print(f" 结论：本次将使用 【{str(device).upper()}】")
    print("=" * 62)
    if device.type == "cpu" and avail:
        print(" 你的 GPU 被报告为可用，但真实算子跑不通 —— 请回到本章正文")
        print("「is_available() 的陷阱」一节，按其中的方法修复后重跑本脚本。")


if __name__ == "__main__":
    main()
