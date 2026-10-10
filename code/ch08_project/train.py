"""第 8 章工程版：训练入口 —— 把前四个文件组装起来跑一次。

这才是"工程的入口"该有的样子：**只有编排，没有实现细节**。
读这个文件应该能一口气看完整条流程，不需要理解任何一个函数内部在干什么。

    读配置 -> 选设备 -> 固定随机源 -> 建数据 -> 建模型 -> 建优化器
           -> 训练（内含早停与存 checkpoint）-> 装回最好的权重 -> 测测试集

    （"选设备"在"固定随机源"**前面**，顺序不能换 —— 原因见 main() 里的注释。）

对比一下：第 5 章的 `ch05_hyperparams.py` 有 700 行，里面既有扫参、
又有画图、又有 checkpoint 逻辑。那个文件的存在意义是"教学"，
所以它必须把每件事的来龙去脉都摊开写。工程版把那些搬到别的文件，
入口就瘦下来了 —— **同一个逻辑，两种写法，服务两个不同的目的。**

用法：
    python code/ch08_project/train.py                    # 正式：10 轮
    python code/ch08_project/train.py --smoke            # 冒烟：1 轮 + 少量样本
    python code/ch08_project/train.py --epochs 3 --lr 0.05
"""
import argparse

import torch
import torch.nn as nn

# 同目录 import，**不需要任何 sys.path 技巧**。
#
# 脚本直接执行时，Python 会把「脚本所在目录」放进 sys.path[0]，
# 所以下面这四个 from config/dataset/engine/model 都能直接找到。
# 验证过两种调用方式都能跑：
#     python code/ch08_project/train.py        # 从仓库根
#     cd code/ch08_project && python train.py  # 在自己目录里
#
# 初版这里写了一行 `sys.path.insert(0, str(Path(__file__).resolve().parent))`，
# 注释还振振有词地解释"为什么要加"。**实测把那行删掉，两种方式都还跑得通** ——
# 也就是说那行和它的注释都是多余的。这属于"看起来很专业的废代码"，
# 比明显的 bug 更难发现：删掉它不会报错，留着它也没人质疑。
# 所以这里的选择是**删掉**，而不是留着"以防万一"。
#
# （真的需要动 sys.path 的情形是"用 -m 跑包"或者"从别处 import 这个脚本"，
#   本章都不是 —— 它是一个直接运行的独立程序，所以也不建 __init__.py。）
from config import Config
from dataset import get_loaders
from engine import evaluate, fit, get_device, load_checkpoint, set_seed
from model import build_model, count_params


def parse_args() -> argparse.Namespace:
    """只声明"能改什么"。

    默认值**不在**这里写字面量，而是从 `Config` 的默认实例上取。
    这样"默认是多少"只有 `Config` 一处定义：改 `Config.lr`，
    命令行不传 `--lr` 时的默认值跟着变，不会两边打架。

    （初版这里写的是 `default=10`、`default=0.01` 这样的字面量，
     和 `Config` 里的值重复了一遍 —— 改一处忘另一处就静默不一致。
     注意第 5 章反而没这个问题：它写的是 `default=CANONICAL_LR`，
     引用了常量。**所以这一条不是"工程版比教学版好"，
     是初版写漏了、补上而已。** 别把凡是工程版的都当成更优。）
    """
    d = Config()
    p = argparse.ArgumentParser(description="第 8 章：工程结构版训练入口")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：1 轮 + 少量样本")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--patience", type=int, default=d.patience, help="早停耐心值，0 表示不早停")
    p.add_argument("--limit", type=int, default=d.limit, help="限制训练样本数，0 表示全部")
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.from_args(args)

    # 冒烟的两条约束在这里显式写出来，不藏在别处 —— 读 main 的人一眼能看见
    # "冒烟到底哪里变小了"，而不是去猜。
    if cfg.smoke:
        cfg.epochs = min(cfg.epochs, 1)
        if cfg.limit == 0:
            cfg.limit = 512

    print("=" * 70)
    print("第 8 章：工程结构版")
    print("=" * 70)
    # dataclass 白送的 __repr__：打印出来就是"这次跑的完整配置"
    print(f"[配置] {cfg}")

    # ------------------------------------------------------------------
    # 顺序有讲究：**先探测设备，再固定随机源。**
    # ------------------------------------------------------------------
    # `get_device()` 为了验证 CUDA 真的能算，会跑一次
    # `torch.randn(2, 1, 28, 28)` —— 这**消耗全局随机数流**。
    # 如果先 set_seed 再 get_device，那 set_seed 定下的"起点"就被
    # 这次探测挪走了，后面 kaiming 初始化拿到的是一段**偏移过的**随机数，
    # 权重和"不探测设备时"完全不同。
    #
    # 这不是理论担忧：初版就是这么写的，同一份模型、同一个种子，
    # 实测第 1 轮 val 96.76%（对照第 5 章正式命令的 96.68%），
    # 最终测试集 98.53% —— 而书里应该是 98.62%。差 0.09pp 看着像噪声，
    # 但整条 val 曲线（96.76/97.39/97.97… vs 96.68/97.38/98.07…）
    # 从第 1 轮起就不一样，说明是**初始化变了**，不是随机波动。
    #
    # 规矩：**凡是在"该确定的流程"之前动过随机数流的东西，要么放到 set_seed
    # 前面，要么自己在内部用独立的 generator。** 第 5 章之所以没踩到，
    # 是因为它的 run_trial 开头又 set_seed 了一次，把这次消耗覆盖掉了 ——
    # 那是巧合，不是设计。
    device = get_device()
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader = get_loaders(cfg)
    print(f"[数据] 训练 {len(train_loader.dataset)} / 验证 {len(val_loader.dataset)}"
          f" / 测试 {len(test_loader.dataset)} 张")

    model = build_model(cfg).to(device)
    print(f"[模型] {type(model).__name__}，可训练参数 {count_params(model):,}")

    # 第 5 章的正式配方：SGD + Momentum
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    print(f"[优化] SGD(lr={cfg.lr:g}, momentum={cfg.momentum:g})"
          f"，最多 {cfg.epochs} 轮，patience={cfg.patience}")
    print()

    history, best_val_acc, best_epoch = fit(
        model, (train_loader, val_loader, test_loader),
        optimizer, criterion, device, cfg,
    )

    # ------------------------------------------------------------------
    # 关键一步：**把最好的那份权装修回模型**，再测测试集。
    # ------------------------------------------------------------------
    # 训练到最后一轮时，模型里装的是"最后一轮"的权重，而早停意味着
    # 最后几轮的验证集成绩其实在退步（这正是它触发早停的原因）。
    #
    # 第 5 章的教学脚本没做这一步 —— 它直接评末轮模型，所以打印出的
    # 测试集准确率（98.80%）和 `best_cnn.pt` 本身能跑出的成绩（98.62%）
    # 差了 0.18pp。那个"差距"本身就是教材：**你以为你交付的是那个模型，
    # 其实不是。**
    #
    # 工程版不存在这个歧义：checkpoint 存的就是 best，评之前装回来，
    # 评的就是交付的那个模型。
    load_checkpoint(model, cfg.checkpoint, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print()
    print("=" * 70)
    print(f"  验证集最好 {best_val_acc:.2%}（第 {best_epoch} 轮）"
          f" | 测试集 {test_acc:.2%}（损失 {test_loss:.4f}）")
    print(f"  checkpoint: {cfg.checkpoint}")
    print("=" * 70)

    # 一句话解释为什么最后几行和验证集最好值往往对不上 ——
    # 这是被问得最多的问题，直接印出来省得读者怀疑代码写错了。
    if best_epoch < cfg.epochs:
        print(f"  注：验证集最好在第 {best_epoch} 轮、测试集是另一个数 —— 两者不看同一批数据，"
              f"本来就不该相等（第 5 章 §5.6）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
