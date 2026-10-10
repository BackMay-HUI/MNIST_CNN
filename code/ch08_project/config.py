"""第 8 章工程版：超参数与路径的集中管理。

前七章每个脚本都自己写一遍 `lr = 0.01`、`batch_size = 128`，
散在各处。想改一个值时，得先记住它出现在哪几个文件里。

工程做法是**把"可调的东西"全部收到一个地方**：别的文件不写死任何数字，
一律 import 这个文件里的 `Config`。这样：

  - 改超参 = 改一个文件（或者干脆不改代码，从命令行传）
  - 想知道"这次跑的是什么配置" = 看一个对象
  - 想把配置存进日志/传给别的进程 = 一个 dataclass 就能序列化

注意 `Config` 里**没有**模型结构和损失函数 —— 那些是"代码"，不是"配置"。
能被数字表达、且换个值不影响逻辑的，才叫超参。
"""
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 本文件在 code/ch08_project/config.py，所以：
#     __file__.parent                    -> code/ch08_project
#     __file__.parent.parent             -> code
#     __file__.parent.parent.parent      -> 仓库根
#
# 第 1~7 章的脚本都直接放在 code/ 下，用两层 parent 就够；本章的工程版
# **深了一层，必须三层**。
#
# 这是全项目唯一一处路径约定的例外。处理办法不是"让每个文件都记住要数几层"
# —— 那样迟早有人数错 —— 而是**只在这里数一次**，别的文件一律
# `from config import ROOT`。所以你会看到 model.py / dataset.py / engine.py
# 里一次 `__file__` 都没出现。
ROOT = Path(__file__).resolve().parent.parent.parent

DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
N_VAL = 10000


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """一次训练需要的全部可调项。

    dataclass 的好处是白送你 __init__ / __repr__：打印出来就是"这次跑的配置"，
    不用自己写一段格式化代码去拼字符串。
    """

    # --- 数据 ---
    batch_size: int = 128
    limit: int = 0                 # >0 时只取训练集前 N 张（冒烟用）

    # --- 优化 ---
    lr: float = 0.01               # SGD + Momentum 的配方，和第 5 章正式命令一致
    momentum: float = 0.9
    epochs: int = 10
    patience: int = 3              # 连续几轮验证集没进步就早停

    # --- 运行 ---
    seed: int = 42
    smoke: bool = False

    # --- 路径：由上面的常量派生，不从命令行传 ---
    data_dir: Path = DATA
    results_dir: Path = RESULTS

    @property
    def is_canonical(self) -> bool:
        """这次跑的是不是**书上那条命令** —— 也就是"一个开关都没改"。

            书上那条命令：python code/ch08_project/train.py

        为什么判断条件能简单成一行：dataclass 白送 `__eq__`，
        "全是默认值"就等于"和 `Config()` 相等"。而 `Config()` 的默认值
        是**唯一的**那一份（连 argparse 的默认值都从这里取），
        所以**往 `Config` 里加一个新的超参，这个判断自动跟着变准**，
        不存在"忘了把它加进判断里"这回事。

        这一点值得说清楚：第 2~7 章每章都手写一个 `is_canonical`，
        把"哪些开关算数"一项一项列出来。那个写法翻过四次车 ——
        第 2 章漏 `--epochs`/`--limit`，第 4 章漏 `--level`，第 6 章漏 `--depth`，
        验收时又发现每一章都漏了 `--seed`。**不是谁不小心，是那种写法本身
        要求人每加一个参数就记得回去补一行。** 本章换成"和默认值比"，
        把这件事从"记得"变成了"不可能忘"。
        """
        return self == Config()

    @property
    def checkpoint(self) -> Path:
        """这个工程的产物 checkpoint。

        **注意它不叫 best_cnn.pt。**

        第 5 章的正式命令会写 `results/best_cnn.pt`，第 7 章又去**加载**它。
        如果本章也往同一个文件写，这个文件就有了两个写入者 ——
        以后看到它时说不清"这是哪条命令、哪份代码产出的"
        （第 5 章 Ruling 34 就是这么栽的）。

        所以本章的产物叫 `ch08_best.pt`，与第 5 章并存、互不干扰。

        非正式运行（改过任何开关）一律另起名字，绝不碰 `ch08_best.pt` ——
        书里 8.4 节引用了它的 md5（`ee167b40…`），被换掉就说不清了。
        后缀里的 `e{epochs}_s{seed}` 只是个"能认出来"的标记，不是配置摘要；
        这次跑的**完整**配置在 `train.py` 开头那行 `[配置] {cfg}` 里。
        """
        if self.smoke:
            return self.results_dir / "ch08_best_smoke.pt"
        if self.is_canonical:
            return self.results_dir / "ch08_best.pt"
        return self.results_dir / f"ch08_best_e{self.epochs}_s{self.seed}.pt"

    @classmethod
    def from_args(cls, args) -> "Config":
        """把命令行参数并进来。命令行没给的项，用 dataclass 的默认值。

        这是"配置"相对"散落的变量"的第二个好处：**默认值只有一份**。
        第 5 章的脚本里 `--lr` 的默认值写在 argparse 的 add_argument 上，
        而"正式命令用的是 0.01"这件事写在另一个常量 CANONICAL_LR 上 ——
        两处要保持一致，靠的是人记得。这里默认值只在 dataclass 里写一次。
        """
        return cls(
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            epochs=args.epochs,
            patience=args.patience,
            limit=args.limit,
            seed=args.seed,
            smoke=args.smoke,
        )
