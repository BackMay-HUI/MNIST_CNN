"""守卫回归测试：除了书里写的那条正式命令，任何跑法都不许写正式文件名。

## 为什么要有这个文件

书里每一章的产物（图、CSV、checkpoint）都用同一个名字：正式运行写原名，
其余一律加后缀。判断"这次算不算正式"的那个函数（`is_canonical`）一旦漏判
某个开关，读者照着书做就会**静默覆盖掉书里引用的那张图** —— 不报错、不提示。

这个坑在项目里复发过四次，每次都是"漏了一个开关"：

| 漏掉的开关 | 哪一章 | 怎么发现的 |
|---|---|---|
| `--epochs` / `--limit` | 第 2 章 | 跑 `--epochs 1 --limit 512` 把正式曲线换掉了 |
| `--level` | 第 4 章 | 加这个参数时忘了同步加进判断 |
| `--depth` | 第 6 章 | 同上 |
| **`--seed`** | **第 2~7 章全部** | **验收时发现：书里第 4/5/6 章的练习 3 就叫读者跑 `--seed 0/1/2`** |

最后那条最严重：**书里的练习本身会把书里的图弄坏**，而且第 2 章正文还专门
写了"任何一个能改变产物的开关都必须进这个判断，漏一个就等于没防" ——
等于书自己犯了自己警告过的错。

所以这条规矩不能只写在正文里靠自觉，得变成可以执行的东西：**每个开关都
当成一个用例列出来**，漏了哪个，这个测试就会红。

用法：
    python tests/test_canonical_guard.py
"""
import inspect
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "ch08_project"))

import ch02_mlp          # noqa: E402
import ch03_cnn          # noqa: E402
import ch04_augment      # noqa: E402
import ch05_hyperparams  # noqa: E402
import ch06_regularize   # noqa: E402
import ch07_evaluate     # noqa: E402


def parse(mod, argv):
    """借脚本自己的 parser 造一个 Namespace（不跑训练）。"""
    saved = sys.argv
    sys.argv = ["script"] + argv
    try:
        # 第 2/3/4 章的 parse_args 要一个描述字符串，其余不要
        if inspect.signature(mod.parse_args).parameters:
            return mod.parse_args("守卫测试")
        return mod.parse_args()
    finally:
        sys.argv = saved


# (标签, 模块, 守卫函数名, 书上那条命令的 argv, 必须判为「非正式」的 argv 列表)
GUARDS = [
    (
        "第 2 章 训练曲线", ch02_mlp, "is_canonical",
        ["--epochs", "10"],
        [["--smoke"], ["--epochs", "1"], ["--epochs", "20"], ["--limit", "512"],
         ["--seed", "0"], ["--seed", "1"], ["--batch-size", "64"], ["--lr", "0.001"],
         ["--optimizer", "adam"]],
    ),
    (
        "第 3 章 训练曲线", ch03_cnn, "is_canonical_curve",
        ["--epochs", "10"],
        [["--smoke"], ["--epochs", "3"], ["--limit", "512"], ["--seed", "0"],
         ["--compare-inits", "--epochs", "5"], ["--init", "default"], ["--init", "zeros"],
         ["--batch-size", "64"], ["--lr", "0.001"]],
    ),
    (
        "第 3 章 初始化对比图", ch03_cnn, "is_canonical_compare",
        ["--compare-inits", "--epochs", "5"],
        [["--smoke", "--compare-inits"], ["--compare-inits", "--epochs", "3"],
         ["--compare-inits", "--epochs", "5", "--seed", "1"],
         ["--compare-inits", "--epochs", "5", "--limit", "512"]],
    ),
    (
        "第 4 章 增强对照", ch04_augment, "is_canonical",
        ["--epochs", "10"],
        [["--smoke"], ["--epochs", "5"], ["--level", "light"], ["--level", "strong"],
         ["--seed", "1"], ["--limit", "512"], ["--batch-size", "64"]],
    ),
    (
        "第 5 章 单次训练（产出 best_cnn.pt）", ch05_hyperparams, "is_canonical",
        ["--epochs", "10"],
        [["--smoke"], ["--epochs", "8"], ["--seed", "1"], ["--limit", "512"],
         ["--batch-size", "64"], ["--lr", "0.1"], ["--momentum", "0"],
         ["--patience", "0"], ["--sweep", "lr"]],
    ),
    (
        "第 5 章 扫参（产出 ch05_sweep.csv）", ch05_hyperparams, "is_canonical",
        ["--epochs", "8", "--sweep", "all"],
        [["--epochs", "8", "--sweep", "lr"], ["--epochs", "8", "--sweep", "batch"],
         ["--epochs", "10", "--sweep", "all"], ["--epochs", "8", "--sweep", "all", "--seed", "1"]],
    ),
    (
        "第 6 章 正则化对照", ch06_regularize, "is_canonical",
        [],
        [["--smoke"], ["--epochs", "5"], ["--seed", "0"], ["--limit", "512"],
         ["--bn"], ["--dropout"], ["--depth", "3"], ["--batch-size", "64"],
         ["--lr", "0.05"], ["--momentum", "0.5"], ["--dropout-p", "0.3"]],
    ),
    (
        "第 7 章 评测产物", ch07_evaluate, "is_canonical",
        [],
        [["--smoke"], ["--epochs", "5"], ["--seed", "1"], ["--limit", "5000"],
         ["--batch-size", "128"], ["--n-errors", "10"], ["--tsne-samples", "1000"],
         ["--ckpt", "results/other.pt"]],
    ),
]


# 第 8 章是工程结构版：判断不写在 `is_canonical(args)` 里，而是
# `Config.is_canonical`（"一个开关都没改" = 和 `Config()` 相等）。
# 所以它没法塞进上面那张表，单独列一组。
# (标签, 改哪个开关)
CH08_CASES = [
    ("--epochs 1", dict(epochs=1)),
    ("--epochs 5", dict(epochs=5)),
    ("--seed 7", dict(seed=7)),
    ("--lr 0.05", dict(lr=0.05)),
    ("--batch-size 64", dict(batch_size=64)),
    ("--momentum 0.5", dict(momentum=0.5)),
    ("--patience 0", dict(patience=0)),
    ("--limit 512", dict(limit=512)),
    ("--smoke", dict(smoke=True)),
]


def check_ch08(failures: list) -> int:
    """第 8 章的 checkpoint（书里 8.4 节引用了它的 md5）有没有被守住。"""
    from config import Config

    checks = 0
    canonical = Config().checkpoint

    checks += 1
    if not Config().is_canonical:
        failures.append("第 8 章 checkpoint：书上那条命令（不带参数）反被判成了非正式运行")

    for label, changed in CH08_CASES:
        checks += 1
        cfg = Config(**changed)
        if cfg.is_canonical:
            failures.append(
                f"第 8 章 checkpoint：`{label}` 会被当成正式运行 —— "
                f"照着跑就会覆盖书里的产物"
            )
        elif cfg.checkpoint == canonical:
            failures.append(f"第 8 章 checkpoint：`{label}` 的产物名和正式的一样：{canonical.name}")
    return checks


def main() -> int:
    failures = []
    checks = 0

    for label, mod, guard_name, canonical_argv, exploratory in GUARDS:
        guard = getattr(mod, guard_name)

        checks += 1
        if not guard(parse(mod, canonical_argv)):
            shown = " ".join(canonical_argv) or "（不带参数）"
            failures.append(f"{label}：书上那条命令 `{shown}` 反被判成了非正式运行")

        for argv in exploratory:
            checks += 1
            if guard(parse(mod, argv)):
                shown = " ".join(argv) or "（不带参数）"
                failures.append(
                    f"{label}：`{shown}` 会被当成正式运行 —— "
                    f"照着跑就会覆盖书里的产物"
                )

    checks += check_ch08(failures)

    print("=" * 66)
    if failures:
        print(f"结果：{len(failures)}/{checks} 个用例不通过")
        print("=" * 66)
        for f in failures:
            print(f"  [FAIL] {f}")
        print()
        print("改法：把漏掉的那个开关加进该章 `is_canonical*` 的判断里。")
        print("判断的写法见第 2 章常见坑 5、第 3 章常见坑 6。")
        return 1

    print(f"结果：{checks}/{checks} 个用例全部通过")
    print("（每条正式命令判为正式；列出的每一种探索性跑法都判为非正式）")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
