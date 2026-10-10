"""冒烟测试自身的守卫：**基线为空时不许"空过"**。

## 为什么要有这个文件

`smoke_test.py` 干两件事：跑 `--smoke` 看退出码，以及**比对正文引用产物的 md5**
—— 后者才是它能抓到"某个脚本把探索性产物写成了正式名字"的唯一手段。

问题在于它的 md5 基线是这么建的：

    refs = 从 book/*.md 里正则捞出来的 figures/xxx.png
    return {rel: md5(rel) for rel in refs if (ROOT / rel).exists()}

**末尾那个 `if ... .exists()` 会在全新克隆上把基线清空** —— `figures/` 是
git-ignored 的，刚 clone 下来根本不存在。于是：

| | 基线大小 | 冒烟跑完 | 结论 |
|---|---|---|---|
| 验收机 | 21 | 21 | 「21 份正式产物 md5 全部未变」✓ |
| 全新克隆 | **0** | **0** | **「0 份正式产物 md5 全部未变」✓** ← 空过 |

第二行的 ✓ 是假的：一份产物都没比，却打印了"全部未变"，`--strict` 下照样
exit 0。**越是干净的机器，这个测试越安静** —— 而干净机器恰恰是别人复现这本书
时的起点，也正是最需要它别撒谎的时候。

所以"基线是不是空的"必须自己去查，不能靠 `.exists()` 悄悄兜住。

用法：
    python tests/test_smoke_guard.py
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import smoke_test  # noqa: E402

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}")
        if detail:
            print(f"         {detail}")
        failures.append(name)


def main() -> int:
    print("=" * 62)
    print("冒烟测试守卫：基线为空必须判失败，不许「0 份产物全部未变」")
    print("=" * 62)

    # ① 正文一份产物都没引用 —— 正则或书稿结构坏了，md5 校验无从谈起
    check(
        "引用数为 0 时报错",
        smoke_test.coverage_error(0, 0) is not None,
        "coverage_error(0, 0) 返回了 None —— 空基线会被当成通过",
    )

    # ② 正文引用了 21 份，磁盘上一份都没有 —— 就是全新克隆的样子
    err = smoke_test.coverage_error(21, 0)
    check(
        "有引用、零产物时报错（全新克隆）",
        err is not None,
        "coverage_error(21, 0) 返回了 None —— 「0 份产物 md5 全部未变」会空过",
    )
    check(
        "错误信息里点名 regen_all.sh",
        err is not None and "regen_all" in err,
        f"错误信息没给出补救命令：{err!r}",
    )

    # ③ 少一份也是漏 —— 那一份的 md5 根本没比过
    err = smoke_test.coverage_error(21, 20)
    check(
        "缺 1 份也报错并点名",
        err is not None and "1" in err,
        f"coverage_error(21, 20) 没报出缺的那一份：{err!r}",
    )

    # ④ 21/21 全在 —— 这才是验收机上的正常状态，不许报错
    check(
        "21/21 齐全时放行",
        smoke_test.coverage_error(21, 21) is None,
        f"齐全时误报：{smoke_test.coverage_error(21, 21)!r}",
    )

    # ⑤ 真实基线现状：引用数与磁盘上的产物数必须相等
    refs = smoke_test.book_figure_refs()
    hashed = smoke_test.book_figure_hashes()
    check(
        f"当前书稿：引用 {len(refs)} 份，磁盘 {len(hashed)} 份，相等",
        len(refs) == len(hashed) and len(refs) > 0,
        f"引用 {len(refs)} 份但只有 {len(hashed)} 份在盘上；"
        f"缺：{sorted(set(refs) - set(hashed))}",
    )
    check(
        "当前书稿通过 coverage_error（验收机基线非空）",
        smoke_test.coverage_error(len(refs), len(hashed)) is None,
    )

    # ⑥ 同一个毛病也长在 `regen_all.sh` 上，一起钉住。
    #    它的每一步都是 `python ... | tee 日志`，而**管道的退出码取的是 tee 的** ——
    #    没有 `set -o pipefail`，python 崩了 tee 照样返回 0，脚本会打印
    #    "全部重生成完毕，无失败"并 exit 0，验收时拿去对的数字是上一轮的旧内容。
    #    `set -u` 管不了这件事，必须是 `set -o pipefail`。
    regen = (ROOT / "tests" / "regen_all.sh").read_text(encoding="utf-8")
    check(
        "regen_all.sh 设了 set -o pipefail",
        "set -o pipefail" in regen,
        "没有 pipefail：任一步骤崩掉都会被 tee 的退出码吞掉，脚本仍报「无失败」",
    )

    print("=" * 62)
    if failures:
        print(f"结果：{len(failures)} 项失败")
        return 1
    print("结果：全部通过")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
