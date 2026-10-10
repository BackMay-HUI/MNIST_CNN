"""冒烟测试：对 code/ 下每个章节脚本跑一次 --smoke。

冒烟测试不是要验证模型准不准，而是验证**脚本没坏**：
能不能跑完、退出码是不是 0。训练几分钟才出结果的完整实验不适合每次
改动都跑，但几秒钟的冒烟可以。

除了退出码，还检查第二件事：**跑完之后，书里的正式图有没有被改掉。**
正文引用到的每一张 `figures/xxx.png|txt` 都会先算一遍 md5，全部冒烟跑完
再算一遍 —— 只要有一张变了，就是有脚本把探索性运行的产物写成了正式名字
（第 2 章常见坑 5、第 3 章常见坑 6）。这类覆盖**不报错、不提示**，光看
退出码是发现不了的，所以必须单独查。

**md5 变了才算失败**；"这份产物跑完才第一次出现"只打印成提示 ——
因为第 4 章有两张增强示意图**故意**不加后缀（它们不训练，冒烟和完整跑
出来一模一样），在没跑过 regen_all.sh 的机器上必然"首次出现"。
"漏判"那一种由 tests/test_canonical_guard.py 逐开关确定性地查。

**基线为空 = 不通过，不是通过。** 这条曾经是反的：建基线时那个
`if (ROOT / rel).exists()` 会在全新克隆上把基线清空（`figures/` 是
git-ignored 的，clone 下来没有），于是打印「0 份正式产物 md5 全部未变」
并 exit 0 —— 一份都没比，却说"全部未变"，`--strict` 也拦不住。
**越干净的机器这个测试越安静**，而干净机器正是别人复现这本书的起点。
现在由 `coverage_error()` 明判：引用数 0 或产物数 < 引用数，`--strict` 直接
失败，默认模式也会明确写出"这一条不作数"。守卫在 tests/test_smoke_guard.py。

用法：
    python tests/smoke_test.py             # 只跑已存在的脚本，缺失的列出来
    python tests/smoke_test.py --strict    # 缺失即失败（最终验收用）

为什么有 --strict 这个开关：写书的过程中脚本是一章一章加的，中途必然
有脚本还没写。默认模式让冒烟测试在增量开发期间也能跑；--strict 则用于
全书完成后确认一个都没漏。
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# 全书应当存在的脚本清单
EXPECTED = [
    "code/ch00_env_check.py",
    "code/ch01_data.py",
    "code/ch02_mlp.py",
    "code/ch03_cnn.py",
    "code/ch04_augment.py",
    "code/ch05_hyperparams.py",
    "code/ch06_regularize.py",
    "code/ch07_evaluate.py",
    "code/ch08_project/train.py",
]


def book_figure_refs() -> list:
    """正文引用到的**全部**正式产物路径 —— 不管文件在不在盘上。

    正文里写的是 `figures/xxx.png` 这种相对路径，直接从 markdown 里捞，
    所以书稿加了新图不用来改这个文件。

    和 `book_figure_hashes()` 分开是为了能回答"**引用了几份**"这个问题：
    两者的差就是"书里提到了、但磁盘上还没有"的那几份，见 `coverage_error()`。
    """
    pat = re.compile(r"figures/[A-Za-z0-9_./-]+\.(?:png|txt)")
    refs = set()
    for md in (ROOT / "book").glob("*.md"):
        refs.update(pat.findall(md.read_text(encoding="utf-8")))
    return sorted(refs)


def book_figure_hashes() -> dict:
    """扫出正文引用、且**磁盘上确实存在**的正式产物，返回 {相对路径: md5}。

    注意这个函数**会悄悄丢掉不存在的引用**（`if ... .exists()`）—— 单看它的
    返回值分不清"书里只引用了 3 份"和"引用了 21 份但 18 份没生成"。所以基线
    是不是空的，必须另外用 `coverage_error()` 查，不能拿它的长度当依据。
    """
    return {
        rel: hashlib.md5((ROOT / rel).read_bytes()).hexdigest()
        for rel in book_figure_refs()
        if (ROOT / rel).exists()
    }


def coverage_error(n_refs: int, n_hashed: int) -> Optional[str]:
    # 注解用 typing.Optional 而不是 `str | None`：本机跑测试的解释器是
    # Python 3.9，`X | Y` 写在函数签名里会被**求值**，直接 TypeError。
    """md5 基线覆盖得够不够？够返回 None，不够返回一句人话。

    **这个函数存在的原因是"空过"**：基线为空时，冒烟前后都是空集合，
    比对结果必然是"没有任何产物被改动" —— 测试打印「0 份正式产物 md5 全部未变」
    并 exit 0。而基线为空恰恰是**全新克隆**的常态（`figures/` 是 git-ignored 的），
    也就是别人复现这本书的起点。越是干净的机器，这个测试越安静。

    所以：一份都没比过 = 没通过，而不是通过。
    """
    if n_refs == 0:
        return ("正文里一份 `figures/` 产物都没引用 —— 要么书稿结构变了，"
                "要么上面那条正则失效了。md5 校验无从谈起。")
    if n_hashed < n_refs:
        missing = n_refs - n_hashed
        return (f"正文引用了 {n_refs} 份产物，磁盘上只有 {n_hashed} 份，"
                f"缺 {missing} 份 —— 这 {missing} 份的 md5 根本没比过，"
                f"「全部未变」是空话。先跑 `bash tests/regen_all.sh` 生成。")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="对全部章节脚本跑冒烟")
    ap.add_argument("--strict", action="store_true", help="脚本缺失即判定失败")
    args = ap.parse_args()

    present = [s for s in EXPECTED if (ROOT / s).exists()]
    missing = [s for s in EXPECTED if not (ROOT / s).exists()]

    if missing:
        print(f"[缺失] {len(missing)} 个脚本尚未编写：")
        for m in missing:
            print(f"    {m}")
        if args.strict:
            print("\n--strict 模式下缺失即失败。")
            return 1
        print()

    if not present:
        print("没有可运行的脚本。")
        return 1 if args.strict else 0

    figures_before = book_figure_hashes()
    n_refs = len(book_figure_refs())
    gap = coverage_error(n_refs, len(figures_before))
    if gap:
        print(f"[基线不足] {gap}")
        if args.strict:
            print("\n--strict 模式下，比不了的基线不算通过 —— 不跑了。")
            return 1
        print("（默认模式下继续跑，但这次「正式产物未被改动」这一条**不作数**。）\n")
    else:
        print(f"[基线] 正文引用的 {len(figures_before)} 份正式产物已记录 md5\n")

    failed = []
    for rel in present:
        script = ROOT / rel
        print(f"[运行] {rel} --smoke ...", end=" ", flush=True)
        r = subprocess.run(
            [PY, str(script), "--smoke"],
            capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8",
            errors="replace",
        )
        if r.returncode == 0:
            print("OK")
        else:
            print(f"失败（退出码 {r.returncode}）")
            failed.append((rel, r))

    # 正式产物有没有被冒烟运行改掉 / 冒出来
    figures_after = book_figure_hashes()
    overwritten = sorted(
        rel for rel, h in figures_before.items() if figures_after.get(rel) != h
    )
    appeared = sorted(set(figures_after) - set(figures_before))

    print()
    print("=" * 62)
    if overwritten:
        print("结果：冒烟运行改动了书里的正式产物 —— 冒烟测试不通过")
        for rel in overwritten:
            print(f"  [被覆盖] {rel}")
            print(f"           {figures_before[rel]} -> {figures_after[rel]}")
        print()
        print("原因：某个脚本的 is_canonical 判断漏了开关，或者根本没做判断。")
        print("      详见第 2 章常见坑 5、第 3 章常见坑 6。")
        print("修好之后跑 `bash tests/regen_all.sh` 把正式产物重新生成。")
        print("=" * 62)
        return 1

    if appeared:
        # 只提示，不判失败。因为"冒烟写出了某个正文引用的文件"有两种可能：
        #   ① 脚本漏判，把探索性产物写成了正式名字 —— 这是 bug；
        #   ② 这张图本来就该由冒烟写（比如第 4 章的增强示意图，它不训练，
        #      冒烟和完整跑出来一模一样，所以**故意**不加后缀）。
        # 单看"文件冒出来了"分不清是哪一种 —— 在没跑过 regen_all.sh 的机器上
        # 更是必然出现。①由 tests/test_canonical_guard.py 逐开关确定性地查，
        # 所以这里降级成提示。
        print("提示：本次冒烟之后，下列正文引用的产物才出现（此前不存在）：")
        for rel in appeared:
            print(f"  [新增] {rel}")
        print("      若其中某一项本该是正式产物、由完整命令产出，那就有问题；")
        print("      参照物是 `bash tests/regen_all.sh` 生成的那一份。")
        print()

    if failed:
        print(f"结果：{len(present) - len(failed)}/{len(present)} 通过，{len(failed)} 个失败")
        for rel, r in failed:
            print("-" * 62)
            print(f"### {rel}")
            print("--- stdout 末尾 ---")
            print("\n".join(r.stdout.splitlines()[-25:]))
            print("--- stderr 末尾 ---")
            print("\n".join(r.stderr.splitlines()[-25:]))
        print("=" * 62)
        return 1

    print(f"结果：{len(present)}/{len(present)} 全部通过")
    if gap:
        print("（注意：基线不足，**没有**比对任何正式产物的 md5 —— 见上面的[基线不足]）")
    else:
        print(f"（正文引用的 {len(figures_before)} 份正式产物 md5 全部未变）")
    if missing:
        print(f"（另有 {len(missing)} 个脚本尚未编写，未纳入本次运行）")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
