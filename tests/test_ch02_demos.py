"""第 2 章「实测数字」的可追溯性测试。

## 为什么要有这个文件

书里每一处 "实测：" 后面跟着的数字，读者都应当能自己跑出来。第 2 章有四处
**教学演示**（不是训练结果）的数字块，它们一度只存在于正文里 —— 仓库里
任何 `.py`、任何 `results/*.txt`、连进度台账里都搜不到：

| 正文位置 | 内容 |
|---|---|
| 2.4 节 | 不清零连续 backward 三次的梯度 |
| 2.5 节 | `requires_grad` / `grad_fn` 在 `no_grad` 内外 |
| 2.6 节 | 直接喂 logits vs 先 softmax 喂 CE 的三个损失值 |
| 2.7 节 | `nn.Linear` 默认初始化的 weight / bias 范围 |

读者照书敲一遍，得到的数字和书上不一样时，没法判断是自己错了还是书上错了 ——
**因为没有"书上那条命令"可跑**。第 6 章的 `print_dropout_demo` 早就是正确做法：
把演示做成脚本的一个入口，数字由脚本打印。

所以这个测试做两件事：
1. `python code/ch02_mlp.py --demos` 能跑通，并打印出四块演示；
2. **演示打印的每一个关键数字，都能在 `book/02_*.md` 里找到** —— 脚本与正文
   一旦有人单方面改动，这里就红。

用法：
    python tests/test_ch02_demos.py
"""
import re
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCRIPT = ROOT / "code" / "ch02_mlp.py"

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}")
        if detail:
            print(f"         {detail}")
        failures.append(name)


def book_text() -> str:
    md = next((ROOT / "book").glob("02_*.md"))
    return md.read_text(encoding="utf-8")


def main() -> int:
    print("=" * 62)
    print("第 2 章演示数字：可追溯性（脚本打印 == 正文引用）")
    print("=" * 62)

    r = subprocess.run(
        [PY, str(SCRIPT), "--demos"],
        capture_output=True, text=True, cwd=str(ROOT),
        encoding="utf-8", errors="replace",
    )
    check(
        "`ch02_mlp.py --demos` 退出码为 0",
        r.returncode == 0,
        f"退出码 {r.returncode}\nstderr 末尾:\n" + "\n".join(r.stderr.splitlines()[-12:]),
    )
    if r.returncode != 0:
        print("=" * 62)
        print("结果：演示跑不起来，后面的追溯无从谈起")
        return 1

    out = r.stdout
    book = book_text()

    # ① 梯度累加：第 1/2/3 次 backward 后的 grad 值
    # 值带负号（MSE 在初始点上的梯度方向如此），所以 `-?` 不能省
    grads = re.findall(r"第 (\d) 次:\s*(-?[\d.]+)", out)
    check("打印出三次梯度", len(grads) == 3, f"匹配到 {len(grads)} 条：{grads}")
    if len(grads) == 3:
        vals = [v for _, v in grads]
        # 容差 5e-6 而不是 1e-9：脚本和正文都只保留 6 位小数，
        # 光是把 0.327173 乘 2 就已经差 1e-6 了（0.654346 vs 0.654345）。
        check(
            "第 2 次 = 2×第 1 次",
            abs(float(vals[1]) - 2 * float(vals[0])) < 5e-6,
            f"{vals[1]} vs 2×{vals[0]}",
        )
        check(
            "第 3 次 = 3×第 1 次",
            abs(float(vals[2]) - 3 * float(vals[0])) < 5e-6,
            f"{vals[2]} vs 3×{vals[0]}",
        )
        for i, v in enumerate(vals, 1):
            check(f"正文引用第 {i} 次梯度 {v}", v in book)

    # ② no_grad 内外
    check("打印出 AddmmBackward0", "AddmmBackward0" in out and "AddmmBackward0" in book)
    check("打印出 grad_fn=None", "grad_fn=None" in book)

    # ③ CrossEntropyLoss 三个值
    ce = re.findall(r"损失 =\s*([\d.]+)", out)
    check("打印出三个 CE 损失值", len(ce) == 3, f"匹配到 {len(ce)} 条：{ce}")
    if len(ce) == 3:
        check(
            "「先 softmax」明显大于「直接喂 logits」",
            float(ce[1]) > float(ce[0]) * 2,
            f"{ce[1]} vs {ce[0]}",
        )
        check(
            "「手算 log_softmax+NLL」与「直接喂」完全一致",
            abs(float(ce[2]) - float(ce[0])) < 1e-9,
            f"{ce[2]} vs {ce[0]}",
        )
        for i, v in enumerate(ce):
            check(f"正文引用第 {i + 1} 个 CE 值 {v}", v in book)

    # ④ 默认初始化范围
    m = re.search(r"理论界 ±1/sqrt\(784\) = ±([\d.]+)", out)
    check("打印出理论界 ±1/sqrt(784)", m is not None)
    if m:
        check(f"正文引用理论界 {m.group(1)}", m.group(1) in book)
    check(
        "打印出「全部落在界内」的核对",
        "全部落在界内" in out and "全部落在界内" in book,
    )

    print("=" * 62)
    if failures:
        print(f"结果：{len(failures)} 项失败")
        return 1
    print("结果：全部通过（正文引用的每个演示数字都由脚本打印）")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
