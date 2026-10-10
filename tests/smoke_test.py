"""冒烟测试：对 code/ 下每个章节脚本跑一次 --smoke。

冒烟测试不是要验证模型准不准，而是验证**脚本没坏**：
能不能跑完、退出码是不是 0、该产出的文件有没有产出。
训练几分钟才出结果的完整实验不适合每次改动都跑，但几秒钟的冒烟可以。

用法：
    python tests/smoke_test.py             # 只跑已存在的脚本，缺失的列出来
    python tests/smoke_test.py --strict    # 缺失即失败（最终验收用）

为什么有 --strict 这个开关：写书的过程中脚本是一章一章加的，中途必然
有脚本还没写。默认模式让冒烟测试在增量开发期间也能跑；--strict 则用于
全书完成后确认一个都没漏。
"""
import argparse
import subprocess
import sys
from pathlib import Path

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

    print()
    print("=" * 62)
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
    if missing:
        print(f"（另有 {len(missing)} 个脚本尚未编写，未纳入本次运行）")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
