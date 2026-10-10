"""检查书稿的提示框写法是否合法。

为什么需要这个检查
------------------
本书用三类彩色提示框（⚠️ 常见坑 / 💡 提示 / 📌 结论），靠 CSS 的
`blockquote.warn / .tip / .done` 上色。但 **Markdown 语法没法给引用块加 class**：

    > **💡 提示**          <- 渲染成 <blockquote>，没有 class，加不上样式

所以必须写原始 HTML。而原始 HTML 有个要命的细节 —— 这里的两个坑都是实测出来的
（用 VSCode Markdown PDF 插件自己捆绑的 markdown-it 14.1.1 渲染验证）：

    正确（HTML 块后必须空一行）:          错误（HTML 紧贴内容）:
    <blockquote class="warn">              <blockquote class="warn">
                                           **⚠️ 常见坑**
    **⚠️ 常见坑**                          正文
                                           </blockquote>
    正文

                                           -> 输出字面量 "**⚠️ 常见坑**"

原因：markdown-it 的 html:true 模式下，CommonMark 规定 HTML 块遇到**空行**才结束。
空行之后的内容才会被当成 Markdown 解析。不空行的话，里头全是死文本。

另一种常见错法是用 <div class="warn-box"></div> 当占位符 —— div 是空元素，
既上不了色也不会影响后面的引用块，后面那个普通引用块依然没 class。

本脚本把这三条规则固化成检查，免得 9 章书稿里手写出错。

用法：
    python tests/check_book.py            # 检查 book/ 下所有 .md
    python tests/check_book.py --strict   # book/ 为空也算失败
"""
import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"

VALID_CLASSES = {"warn", "tip", "done"}

# 开标签必须独占一行
OPEN_RE = re.compile(r'^<blockquote\s+class="([^"]*)"\s*>\s*$')
CLOSE_RE = re.compile(r'^</blockquote>\s*$')
# 这些占位 div 是错误写法
PLACEHOLDER_DIV_RE = re.compile(r'^<div\s+class="[^"]*-box"\s*>\s*</div>\s*$')


def check_file(path: Path) -> list[str]:
    """返回该文件的违规列表，每条含行号。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    problems: list[str] = []

    for i, line in enumerate(lines):
        lineno = i + 1

        # 规则 1：不允许用占位 div
        if PLACEHOLDER_DIV_RE.match(line):
            problems.append(
                f"{path.name}:{lineno}: 不许用 `<div class=\"...-box\"></div>` 占位符。"
                f"它不会给后面的引用块加样式，请改成 `<blockquote class=\"warn\">` "
                f"（空一行）内容（空一行）`</blockquote>`"
            )
            continue

        m = OPEN_RE.match(line)
        if not m:
            continue

        # 规则 2：class 名必须是已知的三个之一
        cls = m.group(1)
        if cls not in VALID_CLASSES:
            problems.append(
                f"{path.name}:{lineno}: 未知的 class `{cls}`，"
                f"只能是 {sorted(VALID_CLASSES)} 之一（见 styles/book.css）"
            )

        # 规则 3：开标签后必须紧跟空行，否则内部 Markdown 不会被解析
        if i + 1 >= len(lines) or lines[i + 1].strip() != "":
            problems.append(
                f"{path.name}:{lineno}: `<blockquote class=\"{cls}\">` 后面必须空一行。"
                f"否则内部的 **粗体** 等 Markdown 语法会原样显示成字面量"
            )

        # 规则 4：必须能找到闭标签，且闭标签前必须空一行
        j = i + 1
        while j < len(lines) and not CLOSE_RE.match(lines[j]):
            j += 1
        if j >= len(lines):
            problems.append(f"{path.name}:{lineno}: 找不到配对的 `</blockquote>`")
            continue
        if j == i + 1 or lines[j - 1].strip() != "":
            problems.append(
                f"{path.name}:{j + 1}: `</blockquote>` 前面必须空一行（同上，"
                f"否则最后一段 Markdown 不会被解析）"
            )

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="检查书稿提示框写法")
    ap.add_argument("--strict", action="store_true", help="book/ 为空也算失败")
    args = ap.parse_args()

    if not BOOK.exists():
        print(f"找不到目录 {BOOK}")
        return 1 if args.strict else 0

    files = sorted(BOOK.glob("*.md"))
    if not files:
        print("book/ 下还没有 .md 文件。")
        return 1 if args.strict else 0

    all_problems: list[str] = []
    for f in files:
        all_problems.extend(check_file(f))

    print(f"检查了 {len(files)} 个章节文件：")
    for f in files:
        n = len(check_file(f))
        print(f"  {'OK  ' if n == 0 else 'FAIL'} {f.name}" + (f"  ({n} 处问题)" if n else ""))
    print()

    if all_problems:
        print("=" * 62)
        print(f"发现 {len(all_problems)} 处问题：")
        print("=" * 62)
        for p in all_problems:
            print(f"  - {p}")
        return 1

    print("提示框写法全部合法。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
