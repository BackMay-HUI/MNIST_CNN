#!/usr/bin/env bash
#
# 把 book/ 下的 12 个 .md 合并成**一本**可直接交付的 PDF（带封面、目录、页码）。
#
#     bash tests/build_book_pdf.sh                    # 出书
#     bash tests/build_book_pdf.sh 输出.pdf           # 换输出路径
#     bash tests/build_book_pdf.sh 输出.pdf 截图目录    # 顺带截几张图核对排版
#
# 管线三步：
#
#     book/*.md  --拼接-->  _book_merged.md
#                --pandoc--> _book_merged.html
#                --Edge/CDP-->  *.pdf
#
# 中间两个文件必须落在**仓库根目录**：正文里的图是 `../figures/xxx.png`，
# 从 book/ 出发指向仓库根的 figures/。合并后要把它改写成 `figures/`，
# 而 HTML 里相对路径是相对 HTML 自己算的 —— 所以 HTML 得待在仓库根。
#
# 和 VSCode 插件那条路的关系：插件一次只导一章、没有封面目录页码。
# 这份脚本出的是"整本书"。两边共用同一份 styles/book.css。
#
# 依赖：pandoc、node（要 Node 22+，用到全局 WebSocket）、Edge。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_PDF="${1:-book/PyTorch_MNIST_全书.pdf}"
SHOT_DIR="${2:-}"

MERGED="_book_merged.md"
HTML="_book_merged.html"

# 不管中途哪一步挂了，这两个中间文件都不留在仓库里
cleanup() { rm -f "$MERGED" "$HTML"; }
trap cleanup EXIT

for tool in pandoc node; do
  command -v "$tool" >/dev/null 2>&1 || { echo "缺 $tool，先装它" >&2; exit 1; }
done

if [ ! -d figures ]; then
  echo "警告：没有 figures/ 目录，正文里的图会全部缺失。" >&2
  echo "      先跑 bash tests/regen_all.sh 生成。" >&2
fi

# ---------------------------------------------------------------------------
# 1. 拼接。LC_ALL=C 是为了让排序确定：按字节序，数字开头的 00~08 排在
#    中文的「附录A/B/C」前面，正是想要的顺序。跟着 locale 走的话，
#    zh_CN 下中文可能排到数字前面，章序就乱了。
# ---------------------------------------------------------------------------
echo "[1/3] 拼接 $(LC_ALL=C ls book/*.md | wc -l) 个章节 -> $MERGED"
{
  for f in $(LC_ALL=C ls book/*.md); do
    # 图片路径改写成相对仓库根
    sed 's|\.\./figures/|figures/|g' "$f"
    printf '\n\n'
  done
} > "$MERGED"

# ---------------------------------------------------------------------------
# 2. 渲染 HTML
#
#    --template / --include-before-body 都是必需的，见 styles/pdf-template.html
#    里的注释（默认模板会带一份自己的样式表来和 book.css 抢）。
#
#    --highlight-style=kate：挑它不是因为好看，是**只有它给 import / print /
#    conda 上色**。实测各主题的 Import、BuiltIn、Extension 三类 token：
#
#        tango      ·空·  ·空·  ·空·
#        pygments   ·空·  ·空·  ·空·
#        haddock    ·空·  ·空·  ·空·
#        espresso   深色底，和浅色正文打架
#        kate       #ff5500  #644a9b  #0095ff   ← 唯一可用的浅色主题
#
#    Python 代码里最常见的就是 `import xxx` 和 `print(...)`，shell 片段里是
#    `conda` / `python`。用 tango 的话这些全是黑的，高亮等于只做了一半。
#    kate 的白底由 book.css 的 `div.sourceCode { background: none }` 清掉。
# ---------------------------------------------------------------------------
#    `-implicit_figures`：**这一条必须关掉**。pandoc 默认会把「单独成段的
#    图片」包成 <figure> 并把 alt 文字渲染成 <figcaption>。本书 19 张图全是
#    这种写法，于是每张图下面会多出一行拿 alt 当的说明，和书里自己的图注
#    重复成两行。markdown-it（VSCode 插件那条路）不会这样，关掉两边才一致。
# ---------------------------------------------------------------------------
echo "[2/3] pandoc 渲染 -> $HTML"
pandoc "$MERGED" \
  -f markdown-implicit_figures -t html --standalone \
  --template=styles/pdf-template.html \
  --include-before-body=styles/cover.html \
  --css=styles/book.css \
  --metadata title="PyTorch 实战：MNIST 手写数字识别" \
  --toc --toc-depth=2 \
  -V toc-title=目录 \
  --highlight-style=kate \
  -o "$HTML"

# ---------------------------------------------------------------------------
# 3. 出 PDF
# ---------------------------------------------------------------------------
echo "[3/3] Edge 出 PDF -> $OUT_PDF"
# "${@:3}" 把第 3 个往后的参数原样传给 node（就是截图用的 scrollY 列表）。
# 少了它，调用方写的一串偏移会被 bash 悄悄吃掉，只截出第一张。
node tests/print_pdf.js "$HTML" "$OUT_PDF" $SHOT_DIR "${@:3}"

echo
echo "================================================================"
echo "  完成：$OUT_PDF"
# 页数：PDF 页树里那个 `/Count N`。**取最大值**而不是第一条 ——
# 文档里可能还有别的 /Count（比如书签树），head -1 抓到的未必是页数，
# 之前就是这么报出「8 页」的。
PAGES="$(grep -a -o '/Count [0-9]\+' "$OUT_PDF" | grep -o '[0-9]\+' | sort -n | tail -1 || true)"
echo "  页数：${PAGES:-未知}"
echo "================================================================"
