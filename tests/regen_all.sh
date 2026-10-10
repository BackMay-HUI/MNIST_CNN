#!/usr/bin/env bash
# 用「书里写的那条命令」重新生成全书所有被引用的产物（图 + 日志）。
#
# 为什么需要这个脚本：开发期间跑过大量探索性命令（--smoke / --epochs 1 /
# --limit 512 …）。第 2、3 章初版只按 --smoke 加后缀，这类运行会把书里的
# 正式图**静默覆盖**掉。虽然现在已经修好（is_canonical / tagged），
# 但**已经被覆盖的那几张图不会自己变回来** —— 必须重跑正式命令。
#
# PDF 导出前跑一遍这个脚本，保证每一张图都是「书上那条命令」的产物。
#
# 注意：`results/*_full.txt` 这类日志**不是脚本自己写的**，而是把脚本的
# stdout 重定向存下来的（各章正文里的命令后面都跟着 `| tee results/xxx.txt`）。
# 所以这里也必须 tee，否则重跑只刷新了图、日志还停在旧版本。
#
# 用法：  bash tests/regen_all.sh 2>&1 | tee results/_regen_log.txt
#         （脚本内部只把每一步单独存成 results/<名字>.txt；要留总日志得自己 tee）
#
# set -o pipefail 不能省：下面每步都是 `python ... | tee 日志`，而**管道的退出码
# 取的是最后一个命令（tee）的** —— 不设它，python 崩了 tee 照样返回 0，
# 脚本会打印"全部重生成完毕，无失败"并 exit 0。验收时拿这种日志去对数字，
# 对着的是上一轮的旧内容。
set -u
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY="/c/Users/ASUS/.conda/envs/pytorch/python.exe"
export PYTHONIOENCODING=utf-8

FAILED=0

# run <日志文件名|-> <说明> -- <命令...>
run() {
    local log="$1"; shift
    local desc="$1"; shift
    [ "$1" = "--" ] && shift

    echo "================================================================"
    echo ">>> $desc"
    echo ">>> \$ $*"
    echo "================================================================"

    if [ "$log" = "-" ]; then
        "$@" 2>&1 || { echo "!!!! 失败：$desc"; FAILED=1; }
    else
        mkdir -p results
        { "$@" ; } 2>&1 | tee "results/$log" || { echo "!!!! 失败：$desc"; FAILED=1; }
    fi
    echo
}

# --- 第 1 章：样本网格（不训练） ---
run ch01_stats_full.txt "第 1 章 数据加载" -- \
    $PY code/ch01_data.py

# --- 第 2 章：10 轮全连接 -> figures/ch02_curve.png ---
run ch02_full.txt "第 2 章 全连接网络（--epochs 10）" -- \
    $PY code/ch02_mlp.py --epochs 10

# --- 第 3 章：两条正式命令 ---
run ch03_full.txt "第 3 章 CNN（--epochs 10）" -- \
    $PY code/ch03_cnn.py --epochs 10
run ch03_init_compare.txt "第 3 章 初始化对比（--compare-inits --epochs 5）" -- \
    $PY code/ch03_cnn.py --compare-inits --epochs 5

# --- 第 4 章：三档增强 -> ch04_compare.png/csv + 示意图两张 ---
run ch04_full.txt "第 4 章 数据增强（--epochs 10，三档全跑）" -- \
    $PY code/ch04_augment.py --epochs 10

# --- 第 5 章：先扫参、再单次训练 ---
# 顺序有讲究：单次训练那条才写 results/best_cnn.pt，放最后跑，
# 保证最终留在磁盘上的是「10 轮」那份。
run ch05_sweep_log.txt "第 5 章 扫参（--epochs 8 --sweep all）" -- \
    $PY code/ch05_hyperparams.py --epochs 8 --sweep all
run ch05_full.txt "第 5 章 正式训练（--epochs 10）" -- \
    $PY code/ch05_hyperparams.py --epochs 10

# --- 第 6 章：四组全跑 -> figures/ch06_gap.png ---
run ch06_full.txt "第 6 章 正则化（--epochs 10）" -- \
    $PY code/ch06_regularize.py --epochs 10

# --- 第 7 章：加载第 5 章的 best_cnn.pt 做误差分析 -> 六张图 ---
run ch07_eval.txt "第 7 章 性能测试（默认）" -- \
    $PY code/ch07_evaluate.py

# --- 第 8 章：工程版入口 ---
run ch08_train.txt "第 8 章 工程结构版" -- \
    $PY code/ch08_project/train.py

echo "================================================================"
if [ "$FAILED" = 0 ]; then
    echo "全部重生成完毕，无失败。"
else
    echo "有步骤失败 —— 往上翻找 '!!!!' 那几行。"
fi
echo "================================================================"
exit "$FAILED"
