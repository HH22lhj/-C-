"""
DP 的 SOC 离散步长（--soc-step）在 25-50 区间扫描，
找出使问题二正式评价期总费用最低的步长。

用法：
    python sweep_soc_step.py

默认对 25, 30, 35, 40, 45, 50 六个步长做全量评价，
每个步长用独立输出子目录 problem2_same_type_sweep_{step}，
结束后把每个步长的总费用写入 sweep_soc_step_results.csv，
并打印最低费用对应的步长。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"C:/Users/LENOVO/Desktop/数模/C题")
CODE_DIR = Path(r"C:/Users/LENOVO/Desktop/数模/C题/代码")
SCRIPT = CODE_DIR / "solve_problem2_同类型分解预测.py"
OUTPUT_ROOT = BASE_DIR / "outputs"
RESULTS_CSV = CODE_DIR / "sweep_soc_step_results_near25.csv"

# 扫描的步长集合（可自行修改；命令行传入时会被覆盖）
STEPS = [25, 30, 35, 40, 45, 50]


def total_cost_from_summary(output_subdir: str) -> float:
    """从某次运行的每日汇总表读取正式评价期总费用。"""
    summary_path = (
        OUTPUT_ROOT / output_subdir / "problem2_daily_summary.csv"
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到汇总表：{summary_path}")

    daily = pd.read_csv(summary_path)
    return float(daily["total_cost_yuan"].sum())


def run_one_step(step: float) -> dict:
    """运行一个步长并返回其总费用。"""
    output_subdir = f"problem2_same_type_sweep_{step:g}"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--soc-step",
        f"{step:g}",
        "--output-subdir",
        output_subdir,
    ]
    print(f"\n===== 开始运行 soc-step={step:g} =====", flush=True)
    result = subprocess.run(cmd, cwd=str(CODE_DIR))
    if result.returncode != 0:
        raise RuntimeError(
            f"soc-step={step:g} 运行失败，返回码 {result.returncode}"
        )

    cost = total_cost_from_summary(output_subdir)
    print(f"===== soc-step={step:g} 总费用 = {cost:.2f} 元 =====", flush=True)
    return {"soc_step": step, "total_cost_yuan": cost}


def main() -> None:
    # 允许从命令行传入步长：python sweep_soc_step.py 22.5 27.5
    if len(sys.argv) > 1:
        steps = [float(arg) for arg in sys.argv[1:]]
    else:
        steps = [float(step) for step in STEPS]

    results = []

    for step in steps:
        row = run_one_step(step)
        results.append(row)

        # 每次运行后立即落盘，方便中途查看进度。
        pd.DataFrame(results).to_csv(
            RESULTS_CSV,
            index=False,
            encoding="utf-8-sig",
        )

    df = pd.DataFrame(results)
    best = df.loc[df["total_cost_yuan"].idxmin()]

    print("\n" + "=" * 50)
    print("扫描结果汇总（SOC 离散步长 -> 正式评价期总费用）")
    print("=" * 50)
    for _, row in df.iterrows():
        marker = "  <-- 最低" if row["soc_step"] == best["soc_step"] else ""
        print(
            f"步长 {row['soc_step']:g} kWh : "
            f"{row['total_cost_yuan']:,.2f} 元{marker}"
        )
    print("-" * 50)
    print(
        f"最低费用点：SOC 步长 = {best['soc_step']:g} kWh，"
        f"总费用 = {best['total_cost_yuan']:,.2f} 元"
    )
    print(f"结果已保存至：{RESULTS_CSV}")


if __name__ == "__main__":
    main()
