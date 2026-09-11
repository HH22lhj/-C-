# -*- coding: utf-8 -*-
"""补生成 result2.xlsx 提交表格（不重跑 DP，直接读已有 CSV）。"""
import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(
    r"C:\Users\LENOVO\Desktop\数模\C题\代码\solve_problem2 - 副本.py"
)
OUT_DIR = Path(r"C:\Users\LENOVO\Desktop\数模\C题\outputs\problem2")


def load_module():
    spec = importlib.util.spec_from_file_location("p2_copy", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    detail = pd.read_csv(OUT_DIR / "problem2_detail.csv")
    daily_summary = pd.read_csv(OUT_DIR / "problem2_daily_summary.csv")

    n_slots = int(detail["slot_index"].max()) + 1
    time_labels = list(range(n_slots))

    print(f"时段数={n_slots}, 明细={len(detail)} 行, 汇总={len(daily_summary)} 天")

    mod = load_module()
    submission_path = mod.generate_submission_tables(
        detail=detail,
        daily_summary=daily_summary,
        time_labels=time_labels,
        output_dir=OUT_DIR,
    )
    print("已生成:", submission_path)


if __name__ == "__main__":
    main()
