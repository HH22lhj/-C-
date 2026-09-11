# -*- coding: utf-8 -*-
"""
问题1：单日确定性调度
LP 基准模型 + DP 对比模型

数据：
附件1.xlsx
每 10 分钟一个时段，共 144 个时段。

单位约定：
小区负载、光伏原始数据为 kW；
每时段能量 = 功率 / 6，单位 kWh。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog


# =========================
# 1. 基本参数
# =========================

DATA_PATH = Path(r"C:\Users\LENOVO\Desktop\数模\C题\附件\附件1.xlsx")
OUT_DIR = Path(r"C:\Users\LENOVO\Desktop\数模\C题\analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 144
DELTA_H = 1 / 6

E_CAP = 12000.0
SOC_MIN = 0.10 * E_CAP
SOC_MAX = 0.90 * E_CAP
SOC_INIT = 0.50 * E_CAP
SOC_FINAL = 0.50 * E_CAP

P_CH_MAX = 5000.0
P_DIS_MAX = 5000.0
C_MAX = P_CH_MAX * DELTA_H
D_MAX = P_DIS_MAX * DELTA_H

ETA_C = 0.9
ETA_D = 0.9


# =========================
# 2. 读取并整理数据
# =========================

def find_col(columns, keywords):
    for col in columns:
        name = str(col)
        if all(k in name for k in keywords):
            return col
    raise ValueError(f"没有找到包含关键词 {keywords} 的列，请检查表头：{columns}")


def read_data(path):
    df = pd.read_excel(path)

    time_col = find_col(df.columns, ["时间"])
    price_col = find_col(df.columns, ["电价"])
    load_col = find_col(df.columns, ["负载"])
    pv_col = find_col(df.columns, ["光伏"])

    df = df[[time_col, price_col, load_col, pv_col]].copy()
    df.columns = ["time", "price", "load_kw", "pv_kw"]

    df = df.dropna().reset_index(drop=True)

    if len(df) != T:
        raise ValueError(f"附件1应有 {T} 个时段，但当前读取到 {len(df)} 行")

    df["load_kwh"] = df["load_kw"] * DELTA_H
    df["pv_kwh"] = df["pv_kw"] * DELTA_H

    return df


# =========================
# 3. LP 模型
# =========================

def solve_lp(df):
    price = df["price"].to_numpy(dtype=float)
    load = df["load_kwh"].to_numpy(dtype=float)
    pv = df["pv_kwh"].to_numpy(dtype=float)

    # 变量顺序：
    # x = [g_1...g_T, c_1...c_T, d_1...d_T, s_1...s_T, u_1...u_T]
    n_var = 5 * T

    def idx_g(t): return t
    def idx_c(t): return T + t
    def idx_d(t): return 2 * T + t
    def idx_s(t): return 3 * T + t
    def idx_u(t): return 4 * T + t

    # 目标函数：min sum p_t g_t
    obj_cost = np.zeros(n_var)
    for t in range(T):
        obj_cost[idx_g(t)] = price[t]

    A_eq = []
    b_eq = []

    # 功率平衡：
    # g_t + v_t + d_t = l_t + c_t + u_t
    # 等价于 g_t - c_t + d_t - u_t = l_t - v_t
    for t in range(T):
        row = np.zeros(n_var)
        row[idx_g(t)] = 1
        row[idx_c(t)] = -1
        row[idx_d(t)] = 1
        row[idx_u(t)] = -1
        A_eq.append(row)
        b_eq.append(load[t] - pv[t])

    # SOC 递推：
    # s_t = s_{t-1} + eta_c c_t - d_t / eta_d
    for t in range(T):
        row = np.zeros(n_var)
        row[idx_s(t)] = 1
        row[idx_c(t)] = -ETA_C
        row[idx_d(t)] = 1 / ETA_D

        if t == 0:
            b_eq.append(SOC_INIT)
        else:
            row[idx_s(t - 1)] = -1
            b_eq.append(0)

        A_eq.append(row)

    # 终端 SOC：
    # s_T = SOC_FINAL
    row = np.zeros(n_var)
    row[idx_s(T - 1)] = 1
    A_eq.append(row)
    b_eq.append(SOC_FINAL)

    bounds = []

    # g_t >= 0
    for _ in range(T):
        bounds.append((0, None))

    # 0 <= c_t <= C_MAX
    for _ in range(T):
        bounds.append((0, C_MAX))

    # 0 <= d_t <= D_MAX
    for _ in range(T):
        bounds.append((0, D_MAX))

    # SOC_MIN <= s_t <= SOC_MAX
    for _ in range(T):
        bounds.append((SOC_MIN, SOC_MAX))

    # u_t >= 0
    for _ in range(T):
        bounds.append((0, None))

    res1 = linprog(
        c=obj_cost,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs"
    )

    if not res1.success:
        raise RuntimeError("LP 第一阶段求解失败：" + res1.message)

    best_cost = res1.fun

    # 第二阶段：在购电费用最优的前提下，最小化充放电总量，避免无意义同时充放电
    obj_cycle = np.zeros(n_var)
    for t in range(T):
        obj_cycle[idx_c(t)] = 1
        obj_cycle[idx_d(t)] = 1

    A_ub = [obj_cost]
    b_ub = [best_cost + 1e-6]

    res2 = linprog(
        c=obj_cycle,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs"
    )

    if not res2.success:
        raise RuntimeError("LP 第二阶段求解失败：" + res2.message)

    x = res2.x

    result = df.copy()
    result["g_kwh"] = x[0:T]
    result["c_kwh"] = x[T:2*T]
    result["d_kwh"] = x[2*T:3*T]
    result["soc_kwh"] = x[3*T:4*T]
    result["u_kwh"] = x[4*T:5*T]

    result["grid_kw"] = result["g_kwh"] / DELTA_H
    result["charge_kw"] = result["c_kwh"] / DELTA_H
    result["discharge_kw"] = result["d_kwh"] / DELTA_H
    result["unused_pv_kw"] = result["u_kwh"] / DELTA_H

    true_cost = float(np.sum(result["price"] * result["g_kwh"]))

    return result, true_cost


# =========================
# 4. DP 对比模型
# =========================

def solve_dp(df, soc_step=100.0):
    price = df["price"].to_numpy(dtype=float)
    load = df["load_kwh"].to_numpy(dtype=float)
    pv = df["pv_kwh"].to_numpy(dtype=float)
    net_load = load - pv

    soc_grid = np.arange(SOC_MIN, SOC_MAX + soc_step, soc_step)

    if SOC_INIT not in soc_grid:
        soc_grid = np.sort(np.append(soc_grid, SOC_INIT))
    if SOC_FINAL not in soc_grid:
        soc_grid = np.sort(np.append(soc_grid, SOC_FINAL))

    N = len(soc_grid)

    init_idx = int(np.where(np.isclose(soc_grid, SOC_INIT))[0][0])
    final_idx = int(np.where(np.isclose(soc_grid, SOC_FINAL))[0][0])

    V_next = np.full(N, np.inf)
    V_next[final_idx] = 0.0

    policy = np.full((T, N), -1, dtype=int)

    S_now = soc_grid.reshape(-1, 1)
    S_next = soc_grid.reshape(1, -1)
    X = S_next - S_now

    feasible_transition = (
        (X >= -D_MAX / ETA_D - 1e-9) &
        (X <= ETA_C * C_MAX + 1e-9)
    )

    psi = np.where(X >= 0, X / ETA_C, ETA_D * X)

    for t in range(T - 1, -1, -1):
        grid_purchase = np.maximum(net_load[t] + psi, 0)
        stage_cost = price[t] * grid_purchase

        total_cost = np.where(
            feasible_transition,
            stage_cost + V_next.reshape(1, -1),
            np.inf
        )

        V = np.min(total_cost, axis=1)
        policy[t, :] = np.argmin(total_cost, axis=1)

        V_next = V

    dp_cost = float(V_next[init_idx])

    soc = SOC_INIT
    rows = []

    state_idx = init_idx

    for t in range(T):
        next_idx = policy[t, state_idx]
        next_soc = soc_grid[next_idx]
        x = next_soc - soc

        if x >= 0:
            c = x / ETA_C
            d = 0.0
        else:
            c = 0.0
            d = -ETA_D * x

        g = max(net_load[t] + (x / ETA_C if x >= 0 else ETA_D * x), 0)
        u = max(-net_load[t] - (x / ETA_C if x >= 0 else ETA_D * x), 0)

        rows.append({
            "time": df.loc[t, "time"],
            "price": price[t],
            "load_kw": df.loc[t, "load_kw"],
            "pv_kw": df.loc[t, "pv_kw"],
            "load_kwh": load[t],
            "pv_kwh": pv[t],
            "g_kwh": g,
            "c_kwh": c,
            "d_kwh": d,
            "soc_kwh": next_soc,
            "u_kwh": u,
            "grid_kw": g / DELTA_H,
            "charge_kw": c / DELTA_H,
            "discharge_kw": d / DELTA_H,
            "unused_pv_kw": u / DELTA_H
        })

        soc = next_soc
        state_idx = next_idx

    result = pd.DataFrame(rows)

    return result, dp_cost


# =========================
# 5. 主程序
# =========================

def main():
    df = read_data(DATA_PATH)

    lp_result, lp_cost = solve_lp(df)
    dp_result, dp_cost = solve_dp(df, soc_step=10)

    output_xlsx = OUT_DIR / "问题1_LP与DP调度结果.xlsx"
    output_summary = OUT_DIR / "问题1_求解结果汇总.txt"

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="原始数据整理", index=False)
        lp_result.to_excel(writer, sheet_name="LP最优调度", index=False)
        dp_result.to_excel(writer, sheet_name="DP对比调度", index=False)

    with open(output_summary, "w", encoding="utf-8") as f:
        f.write("问题1：单日确定性调度求解结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"LP 最优购电成本：{lp_cost:.4f}\n")
        f.write(f"DP 对比购电成本：{dp_cost:.4f}\n")
        f.write(f"DP 使用 SOC 离散步长：10 kWh\n")
        f.write("\n")
        f.write("说明：\n")
        f.write("1. LP 是本文推荐的基准模型，连续变量直接求全局最优。\n")
        f.write("2. DP 用于对比验证，SOC 做了离散化，因此结果可能与 LP 存在小幅差异。\n")
        f.write("3. 若希望 DP 更接近 LP，可将 soc_step 改为 50、20 或 10，但运行时间会增加。\n")

    print("求解完成")
    print(f"LP 最优购电成本：{lp_cost:.4f}")
    print(f"DP 对比购电成本：{dp_cost:.4f}")
    print(f"结果文件：{output_xlsx}")
    print(f"汇总文件：{output_summary}")


if __name__ == "__main__":
    main()

