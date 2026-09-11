"""
问题二：日前购电计划 + 日内因果 DP 执行（跨日连续版本）

关键修正：
1. 日前LP不再强制日末SOC等于日初SOC
2. 日前目标函数和日内DP均加入日末库存价值
3. 相邻日期的SOC自动连续传递

正式评价区间：
2025-02-01 至 2025-12-31

运行示例：
python solve_problem2.py
python solve_problem2.py --max-days 7
python solve_problem2.py --candidate-quantiles 0.6 0.75 0.9
python solve_problem2.py --soc-step 500

输出目录：
C:/Users/LENOVO/Desktop/数模/C题/outputs/problem2/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass
class StorageParams:
    """储能系统参数。"""

    soc_min: float = 1200.0
    soc_max: float = 10800.0
    soc_initial: float = 6000.0

    eta_charge: float = 0.9
    eta_discharge: float = 0.9

    max_power_kw: float = 5000.0
    delta_t: float = 1.0 / 6.0

    @property
    def max_charge_energy(self) -> float:
        """每个10分钟时段电网侧最大充电电量，单位kWh。"""
        return self.max_power_kw * self.delta_t

    @property
    def max_discharge_energy(self) -> float:
        """每个10分钟时段电网侧最大放电电量，单位kWh。"""
        return self.max_power_kw * self.delta_t

    @property
    def max_soc_increase(self) -> float:
        """每个时段SOC最大增加量，单位kWh。"""
        return self.eta_charge * self.max_charge_energy

    @property
    def max_soc_decrease(self) -> float:
        """每个时段SOC最大减少量，单位kWh。"""
        return self.max_discharge_energy / self.eta_discharge


def read_wide_matrix(
    path: Path,
    sheet_name: str | int = 0,
    expected_slots: int = 144,
):
    """
    读取日期为行、时段为列的宽表。

    返回：
        dates：日期序列
        time_labels：144个时段标签
        values：数值矩阵，形状为 天数×144
    """
    raw = pd.read_excel(path, sheet_name=sheet_name)

    if raw.shape[1] < expected_slots + 1:
        raise ValueError(
            f"{path.name} 的列数不足，至少需要1列日期和"
            f"{expected_slots}列时段数据"
        )

    date_col = raw.columns[0]
    dates = pd.to_datetime(raw[date_col], errors="coerce").dt.date

    if dates.isna().any():
        raise ValueError(f"{path.name} 中存在无法识别的日期")

    time_cols = list(raw.columns[1 : expected_slots + 1])

    values = (
        raw.loc[:, time_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    return dates.reset_index(drop=True), time_cols, values


def read_problem1_price(
    path: Path,
    expected_slots: int = 144,
) -> np.ndarray:
    """
    读取附件1中的固定电价。

    约定：
    附件1第二列为问题二使用的固定电价，
    前144行对应一天的144个10分钟时段。
    """
    raw = pd.read_excel(path)

    if raw.shape[1] < 2:
        raise ValueError("附件1至少需要两列，第二列应为电价")

    if raw.shape[0] < expected_slots:
        raise ValueError(
            f"附件1行数不足{expected_slots}行，无法读取完整电价曲线"
        )

    price = pd.to_numeric(
        raw.iloc[:expected_slots, 1],
        errors="coerce",
    ).to_numpy(dtype=float)

    if np.isnan(price).any():
        raise ValueError("附件1电价中存在缺失值")

    if not np.isfinite(price).all():
        raise ValueError("附件1电价中存在无穷值")

    if (price < 0).any():
        raise ValueError("附件1电价中存在负数")

    return price


def check_input_data(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
):
    """检查负荷、光伏和电价数据。"""

    if load_kw.shape != pv_kw.shape:
        raise ValueError(
            f"负荷矩阵和光伏矩阵维度不一致："
            f"{load_kw.shape} vs {pv_kw.shape}"
        )

    if load_kw.shape[1] != 144:
        raise ValueError(
            f"每天应有144个10分钟时段，当前为{load_kw.shape[1]}"
        )

    if price.shape != load_kw.shape:
        raise ValueError(
            f"电价矩阵维度应与负荷一致："
            f"{price.shape} vs {load_kw.shape}"
        )

    for name, array in [
        ("负荷", load_kw),
        ("光伏", pv_kw),
        ("电价", price),
    ]:
        if np.isnan(array).any():
            missing_count = int(np.isnan(array).sum())
            raise ValueError(
                f"{name}存在缺失值，共{missing_count}个"
            )

        if not np.isfinite(array).all():
            raise ValueError(f"{name}存在无穷值")

        if (array < 0).any():
            negative_count = int((array < 0).sum())
            raise ValueError(
                f"{name}存在负数，共{negative_count}个"
            )


def build_historical_scenarios(
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    day_index: int,
    max_scenarios: int,
):
    """
    根据当前日前可获得的历史数据构造预测和场景。

    重要处理规则：

    1. 只能使用当前日之前的数据；
    2. 2025年1月1日不进入正式评价；
    3. 2025年1月1日没有历史预测，因此不进入残差场景库；
    4. 当前实现采用最近历史日的逐时段均值作为预测；
    5. 历史日相对于历史均值的整日残差作为场景，
       以保留一天内144个时段的共同波动结构。
    """

    if day_index <= 1:
        raise ValueError(
            "当前日期之前没有足够历史数据，无法构造正式评价场景"
        )

    if max_scenarios <= 0:
        raise ValueError("max_scenarios必须为正数")

    # 索引0代表2025年1月1日。
    # 由于1月1日没有历史预测，不将它放入残差场景库。
    history_start = max(1, day_index - max_scenarios)
    history_end = day_index

    historical_load = load_energy[history_start:history_end]
    historical_pv = pv_energy[history_start:history_end]

    if historical_load.shape[0] == 0:
        raise ValueError(
            f"第{day_index}天之前没有可用历史数据"
        )

    # 使用历史数据逐时段均值生成日前预测。
    forecast_load = historical_load.mean(axis=0)
    forecast_pv = historical_pv.mean(axis=0)

    # 历史日相对均值的整日残差。
    historical_load_mean = historical_load.mean(axis=0)
    historical_pv_mean = historical_pv.mean(axis=0)

    load_residual = historical_load - historical_load_mean
    pv_residual = historical_pv - historical_pv_mean

    # 将历史残差叠加到当前预测上，构造未来可能出现的整日场景。
    scenario_load = np.maximum(
        forecast_load[None, :] + load_residual,
        0.0,
    )

    scenario_pv = np.maximum(
        forecast_pv[None, :] + pv_residual,
        0.0,
    )

    scenario_count = scenario_load.shape[0]
    scenario_probability = np.full(
        scenario_count,
        1.0 / scenario_count,
    )

    return (
        forecast_load,
        forecast_pv,
        scenario_load,
        scenario_pv,
        scenario_probability,
    )


def normalize_quantiles(
    quantiles: list[float],
    argument_name: str,
) -> list[float]:
    """检查并整理候选分位数。"""

    values = sorted({float(value) for value in quantiles})

    if not values:
        raise ValueError(f"{argument_name}不能为空")

    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(
            f"{argument_name}中的分位数必须在0和1之间"
        )

    return values


def apply_forecast_safety_margin(
    forecast_load: np.ndarray,
    forecast_pv: np.ndarray,
    scenario_load: np.ndarray,
    scenario_pv: np.ndarray,
    load_safety_factor: float,
    pv_safety_factor: float,
):
    """
    对预测和场景做保守修正：
    负荷上调，光伏下调，以降低净负荷被低估的风险。
    """

    adjusted_forecast_load = np.maximum(
        forecast_load * load_safety_factor,
        0.0,
    )
    adjusted_forecast_pv = np.maximum(
        forecast_pv * pv_safety_factor,
        0.0,
    )
    adjusted_scenario_load = np.maximum(
        scenario_load * load_safety_factor,
        0.0,
    )
    adjusted_scenario_pv = np.maximum(
        scenario_pv * pv_safety_factor,
        0.0,
    )

    return (
        adjusted_forecast_load,
        adjusted_forecast_pv,
        adjusted_scenario_load,
        adjusted_scenario_pv,
    )


def select_dynamic_quantiles(
    recent_daily_rows: list[dict],
    normal_quantiles: list[float],
    high_risk_quantiles: list[float],
    lookback_days: int,
    emergency_threshold_kwh: float,
    net_error_threshold_kwh: float,
) -> tuple[list[float], str, str]:
    """
    根据近期执行结果动态选择日前购电分位数。

    近期出现明显临时购电或净负荷正向误差时，
    后续日期使用更高分位数，降低继续买少的风险。
    """

    if lookback_days <= 0 or not recent_daily_rows:
        return normal_quantiles, "常规", "无近期记录"

    recent_rows = recent_daily_rows[-lookback_days:]

    emergency_days = [
        row
        for row in recent_rows
        if row["emergency_purchase_kwh"]
        >= emergency_threshold_kwh
    ]
    positive_error_days = [
        row
        for row in recent_rows
        if row["net_load_positive_error_kwh"]
        >= net_error_threshold_kwh
    ]

    if emergency_days or positive_error_days:
        reasons = []

        if emergency_days:
            reasons.append(
                f"近{lookback_days}天出现"
                f"{len(emergency_days)}天临时购电偏高"
            )

        if positive_error_days:
            reasons.append(
                f"近{lookback_days}天出现"
                f"{len(positive_error_days)}天净负荷低估"
            )

        return high_risk_quantiles, "高风险", "，".join(reasons)

    return normal_quantiles, "常规", "近期误差可控"


def solve_day_ahead_lp(
    target_net_load: np.ndarray,
    price: np.ndarray,
    soc_start: float,
    storage_value: float,
    params: StorageParams,
) -> dict:
    """
    根据目标净负荷求解日前线性规划。

    决策变量：

        g_t：日前计划购电量
        c_t：交流侧充电电量
        r_t：交流侧放电电量
        s_t：时段末储能电量
        u_t：未利用电量

    目标：

        最小化 (日前购电费用 - 日末库存价值)

    关键修正：
    1. 不再强制 s_T = s_0
    2. 目标函数中加入日末SOC的价值项 -storage_value * s_T
    """

    time_count = len(target_net_load)
    variable_count = 5 * time_count

    index_g = slice(0, time_count)
    index_c = slice(time_count, 2 * time_count)
    index_r = slice(2 * time_count, 3 * time_count)
    index_s = slice(3 * time_count, 4 * time_count)
    index_u = slice(4 * time_count, 5 * time_count)

    objective = np.zeros(variable_count)
    objective[index_g] = price
    # 日末库存价值（负号表示这是收益）
    objective[index_s.start + time_count - 1] = -storage_value

    equality_rows = []
    equality_rhs = []

    # 电力平衡：
    #
    # g_t + r_t = n_t + c_t + u_t
    #
    # 即：
    # 日前购电 + 储能放电
    # =
    # 净负荷 + 储能充电 + 未利用电量
    for t in range(time_count):
        row = np.zeros(variable_count)

        row[index_g.start + t] = 1.0
        row[index_r.start + t] = 1.0
        row[index_c.start + t] = -1.0
        row[index_u.start + t] = -1.0

        equality_rows.append(row)
        equality_rhs.append(target_net_load[t])

    # 储能状态转移：
    #
    # s_t = s_{t-1} + eta_c*c_t - r_t/eta_d
    for t in range(time_count):
        row = np.zeros(variable_count)

        row[index_s.start + t] = 1.0
        row[index_c.start + t] = -params.eta_charge
        row[index_r.start + t] = 1.0 / params.eta_discharge

        if t > 0:
            row[index_s.start + t - 1] = -1.0
            right_side = 0.0
        else:
            right_side = soc_start

        equality_rows.append(row)
        equality_rhs.append(right_side)

    bounds = []

    # g_t >= 0
    bounds.extend([(0.0, None)] * time_count)

    # 0 <= c_t <= 最大交流侧充电电量
    bounds.extend(
        [(0.0, params.max_charge_energy)] * time_count
    )

    # 0 <= r_t <= 最大交流侧放电电量
    bounds.extend(
        [(0.0, params.max_discharge_energy)] * time_count
    )

    # soc_min <= s_t <= soc_max
    bounds.extend(
        [(params.soc_min, params.soc_max)] * time_count
    )

    # u_t >= 0
    bounds.extend([(0.0, None)] * time_count)

    result = linprog(
        objective,
        A_eq=np.vstack(equality_rows),
        b_eq=np.asarray(equality_rhs),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            f"日前LP求解失败：{result.message}"
        )

    solution = result.x

    # 提取日末SOC
    terminal_soc = solution[index_s.start + time_count - 1]

    # 纯购电费用（不含库存价值）
    pure_purchase_cost = float(np.sum(price * solution[index_g]))

    return {
        "objective_value": float(result.fun),
        "pure_purchase_cost": pure_purchase_cost,
        "terminal_soc": float(terminal_soc),
        "planned_purchase": solution[index_g],
        "planned_charge": solution[index_c],
        "planned_discharge": solution[index_r],
        "planned_soc": solution[index_s],
        "planned_unused": solution[index_u],
    }


def make_soc_grid(
    params: StorageParams,
    soc_step: float,
) -> np.ndarray:
    """建立SOC离散网格。"""

    if soc_step <= 0:
        raise ValueError("soc_step必须为正数")

    grid = np.arange(
        params.soc_min,
        params.soc_max + 0.5 * soc_step,
        soc_step,
    )

    if not np.any(
        np.isclose(grid, params.soc_initial)
    ):
        grid = np.sort(
            np.append(grid, params.soc_initial)
        )

    return grid


def nearest_grid_index(
    grid: np.ndarray,
    value: float,
) -> int:
    """寻找距离给定SOC最近的网格位置。"""

    return int(np.argmin(np.abs(grid - value)))


def storage_grid_matrices(
    grid: np.ndarray,
    params: StorageParams,
):
    """
    预计算SOC网格转移矩阵。

    返回：

        delta[i,j]：
            从当前SOC grid[i] 转移到下一SOC grid[j] 的SOC变化量。

        psi[i,j]：
            储能动作对应的电网侧净用电变化量。

        feasible[i,j]：
            是否允许从状态i转移到状态j。
    """

    delta = grid[None, :] - grid[:, None]

    feasible = (
        (delta <= params.max_soc_increase + 1e-9)
        & (delta >= -params.max_soc_decrease - 1e-9)
    )

    # delta > 0表示充电。
    # delta < 0表示放电。
    psi = np.where(
        delta >= 0.0,
        delta / params.eta_charge,
        params.eta_discharge * delta,
    )

    return delta, psi, feasible


def compute_dp_value(
    planned_purchase: np.ndarray,
    scenario_net_load: np.ndarray,
    scenario_probability: np.ndarray,
    price: np.ndarray,
    grid: np.ndarray,
    psi: np.ndarray,
    feasible: np.ndarray,
    emergency_multiplier: float,
    storage_value: float,
) -> np.ndarray:
    """
    计算因果DP未来价值函数。

    F[t,i]表示：
    第t个时段开始、SOC为grid[i]时，
    从t到当天结束的最低期望临时购电费用（减去日末库存价值）。

    关键修正：
    终端价值函数从0改为 -storage_value * s
    """

    time_count = len(planned_purchase)
    state_count = len(grid)

    future_value = np.zeros(
        (time_count + 1, state_count),
        dtype=float,
    )

    # 终端价值：日末库存价值
    future_value[time_count] = -storage_value * grid

    for t in range(time_count - 1, -1, -1):

        # 形状：
        # 场景数 × 当前SOC状态数 × 下一SOC状态数
        shortage = np.maximum(
            scenario_net_load[:, t, None, None]
            + psi[None, :, :]
            - planned_purchase[t],
            0.0,
        )

        expected_shortage = np.tensordot(
            scenario_probability,
            shortage,
            axes=(0, 0),
        )

        immediate_cost = (
            emergency_multiplier
            * price[t]
            * expected_shortage
        )

        # 下一时段价值由下一SOC状态决定。
        total_cost = (
            immediate_cost
            + future_value[t + 1][None, :]
        )

        total_cost = np.where(
            feasible,
            total_cost,
            np.inf,
        )

        future_value[t] = np.min(
            total_cost,
            axis=1,
        )

    return future_value


def execute_one_day(
    planned_purchase: np.ndarray,
    actual_net_load: np.ndarray,
    price: np.ndarray,
    soc_start: float,
    params: StorageParams,
    grid: np.ndarray,
    psi: np.ndarray,
    feasible: np.ndarray,
    future_value: np.ndarray,
    emergency_multiplier: float,
) -> dict:
    """
    固定日前购电计划后，使用实际净负荷进行因果DP执行。
    """

    time_count = len(planned_purchase)

    soc_index = nearest_grid_index(
        grid,
        soc_start,
    )

    soc = np.zeros(time_count)
    charge = np.zeros(time_count)
    discharge = np.zeros(time_count)
    emergency_purchase = np.zeros(time_count)
    unused_energy = np.zeros(time_count)
    soc_change = np.zeros(time_count)

    for t in range(time_count):

        immediate_shortage = np.maximum(
            actual_net_load[t]
            + psi[soc_index, :]
            - planned_purchase[t],
            0.0,
        )

        total_cost = (
            emergency_multiplier
            * price[t]
            * immediate_shortage
            + future_value[t + 1]
        )

        total_cost = np.where(
            feasible[soc_index, :],
            total_cost,
            np.inf,
        )

        next_index = int(
            np.argmin(total_cost)
        )

        soc_change[t] = (
            grid[next_index] - grid[soc_index]
        )

        if soc_change[t] >= 0:
            charge[t] = (
                soc_change[t]
                / params.eta_charge
            )
            discharge[t] = 0.0
        else:
            charge[t] = 0.0
            discharge[t] = (
                -params.eta_discharge
                * soc_change[t]
            )

        actual_grid_effect = psi[
            soc_index,
            next_index,
        ]

        emergency_purchase[t] = max(
            actual_net_load[t]
            + actual_grid_effect
            - planned_purchase[t],
            0.0,
        )

        unused_energy[t] = max(
            planned_purchase[t]
            - actual_net_load[t]
            - actual_grid_effect,
            0.0,
        )

        soc_index = next_index
        soc[t] = grid[soc_index]

    normal_cost = float(
        np.sum(price * planned_purchase)
    )

    emergency_cost = float(
        np.sum(
            emergency_multiplier
            * price
            * emergency_purchase
        )
    )

    return {
        "soc": soc,
        "charge": charge,
        "discharge": discharge,
        "emergency_purchase": emergency_purchase,
        "unused_energy": unused_energy,
        "soc_change": soc_change,
        "normal_cost": normal_cost,
        "emergency_cost": emergency_cost,
        "total_cost": normal_cost + emergency_cost,
    }


def build_detail_rows(
    current_date,
    time_labels,
    price,
    load_energy,
    pv_energy,
    planned_purchase,
    execution,
):
    """生成10分钟级结果明细。"""

    rows = []

    for t, label in enumerate(time_labels):
        rows.append(
            {
                "date": current_date,
                "slot_index": t,
                "time": str(label),
                "price_yuan_per_kwh": price[t],
                "load_kwh": load_energy[t],
                "pv_kwh": pv_energy[t],
                "net_load_kwh": (
                    load_energy[t]
                    - pv_energy[t]
                ),
                "planned_purchase_kwh": (
                    planned_purchase[t]
                ),
                "charge_kwh": (
                    execution["charge"][t]
                ),
                "discharge_kwh": (
                    execution["discharge"][t]
                ),
                "soc_end_kwh": (
                    execution["soc"][t]
                ),
                "emergency_purchase_kwh": (
                    execution[
                        "emergency_purchase"
                    ][t]
                ),
                "unused_energy_kwh": (
                    execution[
                        "unused_energy"
                    ][t]
                ),
            }
        )

    return rows


def plot_results(
    output_dir: Path,
    daily_summary: pd.DataFrame,
    detail: pd.DataFrame,
):
    """绘制正式评价期结果图。"""

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    if daily_summary.empty or detail.empty:
        return

    # 每日费用图
    fig, ax = plt.subplots(figsize=(11, 4))

    dates_for_plot = pd.to_datetime(
        daily_summary["date"]
    )

    ax.plot(
        dates_for_plot,
        daily_summary["total_cost_yuan"],
        label="总费用",
    )

    ax.plot(
        dates_for_plot,
        daily_summary["emergency_cost_yuan"],
        label="临时购电费用",
    )

    ax.set_title(
        "问题二正式评价期每日费用"
    )
    ax.set_xlabel("日期")
    ax.set_ylabel("费用/元")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "problem2_daily_cost.png",
        dpi=180,
    )
    plt.close(fig)

    # 正式评价期第一天的能量曲线
    first_date = detail["date"].iloc[0]
    first_day = detail[
        detail["date"] == first_date
    ].copy()

    x = np.arange(len(first_day))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        x,
        first_day["load_kwh"],
        label="负荷",
    )
    ax.plot(
        x,
        first_day["pv_kwh"],
        label="光伏",
    )
    ax.plot(
        x,
        first_day["planned_purchase_kwh"],
        label="计划购电",
    )
    ax.plot(
        x,
        first_day["emergency_purchase_kwh"],
        label="临时购电",
    )

    ax.set_title(
        f"{first_date} 正式评价日能量曲线"
    )
    ax.set_xlabel("10分钟时段")
    ax.set_ylabel("电量/kWh")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "problem2_first_formal_day_energy.png",
        dpi=180,
    )
    plt.close(fig)

    # 正式评价期第一天的SOC曲线
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        x,
        first_day["soc_end_kwh"],
        label="储能SOC",
    )

    ax.set_title(
        f"{first_date} 正式评价日储能电量变化"
    )
    ax.set_xlabel("10分钟时段")
    ax.set_ylabel("储能电量/kWh")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "problem2_first_formal_day_soc.png",
        dpi=180,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="问题二：日前LP + 日内因果DP（跨日连续版本）"
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            r"C:/Users/LENOVO/Desktop/数模/C题"
        ),
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="调试用：从2025-02-01开始只计算前N个正式评价日",
    )

    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=30,
        help="每天最多使用最近多少个历史日场景",
    )

    parser.add_argument(
        "--soc-step",
        type=float,
        default=100.0,
        help="DP的SOC离散步长，越小越精细但越慢",
    )

    parser.add_argument(
        "--candidate-quantiles",
        type=float,
        nargs="+",
        default=[0.75],
        help=(
            "常规状态下日前LP使用的净负荷分位数；"
            "多个值会逐个试算并选择估计费用最低者"
        ),
    )

    parser.add_argument(
        "--high-risk-quantiles",
        type=float,
        nargs="+",
        default=[0.85, 0.90, 0.95],
        help=(
            "近期出现明显临时购电或净负荷低估时使用的"
            "高风险净负荷分位数"
        ),
    )

    parser.add_argument(
        "--load-safety-factor",
        type=float,
        default=1.05,
        help="日前预测中负荷的安全上调系数，默认1.05",
    )

    parser.add_argument(
        "--pv-safety-factor",
        type=float,
        default=0.90,
        help="日前预测中光伏的安全下调系数，默认0.90",
    )

    parser.add_argument(
        "--dynamic-quantile-lookback",
        type=int,
        default=3,
        help="动态分位数判断时回看最近多少个正式评价日",
    )

    parser.add_argument(
        "--dynamic-emergency-threshold",
        type=float,
        default=1000.0,
        help="触发高风险分位数的近期单日临时购电量阈值(kWh)",
    )

    parser.add_argument(
        "--dynamic-net-error-threshold",
        type=float,
        default=5000.0,
        help="触发高风险分位数的近期单日净负荷低估阈值(kWh)",
    )

    parser.add_argument(
        "--storage-value",
        type=float,
        default=0.481548,
        help="日末库存价值系数（元/kWh），参考论文取值",
    )

    parser.add_argument(
        "--use-attachment4-price",
        action="store_true",
        help=(
            "问题四才建议打开；"
            "问题二默认使用附件1固定电价曲线"
        ),
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-02-01",
        help="正式评价开始日期，默认2025-02-01",
    )

    args = parser.parse_args()

    if args.max_days is not None and args.max_days <= 0:
        raise ValueError("--max-days必须为正整数")

    if args.max_scenarios <= 0:
        raise ValueError("--max-scenarios必须为正整数")

    normal_quantiles = normalize_quantiles(
        args.candidate_quantiles,
        "--candidate-quantiles",
    )

    high_risk_quantiles = normalize_quantiles(
        args.high_risk_quantiles,
        "--high-risk-quantiles",
    )

    if args.storage_value < 0:
        raise ValueError("--storage-value不能为负数")

    if args.load_safety_factor < 1.0:
        raise ValueError("--load-safety-factor不能小于1")

    if (
        args.pv_safety_factor < 0.0
        or args.pv_safety_factor > 1.0
    ):
        raise ValueError("--pv-safety-factor必须在0和1之间")

    if args.dynamic_quantile_lookback < 0:
        raise ValueError("--dynamic-quantile-lookback不能为负数")

    if args.dynamic_emergency_threshold < 0:
        raise ValueError("--dynamic-emergency-threshold不能为负数")

    if args.dynamic_net_error_threshold < 0:
        raise ValueError("--dynamic-net-error-threshold不能为负数")

    params = StorageParams()
    emergency_multiplier = 5.0

    base_dir = args.base_dir
    attachment_dir = base_dir / "附件"
    output_dir = (
        base_dir
        / "outputs"
        / "problem2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_attachment1 = (
        attachment_dir / "附件1.xlsx"
    )
    path_attachment2 = (
        attachment_dir / "附件2.xlsx"
    )
    path_attachment4 = (
        attachment_dir / "附件4.xlsx"
    )

    # =====================================================
    # 1. 数据读取
    # =====================================================

    dates, time_labels, load_kw = read_wide_matrix(
        path_attachment2,
        sheet_name=0,
    )

    pv_dates, _, pv_kw = read_wide_matrix(
        path_attachment2,
        sheet_name=1,
    )

    if not dates.equals(pv_dates):
        raise ValueError(
            "附件2的负荷日期和光伏日期不一致"
        )

    fixed_price = read_problem1_price(
        path_attachment1
    )

    if args.use_attachment4_price:
        price_dates, _, price = read_wide_matrix(
            path_attachment4,
            sheet_name=0,
        )

        if not dates.equals(price_dates):
            raise ValueError(
                "附件2日期与附件4电价日期不一致"
            )
    else:
        price = np.tile(
            fixed_price[None, :],
            (load_kw.shape[0], 1),
        )

    # =====================================================
    # 2. 数据处理与检查
    # =====================================================

    check_input_data(
        load_kw,
        pv_kw,
        price,
    )

    load_energy = (
        load_kw * params.delta_t
    )

    pv_energy = (
        pv_kw * params.delta_t
    )

    all_dates = np.asarray(
        dates.tolist(),
        dtype=object,
    )

    # =====================================================
    # 3. 确定正式评价区间
    # =====================================================

    try:
        formal_start_date = pd.Timestamp(
            args.start_date
        ).date()
    except Exception as exc:
        raise ValueError(
            f"无法解析正式评价开始日期：{args.start_date}"
        ) from exc

    formal_indices = np.flatnonzero(
        all_dates >= formal_start_date
    )

    if len(formal_indices) == 0:
        raise ValueError(
            f"数据中没有不早于{formal_start_date}的日期"
        )

    # 只保留正式评价期。
    # 因此--max-days 代表从正式评价开始日往后的N天。
    if args.max_days is not None:
        formal_indices = formal_indices[
            : args.max_days
        ]

    first_formal_index = int(
        formal_indices[0]
    )

    excluded_dates = all_dates[
        :first_formal_index
    ]

    if first_formal_index == 0:
        raise ValueError(
            "正式评价开始日之前没有历史数据，"
            "无法构造日前预测场景"
        )

    print(
        f"正式评价区间："
        f"{all_dates[formal_indices[0]]} 至 "
        f"{all_dates[formal_indices[-1]]}"
    )

    print(
        f"正式评价天数：{len(formal_indices)}"
    )

    print(
        "未纳入正式评价的预热日期："
        f"{len(excluded_dates)}天"
    )

    print(
        f"日末库存价值系数："
        f"{args.storage_value:.6f} 元/kWh"
    )

    print(
        "预测安全修正："
        f"负荷×{args.load_safety_factor:.3f}，"
        f"光伏×{args.pv_safety_factor:.3f}"
    )

    print(
        "动态分位数："
        f"常规{normal_quantiles}，"
        f"高风险{high_risk_quantiles}"
    )

    # =====================================================
    # 4. 参数计算
    # =====================================================

    soc_grid = make_soc_grid(
        params,
        args.soc_step,
    )

    _, psi, feasible = storage_grid_matrices(
        soc_grid,
        params,
    )

    # 2月1日没有执行1月1日的控制策略，
    # 因此正式评价期起始SOC采用题目给定初始值6000kWh。
    soc_start = params.soc_initial

    detail_rows = []
    daily_rows = []

    # =====================================================
    # 5. 模型求解
    # =====================================================

    for day_index in formal_indices:

        current_date = all_dates[day_index]

        (
            forecast_load,
            forecast_pv,
            scenario_load,
            scenario_pv,
            scenario_probability,
        ) = build_historical_scenarios(
            load_energy=load_energy,
            pv_energy=pv_energy,
            day_index=int(day_index),
            max_scenarios=args.max_scenarios,
        )

        (
            forecast_load,
            forecast_pv,
            scenario_load,
            scenario_pv,
        ) = apply_forecast_safety_margin(
            forecast_load=forecast_load,
            forecast_pv=forecast_pv,
            scenario_load=scenario_load,
            scenario_pv=scenario_pv,
            load_safety_factor=args.load_safety_factor,
            pv_safety_factor=args.pv_safety_factor,
        )

        scenario_net_load = (
            scenario_load - scenario_pv
        )

        active_quantiles, risk_mode, risk_reason = (
            select_dynamic_quantiles(
                recent_daily_rows=daily_rows,
                normal_quantiles=normal_quantiles,
                high_risk_quantiles=high_risk_quantiles,
                lookback_days=(
                    args.dynamic_quantile_lookback
                ),
                emergency_threshold_kwh=(
                    args.dynamic_emergency_threshold
                ),
                net_error_threshold_kwh=(
                    args.dynamic_net_error_threshold
                ),
            )
        )

        best_solution = None

        for quantile in active_quantiles:

            target_net_load = np.quantile(
                scenario_net_load,
                quantile,
                axis=0,
            )

            lp_solution = solve_day_ahead_lp(
                target_net_load=target_net_load,
                price=price[day_index],
                soc_start=soc_start,
                storage_value=args.storage_value,
                params=params,
            )

            future_value = compute_dp_value(
                planned_purchase=(
                    lp_solution[
                        "planned_purchase"
                    ]
                ),
                scenario_net_load=scenario_net_load,
                scenario_probability=(
                    scenario_probability
                ),
                price=price[day_index],
                grid=soc_grid,
                psi=psi,
                feasible=feasible,
                emergency_multiplier=(
                    emergency_multiplier
                ),
                storage_value=args.storage_value,
            )

            start_state_index = nearest_grid_index(
                soc_grid,
                soc_start,
            )

            # 估计的总成本 = LP目标值（已包含库存价值）+ DP期望紧急购电费用
            estimated_total_cost = (
                lp_solution["objective_value"]
                + future_value[
                    0,
                    start_state_index,
                ]
                + args.storage_value * soc_start  # 补偿起始库存价值
            )

            candidate_solution = {
                "quantile": quantile,
                "lp_solution": lp_solution,
                "future_value": future_value,
                "target_net_load_sum": float(
                    np.sum(target_net_load)
                ),
                "estimated_total_cost": float(
                    estimated_total_cost
                ),
            }

            if (
                best_solution is None
                or candidate_solution[
                    "estimated_total_cost"
                ]
                < best_solution[
                    "estimated_total_cost"
                ]
            ):
                best_solution = candidate_solution

        if best_solution is None:
            raise RuntimeError(
                f"{current_date}没有得到可行日前计划"
            )

        planned_purchase = best_solution[
            "lp_solution"
        ]["planned_purchase"]

        actual_net_load = (
            load_energy[day_index]
            - pv_energy[day_index]
        )

        execution = execute_one_day(
            planned_purchase=planned_purchase,
            actual_net_load=actual_net_load,
            price=price[day_index],
            soc_start=soc_start,
            params=params,
            grid=soc_grid,
            psi=psi,
            feasible=feasible,
            future_value=best_solution[
                "future_value"
            ],
            emergency_multiplier=(
                emergency_multiplier
            ),
        )

        actual_net_load_sum = float(
            np.sum(actual_net_load)
        )
        forecast_load_sum = float(
            np.sum(forecast_load)
        )
        forecast_pv_sum = float(
            np.sum(forecast_pv)
        )
        forecast_net_load_sum = (
            forecast_load_sum - forecast_pv_sum
        )
        net_load_positive_error = max(
            actual_net_load_sum
            - best_solution["target_net_load_sum"],
            0.0,
        )

        detail_rows.extend(
            build_detail_rows(
                current_date=current_date,
                time_labels=time_labels,
                price=price[day_index],
                load_energy=load_energy[
                    day_index
                ],
                pv_energy=pv_energy[
                    day_index
                ],
                planned_purchase=planned_purchase,
                execution=execution,
            )
        )

        daily_rows.append(
            {
                "date": current_date,
                "risk_mode": risk_mode,
                "risk_reason": risk_reason,
                "active_quantiles": ",".join(
                    f"{quantile:.2f}"
                    for quantile in active_quantiles
                ),
                "selected_quantile": best_solution[
                    "quantile"
                ],
                "scenario_count": len(
                    scenario_probability
                ),
                "forecast_load_kwh": forecast_load_sum,
                "forecast_pv_kwh": forecast_pv_sum,
                "forecast_net_load_kwh": (
                    forecast_net_load_sum
                ),
                "target_net_load_kwh": best_solution[
                    "target_net_load_sum"
                ],
                "actual_net_load_kwh": (
                    actual_net_load_sum
                ),
                "net_load_positive_error_kwh": (
                    net_load_positive_error
                ),
                "soc_start_kwh": soc_start,
                "soc_end_kwh": execution[
                    "soc"
                ][-1],
                "planned_purchase_kwh": (
                    planned_purchase.sum()
                ),
                "emergency_purchase_kwh": (
                    execution[
                        "emergency_purchase"
                    ].sum()
                ),
                "unused_energy_kwh": (
                    execution[
                        "unused_energy"
                    ].sum()
                ),
                "normal_cost_yuan": execution[
                    "normal_cost"
                ],
                "emergency_cost_yuan": execution[
                    "emergency_cost"
                ],
                "total_cost_yuan": execution[
                    "total_cost"
                ],
                "estimated_total_cost_yuan": (
                    best_solution[
                        "estimated_total_cost"
                    ]
                ),
            }
        )

        # 跨日SOC连续传递
        soc_start = float(
            execution["soc"][-1]
        )

        latest_day = daily_rows[-1]

        print(
            f"{current_date}完成 | "
            f"{risk_mode} | "
            f"q={best_solution['quantile']:.2f} | "
            f"SOC："
            f"{latest_day['soc_start_kwh']:.1f}"
            f"->{soc_start:.1f} | "
            f"临时购电："
            f"{latest_day['emergency_purchase_kwh']:.1f}"
            f"kWh | "
            f"总费用："
            f"{latest_day['total_cost_yuan']:.2f}元"
        )

    # =====================================================
    # 6. 结果检验
    # =====================================================

    detail = pd.DataFrame(detail_rows)
    daily_summary = pd.DataFrame(daily_rows)

    if detail.empty or daily_summary.empty:
        raise RuntimeError(
            "正式评价期没有生成任何结果"
        )

    if (
        detail["soc_end_kwh"]
        .lt(params.soc_min - 1e-6)
        .any()
        or detail["soc_end_kwh"]
        .gt(params.soc_max + 1e-6)
        .any()
    ):
        raise RuntimeError(
            "结果检验失败：SOC超出允许范围"
        )

    if (
        detail["charge_kwh"]
        .gt(params.max_charge_energy + 1e-6)
        .any()
    ):
        raise RuntimeError(
            "结果检验失败：充电功率约束被违反"
        )

    if (
        detail["discharge_kwh"]
        .gt(params.max_discharge_energy + 1e-6)
        .any()
    ):
        raise RuntimeError(
            "结果检验失败：放电功率约束被违反"
        )

    nonnegative_columns = [
        "planned_purchase_kwh",
        "emergency_purchase_kwh",
        "unused_energy_kwh",
    ]

    if (
        detail[nonnegative_columns] < -1e-6
    ).any().any():
        raise RuntimeError(
            "结果检验失败：出现负的购电量、"
            "临时购电量或弃电量"
        )

    # 检查正式评价日之间的SOC是否连续。
    previous_soc_end = None

    for _, row in daily_summary.iterrows():
        if previous_soc_end is not None:
            if not np.isclose(
                row["soc_start_kwh"],
                previous_soc_end,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "结果检验失败：相邻正式评价日SOC不连续"
                )

        previous_soc_end = row[
            "soc_end_kwh"
        ]

    # =====================================================
    # 7. 可视化
    # =====================================================

    plot_results(
        output_dir=output_dir,
        daily_summary=daily_summary,
        detail=detail,
    )

    # =====================================================
    # 8. 结果导出
    # =====================================================

    detail_path = (
        output_dir / "problem2_detail.csv"
    )

    daily_path = (
        output_dir
        / "problem2_daily_summary.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily_summary.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    submission_path = generate_submission_tables(
        detail=detail,
        daily_summary=daily_summary,
        time_labels=time_labels,
        output_dir=output_dir,
    )

    total_normal_cost = float(
        daily_summary[
            "normal_cost_yuan"
        ].sum()
    )

    total_emergency_cost = float(
        daily_summary[
            "emergency_cost_yuan"
        ].sum()
    )

    total_cost = float(
        daily_summary[
            "total_cost_yuan"
        ].sum()
    )

    total_emergency_energy = float(
        daily_summary[
            "emergency_purchase_kwh"
        ].sum()
    )

    print()
    print("========== 问题二正式评价完成 ==========")
    print(
        f"正式评价区间："
        f"{daily_summary['date'].iloc[0]} 至 "
        f"{daily_summary['date'].iloc[-1]}"
    )
    print(
        f"正式评价天数：{len(daily_summary)}"
    )
    print(
        f"未纳入评价的预热天数："
        f"{len(excluded_dates)}"
    )
    print(
        f"日末库存价值系数："
        f"{args.storage_value:.6f} 元/kWh"
    )
    print(
        f"计划购电费用："
        f"{total_normal_cost:.2f} 元"
    )
    print(
        f"临时购电费用："
        f"{total_emergency_cost:.2f} 元"
    )
    print(
        f"总费用："
        f"{total_cost:.2f} 元"
    )
    print(
        f"临时购电总量："
        f"{total_emergency_energy:.2f} kWh"
    )
    print(f"结果明细：{detail_path}")
    print(f"每日汇总：{daily_path}")
    print(f"提交表格：{submission_path}")
    print("包含三张表：")
    print("  1. 计划购电量（日期×144时段）")
    print("  2. 充放电量（6个4小时时段）")
    print("  3. 紧急购电量（日期、购电时间段、购电量）")

def format_clock_label(
    total_minutes: int,
    mark_next_day: bool = False,
) -> str:
    """生成类似 0:10 或 0:10+1 的时刻标签。"""

    minutes_in_day = 24 * 60
    wrapped_minutes = total_minutes % minutes_in_day
    hour, minute = divmod(wrapped_minutes, 60)
    label = f"{hour}:{minute:02d}"

    if mark_next_day and total_minutes >= minutes_in_day:
        label += "+1"

    return label


def make_10min_interval_label(slot_index: int) -> str:
    """按附件5示意格式生成第slot_index个10分钟购电时段。"""

    start_minute = (slot_index + 1) * 10
    end_minute = start_minute + 10

    return (
        f"{format_clock_label(start_minute)}-"
        f"{format_clock_label(end_minute, mark_next_day=True)}"
    )


def make_slot_range_label(
    start_slot: int,
    end_slot: int,
) -> str:
    """把连续10分钟时段合并成一个购电时间段标签。"""

    start_minute = (start_slot + 1) * 10
    end_minute = (end_slot + 2) * 10

    return (
        f"{format_clock_label(start_minute)}-"
        f"{format_clock_label(end_minute, mark_next_day=True)}"
    )


def build_emergency_purchase_table(
    detail: pd.DataFrame,
    dates,
    threshold: float = 1e-6,
) -> pd.DataFrame:
    """生成紧急购电量表，连续时段合并为一行。"""

    positive_detail = (
        detail[detail["emergency_purchase_kwh"] > threshold]
        .copy()
        .sort_values(["date", "slot_index"])
    )

    rows = []

    for current_date in dates:
        day_detail = positive_detail[
            positive_detail["date"] == current_date
        ]

        if day_detail.empty:
            rows.append(
                {
                    "日期": current_date,
                    "购电时间段": None,
                    "购电量": None,
                }
            )
            continue

        first_row_for_date = True
        start_slot = None
        end_slot = None
        purchase_amount = 0.0

        for _, row in day_detail.iterrows():
            slot_index = int(row["slot_index"])
            amount = float(row["emergency_purchase_kwh"])

            if start_slot is None:
                start_slot = slot_index
                end_slot = slot_index
                purchase_amount = amount
                continue

            if slot_index == end_slot + 1:
                end_slot = slot_index
                purchase_amount += amount
                continue

            rows.append(
                {
                    "日期": (
                        current_date
                        if first_row_for_date
                        else None
                    ),
                    "购电时间段": make_slot_range_label(
                        start_slot,
                        end_slot,
                    ),
                    "购电量": purchase_amount,
                }
            )
            first_row_for_date = False
            start_slot = slot_index
            end_slot = slot_index
            purchase_amount = amount

        rows.append(
            {
                "日期": (
                    current_date
                    if first_row_for_date
                    else None
                ),
                "购电时间段": make_slot_range_label(
                    start_slot,
                    end_slot,
                ),
                "购电量": purchase_amount,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["日期", "购电时间段", "购电量"],
    )


def build_storage_table(
    detail: pd.DataFrame,
    daily_summary: pd.DataFrame,
) -> pd.DataFrame:
    """生成每天6个4小时时段的储能充放电量表。"""

    four_hour_periods = [
        "0:00-4:00",
        "4:00-8:00",
        "8:00-12:00",
        "12:00-16:00",
        "16:00-20:00",
        "20:00-24:00",
    ]

    detail_with_period = detail.copy()
    detail_with_period["period_idx"] = (
        detail_with_period["slot_index"].astype(int) // 24
    )

    storage_summary = (
        detail_with_period.groupby(
            ["date", "period_idx"],
            as_index=False,
        )
        .agg(
            charge_kwh=("charge_kwh", "sum"),
            discharge_kwh=("discharge_kwh", "sum"),
        )
    )

    storage_lookup = {
        (row["date"], int(row["period_idx"])): row
        for _, row in storage_summary.iterrows()
    }

    rows = []

    for _, day_row in daily_summary.iterrows():
        current_date = day_row["date"]

        for period_idx, period_label in enumerate(four_hour_periods):
            storage_row = storage_lookup.get(
                (current_date, period_idx)
            )

            if storage_row is None:
                charge_amount = 0.0
                discharge_amount = 0.0
            else:
                charge_amount = float(
                    storage_row["charge_kwh"]
                )
                discharge_amount = float(
                    storage_row["discharge_kwh"]
                )

            if period_idx == 0:
                time_mark = "0:00"
                soc_value = day_row["soc_start_kwh"]
            elif period_idx == 1:
                time_mark = "24:00"
                soc_value = day_row["soc_end_kwh"]
            else:
                time_mark = None
                soc_value = None

            rows.append(
                {
                    "日期": (
                        current_date
                        if period_idx == 0
                        else None
                    ),
                    "时间段": period_label,
                    "充电量": charge_amount,
                    "放电量": discharge_amount,
                    "时刻": time_mark,
                    "储电量": soc_value,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "日期",
            "时间段",
            "充电量",
            "放电量",
            "时刻",
            "储电量",
        ],
    )


def format_submission_workbook(
    writer,
    sheet_names: list[str],
):
    """按附件5示意表做基础格式整理。"""

    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    workbook = writer.book
    thin_side = Side(style="thin", color="000000")
    thin_border = Border(
        left=thin_side,
        right=thin_side,
        top=thin_side,
        bottom=thin_side,
    )

    for sheet_name in sheet_names:
        worksheet = workbook[sheet_name]

        for row in worksheet.iter_rows():
            for cell in row:
                cell.font = Font(name="宋体", size=10)
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )

        for cell in worksheet[1]:
            cell.border = thin_border

        worksheet.row_dimensions[1].height = 22

    planned_sheet = workbook["计划购电量"]
    planned_sheet.column_dimensions["A"].width = 12

    for col_idx in range(2, planned_sheet.max_column + 1):
        planned_sheet.column_dimensions[
            get_column_letter(col_idx)
        ].width = 12

    for row in planned_sheet.iter_rows(
        min_row=2,
        max_row=planned_sheet.max_row,
    ):
        row[0].number_format = "yyyy/m/d"
        for cell in row[1:]:
            cell.number_format = "0.00"

    storage_sheet = workbook["充放电量"]
    for col_idx in range(1, storage_sheet.max_column + 1):
        storage_sheet.column_dimensions[
            get_column_letter(col_idx)
        ].width = 12

    for row in storage_sheet.iter_rows(
        min_row=1,
        max_row=storage_sheet.max_row,
        max_col=storage_sheet.max_column,
    ):
        for cell in row:
            cell.border = thin_border

    for row in storage_sheet.iter_rows(
        min_row=2,
        max_row=storage_sheet.max_row,
    ):
        row[0].number_format = "yyyy/m/d"
        row[2].number_format = "0.00"
        row[3].number_format = "0.00"
        row[5].number_format = "0.00"

    emergency_sheet = workbook["紧急购电量"]
    for col_idx in range(1, emergency_sheet.max_column + 1):
        emergency_sheet.column_dimensions[
            get_column_letter(col_idx)
        ].width = 14

    for row in emergency_sheet.iter_rows(
        min_row=1,
        max_row=emergency_sheet.max_row,
        max_col=emergency_sheet.max_column,
    ):
        for cell in row:
            cell.border = thin_border

    for row in emergency_sheet.iter_rows(
        min_row=2,
        max_row=emergency_sheet.max_row,
    ):
        row[0].number_format = "yyyy/m/d"
        row[2].number_format = "0.00"


def generate_submission_tables(
    detail: pd.DataFrame,
    daily_summary: pd.DataFrame,
    time_labels: list,
    output_dir: Path,
):
    """
    生成附件5格式的result2.xlsx，包含三张表：
    1. 计划购电量：日期、144个10分钟时段、全天购电量、全天购电费。
    2. 充放电量：每天6个4小时时段的充电量和放电量。
    3. 紧急购电量：按日期汇总连续紧急购电时段和购电量。
    """

    dates = list(daily_summary["date"])
    interval_labels = [
        make_10min_interval_label(slot_index)
        for slot_index in range(len(time_labels))
    ]

    planned_purchase_wide = detail.pivot(
        index="date",
        columns="slot_index",
        values="planned_purchase_kwh",
    )
    planned_purchase_wide = planned_purchase_wide.reindex(
        index=dates,
        columns=range(len(time_labels)),
    )
    planned_purchase_wide.columns = interval_labels
    planned_purchase_wide.index.name = "日期\\时间"

    daily_lookup = daily_summary.set_index("date")
    planned_purchase_wide["全天购电量"] = daily_lookup.loc[
        planned_purchase_wide.index,
        "planned_purchase_kwh",
    ].to_numpy()
    planned_purchase_wide["全天购电费"] = daily_lookup.loc[
        planned_purchase_wide.index,
        "normal_cost_yuan",
    ].to_numpy()

    storage_table = build_storage_table(
        detail=detail,
        daily_summary=daily_summary,
    )

    emergency_table = build_emergency_purchase_table(
        detail=detail,
        dates=dates,
    )

    submission_path = output_dir / "result2.xlsx"
    sheet_names = [
        "计划购电量",
        "充放电量",
        "紧急购电量",
    ]

    with pd.ExcelWriter(
        submission_path,
        engine="openpyxl",
        date_format="yyyy/m/d",
        datetime_format="yyyy/m/d",
    ) as writer:
        planned_purchase_wide.to_excel(
            writer,
            sheet_name=sheet_names[0],
            index=True,
        )

        storage_table.to_excel(
            writer,
            sheet_name=sheet_names[1],
            index=False,
        )

        emergency_table.to_excel(
            writer,
            sheet_name=sheet_names[2],
            index=False,
        )

        format_submission_workbook(
            writer=writer,
            sheet_names=sheet_names,
        )

    return submission_path

if __name__ == "__main__":
    main()
