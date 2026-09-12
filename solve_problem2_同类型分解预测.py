"""
问题二：日前购电计划 + 日内因果 DP 执行（跨日连续版本）

关键修正：
1. 日前LP不再强制日末SOC等于日初SOC
2. 日前目标函数和日内DP均加入日末库存价值
3. 相邻日期的SOC自动连续传递
4. 日前预测改为“日总量 + 日内形状”分解
5. 负荷预测优先使用最近同类型日修正
6. 分位数由最近历史误差滚动校准

正式评价区间：
2025-02-01 至 2025-12-31

运行示例：
python solve_problem2.py
python solve_problem2.py --max-days 7
python solve_problem2.py --candidate-quantiles 0.6 0.75 0.9
python solve_problem2.py --soc-step 500

输出目录：
C:/Users/LENOVO/Desktop/数模/C题/outputs/problem2_same_type_forecast/
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


def get_date_value(
    all_dates: pd.Series | np.ndarray | list,
    index: int,
) -> pd.Timestamp:
    """兼容Series、数组和列表的日期读取。"""

    if hasattr(all_dates, "iloc"):
        value = all_dates.iloc[index]
    else:
        value = all_dates[index]

    return pd.Timestamp(value)


def make_recency_weights(
    history_indices: np.ndarray,
    day_index: int,
    recency_decay: float,
) -> np.ndarray:
    """
    生成时间衰减权重。

    距离预测日越近的历史日权重越大；
    recency_decay=1时退化为等权平均。
    """

    ages = np.maximum(
        day_index - history_indices,
        1,
    ).astype(float)
    weights = recency_decay ** (ages - 1.0)
    weights_sum = weights.sum()

    if weights_sum <= 0:
        return np.full(
            len(history_indices),
            1.0 / len(history_indices),
        )

    return weights / weights_sum


def select_similar_history_indices(
    all_dates: pd.Series | np.ndarray | list | None,
    day_index: int,
    max_scenarios: int,
    min_similar_days: int,
) -> np.ndarray:
    """
    为预测日选择最近同类型历史日。

    优先级：
    1. 同一星期几；
    2. 同为工作日或同为周末；
    3. 最近全部历史日。

    这些类型只由日期决定，因此不会使用未来真实负荷。
    """

    history_start = max(1, day_index - max_scenarios)
    history_indices = np.arange(
        history_start,
        day_index,
        dtype=int,
    )

    if len(history_indices) == 0:
        raise ValueError(
            f"第{day_index}天之前没有可用历史数据"
        )

    if all_dates is None:
        return history_indices

    current_date = get_date_value(all_dates, day_index)
    current_weekday = current_date.weekday()
    current_is_weekend = current_weekday >= 5

    same_weekday = np.array(
        [
            idx
            for idx in history_indices
            if get_date_value(all_dates, int(idx)).weekday()
            == current_weekday
        ],
        dtype=int,
    )

    if len(same_weekday) >= min_similar_days:
        return same_weekday

    same_workday_type = np.array(
        [
            idx
            for idx in history_indices
            if (
                get_date_value(all_dates, int(idx)).weekday()
                >= 5
            )
            == current_is_weekend
        ],
        dtype=int,
    )

    if len(same_workday_type) >= min_similar_days:
        return same_workday_type

    return history_indices


def normalize_profiles(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    把每天的144个时段曲线拆成日总量和日内形状。

    日内形状的每一行加总为1；日总量为0时，
    对应形状置为0，避免除零。
    """

    daily_totals = values.sum(axis=1)
    profiles = np.divide(
        values,
        daily_totals[:, None],
        out=np.zeros_like(values),
        where=daily_totals[:, None] > 1e-12,
    )

    return daily_totals, profiles


def estimate_same_type_daily_total(
    daily_totals: np.ndarray,
    history_indices: np.ndarray,
    weights: np.ndarray,
    day_index: int,
    daily_trend_clip: float,
) -> float:
    """
    用同类型日估计预测日的日总量。

    若同类型历史日足够多，则用相邻同类型日之间的
    中位增长率外推到预测日；若样本太少，则退回到
    时间衰减加权平均。
    """

    selected_totals = daily_totals[history_indices]
    weighted_average_total = float(
        np.average(
            selected_totals,
            weights=weights,
        )
    )

    if len(history_indices) < 2:
        return weighted_average_total

    growth_rates = []

    for previous_index, current_index in zip(
        history_indices[:-1],
        history_indices[1:],
    ):
        previous_total = daily_totals[previous_index]
        current_total = daily_totals[current_index]

        if previous_total <= 1e-12 or current_total <= 1e-12:
            continue

        day_gap = max(
            int(current_index - previous_index),
            1,
        )
        growth_rates.append(
            np.log(current_total / previous_total) / day_gap
        )

    if not growth_rates:
        return weighted_average_total

    median_growth = float(np.median(growth_rates))
    forecast_gap = max(
        int(day_index - history_indices[-1]),
        1,
    )
    trend_exponent = median_growth * forecast_gap

    if daily_trend_clip > 0:
        trend_exponent = float(
            np.clip(
                trend_exponent,
                -daily_trend_clip,
                daily_trend_clip,
            )
        )

    trend_total = float(
        daily_totals[history_indices[-1]]
        * np.exp(trend_exponent)
    )

    if not np.isfinite(trend_total) or trend_total < 0:
        return weighted_average_total

    return trend_total


def build_historical_scenarios(
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    all_dates: pd.Series | np.ndarray | list | None,
    day_index: int,
    max_scenarios: int,
    recency_decay: float,
    min_similar_days: int,
    daily_trend_clip: float,
    pv_recent_days: int,
    calendar_correction_enabled: bool = False,
    calendar_correction_strength: float = 0.0,
):
    """
    根据当前日前可获得的历史数据构造预测和场景。

    重要处理规则：

    1. 只能使用当前日之前的数据；
    2. 2025年1月1日不进入正式评价；
    3. 2025年1月1日没有历史预测，因此不进入残差场景库；
    4. 负荷预测拆成“日总量 + 日内形状”：
       基础样本仍由最近同星期/同类型日确定；
       月份、季节、节气位置只作为弱日总量修正；
    5. 光伏预测同样拆成“日总量 + 日内形状”，
       基础样本仍保留最近历史日，日历信息只做弱日总量修正；
    6. 历史日相对于同类型基准曲线的整日残差作为场景，
       以保留一天内144个时段的共同波动结构。
    """

    if day_index <= 1:
        raise ValueError(
            "当前日期之前没有足够历史数据，无法构造正式评价场景"
        )

    if max_scenarios <= 0:
        raise ValueError("max_scenarios必须为正数")

    if not 0.0 < recency_decay <= 1.0:
        raise ValueError("recency_decay必须大于0且不大于1")

    if min_similar_days <= 0:
        raise ValueError("min_similar_days必须为正数")

    if pv_recent_days <= 0:
        raise ValueError("pv_recent_days必须为正数")

    # =====================================================
    # 1. 负荷预测：同类型日的日总量 + 日内形状
    # =====================================================

    load_history_indices = select_similar_history_indices(
        all_dates=all_dates,
        day_index=day_index,
        max_scenarios=max_scenarios,
        min_similar_days=min_similar_days,
    )
    load_weights = make_recency_weights(
        history_indices=load_history_indices,
        day_index=day_index,
        recency_decay=recency_decay,
    )

    if len(load_history_indices) < min_similar_days:
        load_history_indices = select_similar_history_indices(
            all_dates=all_dates,
            day_index=day_index,
            max_scenarios=max_scenarios,
            min_similar_days=min_similar_days,
        )
        load_weights = make_recency_weights(
            history_indices=load_history_indices,
            day_index=day_index,
            recency_decay=recency_decay,
        )

    load_calendar_history_days = max(
        max_scenarios * 3,
        90,
    )
    pv_calendar_history_days = max(
        max_scenarios * 4,
        pv_recent_days * 4,
        120,
    )

    all_load_totals = load_energy.sum(axis=1)
    historical_load = load_energy[load_history_indices]
    historical_load_totals, load_profiles = normalize_profiles(
        historical_load
    )

    weighted_forecast_load_total = float(
        np.average(
            historical_load_totals,
            weights=load_weights,
        )
    )

    trend_forecast_load_total = estimate_same_type_daily_total(
        daily_totals=all_load_totals,
        history_indices=load_history_indices,
        weights=load_weights,
        day_index=day_index,
        daily_trend_clip=daily_trend_clip,
    )

    forecast_load_total = trend_forecast_load_total

    forecast_load_total = float(
        np.clip(
            forecast_load_total,
            0.85 * weighted_forecast_load_total,
            1.15 * weighted_forecast_load_total,
        )
    )

    baseline_load_total = float(
        np.average(
        historical_load_totals,
        weights=load_weights,
        )
    )

    forecast_load_profile = np.average(
        load_profiles,
        axis=0,
        weights=load_weights,
    )
    forecast_load_profile /= max(
        forecast_load_profile.sum(),
        1e-12,
    )
    forecast_load = (
        forecast_load_total * forecast_load_profile
    )

    baseline_load = baseline_load_total * forecast_load_profile

    if calendar_correction_enabled:
        load_calendar_factor = estimate_calendar_total_factor(
            daily_totals=all_load_totals,
            all_dates=all_dates,
            day_index=day_index,
            baseline_indices=load_history_indices,
            baseline_weights=load_weights,
            max_scenarios=max_scenarios,
            recency_decay=recency_decay,
            purpose="load",
            history_days=load_calendar_history_days,
        )
        friday_saturday_factor = (
            estimate_friday_saturday_load_factor(
                load_energy=load_energy,
                all_dates=all_dates,
                day_index=day_index,
                recency_decay=recency_decay,
                history_days=load_calendar_history_days,
                min_samples=min_similar_days,
            )
        )
        load_calendar_factor = (
            load_calendar_factor
            * friday_saturday_factor
        )
        load_calendar_factor = 1.0 + calendar_correction_strength * (
            load_calendar_factor - 1.0
        )
        forecast_load_total *= load_calendar_factor
        forecast_load = forecast_load_total * forecast_load_profile

    # =====================================================
    # 2. 光伏预测：日历相似日的日总量 + 日内形状
    # =====================================================

    pv_history_start = max(
        1,
        day_index - min(max_scenarios, pv_recent_days),
    )
    pv_history_indices = np.arange(
        pv_history_start,
        day_index,
        dtype=int,
    )
    pv_weights = make_recency_weights(
        history_indices=pv_history_indices,
        day_index=day_index,
        recency_decay=recency_decay,
    )

    if len(pv_history_indices) == 0:
        pv_history_indices, pv_weights = (
            load_history_indices,
            load_weights,
        )

    if len(pv_history_indices) == 0:
        pv_history_indices = load_history_indices
        pv_weights = load_weights

    historical_pv_for_forecast = pv_energy[pv_history_indices]
    historical_pv_totals, pv_profiles = normalize_profiles(
        historical_pv_for_forecast
    )

    weighted_forecast_pv_total = float(
        np.average(
            historical_pv_totals,
            weights=pv_weights,
        )
    )

    forecast_pv_total = weighted_forecast_pv_total

    forecast_pv_total = float(
        np.clip(
            forecast_pv_total,
            0.75 * weighted_forecast_pv_total,
            1.25 * weighted_forecast_pv_total,
        )
    )

    forecast_pv_profile = np.average(
        pv_profiles,
        axis=0,
        weights=pv_weights,
    )
    forecast_pv_profile /= max(
        forecast_pv_profile.sum(),
        1e-12,
    )
    forecast_pv = np.average(
        historical_pv_for_forecast,
        axis=0,
        weights=pv_weights,
    )

    if calendar_correction_enabled:
        pv_calendar_factor = estimate_calendar_total_factor(
            daily_totals=pv_energy.sum(axis=1),
            all_dates=all_dates,
            day_index=day_index,
            baseline_indices=pv_history_indices,
            baseline_weights=pv_weights,
            max_scenarios=max_scenarios,
            recency_decay=recency_decay,
            purpose="pv",
            history_days=pv_calendar_history_days,
        )
        pv_calendar_factor = 1.0 + calendar_correction_strength * (
            pv_calendar_factor - 1.0
        )
        forecast_pv_total *= pv_calendar_factor

    if forecast_pv_profile.sum() > 1e-12:
        forecast_pv = forecast_pv_total * forecast_pv_profile

    # =====================================================
    # 3. 场景构造：同类型历史日的整日残差
    # =====================================================

    historical_pv = pv_energy[load_history_indices]
    baseline_pv = np.average(
        historical_pv,
        axis=0,
        weights=load_weights,
    )

    load_residual = historical_load - baseline_load
    pv_residual = historical_pv - baseline_pv

    # 将历史残差叠加到当前预测上，构造未来可能出现的整日场景。
    scenario_load = np.maximum(
        forecast_load[None, :] + load_residual,
        0.0,
    )

    scenario_pv = np.maximum(
        forecast_pv[None, :] + pv_residual,
        0.0,
    )

    # 历史全部无光伏的时段视为夜间，预测和场景都强制为零。
    history_start = max(1, day_index - max_scenarios)
    recent_pv = pv_energy[history_start:day_index]
    night_slots = np.all(recent_pv <= 1e-12, axis=0)
    forecast_pv[night_slots] = 0.0
    scenario_pv[:, night_slots] = 0.0

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


def apply_rolling_forecast_bias_correction(
    forecast_load: np.ndarray,
    forecast_pv: np.ndarray,
    scenario_load: np.ndarray,
    scenario_pv: np.ndarray,
    forecast_history: list[dict],
    lookback_days: int,
    min_history_days: int,
    correction_alpha: float,
    load_error_quantile: float,
    pv_error_quantile: float,
):
    """按时段用最近历史误差分位数分别校准负荷和光伏预测。"""

    if (
        len(forecast_history) < min_history_days
        or lookback_days <= 0
    ):
        load_bias = np.zeros_like(forecast_load)
        pv_bias = np.zeros_like(forecast_pv)
    else:
        recent_history = forecast_history[-lookback_days:]
        load_bias = correction_alpha * np.quantile(
            np.stack(
                [
                    row["actual_load"] - row["forecast_load"]
                    for row in recent_history
                ],
                axis=0,
            ),
            load_error_quantile,
            axis=0,
        )
        pv_bias = correction_alpha * np.quantile(
            np.stack(
                [
                    row["actual_pv"] - row["forecast_pv"]
                    for row in recent_history
                ],
                axis=0,
            ),
            pv_error_quantile,
            axis=0,
        )

    corrected_forecast_load = np.maximum(
        forecast_load + load_bias,
        0.0,
    )
    corrected_forecast_pv = np.maximum(
        forecast_pv + pv_bias,
        0.0,
    )
    corrected_scenario_load = np.maximum(
        scenario_load + load_bias[None, :],
        0.0,
    )
    corrected_scenario_pv = np.maximum(
        scenario_pv + pv_bias[None, :],
        0.0,
    )

    return (
        corrected_forecast_load,
        corrected_forecast_pv,
        corrected_scenario_load,
        corrected_scenario_pv,
        load_bias,
        pv_bias,
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


def get_base_day_forecast(
    forecast_cache: dict[tuple, tuple] | None,
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    all_dates: pd.Series | np.ndarray | list | None,
    day_index: int,
    max_scenarios: int,
    recency_decay: float,
    min_similar_days: int,
    daily_trend_clip: float,
    pv_recent_days: int,
    calendar_correction_enabled: bool = False,
    calendar_correction_strength: float = 0.0,
):
    """
    取得某一天的基础预测和历史场景。

    这里的基础预测只使用 day_index 之前的数据。
    加缓存只是为了避免滚动回测时反复计算同一天，
    不改变可用信息范围。
    """

    cache_key = (
        int(day_index),
        bool(calendar_correction_enabled),
        round(float(calendar_correction_strength), 6),
    )

    if forecast_cache is not None and cache_key in forecast_cache:
        return forecast_cache[cache_key]

    result = build_historical_scenarios(
        load_energy=load_energy,
        pv_energy=pv_energy,
        all_dates=all_dates,
        day_index=day_index,
        max_scenarios=max_scenarios,
        recency_decay=recency_decay,
        min_similar_days=min_similar_days,
        daily_trend_clip=daily_trend_clip,
        pv_recent_days=pv_recent_days,
        calendar_correction_enabled=calendar_correction_enabled,
        calendar_correction_strength=calendar_correction_strength,
    )

    if forecast_cache is not None:
        forecast_cache[cache_key] = result

    return result


def collect_walk_forward_net_errors(
    current_day_index: int,
    all_dates: pd.Series | np.ndarray | list,
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    lookback_days: int,
    max_scenarios: int,
    recency_decay: float,
    min_similar_days: int,
    daily_trend_clip: float,
    pv_recent_days: int,
    forecast_cache: dict[tuple, tuple] | None = None,
    calendar_correction_enabled: bool = False,
    calendar_correction_strength: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """
    收集 current_day_index 之前的净负荷预测误差。

    对每个历史日 i，先用 i 之前的数据生成预测，
    再计算：

        误差 = 实际净负荷 - 预测净负荷

    因此该误差库是严格滚动的，不使用预测日之后的数据。
    """

    if lookback_days <= 0:
        return (
            np.empty((0, load_energy.shape[1]), dtype=float),
            [],
        )

    history_start = max(2, current_day_index - lookback_days)
    errors = []
    used_indices = []

    for historical_day_index in range(
        history_start,
        current_day_index,
    ):
        (
            historical_forecast_load,
            historical_forecast_pv,
            _,
            _,
            _,
        ) = get_base_day_forecast(
            forecast_cache=forecast_cache,
            load_energy=load_energy,
            pv_energy=pv_energy,
            all_dates=all_dates,
            day_index=historical_day_index,
            max_scenarios=max_scenarios,
            recency_decay=recency_decay,
            min_similar_days=min_similar_days,
            daily_trend_clip=daily_trend_clip,
            pv_recent_days=pv_recent_days,
            calendar_correction_enabled=calendar_correction_enabled,
            calendar_correction_strength=calendar_correction_strength,
        )

        historical_forecast_net = (
            historical_forecast_load
            - historical_forecast_pv
        )
        historical_actual_net = (
            load_energy[historical_day_index]
            - pv_energy[historical_day_index]
        )

        errors.append(
            historical_actual_net
            - historical_forecast_net
        )
        used_indices.append(historical_day_index)

    if not errors:
        return (
            np.empty((0, load_energy.shape[1]), dtype=float),
            [],
        )

    return np.stack(errors, axis=0), used_indices


def select_rolling_calendar_correction(
    current_day_index: int,
    all_dates: pd.Series | np.ndarray | list,
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    price: np.ndarray,
    lookback_days: int,
    min_history_days: int,
    max_scenarios: int,
    recency_decay: float,
    min_similar_days: int,
    daily_trend_clip: float,
    pv_recent_days: int,
    calendar_correction_strength: float,
    emergency_multiplier: float,
    forecast_cache: dict[tuple, tuple] | None = None,
) -> tuple[bool, str]:
    """
    用最近历史日选择是否启用弱日历修正。

    对每个历史日分别回测两种日前净负荷基础预测：

    1. 不使用月份、季节、节气和周五/周六修正；
    2. 使用指定强度的弱日历修正。

    评分采用“普通购电误差 + 临时购电误差的惩罚倍数”，
    因而会优先避免净负荷被低估，而不是只追求平均误差最小。
    当前预测日只使用当前日前已经可获得的历史日评分。
    """

    history_start = max(2, current_day_index - lookback_days)
    history_indices = list(range(history_start, current_day_index))

    if (
        lookback_days <= 0
        or len(history_indices) < min_history_days
    ):
        return False, (
            f"日历修正回测样本不足{min_history_days}天，"
            "关闭日历修正"
        )

    scores = {
        False: [],
        True: [],
    }

    for historical_day_index in history_indices:
        actual_net_load = (
            load_energy[historical_day_index]
            - pv_energy[historical_day_index]
        )
        historical_price = price[historical_day_index]

        for enabled in (False, True):
            (
                historical_forecast_load,
                historical_forecast_pv,
                _,
                _,
                _,
            ) = get_base_day_forecast(
                forecast_cache=forecast_cache,
                load_energy=load_energy,
                pv_energy=pv_energy,
                all_dates=all_dates,
                day_index=historical_day_index,
                max_scenarios=max_scenarios,
                recency_decay=recency_decay,
                min_similar_days=min_similar_days,
                daily_trend_clip=daily_trend_clip,
                pv_recent_days=pv_recent_days,
                calendar_correction_enabled=enabled,
                calendar_correction_strength=(
                    calendar_correction_strength
                ),
            )

            historical_forecast_net = (
                historical_forecast_load
                - historical_forecast_pv
            )
            over_purchase = np.maximum(
                historical_forecast_net - actual_net_load,
                0.0,
            )
            under_purchase = np.maximum(
                actual_net_load - historical_forecast_net,
                0.0,
            )
            score = np.sum(
                historical_price
                * (
                    over_purchase
                    + emergency_multiplier * under_purchase
                )
            )
            scores[enabled].append(float(score))

    baseline_score = float(np.mean(scores[False]))
    corrected_score = float(np.mean(scores[True]))
    enabled = corrected_score < baseline_score

    decision = "启用" if enabled else "关闭"
    return enabled, (
        f"日历修正回测{len(history_indices)}天："
        f"关闭={baseline_score:.0f}，"
        f"启用={corrected_score:.0f}，"
        f"当前{decision}"
    )


def build_net_load_scenarios_from_errors(
    forecast_load: np.ndarray,
    forecast_pv: np.ndarray,
    scenario_load: np.ndarray,
    scenario_pv: np.ndarray,
    net_error_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    用历史净负荷误差构造净负荷场景。

    优先使用：

        预测净负荷 + 历史净负荷预测误差

    若历史误差为空，则退回原脚本的负荷场景减光伏场景。
    """

    forecast_net_load = forecast_load - forecast_pv

    if len(net_error_history) > 0:
        scenario_net_load = (
            forecast_net_load[None, :]
            + net_error_history
        )
        source = "历史净负荷误差场景"
    else:
        scenario_net_load = scenario_load - scenario_pv
        source = "原始负荷-光伏场景"

    scenario_probability = np.full(
        scenario_net_load.shape[0],
        1.0 / scenario_net_load.shape[0],
    )

    return (
        forecast_net_load,
        scenario_net_load,
        scenario_probability,
        source,
    )


def make_target_net_load_from_error_quantile(
    forecast_net_load: np.ndarray,
    scenario_net_load: np.ndarray,
    net_error_history: np.ndarray,
    quantile: float,
    min_history_days: int,
    correction_alpha: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    由净负荷历史误差分位数生成日前LP目标净负荷。

    核心公式：

        目标净负荷 = 预测净负荷
                 + 修正强度 * Q_q(历史净负荷误差)

    若历史误差样本不足，则退回到场景净负荷的 q 分位数。
    """

    if len(net_error_history) >= min_history_days:
        net_error_quantile = np.quantile(
            net_error_history,
            quantile,
            axis=0,
        )
        net_error_adjustment = (
            correction_alpha * net_error_quantile
        )
        target_net_load = (
            forecast_net_load + net_error_adjustment
        )
        source = "净负荷历史误差分位数"
    else:
        target_net_load = np.quantile(
            scenario_net_load,
            quantile,
            axis=0,
        )
        net_error_adjustment = (
            target_net_load - forecast_net_load
        )
        source = "样本不足，使用场景净负荷分位数"

    return (
        target_net_load,
        net_error_adjustment,
        source,
    )


def make_time_period_groups(
    time_count: int,
) -> list[tuple[str, str, np.ndarray]]:
    """
    将一天划分为若干时段组，用于分时段选择净负荷风险分位数。

    对144个10分钟时段：
    0:00-6:00   夜间，负荷较平稳；
    6:00-10:00  上午，负荷与光伏同时爬升；
    10:00-16:00 中午，光伏高发，净负荷通常较低；
    16:00-20:00 傍晚，光伏快速下降且负荷仍较高；
    20:00-24:00 晚间，光伏基本为0。
    """

    if time_count != 144:
        return [
            (
                "all_day",
                "全天",
                np.arange(time_count, dtype=int),
            )
        ]

    return [
        ("night", "夜间0:00-6:00", np.arange(0, 36, dtype=int)),
        ("morning", "上午6:00-10:00", np.arange(36, 60, dtype=int)),
        ("midday", "中午10:00-16:00", np.arange(60, 96, dtype=int)),
        (
            "evening_ramp",
            "傍晚16:00-20:00",
            np.arange(96, 120, dtype=int),
        ),
        (
            "late_evening",
            "晚间20:00-24:00",
            np.arange(120, 144, dtype=int),
        ),
    ]


def format_period_quantiles(
    period_quantiles: dict[str, float],
    period_groups: list[tuple[str, str, np.ndarray]],
) -> str:
    """将分时段q整理为便于日志和结果表查看的文本。"""

    labels = {
        key: label
        for key, label, _ in period_groups
    }

    return "；".join(
        f"{labels.get(key, key)}={value:.2f}"
        for key, value in period_quantiles.items()
    )


def enforce_period_quantile_floors(
    period_quantiles: dict[str, float],
    candidate_quantiles: list[float],
    dynamic_risk_mode: str,
    normal_quantile_floor: float,
    high_risk_quantile_floor: float,
) -> dict[str, float]:
    """
    对分时段q施加少量风险下限。

    中午光伏高发，允许分位数降到候选集合下界；
    傍晚和晚间是净负荷爬坡及高位区间，高风险状态下使用较高下限。
    """

    minimum_candidate = min(candidate_quantiles)
    high_risk_periods = {
        "evening_ramp",
        "late_evening",
    }
    low_risk_periods = {
        "midday",
    }

    adjusted = {}

    for key, quantile in period_quantiles.items():
        if key in low_risk_periods:
            floor = minimum_candidate
        else:
            floor = normal_quantile_floor

        if (
            dynamic_risk_mode == "高风险"
            and key in high_risk_periods
        ):
            floor = max(
                floor,
                high_risk_quantile_floor,
            )

        adjusted[key] = max(
            quantile,
            floor,
        )

    return adjusted


def make_grouped_target_net_load_from_error_quantiles(
    forecast_net_load: np.ndarray,
    scenario_net_load: np.ndarray,
    net_error_history: np.ndarray,
    period_quantiles: dict[str, float],
    period_groups: list[tuple[str, str, np.ndarray]],
    min_history_days: int,
    correction_alpha: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    用不同时间段的q生成全天目标净负荷曲线。

    每个时段组单独计算净负荷误差分位数，再把这些片段拼回
    144个10分钟时段。这样中午可以较少保守，傍晚可以较多保守。
    """

    target_net_load = np.zeros_like(
        forecast_net_load,
        dtype=float,
    )
    net_error_adjustment = np.zeros_like(
        forecast_net_load,
        dtype=float,
    )

    use_error_quantile = (
        len(net_error_history) >= min_history_days
    )

    for key, _, slots in period_groups:
        quantile = period_quantiles[key]

        if use_error_quantile:
            period_error_quantile = np.quantile(
                net_error_history[:, slots],
                quantile,
                axis=0,
            )
            period_adjustment = (
                correction_alpha
                * period_error_quantile
            )
            target_net_load[slots] = (
                forecast_net_load[slots]
                + period_adjustment
            )
            net_error_adjustment[slots] = (
                period_adjustment
            )
        else:
            period_target = np.quantile(
                scenario_net_load[:, slots],
                quantile,
                axis=0,
            )
            target_net_load[slots] = period_target
            net_error_adjustment[slots] = (
                period_target
                - forecast_net_load[slots]
            )

    if use_error_quantile:
        source = "分时段净负荷历史误差分位数"
    else:
        source = "样本不足，使用分时段场景净负荷分位数"

    return (
        target_net_load,
        net_error_adjustment,
        source,
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


def select_rolling_optimal_quantile(
    current_day_index: int,
    all_dates: pd.Series | np.ndarray | list,
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    price: np.ndarray,
    candidate_quantiles: list[float],
    lookback_days: int,
    min_history_days: int,
    max_scenarios: int,
    recency_decay: float,
    min_similar_days: int,
    daily_trend_clip: float,
    pv_recent_days: int,
    emergency_multiplier: float,
    correction_alpha: float,
    forecast_cache: dict[tuple, tuple] | None = None,
    calendar_correction_enabled: bool = False,
    calendar_correction_strength: float = 0.0,
) -> tuple[float, str]:
    """用最近历史日净负荷预测误差滚动回测选择当前日前分位数。"""

    history_start = max(2, current_day_index - lookback_days)
    history_indices = list(range(history_start, current_day_index))

    if len(history_indices) < min_history_days:
        fallback_quantile = min(
            candidate_quantiles,
            key=lambda value: abs(value - 0.75),
        )
        return fallback_quantile, (
            f"历史样本不足{min_history_days}天，"
            f"使用q={fallback_quantile:.2f}"
        )

    scores = {quantile: [] for quantile in candidate_quantiles}
    valid_history_days = 0

    for historical_day_index in history_indices:
        (
            forecast_load,
            forecast_pv,
            scenario_load,
            scenario_pv,
            _,
        ) = get_base_day_forecast(
            forecast_cache=forecast_cache,
            load_energy=load_energy,
            pv_energy=pv_energy,
            all_dates=all_dates,
            day_index=historical_day_index,
            max_scenarios=max_scenarios,
            recency_decay=recency_decay,
            min_similar_days=min_similar_days,
            daily_trend_clip=daily_trend_clip,
            pv_recent_days=pv_recent_days,
            calendar_correction_enabled=calendar_correction_enabled,
            calendar_correction_strength=calendar_correction_strength,
        )

        (
            net_error_history,
            _,
        ) = collect_walk_forward_net_errors(
            current_day_index=historical_day_index,
            all_dates=all_dates,
            load_energy=load_energy,
            pv_energy=pv_energy,
            lookback_days=lookback_days,
            max_scenarios=max_scenarios,
            recency_decay=recency_decay,
            min_similar_days=min_similar_days,
            daily_trend_clip=daily_trend_clip,
            pv_recent_days=pv_recent_days,
            forecast_cache=forecast_cache,
            calendar_correction_enabled=calendar_correction_enabled,
            calendar_correction_strength=calendar_correction_strength,
        )

        if len(net_error_history) < min_history_days:
            continue

        valid_history_days += 1

        forecast_net_load, scenario_net_load, _, _ = (
            build_net_load_scenarios_from_errors(
                forecast_load=forecast_load,
                forecast_pv=forecast_pv,
                scenario_load=scenario_load,
                scenario_pv=scenario_pv,
                net_error_history=net_error_history,
            )
        )

        actual_net_load = (
            load_energy[historical_day_index]
            - pv_energy[historical_day_index]
        )
        historical_price = price[historical_day_index]

        for quantile in candidate_quantiles:
            target_net_load, _, _ = (
                make_target_net_load_from_error_quantile(
                    forecast_net_load=forecast_net_load,
                    scenario_net_load=scenario_net_load,
                    net_error_history=net_error_history,
                    quantile=quantile,
                    min_history_days=min_history_days,
                    correction_alpha=correction_alpha,
                )
            )
            over_purchase = np.maximum(
                target_net_load - actual_net_load,
                0.0,
            )
            under_purchase = np.maximum(
                actual_net_load - target_net_load,
                0.0,
            )
            score = np.sum(
                historical_price
                * (
                    over_purchase
                    + emergency_multiplier * under_purchase
                )
            )
            scores[quantile].append(float(score))

    if valid_history_days < min_history_days:
        fallback_quantile = min(
            candidate_quantiles,
            key=lambda value: abs(value - 0.75),
        )
        return fallback_quantile, (
            f"有效回测日不足{min_history_days}天，"
            f"使用q={fallback_quantile:.2f}"
        )

    average_scores = {
        quantile: float(np.mean(values))
        for quantile, values in scores.items()
    }
    selected_quantile = min(
        average_scores,
        key=average_scores.get,
    )
    score_text = ", ".join(
        f"q={quantile:.2f}:{average_scores[quantile]:.0f}"
        for quantile in candidate_quantiles
    )
    return selected_quantile, (
        f"滚动{len(history_indices)}天回测，"
        f"{score_text}"
    )


def select_rolling_optimal_period_quantiles(
    current_day_index: int,
    all_dates: pd.Series | np.ndarray | list,
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    price: np.ndarray,
    candidate_quantiles: list[float],
    period_groups: list[tuple[str, str, np.ndarray]],
    lookback_days: int,
    min_history_days: int,
    max_scenarios: int,
    recency_decay: float,
    min_similar_days: int,
    daily_trend_clip: float,
    pv_recent_days: int,
    emergency_multiplier: float,
    correction_alpha: float,
    forecast_cache: dict[tuple, tuple] | None = None,
    calendar_correction_enabled: bool = False,
    calendar_correction_strength: float = 0.0,
) -> tuple[dict[str, float], str]:
    """
    分时段滚动回测选择净负荷误差分位数。

    和全天统一q相比，这里为每个时段组单独选择q。
    选择依据仍是历史回测中“买多成本 + 买少的5倍惩罚成本”最低。
    """

    fallback_quantile = min(
        candidate_quantiles,
        key=lambda value: abs(value - 0.75),
    )
    fallback = {
        key: fallback_quantile
        for key, _, _ in period_groups
    }

    history_start = max(2, current_day_index - lookback_days)
    history_indices = list(range(history_start, current_day_index))

    if len(history_indices) < min_history_days:
        return fallback, (
            f"历史样本不足{min_history_days}天，"
            f"各时段使用q={fallback_quantile:.2f}"
        )

    scores = {
        key: {quantile: [] for quantile in candidate_quantiles}
        for key, _, _ in period_groups
    }
    valid_history_days = 0

    for historical_day_index in history_indices:
        (
            forecast_load,
            forecast_pv,
            scenario_load,
            scenario_pv,
            _,
        ) = get_base_day_forecast(
            forecast_cache=forecast_cache,
            load_energy=load_energy,
            pv_energy=pv_energy,
            all_dates=all_dates,
            day_index=historical_day_index,
            max_scenarios=max_scenarios,
            recency_decay=recency_decay,
            min_similar_days=min_similar_days,
            daily_trend_clip=daily_trend_clip,
            pv_recent_days=pv_recent_days,
            calendar_correction_enabled=calendar_correction_enabled,
            calendar_correction_strength=calendar_correction_strength,
        )

        (
            net_error_history,
            _,
        ) = collect_walk_forward_net_errors(
            current_day_index=historical_day_index,
            all_dates=all_dates,
            load_energy=load_energy,
            pv_energy=pv_energy,
            lookback_days=lookback_days,
            max_scenarios=max_scenarios,
            recency_decay=recency_decay,
            min_similar_days=min_similar_days,
            daily_trend_clip=daily_trend_clip,
            pv_recent_days=pv_recent_days,
            forecast_cache=forecast_cache,
            calendar_correction_enabled=calendar_correction_enabled,
            calendar_correction_strength=calendar_correction_strength,
        )

        if len(net_error_history) < min_history_days:
            continue

        valid_history_days += 1

        forecast_net_load, scenario_net_load, _, _ = (
            build_net_load_scenarios_from_errors(
                forecast_load=forecast_load,
                forecast_pv=forecast_pv,
                scenario_load=scenario_load,
                scenario_pv=scenario_pv,
                net_error_history=net_error_history,
            )
        )

        actual_net_load = (
            load_energy[historical_day_index]
            - pv_energy[historical_day_index]
        )
        historical_price = price[historical_day_index]

        for key, _, slots in period_groups:
            for quantile in candidate_quantiles:
                period_error_quantile = np.quantile(
                    net_error_history[:, slots],
                    quantile,
                    axis=0,
                )
                target_net_load = (
                    forecast_net_load[slots]
                    + correction_alpha
                    * period_error_quantile
                )

                over_purchase = np.maximum(
                    target_net_load
                    - actual_net_load[slots],
                    0.0,
                )
                under_purchase = np.maximum(
                    actual_net_load[slots]
                    - target_net_load,
                    0.0,
                )
                score = np.sum(
                    historical_price[slots]
                    * (
                        over_purchase
                        + emergency_multiplier
                        * under_purchase
                    )
                )
                scores[key][quantile].append(float(score))

    if valid_history_days < min_history_days:
        return fallback, (
            f"有效回测日不足{min_history_days}天，"
            f"各时段使用q={fallback_quantile:.2f}"
        )

    selected_quantiles = {}
    score_text_parts = []

    for key, label, _ in period_groups:
        average_scores = {
            quantile: float(np.mean(values))
            for quantile, values in scores[key].items()
        }
        selected_quantile = min(
            average_scores,
            key=average_scores.get,
        )
        selected_quantiles[key] = selected_quantile
        score_text_parts.append(
            f"{label}:q={selected_quantile:.2f}"
        )

    return selected_quantiles, (
        f"分时段滚动{valid_history_days}天回测，"
        + "，".join(score_text_parts)
    )


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
        params.soc_max + soc_step,
        soc_step,
    )

    # 裁剪到 [soc_min, soc_max] 以内：
    # 当步长不能整除 SOC 跨度时，np.arange 的末点可能略高于 soc_max，
    # 若保留会导致 DP 把 SOC 顶出允许范围，故这里剔除越界点。
    grid = grid[grid <= params.soc_max + 1e-9]

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
        "--output-subdir",
        type=str,
        default="problem2_same_type_forecast",
        help=(
            "输出到outputs下的子目录，"
            "默认problem2_same_type_forecast"
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
        "--forecast-recency-decay",
        type=float,
        default=0.90,
        help=(
            "历史日前预测的时间衰减系数；"
            "越近的日期权重越大，1.0表示等权平均，默认0.90"
        ),
    )

    parser.add_argument(
        "--min-similar-days",
        type=int,
        default=3,
        help=(
            "同类型历史日的最少样本数；"
            "同一星期几不足时自动退到工作日/周末，默认3"
        ),
    )

    parser.add_argument(
        "--daily-trend-clip",
        type=float,
        default=0.20,
        help=(
            "同类型日总量趋势外推的指数截断幅度，"
            "0.20约等于最多上/下调22%，默认0.20"
        ),
    )

    parser.add_argument(
        "--pv-recent-days",
        type=int,
        default=7,
        help="光伏日总量和日内形状预测使用的最近天数，默认7",
    )

    parser.add_argument(
        "--calendar-correction-strength",
        type=float,
        default=0.5,
        help=(
            "月份、季节、节气和周五/周六日历修正强度，"
            "0表示关闭，1表示完全采用修正，默认0.5"
        ),
    )

    parser.add_argument(
        "--calendar-rolling-lookback",
        type=int,
        default=30,
        help="日历修正滚动回测窗口，单位为天，默认30",
    )

    parser.add_argument(
        "--calendar-rolling-min-history-days",
        type=int,
        default=7,
        help="启用日历修正滚动选择所需的最少历史天数，默认7",
    )

    parser.add_argument(
        "--forecast-bias-lookback",
        type=int,
        default=30,
        help="净负荷历史误差分位数校准回看天数，默认30天",
    )

    parser.add_argument(
        "--forecast-bias-alpha",
        type=float,
        default=1.0,
        help="净负荷误差分位数修正强度，1.0表示完全修正，默认1.0",
    )

    parser.add_argument(
        "--load-error-quantile",
        type=float,
        default=0.75,
        help="常规日负荷误差修正分位数，默认0.75",
    )

    parser.add_argument(
        "--pv-error-quantile",
        type=float,
        default=0.75,
        help="常规日光伏误差修正分位数，默认0.75",
    )

    parser.add_argument(
        "--high-risk-load-error-quantile",
        type=float,
        default=0.90,
        help="高风险日负荷上偏修正分位数，默认0.90",
    )

    parser.add_argument(
        "--normal-quantile-floor",
        type=float,
        default=0.70,
        help="常规日滚动 q 下限，默认0.70",
    )

    parser.add_argument(
        "--high-risk-quantile-floor",
        type=float,
        default=0.85,
        help="高风险日滚动 q 下限，默认0.85",
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
        default=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90],
        help=(
            "滚动回测使用的日前净负荷候选分位数"
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
        default=1.00,
        help="兼容旧参数；当前净负荷误差分位数模型不再使用",
    )

    parser.add_argument(
        "--pv-safety-factor",
        type=float,
        default=1.00,
        help="兼容旧参数；当前净负荷误差分位数模型不再使用",
    )

    parser.add_argument(
        "--summer-pv-factor",
        type=float,
        default=1.00,
        help="兼容旧参数；当前净负荷误差分位数模型不再使用",
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
        "--rolling-quantile-lookback",
        type=int,
        default=30,
        help="滚动最优分位点回测窗口，单位为天",
    )

    parser.add_argument(
        "--rolling-min-history-days",
        type=int,
        default=7,
        help="启用滚动优化所需的最少历史天数",
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

    if not 0.0 < args.forecast_recency_decay <= 1.0:
        raise ValueError(
            "--forecast-recency-decay必须大于0且不大于1"
        )

    if args.min_similar_days <= 0:
        raise ValueError("--min-similar-days必须为正整数")

    if args.daily_trend_clip < 0:
        raise ValueError("--daily-trend-clip不能为负数")

    if args.pv_recent_days <= 0:
        raise ValueError("--pv-recent-days必须为正整数")

    if not 0.0 <= args.calendar_correction_strength <= 1.0:
        raise ValueError(
            "--calendar-correction-strength必须在0和1之间"
        )

    if args.calendar_rolling_lookback <= 0:
        raise ValueError(
            "--calendar-rolling-lookback必须为正整数"
        )

    if args.calendar_rolling_min_history_days <= 0:
        raise ValueError(
            "--calendar-rolling-min-history-days必须为正整数"
        )

    if (
        args.calendar_rolling_min_history_days
        > args.calendar_rolling_lookback
    ):
        raise ValueError(
            "--calendar-rolling-min-history-days不能大于"
            "--calendar-rolling-lookback"
        )

    if args.forecast_bias_lookback <= 0:
        raise ValueError("--forecast-bias-lookback必须为正整数")

    if not 0.0 <= args.forecast_bias_alpha <= 1.0:
        raise ValueError(
            "--forecast-bias-alpha必须在0和1之间"
        )

    for argument_name in (
        "load_error_quantile",
        "pv_error_quantile",
        "high_risk_load_error_quantile",
        "normal_quantile_floor",
        "high_risk_quantile_floor",
    ):
        value = getattr(args, argument_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--{argument_name.replace('_', '-')}必须在0和1之间"
            )

    normal_quantiles = normalize_quantiles(
        args.candidate_quantiles,
        "--candidate-quantiles",
    )

    high_risk_quantiles = normalize_quantiles(
        args.high_risk_quantiles,
        "--high-risk-quantiles",
    )

    if not any(
        quantile >= args.normal_quantile_floor
        for quantile in normal_quantiles
    ):
        raise ValueError("常规候选 q 中没有达到常规 q 下限的值")

    if not any(
        quantile >= args.high_risk_quantile_floor
        for quantile in high_risk_quantiles
    ) and not any(
        quantile >= args.high_risk_quantile_floor
        for quantile in normal_quantiles
    ):
        raise ValueError("候选 q 中没有达到高风险 q 下限的值")

    if args.storage_value < 0:
        raise ValueError("--storage-value不能为负数")

    if args.load_safety_factor < 1.0:
        raise ValueError("--load-safety-factor不能小于1")

    if args.pv_safety_factor < 0.0:
        raise ValueError("--pv-safety-factor不能为负数")

    if args.summer_pv_factor < 0.0:
        raise ValueError("--summer-pv-factor不能为负数")

    if args.dynamic_quantile_lookback < 0:
        raise ValueError("--dynamic-quantile-lookback不能为负数")

    if args.dynamic_emergency_threshold < 0:
        raise ValueError("--dynamic-emergency-threshold不能为负数")

    if args.dynamic_net_error_threshold < 0:
        raise ValueError("--dynamic-net-error-threshold不能为负数")

    if args.rolling_quantile_lookback <= 0:
        raise ValueError("--rolling-quantile-lookback必须为正整数")

    if args.rolling_min_history_days <= 0:
        raise ValueError("--rolling-min-history-days必须为正整数")

    if args.rolling_min_history_days > args.rolling_quantile_lookback:
        raise ValueError(
            "--rolling-min-history-days不能大于"
            "--rolling-quantile-lookback"
        )

    params = StorageParams()
    emergency_multiplier = 5.0

    base_dir = args.base_dir
    attachment_dir = base_dir / "附件"
    output_dir = (
        base_dir
        / "outputs"
        / args.output_subdir
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
        "风险修正："
        "使用净负荷历史误差分位数，"
        "不再使用独立负荷/光伏安全系数"
    )

    print(
        "预测时间衰减系数："
        f"{args.forecast_recency_decay:.3f}"
    )

    print(
        "分解预测设置："
        f"同类型日最少{args.min_similar_days}天，"
        f"日总量趋势截断{args.daily_trend_clip:.3f}，"
        f"光伏最近{args.pv_recent_days}天"
    )

    print(
        "日历修正设置："
        f"强度{args.calendar_correction_strength:.3f}，"
        f"回看{args.calendar_rolling_lookback}天，"
        f"最少历史{args.calendar_rolling_min_history_days}天"
    )

    print(
        "净负荷误差校准："
        f"回看{args.forecast_bias_lookback}天，"
        f"强度{args.forecast_bias_alpha:.3f}"
    )

    print(
        "动态分位数："
        f"常规{normal_quantiles}，"
        f"高风险{high_risk_quantiles}"
    )

    print(
        "滚动最优分位数："
        f"回看{args.rolling_quantile_lookback}天，"
        f"最少历史{args.rolling_min_history_days}天，"
        f"临时购电惩罚×{emergency_multiplier:.1f}"
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

    period_groups = make_time_period_groups(
        len(fixed_price)
    )

    print(
        "分时段q分组："
        + "；".join(
            label
            for _, label, _ in period_groups
        )
    )

    # 2月1日没有执行1月1日的控制策略，
    # 因此正式评价期起始SOC采用题目给定初始值6000kWh。
    soc_start = params.soc_initial

    detail_rows = []
    daily_rows = []
    forecast_history = []
    forecast_cache = {}

    # =====================================================
    # 5. 模型求解
    # =====================================================

    for day_index in formal_indices:

        current_date = all_dates[day_index]

        # 恢复到19:18版本：基础预测不加入月份、季节、节气以及
        # 周五/周六的日历修正，只保留同类型日分解预测和净负荷
        # 分时段滚动分位数。日历修正函数保留在脚本中，便于后续
        # 对照试验，但不参与本次正式评价。
        calendar_correction_enabled = False
        calendar_reason = "本次评价关闭日历修正，使用19:18基准版本"

        (
            forecast_load,
            forecast_pv,
            scenario_load,
            scenario_pv,
            scenario_probability,
        ) = get_base_day_forecast(
            forecast_cache=forecast_cache,
            load_energy=load_energy,
            pv_energy=pv_energy,
            all_dates=all_dates,
            day_index=int(day_index),
            max_scenarios=args.max_scenarios,
            recency_decay=args.forecast_recency_decay,
            min_similar_days=args.min_similar_days,
            daily_trend_clip=args.daily_trend_clip,
            pv_recent_days=args.pv_recent_days,
            calendar_correction_enabled=(
                calendar_correction_enabled
            ),
            calendar_correction_strength=(
                args.calendar_correction_strength
            ),
        )

        dynamic_candidate_quantiles, dynamic_risk_mode, dynamic_risk_reason = (
            select_dynamic_quantiles(
                recent_daily_rows=daily_rows,
                normal_quantiles=normal_quantiles,
                high_risk_quantiles=high_risk_quantiles,
                lookback_days=args.dynamic_quantile_lookback,
                emergency_threshold_kwh=(
                    args.dynamic_emergency_threshold
                ),
                net_error_threshold_kwh=(
                    args.dynamic_net_error_threshold
                ),
            )
        )

        (
            net_error_history,
            net_error_history_indices,
        ) = collect_walk_forward_net_errors(
            current_day_index=int(day_index),
            all_dates=all_dates,
            load_energy=load_energy,
            pv_energy=pv_energy,
            lookback_days=args.forecast_bias_lookback,
            max_scenarios=args.max_scenarios,
            recency_decay=args.forecast_recency_decay,
            min_similar_days=args.min_similar_days,
            daily_trend_clip=args.daily_trend_clip,
            pv_recent_days=args.pv_recent_days,
            forecast_cache=forecast_cache,
            calendar_correction_enabled=(
                calendar_correction_enabled
            ),
            calendar_correction_strength=(
                args.calendar_correction_strength
            ),
        )

        calibrated_forecast_load = forecast_load.copy()
        calibrated_forecast_pv = forecast_pv.copy()

        (
            forecast_net_load,
            scenario_net_load,
            scenario_probability,
            scenario_source,
        ) = build_net_load_scenarios_from_errors(
            forecast_load=forecast_load,
            forecast_pv=forecast_pv,
            scenario_load=scenario_load,
            scenario_pv=scenario_pv,
            net_error_history=net_error_history,
        )

        period_candidate_quantiles = normalize_quantiles(
            normal_quantiles
            + (
                high_risk_quantiles
                if dynamic_risk_mode == "高风险"
                else []
            ),
            "--period-candidate-quantiles",
        )

        period_quantiles, rolling_reason = (
            select_rolling_optimal_period_quantiles(
                current_day_index=int(day_index),
                all_dates=all_dates,
                load_energy=load_energy,
                pv_energy=pv_energy,
                price=price,
                candidate_quantiles=period_candidate_quantiles,
                period_groups=period_groups,
                lookback_days=args.rolling_quantile_lookback,
                min_history_days=args.rolling_min_history_days,
                max_scenarios=args.max_scenarios,
                recency_decay=args.forecast_recency_decay,
                min_similar_days=args.min_similar_days,
                daily_trend_clip=args.daily_trend_clip,
                pv_recent_days=args.pv_recent_days,
                emergency_multiplier=emergency_multiplier,
                correction_alpha=args.forecast_bias_alpha,
                forecast_cache=forecast_cache,
                calendar_correction_enabled=(
                    calendar_correction_enabled
                ),
                calendar_correction_strength=(
                    args.calendar_correction_strength
                ),
            )
        )

        period_quantiles = enforce_period_quantile_floors(
            period_quantiles=period_quantiles,
            candidate_quantiles=period_candidate_quantiles,
            dynamic_risk_mode=dynamic_risk_mode,
            normal_quantile_floor=args.normal_quantile_floor,
            high_risk_quantile_floor=(
                args.high_risk_quantile_floor
            ),
        )

        selected_period_quantiles = format_period_quantiles(
            period_quantiles=period_quantiles,
            period_groups=period_groups,
        )
        selected_rolling_quantile = float(
            np.mean(list(period_quantiles.values()))
        )
        risk_mode = (
            "分时段高风险"
            if dynamic_risk_mode == "高风险"
            else "分时段常规"
        )
        risk_reason = (
            f"{rolling_reason}；动态判断：{dynamic_risk_reason}"
        )

        (
            target_net_load,
            net_error_adjustment,
            target_source,
        ) = make_grouped_target_net_load_from_error_quantiles(
            forecast_net_load=forecast_net_load,
            scenario_net_load=scenario_net_load,
            net_error_history=net_error_history,
            period_quantiles=period_quantiles,
            period_groups=period_groups,
            min_history_days=args.rolling_min_history_days,
            correction_alpha=args.forecast_bias_alpha,
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

        best_solution = {
            "quantile": selected_rolling_quantile,
            "period_quantiles": period_quantiles,
            "selected_period_quantiles": (
                selected_period_quantiles
            ),
            "lp_solution": lp_solution,
            "future_value": future_value,
            "target_net_load_sum": float(
                np.sum(target_net_load)
            ),
            "net_error_adjustment_sum": float(
                np.sum(net_error_adjustment)
            ),
            "target_source": target_source,
            "estimated_total_cost": float(
                estimated_total_cost
            ),
        }

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
        forecast_net_load_sum = float(
            np.sum(forecast_net_load)
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
                "calendar_correction_enabled": (
                    calendar_correction_enabled
                ),
                "calendar_correction_reason": (
                    calendar_reason
                ),
                "risk_mode": risk_mode,
                "risk_reason": risk_reason,
                "active_quantiles": best_solution[
                    "selected_period_quantiles"
                ],
                "selected_quantile": best_solution[
                    "quantile"
                ],
                "selected_period_quantiles": (
                    best_solution[
                        "selected_period_quantiles"
                    ]
                ),
                "scenario_count": len(
                    scenario_probability
                ),
                "forecast_load_kwh": forecast_load_sum,
                "forecast_pv_kwh": forecast_pv_sum,
                "forecast_net_load_kwh": (
                    forecast_net_load_sum
                ),
                "load_bias_correction_kwh": 0.0,
                "pv_bias_correction_kwh": 0.0,
                "net_error_correction_kwh": (
                    best_solution[
                        "net_error_adjustment_sum"
                    ]
                ),
                "net_error_history_days": len(
                    net_error_history_indices
                ),
                "scenario_source": scenario_source,
                "target_source": best_solution[
                    "target_source"
                ],
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

        forecast_history.append(
            {
                "forecast_load": calibrated_forecast_load,
                "forecast_pv": calibrated_forecast_pv,
                "actual_load": load_energy[day_index].copy(),
                "actual_pv": pv_energy[day_index].copy(),
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
            f"q均值={best_solution['quantile']:.2f} | "
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
