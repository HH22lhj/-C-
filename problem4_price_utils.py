# -*- coding: utf-8 -*-
"""问题四电价读取与因果预测工具。

本模块只处理附件4的电价矩阵。核心原则是：
1. 预测未来电价时只能使用当前决策时刻以前已经可获得的数据；
2. 附件4真实电价只用于历史回测、当天已发生价格修正和最终结算；
3. 主预测模型采用“最近同星期同槽衰减加权”。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


SLOTS_PER_DAY = 144
SLOT_MINUTES = 10
YEAR_START = pd.Timestamp("2025-01-01 00:00:00")
YEAR_END = pd.Timestamp("2026-01-01 00:00:00")


@dataclass(frozen=True)
class PriceForecastInfo:
    """一次电价预测的审计信息。"""

    selected_decay: float
    selection_reason: str
    intraday_bias: float
    residual_decay_slots: float
    max_same_week_days: int
    fallback_days: int
    selected_alpha: float | None = None
    model_name: str = "same_week_decay"


RIDGE_START_LAG_DAYS = 14
DEFAULT_RIDGE_ALPHA = 0.5
_RIDGE_MODEL_CACHE: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}


def read_price_matrix(
    path: Path,
    sheet_name: str | int = 0,
    expected_slots: int = SLOTS_PER_DAY,
) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    """读取日期为行、10分钟时段为列的附件4电价矩阵。"""

    raw = pd.read_excel(path, sheet_name=sheet_name)
    if raw.shape[1] < expected_slots + 1:
        raise ValueError(
            f"{path.name} 的列数不足，至少需要1列日期和{expected_slots}列电价"
        )

    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0], errors="coerce"))
    if dates.isna().any():
        raise ValueError(f"{path.name} 中存在无法识别的日期")

    time_labels = [str(value) for value in raw.columns[1 : expected_slots + 1]]
    price = (
        raw.iloc[:, 1 : expected_slots + 1]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    validate_price_matrix(price, context=path.name)
    return dates, time_labels, price


def validate_price_matrix(price: np.ndarray, context: str = "电价矩阵") -> None:
    """检查电价是否为365×144非负有限矩阵。"""

    if price.ndim != 2 or price.shape[1] != SLOTS_PER_DAY:
        raise ValueError(f"{context}应为二维矩阵且每天144个时段，实际为{price.shape}")
    if np.isnan(price).any():
        raise ValueError(f"{context}存在缺失或非数值数据")
    if not np.isfinite(price).all():
        raise ValueError(f"{context}存在NaN或无穷值")
    if (price < 0).any():
        raise ValueError(f"{context}存在负电价")


def day_slot_for_target_time(target_time: pd.Timestamp) -> tuple[date, int]:
    """把交付结束时刻映射到日期和日内槽位。

    例如 2025-01-01 00:10 对应 2025-01-01 第0槽；
    2026-01-01 00:00 对应 2025-12-31 第143槽。
    """

    target_time = pd.Timestamp(target_time)
    minutes = target_time.hour * 60 + target_time.minute
    if minutes == 0:
        return (target_time - pd.Timedelta(days=1)).date(), SLOTS_PER_DAY - 1
    return target_time.date(), minutes // SLOT_MINUTES - 1


def _date_to_index(dates: pd.DatetimeIndex) -> dict[date, int]:
    return {pd.Timestamp(value).date(): idx for idx, value in enumerate(dates)}


def actual_price_for_target(
    price_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    target_time: pd.Timestamp,
) -> float:
    """读取一个交付时刻对应的真实电价。"""

    day, slot = day_slot_for_target_time(target_time)
    lookup = _date_to_index(dates)
    if day not in lookup:
        raise ValueError(f"附件4中不存在目标日期：{day}")
    return float(price_matrix[lookup[day], slot])


def make_same_week_price_forecast(
    price_matrix: np.ndarray,
    target_day_index: int,
    decay: float,
    max_same_week_days: int = 5,
    fallback_days: int = 7,
    latest_allowed_index: int | None = None,
) -> np.ndarray:
    """用最近同星期同槽历史价格预测某一天144个时段价格。

    target_day_index 表示要预测的日期下标。latest_allowed_index 表示在当前
    决策时刻已经完整可用的最后一个历史日下标。日内滚动预测未来日期时，
    当前日还未完整结束，因此不能把当前日作为完整历史日使用。
    """

    if not 0.0 < decay <= 1.0:
        raise ValueError("decay必须大于0且不大于1")
    if target_day_index <= 0:
        raise ValueError("预测第0天没有历史电价可用")

    latest = target_day_index - 1
    if latest_allowed_index is not None:
        latest = min(latest, int(latest_allowed_index))

    if latest < 0:
        raise ValueError("没有可用历史电价")

    same_week_indices: list[int] = []
    for lag_week in range(1, max_same_week_days + 1):
        idx = target_day_index - 7 * lag_week
        if 0 <= idx <= latest:
            same_week_indices.append(idx)

    if same_week_indices:
        weights = decay ** np.arange(len(same_week_indices), dtype=float)
        return np.average(price_matrix[same_week_indices], axis=0, weights=weights)

    fallback_start = max(0, latest - fallback_days + 1)
    fallback = price_matrix[fallback_start : latest + 1]
    if fallback.size == 0:
        raise ValueError("同星期和最近历史回退样本均为空")
    return fallback.mean(axis=0)


def select_price_decay_by_walk_forward(
    price_matrix: np.ndarray,
    current_day_index: int,
    candidate_decays: list[float],
    lookback_days: int = 35,
    min_history_days: int = 7,
    max_same_week_days: int = 5,
    fallback_days: int = 7,
) -> tuple[float, str, dict[float, float]]:
    """用当前日前的历史回测选择同星期预测衰减系数lambda。"""

    candidates = sorted({float(value) for value in candidate_decays})
    if not candidates:
        raise ValueError("candidate_decays不能为空")
    if any(value <= 0.0 or value > 1.0 for value in candidates):
        raise ValueError("candidate_decays必须位于(0, 1]区间")

    start = max(1, current_day_index - lookback_days)
    history_indices = list(range(start, current_day_index))
    if len(history_indices) < min_history_days:
        fallback = min(candidates, key=lambda value: abs(value - 0.90))
        return fallback, f"价格回测样本不足{min_history_days}天，使用lambda={fallback:.2f}", {}

    scores = {value: [] for value in candidates}
    valid_days = 0
    for day_index in history_indices:
        if day_index <= 0:
            continue
        actual = price_matrix[day_index]
        valid_days += 1
        for decay in candidates:
            predicted = make_same_week_price_forecast(
                price_matrix=price_matrix,
                target_day_index=day_index,
                decay=decay,
                max_same_week_days=max_same_week_days,
                fallback_days=fallback_days,
                latest_allowed_index=day_index - 1,
            )
            scores[decay].append(float(np.mean(np.abs(predicted - actual))))

    if valid_days < min_history_days:
        fallback = min(candidates, key=lambda value: abs(value - 0.90))
        return fallback, f"价格有效回测日不足{min_history_days}天，使用lambda={fallback:.2f}", {}

    average_scores = {
        decay: float(np.mean(values))
        for decay, values in scores.items()
        if values
    }
    selected = min(average_scores, key=average_scores.get)
    compact = ", ".join(f"{decay:.2f}:{average_scores[decay]:.4f}" for decay in candidates)
    reason = f"最近{valid_days}天价格MAE回测，{compact}，选择lambda={selected:.2f}"
    return float(selected), reason, average_scores


def _ridge_feature_matrix_for_day(
    price_history: np.ndarray,
    dates: pd.DatetimeIndex,
    day_index: int,
) -> np.ndarray:
    """构造Ridge残差模型在某一天的144个槽位特征。"""

    if day_index < RIDGE_START_LAG_DAYS:
        raise ValueError(f"Ridge残差模型至少需要{RIDGE_START_LAG_DAYS}天历史")

    n_days, n_slots = price_history.shape
    if n_slots != SLOTS_PER_DAY:
        raise ValueError("电价矩阵每天必须包含144个10分钟时段")
    if day_index >= n_days:
        raise ValueError(f"day_index={day_index}超出电价矩阵范围")

    daily_mean = price_history.mean(axis=1)
    daily_std = price_history.std(axis=1)
    daily_min = price_history.min(axis=1)
    daily_max = price_history.max(axis=1)

    current_date = pd.Timestamp(dates[day_index])
    dow = current_date.weekday()
    doy = current_date.dayofyear
    prev7_mean = daily_mean[day_index - 7 : day_index].mean()
    prev7_std = daily_mean[day_index - 7 : day_index].std()
    week_level_diff = daily_mean[day_index - 1] - daily_mean[day_index - 8]

    features = np.zeros((n_slots, 30), dtype=float)
    for slot in range(n_slots):
        left = max(0, slot - 1)
        right = min(n_slots, slot + 2)
        features[slot] = [
            price_history[day_index - 1, slot],
            price_history[day_index - 2, slot],
            price_history[day_index - 3, slot],
            price_history[day_index - 7, slot],
            price_history[day_index - 8, slot],
            price_history[day_index - 9, slot],
            price_history[day_index - 14, slot],
            price_history[day_index - 3 : day_index, slot].mean(),
            price_history[day_index - 7 : day_index, slot].mean(),
            price_history[day_index - 14 : day_index, slot].mean(),
            price_history[day_index - 7 : day_index, slot].std(),
            price_history[day_index - 14 : day_index, slot].std(),
            price_history[day_index - 1, slot] - price_history[day_index - 8, slot],
            price_history[day_index - 2, slot] - price_history[day_index - 9, slot],
            price_history[day_index - 7, slot] - price_history[day_index - 14, slot],
            price_history[day_index - 1, left:right].mean(),
            price_history[day_index - 7, left:right].mean(),
            daily_mean[day_index - 1],
            daily_std[day_index - 1],
            daily_min[day_index - 1],
            daily_max[day_index - 1],
            prev7_mean,
            prev7_std,
            week_level_diff,
            np.sin(2 * np.pi * slot / n_slots),
            np.cos(2 * np.pi * slot / n_slots),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365),
            np.cos(2 * np.pi * doy / 365),
        ]
    return features


def _build_ridge_training_data(
    price_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    max_train_day_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成训练样本，目标为 p[d,t] - p[d-7,t]。"""

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for day_index in range(RIDGE_START_LAG_DAYS, max_train_day_index + 1):
        rows.append(_ridge_feature_matrix_for_day(price_matrix, dates, day_index))
        targets.append(price_matrix[day_index] - price_matrix[day_index - 7])

    if not rows:
        return np.empty((0, 30), dtype=float), np.empty(0, dtype=float)
    return np.vstack(rows), np.concatenate(targets)


def _fit_ridge_residual_model(
    price_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    max_train_day_index: int,
    alpha: float,
    min_train_days: int = 7,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    """用闭式解训练标准化Ridge模型，不依赖sklearn。"""

    if alpha <= 0:
        raise ValueError("Ridge alpha必须为正")
    train_day_count = max_train_day_index - RIDGE_START_LAG_DAYS + 1
    if train_day_count < min_train_days:
        return None

    cache_key = (id(price_matrix), int(max_train_day_index), float(alpha))
    cached = _RIDGE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    x_train, residual_train = _build_ridge_training_data(
        price_matrix=price_matrix,
        dates=dates,
        max_train_day_index=max_train_day_index,
    )
    if x_train.size == 0:
        return None

    mean_x = x_train.mean(axis=0)
    std_x = x_train.std(axis=0)
    std_x[std_x < 1e-12] = 1.0
    z_train = (x_train - mean_x) / std_x

    mean_y = float(residual_train.mean())
    centered_y = residual_train - mean_y
    lhs = z_train.T @ z_train
    lhs.flat[:: lhs.shape[0] + 1] += float(alpha)
    rhs = z_train.T @ centered_y
    beta = np.linalg.solve(lhs, rhs)

    model = (mean_x, std_x, mean_y, beta)
    _RIDGE_MODEL_CACHE[cache_key] = model
    return model


def _predict_ridge_residual(
    model: tuple[np.ndarray, np.ndarray, float, np.ndarray],
    x_test: np.ndarray,
) -> np.ndarray:
    mean_x, std_x, mean_y, beta = model
    return mean_y + ((x_test - mean_x) / std_x) @ beta


def _fallback_decay(candidate_decays: list[float] | None, default: float = 0.90) -> float:
    if not candidate_decays:
        return default
    valid = [float(value) for value in candidate_decays if 0.0 < float(value) <= 1.0]
    if not valid:
        return default
    return min(valid, key=lambda value: abs(value - default))


def _forecast_one_day_ridge_residual(
    price_history: np.ndarray,
    dates: pd.DatetimeIndex,
    day_index: int,
    model: tuple[np.ndarray, np.ndarray, float, np.ndarray] | None,
    fallback_decay: float,
    latest_allowed_index: int,
    max_same_week_days: int,
    fallback_days: int,
) -> np.ndarray:
    """预测一天电价；历史不足时回退到同星期加权预测。"""

    if model is None or day_index < RIDGE_START_LAG_DAYS:
        return make_same_week_price_forecast(
            price_matrix=price_history,
            target_day_index=day_index,
            decay=fallback_decay,
            max_same_week_days=max_same_week_days,
            fallback_days=fallback_days,
            latest_allowed_index=latest_allowed_index,
        )

    features = _ridge_feature_matrix_for_day(price_history, dates, day_index)
    residual_hat = _predict_ridge_residual(model, features)
    forecast = features[:, 3] + residual_hat
    return np.maximum(forecast, 0.0)


def day_ahead_ridge_residual_price_forecast(
    price_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    day_index: int,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    candidate_decays: list[float] | None = None,
    min_train_days: int = 7,
    max_same_week_days: int = 5,
    fallback_days: int = 7,
) -> tuple[np.ndarray, PriceForecastInfo]:
    """问题4-2的Ridge残差日前电价预测。"""

    latest_complete_day_index = day_index - 1
    fallback_decay = _fallback_decay(candidate_decays)
    model = _fit_ridge_residual_model(
        price_matrix=price_matrix,
        dates=dates,
        max_train_day_index=latest_complete_day_index,
        alpha=ridge_alpha,
        min_train_days=min_train_days,
    )
    forecast = _forecast_one_day_ridge_residual(
        price_history=price_matrix,
        dates=dates,
        day_index=day_index,
        model=model,
        fallback_decay=fallback_decay,
        latest_allowed_index=latest_complete_day_index,
        max_same_week_days=max_same_week_days,
        fallback_days=fallback_days,
    )
    if model is None:
        reason = (
            f"Ridge训练样本不足{min_train_days}天，回退为同星期同槽加权，"
            f"lambda={fallback_decay:.2f}"
        )
    else:
        reason = (
            "Ridge残差模型：先取上周同日同槽价格p[d-7,t]，"
            f"再用历史样本预测残差，alpha={ridge_alpha:g}，"
            f"训练截止第{latest_complete_day_index}日"
        )
    return forecast, PriceForecastInfo(
        selected_decay=float(fallback_decay),
        selection_reason=reason,
        intraday_bias=0.0,
        residual_decay_slots=0.0,
        max_same_week_days=int(max_same_week_days),
        fallback_days=int(fallback_days),
        selected_alpha=float(ridge_alpha),
        model_name="ridge_residual",
    )


def forecast_price_for_targets_ridge_residual(
    price_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    decision_time: pd.Timestamp,
    targets: pd.DatetimeIndex,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    candidate_decays: list[float] | None = None,
    min_train_days: int = 7,
    max_same_week_days: int = 5,
    fallback_days: int = 7,
    residual_decay_slots: float = 36.0,
) -> tuple[np.ndarray, PriceForecastInfo]:
    """问题4-3的Ridge残差滚动电价预测。

    对当前日以后的日子，若特征需要当天完整价格，则用已经预测出的
    当天曲线递推填充；当天已经发生的真实价格会覆盖对应槽位。
    """

    decision_time = pd.Timestamp(decision_time)
    lookup = _date_to_index(dates)
    decision_day = decision_time.date()
    if decision_day not in lookup:
        raise ValueError(f"附件4中不存在决策日期：{decision_day}")

    target_pairs = [day_slot_for_target_time(pd.Timestamp(target)) for target in targets]
    missing_days = [target_day for target_day, _ in target_pairs if target_day not in lookup]
    if missing_days:
        raise ValueError(f"附件4中不存在目标日期：{missing_days[0]}")

    current_day_index = lookup[decision_day]
    decision_slot = (decision_time.hour * 60 + decision_time.minute) // SLOT_MINUTES
    latest_complete_day_index = current_day_index - 1
    max_target_day_index = max(lookup[target_day] for target_day, _ in target_pairs)
    fallback_decay = _fallback_decay(candidate_decays)

    model = _fit_ridge_residual_model(
        price_matrix=price_matrix,
        dates=dates,
        max_train_day_index=latest_complete_day_index,
        alpha=ridge_alpha,
        min_train_days=min_train_days,
    )

    working_price = price_matrix.copy()
    daily_forecasts: dict[int, np.ndarray] = {}
    for day_index in range(current_day_index, max_target_day_index + 1):
        day_forecast = _forecast_one_day_ridge_residual(
            price_history=working_price,
            dates=dates,
            day_index=day_index,
            model=model,
            fallback_decay=fallback_decay,
            latest_allowed_index=max(day_index - 1, latest_complete_day_index),
            max_same_week_days=max_same_week_days,
            fallback_days=fallback_days,
        )
        daily_forecasts[day_index] = day_forecast
        working_price[day_index] = day_forecast
        if day_index == current_day_index and decision_slot > 0:
            working_price[day_index, :decision_slot] = price_matrix[day_index, :decision_slot]

    intraday_bias = 0.0
    if decision_slot > 0 and current_day_index in daily_forecasts:
        observed_residual = (
            price_matrix[current_day_index, :decision_slot]
            - daily_forecasts[current_day_index][:decision_slot]
        )
        if observed_residual.size:
            intraday_bias = float(np.median(observed_residual))

    forecast = np.zeros(len(targets), dtype=float)
    for position, target in enumerate(targets):
        target_day, target_slot = target_pairs[position]
        target_day_index = lookup[target_day]
        base_value = daily_forecasts[target_day_index][target_slot]
        lead_slots = max(
            int((pd.Timestamp(target) - decision_time).total_seconds() // (SLOT_MINUTES * 60)),
            1,
        )
        if decision_slot > 0 and residual_decay_slots > 0:
            base_value += intraday_bias * np.exp(-lead_slots / residual_decay_slots)
        forecast[position] = max(float(base_value), 0.0)

    if model is None:
        reason = (
            f"Ridge训练样本不足{min_train_days}天，回退为同星期同槽加权，"
            f"lambda={fallback_decay:.2f}"
        )
    else:
        reason = (
            "Ridge残差模型滚动预测：p_hat=p[d-7,t]+残差预测；"
            f"alpha={ridge_alpha:g}；训练截止前一完整日；"
            "跨日目标用预测曲线递推构造可用历史特征"
        )
    if decision_slot > 0:
        reason += f"；日内已发生价格残差中位数={intraday_bias:.4f}"

    info = PriceForecastInfo(
        selected_decay=float(fallback_decay),
        selection_reason=reason,
        intraday_bias=float(intraday_bias),
        residual_decay_slots=float(residual_decay_slots),
        max_same_week_days=int(max_same_week_days),
        fallback_days=int(fallback_days),
        selected_alpha=float(ridge_alpha),
        model_name="ridge_residual",
    )
    return forecast, info


def forecast_price_for_targets(
    price_matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    decision_time: pd.Timestamp,
    targets: pd.DatetimeIndex,
    candidate_decays: list[float],
    price_lookback_days: int = 35,
    min_price_history_days: int = 7,
    max_same_week_days: int = 5,
    fallback_days: int = 7,
    residual_decay_slots: float = 36.0,
) -> tuple[np.ndarray, PriceForecastInfo]:
    """预测任意决策时刻之后若干10分钟目标时刻的电价。

    0:00只使用历史完整日期。6/12/18点以后，允许使用当天已经发生
    的价格误差对未来短期价格作指数衰减修正。
    """

    decision_time = pd.Timestamp(decision_time)
    lookup = _date_to_index(dates)
    decision_day = decision_time.date()
    if decision_day not in lookup:
        raise ValueError(f"附件4中不存在决策日期：{decision_day}")

    current_day_index = lookup[decision_day]
    decision_slot = (decision_time.hour * 60 + decision_time.minute) // SLOT_MINUTES
    latest_complete_day_index = current_day_index - 1

    selected_decay, reason, _ = select_price_decay_by_walk_forward(
        price_matrix=price_matrix,
        current_day_index=current_day_index,
        candidate_decays=candidate_decays,
        lookback_days=price_lookback_days,
        min_history_days=min_price_history_days,
        max_same_week_days=max_same_week_days,
        fallback_days=fallback_days,
    )

    baseline_cache: dict[int, np.ndarray] = {}

    def baseline_for_day(day_index: int) -> np.ndarray:
        if day_index not in baseline_cache:
            baseline_cache[day_index] = make_same_week_price_forecast(
                price_matrix=price_matrix,
                target_day_index=day_index,
                decay=selected_decay,
                max_same_week_days=max_same_week_days,
                fallback_days=fallback_days,
                latest_allowed_index=latest_complete_day_index,
            )
        return baseline_cache[day_index]

    intraday_bias = 0.0
    if decision_slot > 0:
        baseline_today = baseline_for_day(current_day_index)
        observed_residual = (
            price_matrix[current_day_index, :decision_slot]
            - baseline_today[:decision_slot]
        )
        if observed_residual.size:
            intraday_bias = float(np.median(observed_residual))

    forecast = np.zeros(len(targets), dtype=float)
    for position, target in enumerate(targets):
        target_day, target_slot = day_slot_for_target_time(pd.Timestamp(target))
        if target_day not in lookup:
            raise ValueError(f"附件4中不存在目标日期：{target_day}")
        target_day_index = lookup[target_day]
        base_value = baseline_for_day(target_day_index)[target_slot]

        lead_slots = max(
            int((pd.Timestamp(target) - decision_time).total_seconds() // (SLOT_MINUTES * 60)),
            1,
        )
        if decision_slot > 0 and residual_decay_slots > 0:
            base_value += intraday_bias * np.exp(-lead_slots / residual_decay_slots)
        forecast[position] = max(float(base_value), 0.0)

    info = PriceForecastInfo(
        selected_decay=float(selected_decay),
        selection_reason=reason,
        intraday_bias=float(intraday_bias),
        residual_decay_slots=float(residual_decay_slots),
        max_same_week_days=int(max_same_week_days),
        fallback_days=int(fallback_days),
    )
    return forecast, info


def day_ahead_price_forecast(
    price_matrix: np.ndarray,
    day_index: int,
    candidate_decays: list[float],
    price_lookback_days: int = 35,
    min_price_history_days: int = 7,
    max_same_week_days: int = 5,
    fallback_days: int = 7,
) -> tuple[np.ndarray, PriceForecastInfo]:
    """问题4-2的0:00日前电价预测。"""

    selected_decay, reason, _ = select_price_decay_by_walk_forward(
        price_matrix=price_matrix,
        current_day_index=day_index,
        candidate_decays=candidate_decays,
        lookback_days=price_lookback_days,
        min_history_days=min_price_history_days,
        max_same_week_days=max_same_week_days,
        fallback_days=fallback_days,
    )
    forecast = make_same_week_price_forecast(
        price_matrix=price_matrix,
        target_day_index=day_index,
        decay=selected_decay,
        max_same_week_days=max_same_week_days,
        fallback_days=fallback_days,
        latest_allowed_index=day_index - 1,
    )
    return forecast, PriceForecastInfo(
        selected_decay=float(selected_decay),
        selection_reason=reason,
        intraday_bias=0.0,
        residual_decay_slots=0.0,
        max_same_week_days=int(max_same_week_days),
        fallback_days=int(fallback_days),
    )
