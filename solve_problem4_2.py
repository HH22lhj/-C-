# -*- coding: utf-8 -*-
"""问题4-2：波动电价下的日前购电计划。

代码流程：
1. 读取附件2负荷/光伏和附件4真实电价；
2. 用问题二最终版的同类型日分解模型预测净负荷；
3. 用“上周同日同槽基准 + Ridge残差修正”预测电价，优化时只用预测电价；
4. 用预测净负荷误差的分时段滚动分位数生成目标净负荷；
5. 求解日前LP，再用因果DP执行储能动作；
6. 最终费用用附件4真实电价结算，导出result4-2.xlsx和审计明细。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from problem4_price_utils import (
    DEFAULT_RIDGE_ALPHA,
    day_ahead_ridge_residual_price_forecast,
    read_price_matrix,
)


def _load_problem2_module():
    """按文件路径加载问题二最终版脚本，避免中文文件名导入失败。"""

    module_path = Path(__file__).with_name("solve_problem2_同类型分解预测.py")
    spec = importlib.util.spec_from_file_location("problem2_same_type", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载问题二脚本：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p2 = _load_problem2_module()


def execute_one_day_with_price_forecast(
    planned_purchase: np.ndarray,
    actual_net_load: np.ndarray,
    forecast_price: np.ndarray,
    actual_price: np.ndarray,
    soc_start: float,
    params,
    grid: np.ndarray,
    psi: np.ndarray,
    feasible: np.ndarray,
    future_value: np.ndarray,
    emergency_multiplier: float,
) -> dict:
    """固定日前购电后执行储能。

    动作选择使用预测电价，避免提前知道未来真实电价；费用核算使用
    附件4真实电价。
    """

    time_count = len(planned_purchase)
    soc_index = p2.nearest_grid_index(grid, soc_start)

    soc = np.zeros(time_count)
    charge = np.zeros(time_count)
    discharge = np.zeros(time_count)
    emergency_purchase = np.zeros(time_count)
    unused_energy = np.zeros(time_count)
    soc_change = np.zeros(time_count)

    for t in range(time_count):
        immediate_shortage = np.maximum(
            actual_net_load[t] + psi[soc_index, :] - planned_purchase[t],
            0.0,
        )
        total_cost = (
            emergency_multiplier * forecast_price[t] * immediate_shortage
            + future_value[t + 1]
        )
        total_cost = np.where(feasible[soc_index, :], total_cost, np.inf)
        next_index = int(np.argmin(total_cost))
        if not np.isfinite(total_cost[next_index]):
            raise RuntimeError(f"第{t}个时段储能DP没有可行动作")

        soc_change[t] = grid[next_index] - grid[soc_index]
        if soc_change[t] >= 0.0:
            charge[t] = soc_change[t] / params.eta_charge
            discharge[t] = 0.0
        else:
            charge[t] = 0.0
            discharge[t] = -params.eta_discharge * soc_change[t]

        actual_grid_effect = psi[soc_index, next_index]
        emergency_purchase[t] = max(
            actual_net_load[t] + actual_grid_effect - planned_purchase[t],
            0.0,
        )
        unused_energy[t] = max(
            planned_purchase[t] - actual_net_load[t] - actual_grid_effect,
            0.0,
        )

        soc_index = next_index
        soc[t] = grid[soc_index]

    normal_cost = float(np.sum(actual_price * planned_purchase))
    emergency_cost = float(
        np.sum(emergency_multiplier * actual_price * emergency_purchase)
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


def build_detail_rows_q4(
    current_date,
    time_labels,
    forecast_price: np.ndarray,
    actual_price: np.ndarray,
    load_energy: np.ndarray,
    pv_energy: np.ndarray,
    planned_purchase: np.ndarray,
    execution: dict,
) -> list[dict]:
    """生成问题4-2的10分钟级明细。"""

    rows = []
    for t, label in enumerate(time_labels):
        rows.append(
            {
                "date": current_date,
                "slot_index": t,
                "time": str(label),
                "forecast_price_yuan_per_kwh": float(forecast_price[t]),
                "price_yuan_per_kwh": float(actual_price[t]),
                "load_kwh": float(load_energy[t]),
                "pv_kwh": float(pv_energy[t]),
                "net_load_kwh": float(load_energy[t] - pv_energy[t]),
                "planned_purchase_kwh": float(planned_purchase[t]),
                "charge_kwh": float(execution["charge"][t]),
                "discharge_kwh": float(execution["discharge"][t]),
                "soc_end_kwh": float(execution["soc"][t]),
                "emergency_purchase_kwh": float(execution["emergency_purchase"][t]),
                "unused_energy_kwh": float(execution["unused_energy"][t]),
            }
        )
    return rows


def build_result4_2_workbook(
    detail: pd.DataFrame,
    daily_summary: pd.DataFrame,
    time_labels: list,
    output_dir: Path,
) -> Path:
    """生成附件5同结构的result4-2.xlsx。"""

    dates = list(daily_summary["date"])
    interval_labels = [
        p2.make_10min_interval_label(slot_index)
        for slot_index in range(len(time_labels))
    ]

    planned_wide = detail.pivot(
        index="date",
        columns="slot_index",
        values="planned_purchase_kwh",
    )
    planned_wide = planned_wide.reindex(index=dates, columns=range(len(time_labels)))
    planned_wide.columns = interval_labels
    planned_wide.index.name = "日期\\时间"

    daily_lookup = daily_summary.set_index("date")
    planned_wide["全天购电量"] = daily_lookup.loc[
        planned_wide.index, "planned_purchase_kwh"
    ].to_numpy()
    planned_wide["全天购电费"] = daily_lookup.loc[
        planned_wide.index, "normal_cost_yuan"
    ].to_numpy()

    storage_table = p2.build_storage_table(detail=detail, daily_summary=daily_summary)
    emergency_table = p2.build_emergency_purchase_table(detail=detail, dates=dates)

    output_path = output_dir / "result4-2.xlsx"
    sheet_names = ["计划购电量", "充放电量", "紧急购电量"]
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
        date_format="yyyy/m/d",
        datetime_format="yyyy/m/d",
    ) as writer:
        planned_wide.to_excel(writer, sheet_name=sheet_names[0], index=True)
        storage_table.to_excel(writer, sheet_name=sheet_names[1], index=False)
        emergency_table.to_excel(writer, sheet_name=sheet_names[2], index=False)
        p2.format_submission_workbook(writer=writer, sheet_names=sheet_names)
    return output_path


def plot_q4_2_results(
    output_dir: Path,
    daily_summary: pd.DataFrame,
    detail: pd.DataFrame,
) -> None:
    """输出费用曲线和电价预测误差曲线。"""

    if daily_summary.empty or detail.empty:
        return

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    dates = pd.to_datetime(daily_summary["date"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, daily_summary["total_cost_yuan"], label="总费用")
    ax.plot(dates, daily_summary["emergency_cost_yuan"], label="紧急购电费用")
    ax.set_title("问题4-2每日真实结算费用")
    ax.set_xlabel("日期")
    ax.set_ylabel("费用/元")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "problem4_2_daily_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, daily_summary["price_mae_yuan_per_kwh"], label="日前电价MAE")
    ax.set_title("问题4-2日前电价预测误差")
    ax.set_xlabel("日期")
    ax.set_ylabel("元/kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "problem4_2_price_mae.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题4-2：波动电价日前计划")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"C:/Users/LENOVO/Desktop/数模/C题"),
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="problem4_2",
        help="输出到base-dir/outputs下的子目录",
    )
    parser.add_argument("--start-date", type=str, default="2025-02-01")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=30)
    parser.add_argument("--forecast-recency-decay", type=float, default=0.90)
    parser.add_argument("--min-similar-days", type=int, default=3)
    parser.add_argument("--daily-trend-clip", type=float, default=0.20)
    parser.add_argument("--pv-recent-days", type=int, default=7)
    parser.add_argument("--forecast-bias-lookback", type=int, default=30)
    parser.add_argument("--forecast-bias-alpha", type=float, default=1.0)
    parser.add_argument("--candidate-quantiles", type=float, nargs="+", default=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--high-risk-quantiles", type=float, nargs="+", default=[0.85, 0.90, 0.95])
    parser.add_argument("--normal-quantile-floor", type=float, default=0.70)
    parser.add_argument("--high-risk-quantile-floor", type=float, default=0.85)
    parser.add_argument("--dynamic-quantile-lookback", type=int, default=3)
    parser.add_argument("--dynamic-emergency-threshold", type=float, default=1000.0)
    parser.add_argument("--dynamic-net-error-threshold", type=float, default=5000.0)
    parser.add_argument("--rolling-quantile-lookback", type=int, default=30)
    parser.add_argument("--rolling-min-history-days", type=int, default=7)
    parser.add_argument("--scoring-emergency-multiplier", type=float, default=2.5)
    parser.add_argument("--soc-step", type=float, default=100.0)
    parser.add_argument(
        "--price-decay-candidates",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.90, 1.00],
    )
    parser.add_argument("--price-lookback-days", type=int, default=35)
    parser.add_argument("--min-price-history-days", type=int, default=7)
    parser.add_argument("--max-same-week-price-days", type=int, default=5)
    parser.add_argument("--price-fallback-days", type=int, default=7)
    parser.add_argument("--price-ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument(
        "--storage-value-mode",
        choices=["median-price", "fixed"],
        default="median-price",
        help="median-price表示按当天预测电价中位数估计日末库存价值",
    )
    parser.add_argument("--fixed-storage-value", type=float, default=0.481548)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_days is not None and args.max_days <= 0:
        raise ValueError("--max-days必须为正整数")
    if args.max_scenarios <= 0:
        raise ValueError("--max-scenarios必须为正整数")
    if args.price_lookback_days <= 0 or args.min_price_history_days <= 0:
        raise ValueError("电价回测窗口和最少历史天数必须为正")
    if args.price_ridge_alpha <= 0:
        raise ValueError("Ridge alpha必须为正")

    params = p2.StorageParams()
    emergency_multiplier = 5.0
    attachment_dir = args.base_dir / "附件"
    output_dir = args.base_dir / "outputs" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    path_attachment2 = attachment_dir / "附件2.xlsx"
    path_attachment4 = attachment_dir / "附件4.xlsx"

    # 1. 数据读取
    dates, time_labels, load_kw = p2.read_wide_matrix(path_attachment2, sheet_name=0)
    pv_dates, _, pv_kw = p2.read_wide_matrix(path_attachment2, sheet_name=1)
    if not dates.equals(pv_dates):
        raise ValueError("附件2负荷日期和光伏日期不一致")

    price_dates, _, actual_price = read_price_matrix(path_attachment4)
    if not pd.DatetimeIndex(pd.to_datetime(dates)).equals(price_dates):
        raise ValueError("附件2日期与附件4日期不一致")

    # 2. 数据处理与检查
    p2.check_input_data(load_kw=load_kw, pv_kw=pv_kw, price=actual_price)
    load_energy = load_kw * params.delta_t
    pv_energy = pv_kw * params.delta_t
    all_dates = np.asarray(dates.tolist(), dtype=object)

    formal_start_date = pd.Timestamp(args.start_date).date()
    formal_indices = np.flatnonzero(all_dates >= formal_start_date)
    if len(formal_indices) == 0:
        raise ValueError(f"数据中没有不早于{formal_start_date}的日期")
    if args.max_days is not None:
        formal_indices = formal_indices[: args.max_days]
    first_formal_index = int(formal_indices[0])
    if first_formal_index == 0:
        raise ValueError("正式评价日前没有历史数据，无法预测")

    # 3. 参数计算
    normal_quantiles = p2.normalize_quantiles(args.candidate_quantiles, "--candidate-quantiles")
    high_risk_quantiles = p2.normalize_quantiles(args.high_risk_quantiles, "--high-risk-quantiles")
    soc_grid = p2.make_soc_grid(params, args.soc_step)
    _, psi, feasible = p2.storage_grid_matrices(soc_grid, params)
    period_groups = p2.make_time_period_groups(len(time_labels))

    print(f"问题4-2正式区间：{all_dates[formal_indices[0]]} 至 {all_dates[formal_indices[-1]]}")
    print(f"正式天数：{len(formal_indices)}，SOC步长：{args.soc_step:g} kWh")
    print("电价模型：上周同日同槽基准 + Ridge残差修正")
    print("购电模型：问题二同类型分解预测 + 分时段净负荷误差分位数 + 日前LP + 因果DP")

    # 4. 模型求解
    soc_start = params.soc_initial
    detail_rows: list[dict] = []
    daily_rows: list[dict] = []
    price_audit_rows: list[dict] = []
    forecast_cache: dict[tuple, tuple] = {}

    for day_index in formal_indices:
        day_index = int(day_index)
        current_date = all_dates[day_index]

        forecast_price, price_info = day_ahead_ridge_residual_price_forecast(
            price_matrix=actual_price,
            dates=price_dates,
            day_index=day_index,
            ridge_alpha=args.price_ridge_alpha,
            candidate_decays=args.price_decay_candidates,
            max_same_week_days=args.max_same_week_price_days,
            fallback_days=args.price_fallback_days,
            min_train_days=args.min_price_history_days,
        )
        actual_price_day = actual_price[day_index]
        price_mae = float(np.mean(np.abs(forecast_price - actual_price_day)))
        price_bias = float(np.mean(forecast_price - actual_price_day))
        storage_value = (
            params.eta_discharge * float(np.median(forecast_price))
            if args.storage_value_mode == "median-price"
            else float(args.fixed_storage_value)
        )

        forecast_load, forecast_pv, scenario_load, scenario_pv, scenario_probability = (
            p2.get_base_day_forecast(
                forecast_cache=forecast_cache,
                load_energy=load_energy,
                pv_energy=pv_energy,
                all_dates=all_dates,
                day_index=day_index,
                max_scenarios=args.max_scenarios,
                recency_decay=args.forecast_recency_decay,
                min_similar_days=args.min_similar_days,
                daily_trend_clip=args.daily_trend_clip,
                pv_recent_days=args.pv_recent_days,
                calendar_correction_enabled=False,
                calendar_correction_strength=0.0,
            )
        )

        dynamic_candidates, dynamic_mode, dynamic_reason = p2.select_dynamic_quantiles(
            recent_daily_rows=daily_rows,
            normal_quantiles=normal_quantiles,
            high_risk_quantiles=high_risk_quantiles,
            lookback_days=args.dynamic_quantile_lookback,
            emergency_threshold_kwh=args.dynamic_emergency_threshold,
            net_error_threshold_kwh=args.dynamic_net_error_threshold,
        )

        net_error_history, net_error_history_indices = p2.collect_walk_forward_net_errors(
            current_day_index=day_index,
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
            calendar_correction_enabled=False,
            calendar_correction_strength=0.0,
        )

        forecast_net_load, scenario_net_load, scenario_probability, scenario_source = (
            p2.build_net_load_scenarios_from_errors(
                forecast_load=forecast_load,
                forecast_pv=forecast_pv,
                scenario_load=scenario_load,
                scenario_pv=scenario_pv,
                net_error_history=net_error_history,
            )
        )

        period_candidate_quantiles = p2.normalize_quantiles(
            normal_quantiles + (high_risk_quantiles if dynamic_mode == "高风险" else []),
            "--period-candidate-quantiles",
        )
        period_quantiles, rolling_reason = p2.select_rolling_optimal_period_quantiles(
            current_day_index=day_index,
            all_dates=all_dates,
            load_energy=load_energy,
            pv_energy=pv_energy,
            price=actual_price,
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
            scoring_emergency_multiplier=args.scoring_emergency_multiplier,
            correction_alpha=args.forecast_bias_alpha,
            forecast_cache=forecast_cache,
            calendar_correction_enabled=False,
            calendar_correction_strength=0.0,
        )
        period_quantiles = p2.enforce_period_quantile_floors(
            period_quantiles=period_quantiles,
            candidate_quantiles=period_candidate_quantiles,
            dynamic_risk_mode=dynamic_mode,
            normal_quantile_floor=args.normal_quantile_floor,
            high_risk_quantile_floor=args.high_risk_quantile_floor,
        )
        selected_period_quantiles = p2.format_period_quantiles(period_quantiles, period_groups)

        target_net_load, net_error_adjustment, target_source = (
            p2.make_grouped_target_net_load_from_error_quantiles(
                forecast_net_load=forecast_net_load,
                scenario_net_load=scenario_net_load,
                net_error_history=net_error_history,
                period_quantiles=period_quantiles,
                period_groups=period_groups,
                min_history_days=args.rolling_min_history_days,
                correction_alpha=args.forecast_bias_alpha,
            )
        )

        lp_solution = p2.solve_day_ahead_lp(
            target_net_load=target_net_load,
            price=forecast_price,
            soc_start=soc_start,
            storage_value=storage_value,
            params=params,
        )
        future_value = p2.compute_dp_value(
            planned_purchase=lp_solution["planned_purchase"],
            scenario_net_load=scenario_net_load,
            scenario_probability=scenario_probability,
            price=forecast_price,
            grid=soc_grid,
            psi=psi,
            feasible=feasible,
            emergency_multiplier=emergency_multiplier,
            storage_value=storage_value,
        )

        actual_net_load = load_energy[day_index] - pv_energy[day_index]
        execution = execute_one_day_with_price_forecast(
            planned_purchase=lp_solution["planned_purchase"],
            actual_net_load=actual_net_load,
            forecast_price=forecast_price,
            actual_price=actual_price_day,
            soc_start=soc_start,
            params=params,
            grid=soc_grid,
            psi=psi,
            feasible=feasible,
            future_value=future_value,
            emergency_multiplier=emergency_multiplier,
        )

        detail_rows.extend(
            build_detail_rows_q4(
                current_date=current_date,
                time_labels=time_labels,
                forecast_price=forecast_price,
                actual_price=actual_price_day,
                load_energy=load_energy[day_index],
                pv_energy=pv_energy[day_index],
                planned_purchase=lp_solution["planned_purchase"],
                execution=execution,
            )
        )

        actual_net_sum = float(np.sum(actual_net_load))
        target_net_sum = float(np.sum(target_net_load))
        daily_rows.append(
            {
                "date": current_date,
                "price_model": price_info.model_name,
                "selected_price_alpha": price_info.selected_alpha,
                "selected_price_decay": price_info.selected_decay,
                "price_selection_reason": price_info.selection_reason,
                "price_mae_yuan_per_kwh": price_mae,
                "price_bias_yuan_per_kwh": price_bias,
                "storage_value_yuan_per_kwh": storage_value,
                "risk_mode": "分时段高风险" if dynamic_mode == "高风险" else "分时段常规",
                "risk_reason": f"{rolling_reason}；动态判断：{dynamic_reason}",
                "selected_quantile": float(np.mean(list(period_quantiles.values()))),
                "selected_period_quantiles": selected_period_quantiles,
                "scenario_count": int(len(scenario_probability)),
                "forecast_load_kwh": float(np.sum(forecast_load)),
                "forecast_pv_kwh": float(np.sum(forecast_pv)),
                "forecast_net_load_kwh": float(np.sum(forecast_net_load)),
                "net_error_correction_kwh": float(np.sum(net_error_adjustment)),
                "net_error_history_days": int(len(net_error_history_indices)),
                "scenario_source": scenario_source,
                "target_source": target_source,
                "target_net_load_kwh": target_net_sum,
                "actual_net_load_kwh": actual_net_sum,
                "net_load_positive_error_kwh": max(actual_net_sum - target_net_sum, 0.0),
                "soc_start_kwh": soc_start,
                "soc_end_kwh": float(execution["soc"][-1]),
                "planned_purchase_kwh": float(np.sum(lp_solution["planned_purchase"])),
                "emergency_purchase_kwh": float(np.sum(execution["emergency_purchase"])),
                "unused_energy_kwh": float(np.sum(execution["unused_energy"])),
                "normal_cost_yuan": execution["normal_cost"],
                "emergency_cost_yuan": execution["emergency_cost"],
                "total_cost_yuan": execution["total_cost"],
                "estimated_objective_yuan": float(lp_solution["objective_value"]),
            }
        )

        price_audit_rows.append(
            {
                "date": current_date,
                "price_model": price_info.model_name,
                "selected_alpha": price_info.selected_alpha,
                "selected_decay": price_info.selected_decay,
                "selection_reason": price_info.selection_reason,
                "forecast_mean_price": float(np.mean(forecast_price)),
                "actual_mean_price": float(np.mean(actual_price_day)),
                "mae_yuan_per_kwh": price_mae,
                "bias_forecast_minus_actual": price_bias,
            }
        )

        soc_start = float(execution["soc"][-1])
        print(
            f"{current_date}完成 | alpha={price_info.selected_alpha:.2f} | "
            f"价格MAE={price_mae:.4f} | q均值={daily_rows[-1]['selected_quantile']:.2f} | "
            f"SOC {daily_rows[-1]['soc_start_kwh']:.1f}->{soc_start:.1f} | "
            f"紧急购电{daily_rows[-1]['emergency_purchase_kwh']:.1f}kWh | "
            f"总费用{daily_rows[-1]['total_cost_yuan']:.2f}元"
        )

    # 5. 结果检验
    detail = pd.DataFrame(detail_rows)
    daily_summary = pd.DataFrame(daily_rows)
    price_audit = pd.DataFrame(price_audit_rows)
    if detail.empty or daily_summary.empty:
        raise RuntimeError("没有生成问题4-2结果")

    if detail["soc_end_kwh"].lt(params.soc_min - 1e-6).any() or detail[
        "soc_end_kwh"
    ].gt(params.soc_max + 1e-6).any():
        raise RuntimeError("结果检验失败：SOC越界")
    if detail["charge_kwh"].gt(params.max_charge_energy + 1e-6).any():
        raise RuntimeError("结果检验失败：充电功率越界")
    if detail["discharge_kwh"].gt(params.max_discharge_energy + 1e-6).any():
        raise RuntimeError("结果检验失败：放电功率越界")
    nonnegative_columns = ["planned_purchase_kwh", "emergency_purchase_kwh", "unused_energy_kwh"]
    if (detail[nonnegative_columns] < -1e-6).any().any():
        raise RuntimeError("结果检验失败：出现负购电、紧急购电或弃电")

    previous_soc_end = None
    for _, row in daily_summary.iterrows():
        if previous_soc_end is not None and not np.isclose(
            row["soc_start_kwh"], previous_soc_end, atol=1e-6
        ):
            raise RuntimeError("结果检验失败：跨日SOC不连续")
        previous_soc_end = row["soc_end_kwh"]

    # 6. 可视化
    plot_q4_2_results(output_dir=output_dir, daily_summary=daily_summary, detail=detail)

    # 7. 结果导出
    detail_path = output_dir / "problem4_2_detail.csv"
    daily_path = output_dir / "problem4_2_daily_summary.csv"
    audit_path = output_dir / "price_forecast_audit.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    daily_summary.to_csv(daily_path, index=False, encoding="utf-8-sig")
    price_audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    result_path = build_result4_2_workbook(
        detail=detail,
        daily_summary=daily_summary,
        time_labels=time_labels,
        output_dir=output_dir,
    )

    total_normal = float(daily_summary["normal_cost_yuan"].sum())
    total_emergency = float(daily_summary["emergency_cost_yuan"].sum())
    total_cost = float(daily_summary["total_cost_yuan"].sum())
    total_emergency_kwh = float(daily_summary["emergency_purchase_kwh"].sum())

    print()
    print("========== 问题4-2完成 ==========")
    print(f"计划购电费用：{total_normal:.2f} 元")
    print(f"紧急购电费用：{total_emergency:.2f} 元")
    print(f"总费用：{total_cost:.2f} 元")
    print(f"紧急购电总量：{total_emergency_kwh:.2f} kWh")
    print(f"明细：{detail_path}")
    print(f"每日汇总：{daily_path}")
    print(f"电价预测审计：{audit_path}")
    print(f"提交表格：{result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
