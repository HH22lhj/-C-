"""粗算全年负荷由光伏、储能和计划购电完全覆盖所需的购电量。"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(r"C:/Users/LENOVO/Desktop/数模/C题")
ATTACHMENT_DIR = BASE_DIR / "附件"

DELTA_T = 1.0 / 6.0
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_INITIAL = 6000.0
ETA_CHARGE = 0.90
ETA_DISCHARGE = 0.90
MAX_POWER_KW = 5000.0


def read_matrix(path: Path, sheet_name: int) -> np.ndarray:
    raw = pd.read_excel(path, sheet_name=sheet_name)
    values = raw.iloc[:, 1:145].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if values.shape[1] != 144 or not np.isfinite(values).all():
        raise ValueError(f"{path.name} 第{sheet_name}个工作表数据无效")
    return values * DELTA_T


def main() -> None:
    attachment2 = ATTACHMENT_DIR / "附件2.xlsx"
    attachment1 = ATTACHMENT_DIR / "附件1.xlsx"

    load_energy = read_matrix(attachment2, sheet_name=0)
    pv_energy = read_matrix(attachment2, sheet_name=1)

    price_raw = pd.read_excel(attachment1)
    price = pd.to_numeric(
        price_raw.iloc[:144, 1],
        errors="coerce",
    ).to_numpy(dtype=float)

    if load_energy.shape != pv_energy.shape:
        raise ValueError("负荷和光伏数据维度不一致")
    if load_energy.shape[1] != 144 or len(price) != 144:
        raise ValueError("每天必须包含144个10分钟时段")

    max_charge_grid = MAX_POWER_KW * DELTA_T
    max_discharge_grid = MAX_POWER_KW * DELTA_T

    soc = SOC_INITIAL
    total_load = 0.0
    total_pv = 0.0
    direct_pv = 0.0
    battery_charge_grid = 0.0
    battery_discharge_grid = 0.0
    planned_grid = 0.0
    curtailed_pv = 0.0
    planned_cost = 0.0
    detail_rows = []

    for day in range(load_energy.shape[0]):
        for slot in range(144):
            load = load_energy[day, slot]
            pv = pv_energy[day, slot]
            total_load += load
            total_pv += pv

            direct = min(load, pv)
            direct_pv += direct
            surplus = pv - direct
            shortage = load - direct

            # 光伏余电优先充电，超过容量或功率的部分弃掉。
            charge_grid = min(
                surplus,
                max_charge_grid,
                (SOC_MAX - soc) / ETA_CHARGE,
            )
            soc += ETA_CHARGE * charge_grid
            battery_charge_grid += charge_grid
            curtailed_pv += surplus - charge_grid

            # 负荷缺口由储能放电，剩余缺口由计划购电补足。
            discharge_grid = min(
                shortage,
                max_discharge_grid,
                (soc - SOC_MIN) * ETA_DISCHARGE,
            )
            soc -= discharge_grid / ETA_DISCHARGE
            battery_discharge_grid += discharge_grid

            grid = shortage - discharge_grid
            planned_grid += grid
            slot_price = price[slot]
            slot_cost = grid * slot_price
            planned_cost += slot_cost

            detail_rows.append(
                {
                    "day_index": day + 1,
                    "slot_index": slot,
                    "load_kwh": load,
                    "pv_kwh": pv,
                    "direct_pv_kwh": direct,
                    "charge_grid_kwh": charge_grid,
                    "discharge_grid_kwh": discharge_grid,
                    "planned_grid_kwh": grid,
                    "price_yuan_per_kwh": slot_price,
                    "planned_cost_yuan": slot_cost,
                    "soc_end_kwh": soc,
                    "curtailed_pv_kwh": surplus - charge_grid,
                }
            )

    detail = pd.DataFrame(detail_rows)
    detail_path = BASE_DIR / "outputs" / "problem2" / "annual_coverage_detail.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    daily = detail.groupby("day_index", as_index=False).agg(
        load_kwh=("load_kwh", "sum"),
        pv_kwh=("pv_kwh", "sum"),
        charge_grid_kwh=("charge_grid_kwh", "sum"),
        discharge_grid_kwh=("discharge_grid_kwh", "sum"),
        planned_grid_kwh=("planned_grid_kwh", "sum"),
        planned_cost_yuan=("planned_cost_yuan", "sum"),
        curtailed_pv_kwh=("curtailed_pv_kwh", "sum"),
    )
    daily_path = BASE_DIR / "outputs" / "problem2" / "annual_coverage_daily.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    net_load = max(total_load - total_pv, 0.0)
    theoretical_lower_bound = max(
        net_load - battery_discharge_grid + battery_charge_grid,
        0.0,
    )

    print("========== 全年覆盖粗算 ==========")
    print(f"数据天数：{load_energy.shape[0]}")
    print(f"全年负荷：{total_load:.2f} kWh")
    print(f"全年光伏：{total_pv:.2f} kWh")
    print(f"负荷减光伏的能量差：{net_load:.2f} kWh")
    print(f"光伏直接供负荷：{direct_pv:.2f} kWh")
    print(f"储能放电供负荷：{battery_discharge_grid:.2f} kWh")
    print(f"储能充电用电量：{battery_charge_grid:.2f} kWh")
    print(f"弃光电量：{curtailed_pv:.2f} kWh")
    print(f"所需计划购电量：{planned_grid:.2f} kWh")
    print(f"计划购电费用粗算：{planned_cost:.2f} 元")
    print(f"计划购电平均电价：{planned_cost / max(planned_grid, 1e-12):.4f} 元/kWh")
    print(f"能量平衡下界参考：{theoretical_lower_bound:.2f} kWh")
    print(f"年末SOC：{soc:.2f} kWh")
    print(f"逐时段明细：{detail_path}")
    print(f"每日汇总：{daily_path}")
    print("说明：这是光伏优先的可执行粗算，不包含日前价格套利和预测误差临时购电。")


if __name__ == "__main__":
    main()
