# 数模 C 题代码

数学建模 C 题相关代码。

## 目录

- `问题1_单日确定性调度.py`：问题1 单日确定性调度，含 LP 基准模型 + DP 对比模型

## 运行环境

- Python 3.13+
- 依赖：`pandas`、`numpy`、`scipy`、`openpyxl`

安装依赖：

```bash
pip install pandas numpy scipy openpyxl
```

## 运行

```bash
python 问题1_单日确定性调度.py
```

## 说明

- 数据文件 `附件1.xlsx` 与输出目录 `analysis_outputs/` 未上传，请自行放置在 `DATA_PATH` / `OUT_DIR` 指定的路径。
- LP 为连续变量全局最优基准模型；DP 用 SOC 离散化做对比验证，离散步长越小越接近 LP。
