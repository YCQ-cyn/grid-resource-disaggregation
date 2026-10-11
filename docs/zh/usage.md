# 训练与推理

[English](../en/usage.md)

以下命令在仓库根目录执行。所有数据、权重与运行产物由使用者本地提供或生成。

## 安装与检查

```sh
python -m pip install -e '.[dev]'
python -m grid_disaggregation --help
python -m pytest -q
python -m grid_disaggregation.preprocessing.formula_checks
```

COMTRADE预处理可选使用系统7z；没有该程序时安装`.[archives]`。示例：

```sh
python -m grid_disaggregation.preprocessing.extract_features --help
python -m grid_disaggregation.preprocessing.extract_features --input data/recordings --output-dir outputs/features
```

该入口生成基础量、谐波量及56分钟计算项。正式模型读取分钟计算项，不将所有导出字段自动加入模型。

## 配置数据

```sh
cp config/dataset.example.json config/dataset.local.json
```

填写main／pv／energy／charger路径、明确分批身份、分表计量关系和日期划分。资源任务仅使用精确自然分钟数据；PV任务可按配置读取异步采样。新域以new开头，历史辅助域只放TRAIN。辅助充电可配置独立记录，但不能伪造其对应总开。参见[数据接口](data_contract.md)。

配置路径示例是占位，须换成真实自有文件。运行命令不会下载数据或权重。

## 资源训练与推理

```sh
python -m grid_disaggregation resource-train --plan config/dataset.local.json --output outputs/resource
python -m grid_disaggregation resource-predict --input data/main.csv --model-dir outputs/resource --output outputs/resource_predictions.csv
```

训练默认使用三个seed，有辅助充电记录时先进行EV-only训练。保存每seed best Validation checkpoint，再根据Validation选selected.pt与分类阈值。评价只统计已知标签；未覆盖的Test角色跳过。

## PV准备、训练与推理

```sh
python -m grid_disaggregation pv-prepare --plan config/dataset.local.json --output outputs/pv_data
python -m grid_disaggregation pv-train --table outputs/pv_data/training_table.csv --normalizer outputs/pv_data/normalizer.json --method small_s4d --output outputs/pv
python -m grid_disaggregation pv-predict --input data/main.csv --model-dir outputs/pv/release --output outputs/pv_predictions.csv
```

需要TCN-LSTM对照时改为`--method tcn_lstm`，输出到另一个目录。训练使用新域75%与历史25%的采样比例，要求两种TRAIN组都有可用端点。该工作流按显式配置重建数据与normalizer，不会复现未公开的原始私有数据划分；公开模块保留已有模型、编码与loss的数学实现。

## 在线调用

```python
from grid_disaggregation.pv.inference import OnlinePV

runner = OnlinePV('outputs/pv/release')
result = runner.update('2026-01-01 10:00:00', record, observed=True)
```

record是总开字段字典，可含sample_time_main／arrival_time_main。只提交聚合已完成的分钟，时间必须递增。当前运行逻辑要求源采样与可用时间早于该分钟末端；异步源CSV由既有canonical_csv规则分桶。不要将未来记录提前提交。

历史和normalizer常驻内存；缺分钟由内部缺测记录补齐。使用input_available判断当前数据可用性。该Python在线模块不涉及终端硬件适配。
