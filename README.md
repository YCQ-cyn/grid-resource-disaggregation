# 台区资源辨识与光伏功率分离

[English](README.en.md)

从总开分钟级电气数据中辨识光伏、能源站和充电桩的运行信号，并估计当前分钟的光伏发电功率。两个任务独立建模，推理时均只需要总开数据，不读取分表标签、未来观测、天气或地理信息。

## 业务问题

台区总功率同时包含普通负荷、能源站、充电桩和光伏发电的影响。光伏抵消负荷，设备启停形成突变，储能或能源站运行又可能改变功率方向与持续段。仅凭某一分钟功率大小，难以判断变化来自哪一种资源。

本项目利用连续历史中的局部事件和长期变化过程，分别回答：**当前有哪些资源运行信号？当前光伏贡献了多少功率？** 结果用于资源运行监测、光伏出力估计和后续业务核验，不等同于设备物理安装证明。

## 方法

| 任务 | 模型 | 历史上下文 | 输出 |
|---|---|---|---|
| 多标签资源辨识 | 因果TCN＋单向LSTM | 120分钟 | PV／能源站／充电桩三个独立概率与状态 |
| 光伏实时分离 | Small Causal S4D双头；TCN-LSTM双头作对照 | 240分钟 | 当前分钟PV功率，单位kW |

**资源辨识。** TCN提取突增、突降、波动和持续段等局部动态，单向LSTM结合这些变化的历史演化。三个sigmoid输出互不排斥，允许多种资源同时运行。分表仅用于监督；缺失标签保持未知并在loss与评价中屏蔽。未配对的充电记录只用于TRAIN阶段的充电单标签辅助训练，不建立虚假的总开对应关系。

**光伏分离。** 轻量因果状态空间骨干建模历史依赖，activity和magnitude双头分别估计发电活动与功率幅值：

```text
PV功率(kW) = sigmoid(activity) × softplus(magnitude) × target_scale_kw
```

训练联合约束最终功率、活动判断、活跃时幅值和非活跃时零输出。光伏参考来自配对分表的 `max(-TotW_MA, 0)`；推理只使用总开特征，两个任务不串联。这里估计的是已经完成的当前分钟，不是未来功率预测。

**数据与评价。** 56个电气字段按固定顺序组织，并附有效位、缺口和新鲜度信息。资源输入为114通道，PV输入为116通道。按日期划分后再生成窗口，normalizer仅由TRAIN拟合；三seed训练，checkpoint与阈值仅依据Validation选择。PV短输入缺口只允许过去值维持，短标签插值仅用于同一split内离线训练，并降低其权重；主评价使用观测标签。

## 项目结构

```text
grid-resource-disaggregation/
├── README.md
├── README.en.md
├── pyproject.toml
├── config/
│   ├── fields.json                  56字段顺序
│   └── dataset.example.json         自有数据路径与日期划分模板
├── src/grid_disaggregation/
│   ├── cli.py                       命令行工作流
│   ├── common.py                    配置与文件工具
│   ├── resource/
│   │   ├── models.py                因果TCN与TCN-LSTM
│   │   ├── data.py                  精确对齐、标签掩码与窗口
│   │   ├── training.py              三seed训练与Validation选模
│   │   └── inference.py             总开在线辨识
│   ├── pv/
│   │   ├── models.py                TCN-LSTM双头对照
│   │   ├── state_space.py           Small Causal S4D
│   │   ├── data.py                  配对参考、缺测与日期隔离
│   │   ├── features.py              因果输入编码
│   │   ├── training.py              双头训练
│   │   ├── metrics.py               功率、活动与电量评价
│   │   └── inference.py             总开逐分钟PV分离
│   └── preprocessing/
│       ├── extract_features.py      COMTRADE特征提取入口
│       ├── comtrade_io.py           录波与压缩包读取
│       ├── basic_items.py           基础电气量
│       ├── harmonic_items.py        谐波量
│       ├── calculated_items.py      分钟计算项
│       └── formula_checks.py        公式自检
├── docs/
│   ├── zh/                          方法、数据接口与使用说明
│   └── en/
├── tests/                           不依赖真实数据的测试
├── licenses/                        第三方许可
└── .github/workflows/               自动化测试
```

## 快速使用

Python 3.11及以上；CPU可运行，CUDA可选。

```sh
python -m pip install -e '.[dev]'
python -m grid_disaggregation --help
python -m pytest -q
```

使用自有数据，将配置模板另存为 `config/dataset.local.json`，填写文件路径及互不重叠的日期。训练与推理命令见[使用说明](docs/zh/usage.md)。仓库不提供采集数据、训练权重或拟合normalizer。

## 评价与适用边界

资源辨识以PR-AUC、Macro F1和单类Precision／Recall／F1评价；PV分离报告MAE、Generation MAE、RMSE、R²、Activity F1和电量偏差，并单独检查非活跃假电量、低功率误差与日尺度偏差。[已有实验摘要](docs/zh/evaluation.md)明确区分Validation、内部Test和复用时间留出。

能源站标签目前是功率状态代理，不是经独立核实的储能充放电真值。部分充电配对拓扑未独立确认，相关指标属于探索性结果。总开功率分解本身不唯一，跨台区、容量和季节的泛化需要独立数据验证。

- [方法说明](docs/zh/methodology.md)
- [数据接口](docs/zh/data_contract.md)
- [训练与推理](docs/zh/usage.md)
- [评价说明](docs/zh/evaluation.md)
- [第三方声明](NOTICE.md)
