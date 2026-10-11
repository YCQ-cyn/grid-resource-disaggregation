# 数据接口

[English](../en/data_contract.md)

## 自有数据配置

`config/dataset.example.json`只包含路径占位与空日期列表。路径相对配置文件解析；运行前必须填入自有文件与明确日期，不能直接把空模板用于训练。

每批配置包含id、main、pv、可选energy／charger、async、energy_supervised及dates。日期使用YYYY-MM-DD。资源任务使用train／validation／internal_test／domain_test；PV任务使用train／validation／test。任何相同自然日不得同时属于TRAIN和评价；同批日期也不能重复出现在不同列表。语义未确认的能源站应设置energy_supervised=false。

PV新域id必须以`new`开头，历史辅助域使用其他前缀。PV训练需要两种域；新域日期承担Validation和Test，历史数据只作为TRAIN。辅助充电列表元素为device和path，其日期不得进入评价日期集合。

## 自然分钟CSV

每条记录包含唯一、分钟对齐的minute_start、coverage_ratio和[56字段固定顺序](../../config/fields.json)。时间统一使用本地UTC+8口径。字段按名称读取，模型编码顺序由fields.json决定。完整分钟阈值是coverage_ratio≥0.999；不能将低质量标签当作OFF。

总开、光伏、能源站和充电分别提供CSV。资源监督按minute_start精确关联，不使用nearest join或人工时移。功率方向以表计配置为准，PV参考定义为max(-TotW_MA,0)。输入通道只来自总开。

## 异步记录

async=true时，源CSV包含data_time、report_time及56字段。带时区时间戳按实际时区解释；无时区data_time／report_time按UTC解析后转UTC+8。总开按首条到达记录保留同sample_time数据，按report_time所属分钟组织；PV参考按data_time所属分钟组织。它是异步代理参考，不能称为严格同步分钟真值。资源标签准备不接纳该格式。

## 模型输入语义

|通道|资源|PV|
|---|---:|---:|
|标准化总开电气字段|56|56|
|有效位|56|56|
|measurement_mask|1|1|
|gap_indicator|1|1|
|measurement_age_scaled|—|1|
|sample_age_scaled|—|1|

measurement_age_scaled是距最后有效总有功观测的分钟数，裁剪到4后除4；sample_age_scaled是分钟结束距最近有效采样时刻的秒数，裁剪到0～300后除300。PV维持缺失字段的最近有效值最多3分钟，其有效位仍为0。不能从分表标签、未来行或评价日期拟合输入。

## 本地输出

资源训练产生selected.pt、selection.json、data/normalizer.json和分标签指标；PV训练产生release/中的checkpoint、manifest.json、normalizer.json及评价表。所有这些文件均由本地运行生成，不属于仓库交付数据。

PV分离输出pv_power_kw（正发电kW）、内部activity_probability、magnitude_kw、input_available及gap_indicator。资源输出三个独立运行代理概率与状态；PV分离activity不是资源辨识PV输出的替代。
