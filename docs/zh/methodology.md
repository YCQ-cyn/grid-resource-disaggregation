# 方法说明

[English](../en/methodology.md)

## 任务与计量边界

总开有功按购电方向为正时，典型组成可写为：

```text
P_main = P_load + P_energy_station + P_charger − P_pv
```

能源站功率允许双向；实际符号必须依据表计极性确认。监督参考与总开需要相同计量边界及可靠的时间关系，不能把不相关分表平移后当成同步真值。

模型输出针对分钟聚合完成后的当前时刻。资源辨识判断运行信号，PV分离估计当前发电分量，两者不互相依赖。

## 资源辨识

输入为过去120分钟的56项总开字段、对应56有效位、measurement_mask和gap_indicator。归一化使用TRAIN中位数、IQR及0.1%／99.9%裁剪边界。缺测数值在归一化后置零，有效位保留缺测语义。

TCN宽度16／16／24／32，卷积核5，膨胀率1／2／4／8。每块包含两次裁剪后的因果卷积与残差连接。单层、32隐单元、单向LSTM处理TCN序列；仅取末端状态，经三个线性logit和独立sigmoid输出。114通道配置下参数量为38,555。

标签为PV活动、能源站功率≥5kW、充电功率≥1kW。PV活动阈值来自TRAIN夜间参考噪声。能源站和充电标签属于运行代理；无观测、质量不足或语义未确认的标签不参与监督。TRAIN-only独立充电记录仅监督第三个标签。

主loss为逐类归一化的masked BCE；正类权重从TRAIN已知标签计算并裁剪到0.25～10。三seed先作充电辅助预训练，再训练总开样本。每seed根据Validation Macro PR-AUC保存checkpoint，阈值根据Validation F1校准，最终按Validation Macro PR-AUC、Macro F1选seed。

## 光伏分离

输入为过去240分钟的相同56字段与有效位，增加measurement_mask、gap_indicator、measurement_age_scaled、sample_age_scaled。Small Causal S4D包含116→32投影、两个32宽／24状态的对角状态空间块和末端双头，共9,314参数。块内LayerNorm仅沿通道，时间卷积为因果方向，没有未来池化或双向递推。

双头输出activity logit与magnitude logit，最终功率为sigmoid(activity)×softplus(magnitude)×TRAIN拟合尺度。训练对归一化功率采用SmoothL1（beta=0.02），活跃分钟权重2.5，另加0.20活动BCE、0.10活跃幅值监督、0.20非活跃零输出loss。插值目标权重为0.25。

训练batch按新域75%、历史辅助域25%采样；目标尺度和中心／IQR来自新域TRAIN，裁剪边界覆盖全部TRAIN。TCN-LSTM双头共享输入、loss和评价，可作为结构对照。Validation MAE优先，其后依次为Generation MAE、RMSE和日绝对电量偏差；Test不参与选择。

## 缺测、因果与边界

总开最多维持3分钟过去有效值，禁止未来补输入。离线PV参考仅在同split内部、双侧均有观测且缺口不超过3分钟时插值；主评价只用实测参考。数据划分发生在窗口生成之前；不同日期角色的历史状态重置。

当前模型保留00:00～00:14的既有缺口策略。这个约定属于已有数据与模型处理逻辑，不是所有计量终端的普遍规律。短历史以中性零通道补齐，输出应同时检查input_available；有数值结果不代表完整历史或可靠数据。

## 参考

- Bai et al., *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, 2018：[论文](https://arxiv.org/abs/1803.01271)。
- Gu et al., *On the Parameterization and Initialization of Diagonal State Space Models*, 2022：[论文](https://arxiv.org/abs/2206.11893)。
- [S4官方实现](https://github.com/state-spaces/s4)：S4D核参数化参考，见第三方声明。
