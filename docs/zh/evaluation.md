# 评价说明

[English](../en/evaluation.md)

## 指标口径

PR-AUC采用average precision，仅已知标签参与统计；Macro F1是单类F1均值。Generation MAE仅统计参考高于TRAIN活动阈值的分钟。Energy Bias＝预测与参考累计差／参考累计量，kWh累计使用分钟kW除60；不连续时间段的结果仅代表有效评价分钟。

PV还应检查非活跃MAE、非活跃预测总电量、active 0～2kW MAE、每日绝对电量偏差及ramp误差。不能只凭整体相关系数或曲线视觉重合判断可靠性。

## 复现与泛化

训练数据按显式日期角色分块，划分后再创建历史窗口。normalizer、标签噪声阈值和功率尺度均只用TRAIN；分类阈值只用Validation。三seed分别保存best checkpoint。公开测试验证数值、因果和数据路由，不宣称复现私有实验精度。

上线评估应另设未参与开发的新台区／新批次，覆盖低容量、持续发电、复杂负荷扰动和异常采集场景；不能用待评估Test反向调系数或挑seed。
