# ECO65＋ZhiXing单任务数据采集

已完成罐入篮任务的场景覆盖采集和LeRobot转换验证：训练9条/5,226帧，验证3条/1,721帧，共6,947个动作配对。所有轨迹通过原采集审计和LeRobot全帧读取验证，双相机13,894张图像逐像素一致。

- [LeRobot转换验证完成记录](LeRobot转换验证_完成记录_20261003_r1.md)
- [当前版本与文件索引r3](当前版本与文件索引_20261003_r3.md)
- [转换与CPU读取说明](versions/lerobot_20261003_r1/README.md)
- [场景覆盖完成记录](场景覆盖_完成记录_20261003_r1.md)
- [采样退化问题记录](场景采样退化_问题记录_20261003_r1.md)

采集版本`coverage_20261003_r1`，转换版本`lerobot_20261003_r1`。正式LeRobot数据位于服务器`lerobot_exports/coverage_lerobot_r1/datasets/`，按train/validation独立repo保存；原始HDF5保留于`runs/coverage_train_val_r1/`。保留的3个test场景未采集或转换。

本轮仅做CPU转换与数据读取验证，未加载模型、未训练、未执行策略闭环或Spatial验收。小批专家数据通过不代表模型高成功率适配。旧冻结版本及历史完成记录保留；根README更新备份见`maintenance/readme_20261003_r3/`。
