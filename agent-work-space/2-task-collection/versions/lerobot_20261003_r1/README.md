# LeRobot CPU转换与验证

版本`lerobot_20261003_r1`。输入为`coverage_train_val_r1`的已接受数据，输出为两个独立的LeRobot v2.1本地数据集。方案与验收范围见[转换验证方案](转换验证方案.md)。本版本不训练、不加载策略、不需要GPU、不上传Hub。

在服务器仓库根目录运行，输出名必须唯一：

```bash
# 两条开发检查：train与validation各第一条
python3 agent-work-space/2-task-collection/versions/lerobot_20261003_r1/scripts/run.py \
  --name 新的检查目录 --limit-each 1

# 全部12条；先转换，随后独立重载逐帧核对
python3 agent-work-space/2-task-collection/versions/lerobot_20261003_r1/scripts/run.py \
  --name 新的正式目录
```

原始数据挂载只读，只有`lerobot_exports/`允许写入。执行容器禁网，使用两颗CPU、8GB内存上限和独立Python3.10环境。每阶段日志及真实退出码在`{name}_execution/`。转换成功但验证失败不能作为已验收数据。

## 读取

输出位于`lerobot_exports/{name}/datasets/`，repo_id分别为：

- `local/eco65_zhixing_can_coverage_r1_train`
- `local/eco65_zhixing_can_coverage_r1_validation`

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

root = "/app/agent-work-space/2-task-collection/lerobot_exports/{name}/datasets"
repo_id = "local/eco65_zhixing_can_coverage_r1_train"
ds = LeRobotDataset(
    repo_id=repo_id,
    root=f"{root}/{repo_id}",
    delta_timestamps={"actions": [i / 20 for i in range(50)]},
    video_backend="pyav",
)
sample = ds[0]
# state: [8]; actions: [50,7]; image/wrist_image: float32 [3,256,256]
# actions_is_pad: [50]；task为语言文本
```

在未来OpenPI运行环境中，`HF_LEROBOT_HOME`需指向上述`datasets/`根目录，并选择训练repo_id。依赖使用当前仓库锁定的LeRobot提交`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。不要使用最新版LeRobot自动改写数据格式。

这里采用两个独立repo隔离训练与验证，每个repo内部LeRobot默认split标签仍为`train`；业务划分以repo_id和`provenance.json`的split为准，不能将两个repo合并训练。只有训练repo的统计可用于训练；LeRobot的基础统计并非OpenPI已就绪的归一化资产。

图像保持源朝向并无损内嵌Parquet，读取时LeRobot自动返回CHW/[0,1]。7维动作已是归一化OSC增量，后续LIBERO配置应保持`extra_delta_transform=False`；夹爪−1开/+1合，不反转。state最后两维是ZhiXing具名关节角，不能当作Panda指爪米制位移解释。

`next.done`仅标识记录结束；原环境可能提前触发done，但转换保留完整释放后窗口。源末端额外观测保留在HDF5，不伪造额外动作。`provenance.json`记录scene、seed、原文件哈希、帧区间及源仿真起始时间。

CPU PyTorch批读取与50步action chunk验证不等于完整OpenPI模型预处理或训练验证；本轮不实例化模型。未来训练需要独立准备配置和训练集归一化资产。
