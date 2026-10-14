# ECO65＋ZhiXing 单任务试采

版本 `pilot_20261003_r1`。首任务为LIBERO Object task 0：抓起alphabet soup罐，放入篮子。方案与数据字段见[单任务试采方案](单任务试采方案.md)。本阶段是仿真数据链路验证；没有模型训练或Spatial正式验收。

## 运行

在服务器仓库根目录执行；批次名必须全新。先阅读根目录的完成记录确认当前验收状态。

```bash
cd /mnt/data/vla-policy/openpi
python3 agent-work-space/2-task-collection/versions/pilot_20261003_r1/scripts/pipeline.py \
  --batch 新的唯一批次名 --seeds 54100,54101,54102,54103,54104 --workers 2
```

该入口先校验冻结源码及机器人/LIBERO依赖指纹，再采集、回放、审计。使用服务器既有`libero:latest`镜像与1.3目录的隔离Python3.10/MuJoCo3.2.6环境；纯CPU、OSMesa，每次运行最多两个采集进程，每个2 CPU/5GB内存限制。开发实现包含串行`run.py`和处理输出所有权的并行`run_r2.py`及保留启动失败状态的`run_r3.py`；实际复现用上面的完整pipeline入口。

若代码指纹漂移，停止并建立新版本，不直接更新旧manifest。失败批次保留。若采集进程失败，pipeline返回非零；检查该批次`.status.json`/日志/partial文件后建立新批次，不在原目录重跑覆盖。

## 数据位置

`runs/<batch>/can_s<seed>/` 每条包含：

- `trajectory.h5`：双视角RGB、8维state、7维实际动作、完整仿真状态、准备阶段、语言与XML；进行中或异常文件保留为`trajectory.partial.h5`。
- `teacher/`：未修改r20教师输出的任务门限、逐步记录、逐物理步受力和接触证据。
- `replay.json`和`replay.status.json`：动作回放差异与真实退出码，二者必须一起判断。
- `audit.json`：物理审计、数据审计和文件SHA256，`accepted=true`才允许作为本批合格示范。
- `preview.jpg`：抓取、保持、搬运、释放阶段的两个视角拼图。

批次`dataset_manifest.json`明确记录尝试数、接受数及拒收原因；参考种子回归单列，不加入五条新种子的成功统计。文件中的`process_exit_pending=true`指写文件时进程仍运行，后续宿主机status提供最终退出结果，不能只读该字段判断失败或成功。

## 读取示例

```python
import h5py
with h5py.File('trajectory.h5', 'r') as f:
    image = f['observations/image'][:-1]         # T,256,256,3
    wrist = f['observations/wrist_image'][:-1]
    state = f['observations/state'][:-1]         # T,8
    actions = f['transitions/actions'][:]       # T,7
    language = f.attrs['language']
    final_observation = f['observations/state'][-1]
```

第t条动作严格对应第t个观测；第t+1个观测是动作后的真实状态。最后观测没有对应动作。示范停止位置由episode边界确定，不按中间的environment_done裁切。后续做动作块时不可越过episode尾端。

## 复核

```bash
python3 agent-work-space/2-task-collection/versions/pilot_20261003_r1/scripts/freeze.py verify
```

冻结包位于`releases/pilot_20261003_r1/`：归档新采集版本（包含原样复制的专家依赖），并对当前robosuite/LIBERO源码与资产建立指纹。原生产目录、上游1.3脚本和控制器未被修改。冻结通过只说明文件一致性，数据是否合格以各批次审计结果为准。

覆盖限制：罐的初始抓取位置基本固定；新种子只提供不同场景尝试，不能据此声称已具备位置泛化。腕部图像有夹爪遮挡。当前格式为可审计HDF5原始数据，尚未进行LeRobot转换、训练/验证/测试正式划分或π0.5微调。
