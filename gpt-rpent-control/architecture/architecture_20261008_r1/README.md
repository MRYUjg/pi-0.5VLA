# 当前实现架构 PPT 素材 20261008 r1

[可编辑单页 PPTX](output/ECO65_闭环控制架构_20261008_r2.pptx) · [高清 PNG](output/ECO65_闭环控制架构_20261008_r2.png)

16:9，PNG 为 3200×1800。PPTX 中模块、文字和连接线为原生可编辑对象。字体为 PingFang SC。

依据当前版本索引 20261007 r4、eco65_zhixing_20261007_r5 的 run_episode.py / vla_bridge.py / env_server.py 和部署复验记录制作。20261008 通过当前服务器核对：本地与服务器 8 个 Python 实现文件的 SHA256 一致，RPent commit 为 068cd64f4f175179a13c67d429a4d00ca56fb196。

图覆盖自然语言任务、DeepSeek 文本工具规划、RPent、原精度 π0.5、环境 RPC、OSC_POSE、ECO65＋ZhiXing 仿真执行、观测与文本反馈，以及日志和双视角 MP4 留存。

当前实现仅为 LIBERO/MuJoCo 仿真闭环。原模型没有加载微调权重。本次完整任务复用 pi0_doubled，尚未证明 LLM 自主分解/恢复的额外增益。部署配置原模型批次为 0/10，不能将链路接通当作高成功率适配或实机验收。详细接口定义和边界见 PPT 备注。

本次仅新增架构素材和版本索引，不修改控制器、模型、采集或审计实现。旧版本保留。
