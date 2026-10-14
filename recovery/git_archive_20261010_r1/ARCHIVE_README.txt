本地 Git 存档补充，2026-10-10

source_overrides 保存此前 SHA256 核验发现的三份正确源码副本；不覆盖主工作目录。
libero_source 保存 LIBERO 的源码、配置、场景定义与初始化文件，其中机器人注册文件采用已核验的正确副本。
父仓库把 third_party/libero 记录为子模块，所以仅上传父仓库不会保存其本地修改。
恢复时先取得 .gitmodules 中记录的 LIBERO 版本，再将该源码副本合并到其工作目录，并按 manifest.json 核验。
LIBERO 上游网格、纹理等二进制资产未重复复制到本补充目录；robosuite 机器人网格与纹理由主仓库存档。
源版本配置和成功率结论保持原记录；本补充不代表运行验收。
正式训练数据、微调参数、优化器状态和部署引擎未在本地完整保存，详见 maintenance/local_archive_audit_20261010_r1/recovery_scope.json。
