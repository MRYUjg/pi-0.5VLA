# ZhiXing 夹爪抓持能力 —— MuJoCo 可视化

资产：**已落地版**（`zhixing_gripper.xml` sha256 `e7aab50c…`：`forcerange="-2 2"` + 指垫橡胶参数）
试块：实心 60 × 50 × 70 mm，摆在可达区 (−0.22, 0.10)

## 文件

| 文件 | 说明 |
|---|---|
| `grasp_50mm.mp4` | **完整时序**，7.5 s @ 10 fps，75 帧 |
| `stage_50mm_00_initial.png` … `10_released.png` | 9 张关键阶段大图 |
| `frames/` | 逐帧（`--save-frames`） |

画面布局：**左 = agentview 全局**（含工作台全景），**右 = 同视角放大特写**。
左下角实时显示「试块抬升 / 夹爪净距」。

## 本帧实测

| 阶段 | 试块抬升 | 夹爪净距 |
|---|---:|---:|
| 4. 闭合（夹住） | +3.1 mm | 53.45 mm（试块 50 mm → 已咬入） |
| 6. 保持 2 s | **+80.4 mm** | 53.24 mm |
| 放回后偏差 | −4.0 mm | — |

闭合用了 3300 子步（`Left_1 = −0.6174`），抬升走 **OSC 力驱动**（不是运动学瞬移）。

## 复现

```bash
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
  /bin/bash -c "source /.venv/bin/activate && python \
  /app/agent-work-space/zhixing_adaptation_20260927/scripts/n3r_grasp_viz.py \
  --width-mm 50 --save-frames"
```

## 渲染上踩过的坑（已在脚本里规避）

1. **`mjVIS_CONTACTFORCE` 不能开**：力矢量 glyph 体积巨大，会把整个画面糊成橙色块。
2. **不能直接画 robosuite 的碰撞 geom（group 0）**：它们被染成橙色，叠在视觉 geom 上
   会让场景变成橙色。改为**渲染前重新着色**（group-0 → 中性灰，试块 → 红，共 186 个 geom）。
3. **不能把试块 geom 改成 `group="1"`**：robosuite 的 arena merge 会报
   `mass and inertia of moving bodies must be larger than mjMINVAL`（standalone 编译却正常）。
4. **自由相机 `elevation` 符号**很容易把机位放到桌面以下（画面全灰）。
   最终放弃自由相机，改用 **agentview 的 2D 放大裁剪**做特写，稳定可靠。
5. **`cv2.putText` 不支持中文**（渲染成 `?????`），标注一律用 ASCII。
