#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-rework（审核项 R7）：**全程 7 维 OSC 动作链**的抓取—抬升—保持—搬运—放置—释放验证。

审核结论（R7）与本脚本的对应关系
================================
| 审核指出的问题 | 本脚本的处理 |
|---|---|
| 接近/下降用 IK 直写 qpos；闭合时每子步锁臂；放回也用 IK 覆写 | episode 内**只有** `env.step(7 维动作)`；IK 函数在 episode 开始后被运行时守卫替换成抛异常，`ik_calls_during_episode` 必须为 0 |
| `success` 仅 `dz_hold_mm > 30` | 7 条**全部必须成立**的逐 trial 闸门（见 `THRESHOLDS`），抬升只是其中之一 |
| 总 verdict 用 `any_width_pass` | 按宽度分组，**每组独立判定 ≥9/10**；总 verdict 要求**每一个**宽度组都达标 |
| B 项缺少旋转测试 | 12 个闭环探针：±x/±y/±z 平移 + ±rx/±ry/±rz 旋转，报告"命令量 vs 实达量"误差 |
| 仍用 `GRASP_OFFSET_CLOSED = 82.54 mm`（TCP 前移到 0.20254 后失效） | **完全不使用该常数**；目标点全部用 `grip_site` 的实测世界位姿（`N.gripsite`）表达，与 N4 采集链路一致 |
| "无穿透"与 0.82–1.12 mm 实测穿透不符 | 定义**允许的软接触穿透阈值**，用 MuJoCo 接触数据 `d.contact[i].dist` 逐**物理子步**测全程峰值并如实报告 |

物体口径（唯一有意的方法学选择，已写入 JSON 的 `object_substitution`）
------------------------------------------------------------------
试块**顶替 BDDL 目标物体本身**（进程内替换 `get_object_fn` 返回的类），而不是另插一个
与任务无关的 body。理由：

* 原生 LIBERO 成功谓词 `(In alphabet_soup_1 basket_1_contain_region)` 才**真的适用**
  —— 试块**就是** `alphabet_soup_1`，`env.check_success()` 是货真价实的原生谓词；
* 抓取—搬运—投放链路与 N4 采集链路（`libero_object` task 0）**完全同构**；
* 试块是**已知几何的实心长方体**，抓不住时可归因到夹爪而非物体模型缺陷。

注入只发生在测试进程内，**磁盘资产零改动**（与既有 `n3r_block_test.py` 同一手法）。
输出只写 `results/n3_osc_*`，绝不覆盖历史 `n3_*` / `n3r_*` 文件。

用法
----
    python n3_osc_full_chain.py --mode probe            # 诊断 + 1 组 1 trial
    python n3_osc_full_chain.py --mode full             # 正式：3 宽度 × 10 + 探针 + 重定向
"""
import argparse
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N          # noqa: E402  底层几何/接触工具
import n4_collect as N4            # noqa: E402  Episode / 采样 / 目标解析 / TCP 契约

ROOT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927")
OUT_JSON = ROOT / "results" / "n3_osc_full_chain.json"
OUT_TRACE = ROOT / "results" / "n3_osc_full_chain_traces.npz"

# --------------------------------------------------------------------------- #
# 全部阈值集中在这里，并**原样写进结果 JSON**（`thresholds` 字段）
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "lift_min_mm": 30.0,
    "lift_note": "抬升段结束时物体中心相对 episode 起始高度 ≥ 30 mm",

    "hold_min_mm": 25.0,
    "hold_steps": 40,
    "hold_note": "保持段 40 个控制步（20 Hz → 2 s）内**每一步**物体中心都 ≥ 起始 +25 mm",

    "post_release_steps": 20,
    "post_release_settle_max_steps": 90,
    "post_release_drift_max_mm": 5.0,
    "post_release_note": ("释放后再静置（最多 90 步，等到物体线速度 <8 mm/s），随后 20 个控制步内："
                          "物体中心**每一步**都在目标 contain 区域内，且 20 步末位移 < 5 mm"),

    "max_step_disp_m": 0.02,
    "max_step_disp_note": ("全程每个控制步 grip_site 位移上限 0.02 m/step（=0.4 m/s @20 Hz）。"
                           "IK 直写会产生整段位移的瞬移，此闸门专门抓它。"),

    "penetration_soft_limit_mm": 2.0,
    "penetration_note": ("指垫↔物体接触的允许软接触穿透上限。试块 solref=0.02（软接触），"
                         "在需要传递载荷的夹持力下必然有亚毫米级穿透；历史记录 0.82–1.12 mm，"
                         "故取 2.0 mm ≈ 2 倍历史峰值。**峰值无论是否通过都写进结果。**"),

    "trials_per_width": 10,
    "pass_min_per_width": 9,
    "probe_converge_tol_pos_m": 0.005,
    "probe_converge_tol_rot_rad": 0.03,
    "probe_settle_steps": 10,
    "probe_converge_note": ("探针的收敛判据（0.005 m / 0.03 rad）**故意严于**通过闸门"
                            "（0.05 rad），以避免'收敛'与'通过'循环论证；未收敛即判 FAIL，"
                            "即使实测误差已落在闸门内也如实记 FAIL。"),
    "rotation_probe_cmd_rad": 0.15,
    "rotation_probe_err_max_rad": 0.05,
    "translation_probe_cmd_m": 0.02,
    "translation_probe_err_max_mm": 5.0,
    "reorient_group_min_pass": 5,
    "reorient_group_trials": 6,
}

TEACHER_GAIN_POS_M = N4.TEACHER_GAIN_POS_M      # 0.02 —— 与 N4 采集链路同一教师增益
TEACHER_GAIN_ROT_RAD = N4.TEACHER_GAIN_ROT_RAD  # 0.05
GRIP_OPEN, GRIP_CLOSE = N4.GRIP_ACT_OPEN, N4.GRIP_ACT_CLOSE

# 教师"步长限制器"：只限制每个控制步的**目标点推进量**，发出去的仍是合法 7 维 OSC 增量。
# 为什么需要：N4 的 servo 用 gain=1.0，误差 20 mm 就饱和到满动作（OSC 满量程 50 mm/step），
# 那样测不过"无瞬移"闸门是教师自己的问题、不是夹爪的问题。
SERVO_MAX_STEP_POS_M = 0.012
SERVO_MAX_STEP_ROT_RAD = 0.04

# OSC 动作→物理量的**真实**换算：`scale_action(u) = u * output_max`（input 为 ±1）。
# 故"每步最多推进 12 mm / 0.04 rad"对应的动作上限是 max_step/output_max。
# ★ 不这样换算的话，教师增益 TEACHER_GAIN_ROT_RAD=0.05 会把一个 0.05 rad 的姿态
#   误差变成满量程动作 = **0.5 rad/step** 的旋转指令；一旦姿态误差因关节限位无法
#   消除，这个饱和指令会长期存在并把位置通道带偏（实测：下降段 grip_site 横向漂移
#   17 mm，指垫撞到试块顶棱）。这里把限制真正施加在**指令增量**上。
OSC_POS_UNIT_M = 0.05
OSC_ROT_UNIT_RAD = 0.5

# 本测试自己的 episode 帧上限。N4.MAX_FRAMES=900 是为采集 horizon 设的；
# 本测试要在部署速度（gripper speed 0.01 → 全行程闭合约 200 控制步）下跑完整条链，
# 故放宽；超过它只说明流程异常，仍会截断并如实记录。
FRAME_LIMIT = 1700

BLOCK_DEPTH_M = 0.060      # 闭合轴**垂直**方向的深度（物体局部 x）
BLOCK_HEIGHT_M = 0.070     # 高度（物体局部 z）
BLOCK_DENSITY = 700.0      # 与历史 n3r 试块一致

BLOCK_XML = """<mujoco model="n3_osc_block">
  <worldbody>
    <body>
      <body name="object">
        <geom name="g0" type="box" size="{hx} {hy} {hz}" group="0"
              rgba="0.85 0.22 0.18 0.3" contype="1" conaffinity="1" condim="4"
              friction="1 0.005 0.0001" solref="0.02 1" density="{dens}"/>
        <geom name="g1" type="box" size="{hx} {hy} {hz}" group="1"
              rgba="0.85 0.22 0.18 1" contype="0" conaffinity="0"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -{hz}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {hz}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{hx} {hy} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""

_ACTIVE_MON = {"m": None}
_IK_GUARD = {"armed": False, "violations": []}
_BLOCK_CLS_CACHE = {}
_TMP_DIR = pathlib.Path(tempfile.gettempdir())
_ORIG_GET_OBJECT_FN = {"f": None}


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def sha16(p):
    return N4.sha16(p)


def native_success(env):
    """LIBERO 原生成功谓词。

    ★ `OffScreenRenderEnv`（`ControlEnv`）**没有** `_check_success`，只有
    `check_success()`；直接写 `env._check_success()` 会抛 AttributeError，
    被 except 吞掉后会**静默变成 False**（首次实现踩到：谓词恒为 False）。
    """
    f = getattr(env, "check_success", None)
    if callable(f):
        return bool(f())
    return bool(getattr(env, "env", env)._check_success())


def expm_so3(w):
    """旋转向量 → 旋转矩阵（世界系）。"""
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + K
    K = K / th
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pad_geoms(sim):
    import mujoco

    m = sim.model._model
    out = {}
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "left_fingerpad_col" in n:
            out["left"] = g
        elif "right_fingerpad_col" in n:
            out["right"] = g
    return out


def closing_axis_world(env):
    """夹爪**闭合轴**的世界方向（由两指垫 geom 原点差给出，投影到水平面）。"""
    sim, m, d, _ = N.ids(env)
    pads = pad_geoms(sim)
    if "left" not in pads or "right" not in pads:
        return None
    a = np.array(d.geom_xpos[pads["right"]], dtype=float) - \
        np.array(d.geom_xpos[pads["left"]], dtype=float)
    a[2] = 0.0
    n = float(np.linalg.norm(a))
    return None if n < 1e-9 else a / n


def object_yaw(env, bid):
    """物体 free joint 的当前世界偏航（只对纯 z 偏航的姿态有效）。"""
    a, _ = N4.free_joint_adr(env, bid)
    sim, m, d, _ = N.ids(env)
    q = np.array(d.qpos[a + 3:a + 7], dtype=float)
    return float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


def set_object_yaw(env, bid, xy, target_yaw):
    """把物体摆到 xy 并**绝对**设为 target_yaw 的偏航。

    `N4.apply_object_sample` 是**左乘**一个世界 z 偏航（在既有姿态上叠加），
    所以必须先量出当前偏航再补差值；直接再调用一次会得到随机偏航 + 目标偏航
    （首次实现踩到：试块被转了 27.7°，宽度不在闭合轴上）。
    """
    delta = float(target_yaw) - object_yaw(env, bid)
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return N4.apply_object_sample(env, bid, xy, delta)


# --------------------------------------------------------------------------- #
# 物体注入：把 BDDL 目标物体的**类别**换成实心试块（进程内，不落盘）
# --------------------------------------------------------------------------- #
def block_object_class(width_m, depth_m, height_m):
    key = (round(width_m, 6), round(depth_m, 6), round(height_m, 6))
    if key in _BLOCK_CLS_CACHE:
        return _BLOCK_CLS_CACHE[key]
    from robosuite.models.objects import MujocoXMLObject

    # ★ 每个尺寸一个**独立**临时 XML：`MujocoXMLObject.__init__` 在实例化时才读文件，
    #   若共用一个路径，后写的新尺寸会污染已缓存的旧类（首次实现踩到：width_70 的
    #   文件把 reorient 的 30/50 mm 组污染成 70 mm 试块，实测 bbox 70.1×60.1×70.0）。
    xml_path = _TMP_DIR / ("n3_osc_block_%d_%d_%d.xml"
                           % (round(width_m * 1e6), round(depth_m * 1e6),
                              round(height_m * 1e6)))
    xml_path.write_text(
        BLOCK_XML.format(hx=depth_m / 2.0, hy=width_m / 2.0, hz=height_m / 2.0,
                         dens=BLOCK_DENSITY), encoding="utf-8")

    class _OscBlock(MujocoXMLObject):
        """实心长方体试块；接口与 HopeBaseObject 一致。"""

        def __init__(self, name, obj_name=None):
            super().__init__(str(xml_path), name=name,
                             joints=[dict(type="free", damping="0.0005")],
                             obj_type="all", duplicate_collision_geoms=False)
            self.category_name = "n3_osc_block"
            # HOPE 物体靠 rotation/rotation_axis 摆正网格；盒子本身已轴对齐
            self.rotation = (0.0, 0.0)
            self.rotation_axis = "z"
            self.object_properties = {"vis_site_names": {}}

    _BLOCK_CLS_CACHE[key] = _OscBlock
    return _OscBlock


def install_object_substitution(category, block_cls, log=None):
    """让 BDDL 里的 `category` 类实例化出试块（只在当前进程内生效）。"""
    import libero.libero.envs.problems.libero_floor_manipulation as LFM

    if _ORIG_GET_OBJECT_FN["f"] is None:
        _ORIG_GET_OBJECT_FN["f"] = LFM.get_object_fn
    orig = _ORIG_GET_OBJECT_FN["f"]

    def patched(cat_name):
        if str(cat_name).lower() == str(category).lower():
            return block_cls
        return orig(cat_name)

    LFM.get_object_fn = patched
    if log:
        log("[inject] BDDL 类别 %r → 实心试块（进程内 monkeypatch，磁盘资产未改动）" % category)


def target_category(env):
    """从 BDDL 反查目标物体所属类别名（如 alphabet_soup_1 → alphabet_soup）。"""
    domain = getattr(env, "env", env)
    p = domain.parsed_problem
    obj, region = N4.goal_objects(env)
    if obj is None:
        return None, None, None
    for cat, names in (p.get("objects") or {}).items():
        if obj in names:
            return cat, obj, region
    return None, obj, region


# --------------------------------------------------------------------------- #
# 接触穿透监测（**每个物理子步**采样）
# --------------------------------------------------------------------------- #
class ContactMonitor:
    """指垫↔物体 与 物体↔任意 的有符号穿透深度监测。

    `d.contact[i].dist` < 0 即穿透，穿透深度 = −dist（mm）。
    挂到 `env.env._pre_action`：robosuite 每个物理子步都会调用它一次
    （`MujocoEnv.step` 内 `for i in range(control_timestep/model_timestep)`），
    故采样率 = 物理子步率 500 Hz，比控制步 20 Hz 细 25 倍。
    另外 `Episode.act` 在每个控制步末再采一次，两端都不漏。
    """

    def __init__(self):
        self.obj_bid = None
        self.pad_ids = set()
        self.phase = "init"
        self.peak_pad = 0.0
        self.peak_any = 0.0
        self.per_phase = {}
        self.n_samples = 0
        self.n_pad_contacts = 0
        self.cur_pad = 0.0

    def bind(self, env, obj_bid):
        self.env = env
        sim, m, d, _ = N.ids(env)
        self.obj_bid = obj_bid
        self.pad_ids = set(pad_geoms(sim).values())

    def set_phase(self, phase):
        self.phase = phase

    def sample(self):
        if self.obj_bid is None:
            return
        sim = self.env.env.sim
        m, d = sim.model._model, sim.data._data
        pad = 0.0
        anyd = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if int(m.geom_bodyid[g1]) != self.obj_bid and \
                    int(m.geom_bodyid[g2]) != self.obj_bid:
                continue
            dist = float(c.dist)
            if dist < 0.0:
                pen = -dist * 1000.0
                if pen > anyd:
                    anyd = pen
                if g1 in self.pad_ids or g2 in self.pad_ids:
                    if pen > pad:
                        pad = pen
                    self.n_pad_contacts += 1
        self.n_samples += 1
        self.cur_pad = pad
        if pad > self.peak_pad:
            self.peak_pad = pad
        if anyd > self.peak_any:
            self.peak_any = anyd
        ph = self.per_phase.setdefault(self.phase, {"pad_peak_mm": 0.0,
                                                    "any_peak_mm": 0.0, "samples": 0})
        ph["samples"] += 1
        if pad > ph["pad_peak_mm"]:
            ph["pad_peak_mm"] = pad
        if anyd > ph["any_peak_mm"]:
            ph["any_peak_mm"] = anyd


def install_contact_hook(env):
    inner = env.env
    if getattr(inner, "_n3_osc_hooked", False):
        return False
    orig = inner._pre_action

    def hooked(action, policy_step=False):
        orig(action, policy_step)
        mon = _ACTIVE_MON["m"]
        if mon is not None:
            mon.sample()

    inner._pre_action = hooked
    inner._n3_osc_hooked = True
    return True


# --------------------------------------------------------------------------- #
# 运行时 IK 守卫
# --------------------------------------------------------------------------- #
def arm_ik_guard():
    if _IK_GUARD["armed"]:
        return

    def _boom(name):
        def f(*a, **k):
            _IK_GUARD["violations"].append(name)
            raise AssertionError(
                "[R7 guard] episode 内调用了 %s —— 必须全程只用 7 维 OSC 动作链" % name)
        return f

    for nm in ("set_arm", "solve_ik_gripsite", "ik_goto", "step_hold"):
        if hasattr(N, nm):
            setattr(N, nm, _boom(nm))
    _IK_GUARD["armed"] = True


# --------------------------------------------------------------------------- #
# Episode 子类：只加"步长受限伺服"与"逐控制步位移/穿透记录"，其余复用 N4
# --------------------------------------------------------------------------- #
class ChainEpisode(N4.Episode):

    def __init__(self, env, mon=None):
        super().__init__(env)
        self.mon = mon
        self.max_step_disp = 0.0
        self.step_disps = []
        self.site_path = []
        self.obj_path = []
        self.bid = None
        self.pen_trace = []
        self.disp_phase = {}
        self.phase = "init"
        self.native_first_true = None
        self.native_true_steps = 0
        self.native_error = None
        self.last_err = (None, None)
        self.conv = {"pos": False, "rot": False}

    def set_phase(self, ph):
        self.phase = ph
        if self.mon is not None:
            self.mon.set_phase(ph)

    def act(self, action, phase=None):
        if phase is not None:
            self.set_phase(phase)
        # 原生谓词一旦成立 LIBERO 会置 done，之后 robosuite 拒绝再 step。
        # R7 明确要求验证"释放后仍保持稳定"，故必须继续推进仿真；
        # 这里只清 done 标志（不触碰任何物理量），并逐步记录谓词真值。
        try:
            self.env.env.done = False
        except Exception:                                     # noqa: BLE001
            pass
        fr = super().act(action, self.phase)
        sim, m, d, gid = N.ids(self.env)
        p_now = np.array(d.site_xpos[gid], dtype=float)
        disp = float(np.linalg.norm(p_now - fr["grip_site"]))
        self.max_step_disp = max(self.max_step_disp, disp)
        self.step_disps.append(disp)
        self.disp_phase[self.phase] = max(self.disp_phase.get(self.phase, 0.0), disp)
        fr["step_disp_m"] = disp
        self.site_path.append(p_now.copy())
        if self.bid is not None:
            bb = N4.bbox(sim, self.bid)
            self.obj_path.append(np.full(3, np.nan) if bb is None
                                 else 0.5 * (bb[0] + bb[1]))
        self.pen_trace.append(float(self.mon.cur_pad) if self.mon else 0.0)
        try:
            ok = native_success(self.env)
        except Exception as e:                                # noqa: BLE001
            ok = False
            self.native_error = "%s: %s" % (type(e).__name__, e)
        fr["native_success"] = ok
        if ok:
            self.native_true_steps += 1
            if self.native_first_true is None:
                self.native_first_true = self.n
        return fr

    def track(self, target, R_target, grip_cmd, phase, steps,
              max_step_pos=SERVO_MAX_STEP_POS_M,
              max_step_rot=SERVO_MAX_STEP_ROT_RAD):
        """朝 (target, R_target) 推进 `steps` 步，每步**目标推进量受限**。

        发出去的永远是合法 7 维 OSC 增量；限制的只是教师每一步给出的目标点。
        （不限制的话，20 mm 误差就饱和成满量程动作 = 50 mm/step，
        "无瞬移"闸门会变成在测教师而不是测夹爪。）
        """
        err = (0.0, 0.0)
        for _ in range(steps):
            if self.n >= FRAME_LIMIT:
                break
            sim, m, d, gid = N.ids(self.env)
            p, R = N.gripsite(sim, m, d, gid)
            dv = np.asarray(target, dtype=float) - p
            nd = float(np.linalg.norm(dv))
            if nd > max_step_pos:
                dv = dv * (max_step_pos / nd)
            ap = np.clip(dv / TEACHER_GAIN_POS_M, -1.0, 1.0)
            er = N.rot_err_vec(R_target, R)
            ne = float(np.linalg.norm(er))
            if ne > max_step_rot:
                er = er * (max_step_rot / ne)
            ar = np.clip(er / TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
            ap = np.clip(ap, -max_step_pos / OSC_POS_UNIT_M,
                         max_step_pos / OSC_POS_UNIT_M)
            ar = np.clip(ar, -max_step_rot / OSC_ROT_UNIT_RAD,
                         max_step_rot / OSC_ROT_UNIT_RAD)
            self.act(np.concatenate([ap, ar, [grip_cmd]]), phase)
            err = (nd, ne)
        return err

    def servo_limited(self, target, R_target, grip_cmd, phase, max_steps,
                      tol=0.004, rot_tol=0.03,
                      max_step_pos=SERVO_MAX_STEP_POS_M,
                      max_step_rot=SERVO_MAX_STEP_ROT_RAD, verbose=False,
                      stall_patience=30):
        """闭环伺服到 (target, R_target)，每步目标推进量受限。全程只发 7 维 OSC 动作。

        终止条件：到位（|e|<tol 且 |rot_e|<rot_tol）、**停滞**（stall_patience 步内
        误差不再改善 ≥0.2 mm —— OSC 在重力下存在 ~7 mm 稳态误差，且搬运目标点可能
        超出 ECO65 可达域）、或步数/帧数上限。停滞不视为成功，`bool(ok)` 会如实返回 False。
        """
        best = None
        no_gain = 0
        self.last_err = (None, None)
        self.conv = {"pos": False, "rot": False}
        for k in range(max_steps):
            if self.n >= FRAME_LIMIT:
                return False, k
            sim, m, d, gid = N.ids(self.env)
            p, R = N.gripsite(sim, m, d, gid)
            dv = np.asarray(target, dtype=float) - p
            nd = float(np.linalg.norm(dv))
            if nd > max_step_pos:
                dv = dv * (max_step_pos / nd)
            ap = np.clip(dv / TEACHER_GAIN_POS_M, -1.0, 1.0)
            er = N.rot_err_vec(R_target, R)
            ne = float(np.linalg.norm(er))
            if ne > max_step_rot:
                er = er * (max_step_rot / ne)
            ar = np.clip(er / TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
            ap = np.clip(ap, -max_step_pos / OSC_POS_UNIT_M,
                         max_step_pos / OSC_POS_UNIT_M)
            ar = np.clip(ar, -max_step_rot / OSC_ROT_UNIT_RAD,
                         max_step_rot / OSC_ROT_UNIT_RAD)
            self.act(np.concatenate([ap, ar, [grip_cmd]]), phase)
            if verbose and (k < 2 or (k + 1) % 20 == 0):
                bb = N4.bbox(sim, self.bid) if self.bid is not None else None
                print("      [%-9s %3d] |e|=%.4f  |rot_e|=%.4f  disp=%.4f m  objz=%s"
                      % (phase, k + 1, nd, ne, self.step_disps[-1],
                         "n/a" if bb is None else "%.4f" % (0.5 * (bb[0] + bb[1]))[2]))
            self.last_err = (nd, ne)
            self.conv = {"pos": bool(nd < tol), "rot": bool(ne < rot_tol)}
            if nd < tol and ne < rot_tol:
                return True, k + 1
            score = nd + 0.1 * ne
            if best is None or score < best - 2e-4:
                best = score
                no_gain = 0
            else:
                no_gain += 1
                if no_gain >= stall_patience:
                    return False, k + 1
        return False, max_steps


# --------------------------------------------------------------------------- #
# 单 trial：完整 7 维 OSC 链
# --------------------------------------------------------------------------- #
def _drive_to(ep, target, R_ref, grip, phase, steps):
    """在 place/release/settle/window 段用：朝目标保持，连发 steps 步（同样带步长限制）。"""
    return ep.track(target, R_ref, grip, phase, steps)


def run_trial(env, args, spec, log, verbose=False):
    """spec: dict(width_mm, yaw_rad, seed, tag, trial, trial_label)"""
    t0 = time.time()
    mon = ContactMonitor()
    ep = ChainEpisode(env, mon)
    env.seed(spec["seed"])
    ep.start()
    sim, m, d, gid = N.ids(env)

    obj_name, region_name = N4.goal_objects(env)
    bid, bname = N4.find_body(sim, obj_name)
    rid, rname = N4.find_body(sim, region_name)
    if bid is None or rid is None:
        raise RuntimeError("找不到目标物体 %s 或区域 %s" % (obj_name, region_name))

    # 闭合轴（世界系，水平投影）+ 复位姿态 → 决定试块初始偏航，
    # 使"宽度"正好落在**重定向之后**的闭合轴上。
    axis_w = closing_axis_world(env)
    if axis_w is None:
        raise RuntimeError("无法确定夹爪闭合轴")
    R_home0 = N.gripsite(sim, m, d, gid)[1].copy()
    # 重定向：绕**本体 z（工具轴）**旋转 —— 与 N4 的 `R_ref @ rotz(pi/2)` 同一约定。
    # 实测（n3_osc_diag_reorient.py）：绕工具轴可达（残差 0.013–0.065 rad），
    # 而绕世界 z 的等效目标在同一姿态下达不到（残差 0.206 rad = 11.8°，直接导致
    # 夹取面与目标截面不平行、指垫落到试块顶面）。
    R_ref0 = R_home0 @ rotz(spec["yaw_rad"]) if abs(spec["yaw_rad"]) > 1e-9 \
        else R_home0.copy()
    # 试块宽度方向必须对齐**重定向后真实的闭合轴**。用 `R_ref` 把 home 的闭合轴
    # 变换过去（v_local = R_home^T·axis_w；axis_after = R_ref·v_local），
    # 与旋转是左乘/右乘无关，避免"帧语义不一致 → 夹取面不平行"。
    axis_after = R_ref0 @ (R_home0.T @ axis_w)
    phi_eff = float(np.arctan2(axis_after[1], axis_after[0]))
    obj_yaw = float(phi_eff - np.pi / 2.0)     # 试块局部 y（宽度方向）对齐闭合轴

    # ---- 阶段 0（不记录）：BDDL 区域内随机重摆（复用 N4.sample_object_init）+ 静置 ----
    rng = np.random.default_rng(900000 + spec["seed"])
    init_sample = N4.sample_object_init(env, bid, obj_name, bname, region_name, rng)
    if init_sample is None:
        raise RuntimeError("区域初始化采样失败")
    # 采样函数会给随机偏航；试块宽度必须落在闭合轴上，故按已记录的 xy **绝对**重设偏航
    yaw_before = object_yaw(env, bid)
    set_object_yaw(env, bid, init_sample["xy"], obj_yaw)
    yaw_after = object_yaw(env, bid)
    init_sample["yaw_rad"] = round(yaw_after, 6)
    init_sample["yaw_forced_rad"] = round(obj_yaw, 6)
    init_sample["yaw_before_forced_rad"] = round(yaw_before, 6)
    init_sample["yaw_note"] = ("试块为长方体，宽度必须与闭合轴对齐，故偏航按闭合轴"
                               "反解并**绝对**设置；xy 仍来自 N4.sample_object_init")

    prep = args.prep_steps
    obs_prep = N4.quiet_hold(env, prep, GRIP_OPEN)
    ep.raw_steps += prep
    ep.refresh_obs(obs_prep, "post_init_quiet_hold")

    sim, m, d, gid = N.ids(env)
    mon.bind(env, bid)
    ep.bid = bid
    ob, oh = N4.bbox(sim, bid)
    obj_c0 = 0.5 * (ob + oh)
    rb, rh = N4.bbox(sim, rid)
    R_home = N.gripsite(sim, m, d, gid)[1].copy()
    R_ref = R_home @ rotz(spec["yaw_rad"]) if abs(spec["yaw_rad"]) > 1e-9 \
        else R_home.copy()
    # 静置前后复位姿态的一致性（若不一致，说明静置期姿态漂了）
    home_drift_rad = float(np.linalg.norm(N.rot_err_vec(R_home0, R_home)))

    h = float(oh[2] - ob[2])
    z_g = float(min(ob[2] + max(0.028, min(0.5 * h, h - 0.006)), oh[2] - 0.004))
    pin = np.array([0.5 * (ob[0] + oh[0]), 0.5 * (ob[1] + oh[1]), z_g])

    ep.act(np.concatenate([np.zeros(6), [GRIP_OPEN]]), "settle")   # 第 0 帧
    hook_installed = install_contact_hook(env)
    _ACTIVE_MON["m"] = mon
    arm_ik_guard()
    if verbose:
        print("      [init] 试块碰撞几何 x=%.1f y=%.1f z=%.1f mm  中心=%s  闭合轴=%s  "
              "物体偏航=%.2f°（期望 %.2f°）"
              % ((oh[0] - ob[0]) * 1000, (oh[1] - ob[1]) * 1000, (oh[2] - ob[2]) * 1000,
                 np.round(0.5 * (ob + oh), 4).tolist(), np.round(axis_w, 4).tolist(),
                 np.degrees(object_yaw(env, bid)), np.degrees(obj_yaw)))

    rec = {
        "trial": spec["trial"], "trial_label": spec["trial_label"], "tag": spec["tag"],
        "seed": spec["seed"],
        "block_extent_mm": {"x_depth": BLOCK_DEPTH_M * 1000,
                            "y_width_along_closing_axis": spec["width_mm"],
                            "z_height": BLOCK_HEIGHT_M * 1000},
        "block_mass_kg": BLOCK_DEPTH_M * (spec["width_mm"] / 1000.0) * BLOCK_HEIGHT_M
                         * BLOCK_DENSITY,
        "gripper_yaw_deg": float(np.degrees(spec["yaw_rad"])),
        "closing_axis_world_at_home": [float(v) for v in axis_w],
        "home_orientation_drift_during_prep_rad": home_drift_rad,
        "closing_axis_expected_after_reorient": [float(v) for v in axis_after],
        "object_init_sample": init_sample,
        "object_bbox_lo": [float(v) for v in ob], "object_bbox_hi": [float(v) for v in oh],
        "object_start_center": [float(v) for v in obj_c0],
        "grasp_pinch_target": [float(v) for v in pin], "grasp_z": z_g,
        "basket_bbox_lo": [float(v) for v in rb], "basket_bbox_hi": [float(v) for v in rh],
        "contact_hook_installed": bool(hook_installed),
        "phases": [], "gates": {}, "notes": [],
    }
    pe = {}

    def obj_center():
        s2, _, _, _ = N.ids(env)
        bb = N4.bbox(s2, bid)
        return None if bb is None else 0.5 * (bb[0] + bb[1])

    native_err = {"e": None}

    def native():
        try:
            return native_success(env)
        except Exception as e:                                # noqa: BLE001
            native_err["e"] = "%s: %s" % (type(e).__name__, e)
            return False

    try:
        # ---- 1. 可选：命令式重定向（全程 OSC 旋转） ----
        if abs(spec["yaw_rad"]) > 1e-9:
            p0 = N.gripsite(*N.ids(env))[0].copy()
            ok, k = ep.servo_limited(p0, R_ref, GRIP_OPEN, "reorient", 400,
                                     tol=0.005, rot_tol=0.02, verbose=verbose)
            sim, m, d, gid = N.ids(env)
            R_now = N.gripsite(sim, m, d, gid)[1]
            ax_now = closing_axis_world(env)
            pe["reorient_steps"] = k
            pe["reorient_done"] = bool(ok)
            pe["reorient_converged_rot"] = bool(ep.conv["rot"])
            pe["reorient_residual_rad"] = float(np.linalg.norm(N.rot_err_vec(R_ref, R_now)))
            pe["closing_axis_after_reorient"] = None if ax_now is None else \
                [float(v) for v in ax_now]
            # 试块"宽度方向"（局部 y）在世界系的方向：由当前姿态取
            yaw_now = object_yaw(env, bid)
            width_dir = np.array([-np.sin(yaw_now), np.cos(yaw_now), 0.0])
            pe["block_width_dir_world"] = [float(v) for v in width_dir]
            if ax_now is not None:
                cosang = float(np.clip(abs(np.dot(ax_now, width_dir)), 0.0, 1.0))
                pe["closing_axis_vs_block_width_deg"] = float(
                    np.degrees(np.arccos(cosang)))
                # 该偏角下的等效截面宽（决定指垫还能不能跨住试块）
                bw, bd = (spec["width_mm"] / 1000.0, BLOCK_DEPTH_M)
                th = np.arccos(cosang)
                pe["effective_section_width_mm"] = float(
                    (bw * np.cos(th) + bd * np.sin(th)) * 1000)

        # ---- 2. 接近 ----
        ok, k = ep.servo_limited(pin + np.array([0, 0, 0.08]), R_ref, GRIP_OPEN,
                                 "approach", 260, verbose=verbose)
        pe["approach"] = [bool(ok), k]
        pe["approach_residual_m"] = ep.last_err[0]

        # ---- 3. 下降 ----
        ok, k = ep.servo_limited(pin, R_ref, GRIP_OPEN, "descend", 260, tol=0.008,
                                 verbose=verbose)
        pe["descend"] = [bool(ok), k]
        pe["descend_converged_pos"] = bool(ep.conv["pos"])
        pe["descend_residual_m"] = ep.last_err[0]
        _s, _m, _d, _ = N.ids(env)
        _pads = pad_geoms(_s)
        _pl = np.array(_d.geom_xpos[_pads["left"]], dtype=float)
        _pr = np.array(_d.geom_xpos[_pads["right"]], dtype=float)
        _bb = N4.bbox(_s, bid)
        pe["descend_pad_mid_z"] = float(0.5 * (_pl[2] + _pr[2]))
        pe["descend_block_top_z"] = float(_bb[1][2])
        pe["descend_pad_below_block_top_mm"] = float(
            (_bb[1][2] - 0.5 * (_pl[2] + _pr[2])) * 1000)
        pe["grip_site_at_descend"] = [float(v) for v in N.gripsite(*N.ids(env))[0]]

        # ---- 4. 闭合（闭合到停住；臂由 OSC 保持，不逐子步锁臂） ----
        l1p, stable, close_steps = None, 0, 0
        for _ in range(args.close_max_steps):
            if ep.n >= FRAME_LIMIT:
                break
            sim, m, d, gid = N.ids(env)
            p, R = N.gripsite(sim, m, d, gid)
            dv = pin - p
            nd = float(np.linalg.norm(dv))
            if nd > SERVO_MAX_STEP_POS_M:
                dv = dv * (SERVO_MAX_STEP_POS_M / nd)
            ap = np.clip(dv / TEACHER_GAIN_POS_M, -1, 1)
            er = N.rot_err_vec(R_ref, R)
            ne = float(np.linalg.norm(er))
            if ne > SERVO_MAX_STEP_ROT_RAD:
                er = er * (SERVO_MAX_STEP_ROT_RAD / ne)
            ar = np.clip(er / TEACHER_GAIN_ROT_RAD, -1, 1)
            ap = np.clip(ap, -SERVO_MAX_STEP_POS_M / OSC_POS_UNIT_M,
                         SERVO_MAX_STEP_POS_M / OSC_POS_UNIT_M)
            ar = np.clip(ar, -SERVO_MAX_STEP_ROT_RAD / OSC_ROT_UNIT_RAD,
                         SERVO_MAX_STEP_ROT_RAD / OSC_ROT_UNIT_RAD)
            ep.act(np.concatenate([ap, ar, [GRIP_CLOSE]]), "close")
            close_steps += 1
            l1 = N4.left1(env)
            if close_steps >= args.close_min_steps and l1p is not None \
                    and abs(l1 - l1p) < 1e-4:
                stable += 1
                if stable >= args.close_stable_checks:
                    break
            else:
                stable = 0
            l1p = l1
            if verbose and (close_steps <= 3 or close_steps % 20 == 0):
                print("      [close %3d] Left_1=%.4f  jaw=%.1f mm  pen_peak=%.3f mm"
                      % (close_steps, l1, N.jaw_face_separation_mm(env) or -1.0,
                         mon.peak_pad))
        pe["close_steps"] = close_steps
        pe["left1_after_close"] = N4.left1(env)
        pe["jaw_after_close_mm"] = N.jaw_face_separation_mm(env)
        c_close = obj_center()
        pe["obj_shift_on_close_mm"] = (None if c_close is None
                                       else float(np.linalg.norm(c_close - obj_c0) * 1000))
        pe["close_penetration_peak_mm"] = mon.per_phase.get("close", {}).get("pad_peak_mm")

        # ---- 5. 抬升 ----
        lift_target = pin + np.array([0, 0, args.lift_m])
        ok, k = ep.servo_limited(lift_target, R_ref, GRIP_CLOSE, "lift", 260,
                                 tol=0.004, verbose=verbose)
        pe["lift_servo"] = [bool(ok), k]
        pe["lift_residual_m"] = ep.last_err[0]
        c_lift = obj_center()
        pe["lift_mm"] = None if c_lift is None else float((c_lift[2] - obj_c0[2]) * 1000)

        # ---- 6. 保持 2 s（40 控制步），逐步记录高度 ----
        hold_h = []
        for _ in range(THRESHOLDS["hold_steps"]):
            if ep.n >= FRAME_LIMIT:
                break
            _drive_to(ep, lift_target, R_ref, GRIP_CLOSE, "hold", 1)
            c = obj_center()
            hold_h.append(float("nan") if c is None else float((c[2] - obj_c0[2]) * 1000))
        pe["hold_steps_used"] = len(hold_h)
        pe["hold_min_mm"] = float(np.nanmin(hold_h)) if hold_h else None
        pe["hold_max_mm"] = float(np.nanmax(hold_h)) if hold_h else None
        pe["hold_end_mm"] = float(hold_h[-1]) if hold_h else None
        pe["hold_penetration_peak_mm"] = mon.per_phase.get("hold", {}).get("pad_peak_mm")

        # ---- 7. 搬运 → 投放 ----
        base = np.array(env.env.sim.data.body_xpos[
            env.env.sim.model.body_name2id("robot0_base")], dtype=float)
        frac = 0.35
        near_x = rb[0] + frac * (rh[0] - rb[0]) if base[0] < rb[0] else \
            rh[0] - frac * (rh[0] - rb[0])
        bx = np.array([near_x, 0.5 * (rb[1] + rh[1])])
        above = np.array([bx[0], bx[1], float(rh[2]) + 0.12])
        into = np.array([bx[0], bx[1], float(rh[2]) + 0.03])
        pe["place_point"] = [float(v) for v in into]
        pe["basket_inner_size_mm"] = [float(rh[0] - rb[0]) * 1000,
                                      float(rh[1] - rb[1]) * 1000,
                                      float(rh[2] - rb[2]) * 1000]

        pe["sep_at_transport_mm"] = N.jaw_face_separation_mm(env) or 0.0
        ok, k = ep.servo_limited(above, R_ref, GRIP_CLOSE, "transport", 360,
                                 tol=0.008, verbose=verbose)
        pe["transport"] = [bool(ok), k]
        pe["transport_residual_m"] = ep.last_err[0]
        pe["transport_grip_site"] = [float(v) for v in N.gripsite(*N.ids(env))[0]]
        pe["sep_after_transport_mm"] = N.jaw_face_separation_mm(env) or 0.0
        c_tr = obj_center()
        lost = bool(c_tr is None or (c_tr[2] - obj_c0[2]) < 0.02)
        if lost:
            rec["notes"].append("搬运段物体已脱落（高度回落），跳过下潜投放")
        else:
            ok, k = ep.servo_limited(into, R_ref, GRIP_CLOSE, "place", 300,
                                     tol=0.008, verbose=verbose)
            pe["place"] = [bool(ok), k]
            pe["place_residual_m"] = ep.last_err[0]
            pe["place_grip_site"] = [float(v) for v in N.gripsite(*N.ids(env))[0]]

        # ---- 8. 释放 ----
        _drive_to(ep, into, R_ref, GRIP_OPEN, "release", args.release_steps)

        # ---- 9. 落稳（最多 N 步等到线速度小） ----
        settle_used = 0
        _, jadr = N4.free_joint_adr(env, bid)
        for _ in range(THRESHOLDS["post_release_settle_max_steps"]):
            _drive_to(ep, into, R_ref, GRIP_OPEN, "post_release_settle", 1)
            settle_used += 1
            v = float(np.linalg.norm(np.array(N.ids(env)[2].qvel[jadr:jadr + 3])))
            if v < 0.008:
                break
        pe["settle_steps"] = settle_used

        # ---- 10. 释放后稳定窗口 20 步 ----
        c_win0 = obj_center()
        inside_flags, drifts = [], []
        for _ in range(THRESHOLDS["post_release_steps"]):
            if ep.n >= FRAME_LIMIT:
                break
            _drive_to(ep, into, R_ref, GRIP_OPEN, "post_release_window", 1)
            c = obj_center()
            if c is None:
                inside_flags.append(False)
                drifts.append(float("nan"))
                continue
            inside_flags.append(bool(rb[0] - 0.03 <= c[0] <= rh[0] + 0.03
                                     and rb[1] - 0.03 <= c[1] <= rh[1] + 0.03))
            drifts.append(float(np.linalg.norm(c - c_win0) * 1000))
        c_win1 = obj_center()
        pe["post_release_inside_all"] = bool(all(inside_flags)) if inside_flags else False
        pe["post_release_inside_frac"] = (float(np.mean(inside_flags))
                                         if inside_flags else 0.0)
        pe["post_release_drift_mm"] = float(drifts[-1]) if drifts else None
        pe["post_release_drift_max_mm"] = float(np.nanmax(drifts)) if drifts else None
        pe["post_release_window_steps"] = len(drifts)
        pe["post_release_center"] = None if c_win1 is None else [float(v) for v in c_win1]
        pe["post_release_penetration_peak_mm"] = \
            mon.per_phase.get("post_release_window", {}).get("pad_peak_mm")

        # ---- 11. 撤退 ----
        ep.servo_limited(above, R_ref, GRIP_OPEN, "retreat", 200, tol=0.01,
                         verbose=verbose)
        pe["retreat_residual_m"] = ep.last_err[0]
        pe["reach_limited"] = bool(
            (pe.get("transport_residual_m") or 0.0) > 0.02
            or (pe.get("place_residual_m") or 0.0) > 0.02)
    except Exception as e:                                     # noqa: BLE001
        rec["notes"].append("例外：%s" % e)
        log("    [!] trial 例外：%s" % e)
    finally:
        _ACTIVE_MON["m"] = None

    rec["phase_end"] = pe
    rec["n_frames"] = ep.n
    rec["ik_calls_during_episode"] = len(_IK_GUARD["violations"])
    rec["max_step_disp_m"] = ep.max_step_disp
    rec["max_step_disp_per_phase"] = {k: float(v) for k, v in ep.disp_phase.items()}
    rec["penetration_peak_pad_mm"] = mon.peak_pad
    rec["penetration_peak_object_any_mm"] = mon.peak_any
    rec["penetration_sampling"] = ("每个**物理子步**（`_pre_action` 钩子，500 Hz）+ 每控制步末"
                                   "（共 %d 次采样，其中 %d 次存在指垫接触）"
                                   % (mon.n_samples, mon.n_pad_contacts))
    rec["penetration_per_phase_mm"] = {
        k: {"pad_peak_mm": v["pad_peak_mm"], "any_peak_mm": v["any_peak_mm"],
            "samples": v["samples"]} for k, v in mon.per_phase.items()}
    rec["native_success_error"] = native_err["e"] or ep.native_error
    rec["native_success_predicate_fn"] = "OffScreenRenderEnv.check_success() -> env._check_success()"
    rec["predicate_debug"] = predicate_debug(env, obj_name, region_name)
    rec["native_success_first_true_step"] = ep.native_first_true
    rec["native_success_steps"] = ep.native_true_steps
    rec["native_success_at_end"] = native()

    g = {}
    g["lift_ge_30mm"] = bool(pe.get("lift_mm") is not None
                             and pe["lift_mm"] >= THRESHOLDS["lift_min_mm"])
    g["hold_ge_25mm_40steps"] = bool(
        pe.get("hold_steps_used", 0) >= THRESHOLDS["hold_steps"]
        and pe.get("hold_min_mm") is not None
        and pe["hold_min_mm"] >= THRESHOLDS["hold_min_mm"])
    g["place_release_stable"] = bool(
        pe.get("post_release_window_steps", 0) >= THRESHOLDS["post_release_steps"]
        and pe.get("post_release_inside_all")
        and pe.get("post_release_drift_mm") is not None
        and pe["post_release_drift_mm"] < THRESHOLDS["post_release_drift_max_mm"])
    g["native_libero_predicate"] = bool(rec["native_success_at_end"])
    g["no_teleport"] = bool(ep.max_step_disp < THRESHOLDS["max_step_disp_m"])
    g["penetration_within_soft_limit"] = bool(
        mon.peak_pad <= THRESHOLDS["penetration_soft_limit_mm"])
    g["no_ik_qpos_write"] = bool(rec["ik_calls_during_episode"] == 0)
    # [复核 §3.2.4] 接近 / 下降 / 重定向阶段**未收敛即失败**。
    # 复核指出："当前流程可在重定向未收敛后继续抓取，抓取成功不能替代姿态要求通过。"
    # `servo_limited` 的返回值 ok 就是该段的收敛标志（§3.2.3 的时序修正已在其中：
    # 误差在控制步执行后重读，位置与角度同时判）。
    g["approach_converged"] = bool(pe.get("approach", [False])[0])
    g["descend_converged"] = bool(pe.get("descend", [False])[0])
    # 未命令重定向时该项不适用（默认 True）；一旦命令了就必须收敛
    g["reorient_converged"] = bool(pe.get("reorient_converged_rot", True))
    rec["gates"] = g
    rec["success"] = bool(all(g.values()))
    rec["failed_gates"] = [k for k, v in g.items() if not v]
    rec["wall_seconds"] = round(time.time() - t0, 1)

    trace = {
        "grip_site": np.array(ep.site_path, dtype=np.float32),
        "obj_center": np.array(ep.obj_path, dtype=np.float32),
        "step_disp": np.array(ep.step_disps, dtype=np.float32),
        "pen_pad_mm": np.array(ep.pen_trace, dtype=np.float32),
        "phase": np.array([f["phase"] for f in ep.frames], dtype=object),
        "native_success": np.array([f["native_success"] for f in ep.frames], dtype=np.int8),
    }
    ep.frames = []
    return rec, trace


# --------------------------------------------------------------------------- #
# 位姿跟踪探针（±x/±y/±z 平移 + ±rx/±ry/±rz 旋转）
# --------------------------------------------------------------------------- #
def run_probe(env, kind, axis_name, sign, args, log):
    """单个闭环位姿跟踪探针：**真的下发**命令位移/旋转，并记录"命令 vs 实达"。

    判据（两者都必须成立，缺一即 fail；`err` 缺失**不得**判 pass）：
      * 平移：`converged_pos == True` 且 `err_mm <= translation_probe_err_max_mm`
      * 旋转：`converged_rot == True` 且 `err_rad <= rotation_probe_err_max_rad`
             且 `geodesic_err_rad <= rotation_probe_err_max_rad`
    """
    mon = ContactMonitor()
    ep = ChainEpisode(env, mon)
    env.seed(args.seed_base + 700)
    ep.start()
    obs = N4.quiet_hold(env, args.prep_steps, GRIP_OPEN)
    ep.raw_steps += args.prep_steps
    ep.refresh_obs(obs, "probe_prep")
    mon.bind(env, None)                       # 探针不夹物体：穿透闸门不适用
    ep.act(np.concatenate([np.zeros(6), [GRIP_OPEN]]), "probe_settle")
    arm_ik_guard()

    axis = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
            "z": np.array([0, 0, 1.0])}[axis_name]
    p0, R0 = N.gripsite(*N.ids(env))
    p0, R0 = p0.copy(), R0.copy()
    name = "%s%s%s" % ("+" if sign > 0 else "-", axis_name,
                       "" if kind == "trans" else "r")
    if kind == "trans":
        cmd = THRESHOLDS["translation_probe_cmd_m"]
        target, R_t = p0 + axis * sign * cmd, R0
        tol, rot_tol, maxs = 0.002, 0.03, 300
    else:
        # ★ 真的下发旋转：R_t = expm(axis·σ·cmd) @ R0（世界系）
        cmd = THRESHOLDS["rotation_probe_cmd_rad"]
        R_t = expm_so3(axis * sign * cmd) @ R0
        target, tol, rot_tol, maxs = p0, 0.005, 0.03, 300
    ok, steps = ep.servo_limited(target, R_t, GRIP_OPEN, "probe_%s" % name,
                                 maxs, tol=tol, rot_tol=rot_tol,
                                 stall_patience=args.probe_stall_patience)
    conv_pos, conv_rot = bool(ep.conv["pos"]), bool(ep.conv["rot"])
    # 收敛后再驻留 10 步（0.5 s）再读数：OSC 最后一步的过冲属于瞬态
    ep.track(target, R_t, GRIP_OPEN, "probe_%s_settle" % name, 10)
    p1, R1 = N.gripsite(*N.ids(env))
    p1, R1 = p1.copy(), R1.copy()

    rec = {"probe": name, "axis": axis_name, "sign": sign, "kind":
           "translation" if kind == "trans" else "rotation",
           "steps": steps, "converged": bool(ok),
           "converged_pos": conv_pos, "converged_rot": conv_rot,
           "converge_tol_pos_m": tol, "converge_tol_rot_rad": rot_tol,
           "settle_steps": 10,
           "note": "收敛后驻留 10 个控制步再读数（消除最后一步瞬态过冲）",
           "max_step_disp_m": ep.max_step_disp,
           "ik_calls_during_episode": len(_IK_GUARD["violations"]),
           "residual_pos_mm": float(np.linalg.norm(target - p1) * 1000),
           "residual_rot_rad": float(np.linalg.norm(N.rot_err_vec(R_t, R1))),
           "geodesic_err_rad": float(np.linalg.norm(N.rot_err_vec(R_t, R1))),
           "geodesic_err_deg": float(np.degrees(np.linalg.norm(N.rot_err_vec(R_t, R1))))}
    if kind == "trans":
        achieved = float(np.dot(p1 - p0, axis) * sign)
        err = abs(achieved - cmd)
        rec.update({"commanded_mm": cmd * 1000, "commanded_rad": None,
                    "achieved_mm": achieved * 1000, "achieved_rad": None,
                    "err_mm": err * 1000, "err_rad": None})
        # 缺值不得判 pass
        rec["pass"] = bool(rec["err_mm"] is not None and conv_pos
                           and err * 1000 < THRESHOLDS["translation_probe_err_max_mm"])
        rec["criterion"] = ("converged_pos == True 且 err_mm < %.1f mm"
                            % THRESHOLDS["translation_probe_err_max_mm"])
    else:
        # `N.rot_err_vec(A,B)` = "把 B 转到 A"的世界系旋转向量；
        # 测"从 R0 实际转到哪"必须写 rot_err_vec(R1, R0)，否则符号相反。
        rv = N.rot_err_vec(R1, R0)
        achieved = float(np.dot(rv, axis))
        err = abs(achieved - sign * cmd)
        geo = rec["geodesic_err_rad"]
        rec.update({"commanded_rad": cmd, "commanded_deg": float(np.degrees(cmd)),
                    "commanded_mm": None,
                    "achieved_rad": achieved, "achieved_deg": float(np.degrees(achieved)),
                    "err_rad": err, "err_deg": float(np.degrees(err)),
                    "err_mm": None})
        rec["pass"] = bool(conv_rot and rec["err_rad"] is not None
                           and geo is not None
                           and err <= THRESHOLDS["rotation_probe_err_max_rad"]
                           and geo <= THRESHOLDS["rotation_probe_err_max_rad"])
        rec["criterion"] = ("converged_rot == True 且 err_rad <= %.3f rad 且 "
                            "geodesic_err_rad <= %.3f rad"
                            % (THRESHOLDS["rotation_probe_err_max_rad"],
                               THRESHOLDS["rotation_probe_err_max_rad"]))
    ep.frames = []
    return rec

def make_env(args, log):
    env, task = N4.build(args.suite, args.task_id)
    spd = float(env.env.robots[0].gripper.speed)
    if abs(spd - args.gripper_speed) > 1e-9:
        raise RuntimeError("[FATAL] ZHIXING_GRIPPER_SPEED 注入失败：期望 %.4f，实得 %.4f"
                           % (args.gripper_speed, spd))
    return env, task


def _f(v, nd=1):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def predicate_debug(env, obj_name, region_name):
    """把原生谓词 `In(obj, region)` 的每个中间量摊开，便于归因（不参与判据）。"""
    out = {}
    try:
        domain = getattr(env, "env", env)
        out["goal_state"] = domain.parsed_problem.get("goal_state")
        out["object_sites_keys"] = list(domain.object_sites_dict.keys())
        out["has_region_state"] = region_name in domain.object_states_dict
        so = domain.object_sites_dict.get(region_name)
        if so is None:
            out["error"] = "object_sites_dict 缺少 %r" % region_name
            return out
        pos = np.array(domain.sim.data.get_site_xpos(region_name), dtype=float)
        mat = np.array(domain.sim.data.get_site_xmat(region_name),
                       dtype=float).reshape(3, 3)
        bid = domain.obj_body_id.get(obj_name)
        opos = np.array(domain.sim.data.body_xpos[bid], dtype=float) if bid is not None \
            else None
        size = so.size if not isinstance(so.size, str) else None
        out.update({
            "site_name": so.name, "parent_name": so.parent_name,
            "site_pos": pos.tolist(),
            "site_size": None if size is None else [float(v) for v in size],
            "obj_body_id": None if bid is None else int(bid),
            "obj_body_xpos": None if opos is None else opos.tolist(),
            "check_contact": bool(domain.object_states_dict[region_name].check_contact(
                domain.object_states_dict[obj_name])) if out["has_region_state"] else None,
            "in_box": None if opos is None else bool(so.in_box(pos, mat, opos)),
            "eval_predicate": bool(domain._eval_predicate(
                ["in", obj_name, region_name])),
        })
    except Exception as e:                                     # noqa: BLE001
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "full", "probes", "widths", "reorient"],
                    default="full",
                    help="probe=12 探针+1 trial 诊断；probes=只跑探针；widths=只跑宽度组；"
                         "reorient=只跑重定向组；full=全部")
    ap.add_argument("--merge", action="store_true",
                    help="把本次产生的段落合并进已存在的 OUT_JSON（保留其它组的历史结论）")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--widths-mm", default="30,50,70")
    ap.add_argument("--trials", type=int, default=THRESHOLDS["trials_per_width"])
    ap.add_argument("--reorient-trials", type=int, default=0,
                    help="0 = 自动（每宽度 2 个：+90° / −90° 偏航后抓取）；-1 = 跳过该组")
    ap.add_argument("--seed-base", type=int, default=41000)
    ap.add_argument("--gripper-speed", type=float, default=0.01)
    ap.add_argument("--prep-steps", type=int, default=120,
                    help="episode 前静置/全开夹爪的控制步数（speed=0.01 时开满需 100 步）")
    ap.add_argument("--close-max-steps", type=int, default=700)
    ap.add_argument("--close-min-steps", type=int, default=60)
    ap.add_argument("--close-stable-checks", type=int, default=8)
    ap.add_argument("--release-steps", type=int, default=60)
    ap.add_argument("--lift-m", type=float, default=0.08)
    ap.add_argument("--probe-stall-patience", type=int, default=100)
    ap.add_argument("--probes", dest="probes", action="store_true", default=True)
    ap.add_argument("--no-probes", dest="probes", action="store_false")
    a = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    log("=" * 78)
    log("[N3-OSC] 全程 7 维 OSC 动作链抓取—抬升—保持—搬运—放置—释放（审核 R7 返工）")
    log("[N3-OSC] suite=%s task=%d 宽度=%s 每宽度 %d trials mode=%s"
        % (a.suite, a.task_id, a.widths_mm, a.trials, a.mode))

    widths = [float(x) for x in a.widths_mm.split(",") if x.strip()]
    if a.mode == "probe":
        widths = widths[:1]
        a.trials = max(1, a.trials)

    block_cls = block_object_class(widths[0] / 1000.0, BLOCK_DEPTH_M, BLOCK_HEIGHT_M)
    install_object_substitution("alphabet_soup", block_cls, log)
    env, task = N4.build(a.suite, a.task_id)
    cat, obj_name, region_name = target_category(env)
    if cat is None:
        log("[FATAL] 无法从 BDDL 反查目标物体类别")
        return 2
    ctl = N4.controller_action_contract(env)
    tcp = N4.eef_pos_of(env)
    spd = float(env.env.robots[0].gripper.speed)
    log("[bddl] 目标物体=%s 类别=%s 目标区域=%s 任务='%s'"
        % (obj_name, cat, region_name, task.language))
    log("[grip] ZhixingGripper.speed = %.4f（部署值，来自 ZHIXING_GRIPPER_SPEED）" % spd)
    log("[tcp] eef body pos = %s   zhixing xml sha16 = %s" % (tcp, sha16(N4.XML_PATH)))
    log("[ctl] %s output_max=%s control_timestep=%.4f"
        % (ctl["controller_type"], ctl["controller_output_max"],
           ctl["control_timestep"]))
    ax = closing_axis_world(env)
    log("[axis] 复位位姿下闭合轴（世界系）= %s → 试块局部 y（宽度）按此对齐"
        % ("n/a" if ax is None else np.round(ax, 4).tolist()))
    env.close()

    prev = None
    if a.merge and OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            log("[merge] 载入既有结果 %s（生成于 %s），只替换本次跑到的段落"
                % (OUT_JSON.name, prev.get("generated_at")))
        except Exception as e:                                 # noqa: BLE001
            log("[merge] 读取失败，改为全量写出：%s" % e)
            prev = None

    result = {
        "node": "N3-osc-full-chain", "audit_item": "R7",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite": a.suite, "task_id": a.task_id, "task_language": task.language,
        "method": ("全程 7 维 OSC 动作链（无 IK qpos 直写、无逐子步锁臂）；"
                   "reuse n4_collect.Episode，教师增益/采样/TCP 与采集链路一致"),
        "thresholds": dict(THRESHOLDS),
        "controller": ctl,
        "tcp_eef_body_pos": tcp,
        "closing_axis_world_at_home": None if ax is None else [float(v) for v in ax],
        "assets": {"zhixing_gripper_xml_sha16": sha16(N4.XML_PATH),
                   "eco65_robot_xml_sha16": sha16(N4.ROBOT_XML),
                   "script_sha16": sha16(pathlib.Path(__file__)),
                   "collector_sha16": sha16(N4.COLLECTOR)},
        "object_substitution": {
            "category": cat, "bddl_object": obj_name, "goal_region": region_name,
            "block_mm": {"x_depth": BLOCK_DEPTH_M * 1000,
                         "y_width": "per group (30/50/70)",
                         "z_height": BLOCK_HEIGHT_M * 1000},
            "density": BLOCK_DENSITY,
            "geom": ("box, contype/conaffinity=1, condim=4, friction='1 0.005 0.0001', "
                     "solref='0.02 1'（与历史 n3r 试块一致）"),
            "why": ("让试块**就是** BDDL 目标物体，原生谓词 env.check_success() 才真正适用；"
                    "代价是物性沿用试块参数而非 HOPE 罐体参数，已在此显式声明"),
            "disk_assets_modified": False,
            "grasp_offset_closed_used": False,
            "grasp_offset_closed_note": ("历史常数 GRASP_OFFSET_CLOSED=[-0.00006,0.00002,0.08254]"
                                         "（TCP 前移到 0.20254 后失效）**未被使用**；"
                                         "所有目标点由 grip_site 的实测世界位姿表达"),
        },
        "gripper_speed": spd,
        "groups": {}, "probes": [], "verdict": None, "verdict_reason": "",
    }
    # [复核 §3.2.5 / §8C] 单一冻结版本溯源：整份结果必须来自**同一个**代码/参数版本，
    # 因此把生效的 PD profile、速度与 10 个资产哈希写进结果本身，
    # 供事后判定"这一份到底是不是同版本跑出来的"，杜绝再次出现拼接结果。
    try:
        import runtime_config as _RC
        result["runtime"] = _RC.fingerprint()
        log("[runtime] 冻结版本=%s  PD profile=%s  speed=%s"
            % (result["runtime"].get("freeze_version"),
               result["runtime"].get("extra_joint_pd_profile"),
               result["runtime"].get("gripper_speed")))
    except Exception as e:                                     # noqa: BLE001
        result["runtime"] = {"error": "%s: %s" % (type(e).__name__, e)}
        log("[runtime] 无法读取 runtime_config：%s" % e)

    t_start = time.time()
    traces = {}

    if prev is not None:
        result["groups"] = dict(prev.get("groups", {}))
        if a.mode not in ("probes", "probe"):
            result["probes"] = prev.get("probes", [])
            result["probe_summary"] = prev.get("probe_summary")
        if a.mode not in ("reorient",):
            result["_keep_reorient"] = prev.get("groups", {}).get("reorient_grasp")

    # ---------------- 探针 ----------------
    if a.probes and a.mode in ("full", "probe", "probes"):
        log("\n" + "-" * 78)
        log("[probes] 12 个闭环位姿跟踪探针：±x/±y/±z 平移 %.0f mm；±rx/±ry/±rz 旋转 %.3f rad (%.2f°)"
            % (THRESHOLDS["translation_probe_cmd_m"] * 1000,
               THRESHOLDS["rotation_probe_cmd_rad"],
               np.degrees(THRESHOLDS["rotation_probe_cmd_rad"])))
        penv, _ = make_env(a, log)
        inst = install_contact_hook(penv)
        log("[probes] env 复用；contact hook installed=%s" % inst)
        for kind, axes in (("trans", ["x", "y", "z"]), ("rot", ["x", "y", "z"])):
            for axn in axes:
                for sg in (+1, -1):
                    try:
                        r = run_probe(penv, kind, axn, sg, a, log)
                    except Exception as e:                     # noqa: BLE001
                        r = {"probe": "%s%s" % ("+" if sg > 0 else "-", axn),
                             "kind": kind, "pass": False, "error": str(e)}
                    result["probes"].append(r)
                    if kind == "trans":
                        log("  %-3s 平移 命令 %6.1f mm → 实达 %7.2f mm  残差 %5.2f mm  "
                            "| 目标残差 %5.2f mm  %s"
                            % (r["probe"], r.get("commanded_mm", np.nan),
                               r.get("achieved_mm", np.nan), r.get("err_mm", np.nan),
                               r.get("residual_pos_mm", np.nan),
                               "PASS" if r.get("pass") else "FAIL"))
                    else:
                        log("  %-3s 旋转 命令 %.3f rad (%.2f°) → 实达 %.3f rad (%.2f°)  "
                            "误差 %.4f rad (%.2f°)  残差 %.4f rad  %s"
                            % (r["probe"], r.get("commanded_rad", np.nan),
                               r.get("commanded_deg", np.nan),
                               r.get("achieved_rad", np.nan), r.get("achieved_deg", np.nan),
                               r.get("err_rad", np.nan), r.get("err_deg", np.nan),
                               r.get("residual_rot_rad", np.nan),
                               "PASS" if r.get("pass") else "FAIL"))
        penv.close()
        npass = sum(1 for r in result["probes"] if r.get("pass"))
        tr_err = [r.get("err_mm") for r in result["probes"]
                  if r.get("kind") == "translation" and r.get("err_mm") is not None]
        rt_err = [r.get("err_rad") for r in result["probes"]
                  if r.get("kind") == "rotation" and r.get("err_rad") is not None]
        result["probe_summary"] = {
            "n": len(result["probes"]), "n_pass": npass,
            "all_pass": bool(npass == len(result["probes"])),
            "criterion_rotation_rad": THRESHOLDS["rotation_probe_err_max_rad"],
            "criterion_translation_mm": THRESHOLDS["translation_probe_err_max_mm"],
            "max_trans_err_mm": max(tr_err) if tr_err else None,
            "max_rot_err_rad": max(rt_err) if rt_err else None,
            "max_rot_err_deg": float(np.degrees(max(rt_err))) if rt_err else None,
        }
        log("  → 探针 %d/%d 通过（平移最大误差 %s mm；旋转最大误差 %s rad）"
            % (npass, len(result["probes"]),
               _f(result["probe_summary"]["max_trans_err_mm"], 3),
               _f(result["probe_summary"]["max_rot_err_rad"], 4)))

    if a.mode == "probe":
        block_cls = block_object_class(widths[0] / 1000.0, BLOCK_DEPTH_M, BLOCK_HEIGHT_M)
        install_object_substitution(cat, block_cls, log)
        genv, _ = make_env(a, log)
        spec = {"trial": 1, "trial_label": "probe_w%d" % widths[0],
                "tag": "probe", "width_mm": widths[0], "yaw_rad": 0.0,
                "seed": a.seed_base}
        rec, tr = run_trial(genv, a, spec, log, verbose=True)
        log("[probe] 闸门=%s  失败=%s" % (rec["gates"], rec["failed_gates"]))
        result["groups"]["probe"] = {"trials": [rec]}
        genv.close()

    if a.mode == "probe":
        OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        log("[out] %s" % OUT_JSON)
        return 0

    # ---------------- 宽度分组 ----------------
    for wi, w in enumerate(widths if a.mode in ("full", "widths") else []):
        gname = "width_%d" % int(round(w))
        log("\n" + "=" * 78)
        log("[group %s] 试块 x(深)=%.0f mm  y(沿闭合轴宽)=%.0f mm  z(高)=%.0f mm  质量≈%.3f kg"
            % (gname, BLOCK_DEPTH_M * 1000, w, BLOCK_HEIGHT_M * 1000,
               BLOCK_DEPTH_M * (w / 1000.0) * BLOCK_HEIGHT_M * BLOCK_DENSITY))
        bcls = block_object_class(w / 1000.0, BLOCK_DEPTH_M, BLOCK_HEIGHT_M)
        install_object_substitution(cat, bcls, log)
        genv, _ = make_env(a, log)
        rows = []
        for t in range(a.trials):
            spec = {"trial": t + 1, "trial_label": "%s_t%d" % (gname, t + 1),
                    "tag": gname, "width_mm": w, "yaw_rad": 0.0,
                    "seed": a.seed_base + wi * 100 + t}
            try:
                rec, tr = run_trial(genv, a, spec, log)
            except Exception as e:                             # noqa: BLE001
                rec, tr = {"trial": t + 1, "trial_label": spec["trial_label"],
                           "seed": spec["seed"], "success": False,
                           "failed_gates": ["exception"], "notes": [str(e)],
                           "gates": {}}, None
            rows.append(rec)
            if tr is not None:
                traces[spec["trial_label"]] = tr
            pe = rec.get("phase_end", {}) or {}
            log("  [%s %2d/%d] %s  抬升=%s mm  保持min=%s mm  释放漂移=%s mm  "
                "峰值穿透=%.3f mm  max步位移=%.4f m  原生谓词=%s  失败闸门=%s"
                % (gname, t + 1, a.trials, "OK  " if rec.get("success") else "FAIL",
                   _f(pe.get("lift_mm")), _f(pe.get("hold_min_mm")),
                   _f(pe.get("post_release_drift_mm")),
                   rec.get("penetration_peak_pad_mm", float("nan")),
                   rec.get("max_step_disp_m", float("nan")),
                   rec.get("native_success_at_end"),
                   ",".join(rec.get("failed_gates", [])) or "-"))
        genv.close()
        n_ok = sum(1 for r in rows if r.get("success"))
        gate_fail = {}
        for r in rows:
            for g in r.get("failed_gates", []):
                gate_fail[g] = gate_fail.get(g, 0) + 1
        peaks = [r.get("penetration_peak_pad_mm", float("nan")) for r in rows]
        grp = {
            "width_mm": w, "n_total": len(rows), "n_success": n_ok,
            "success_rate": n_ok / max(1, len(rows)),
            "criterion": ">= %d/%d" % (THRESHOLDS["pass_min_per_width"], a.trials),
            "pass": bool(n_ok >= THRESHOLDS["pass_min_per_width"]),
            "gate_failure_counts": gate_fail,
            "penetration_peak_pad_mm": {
                "max": float(np.nanmax(peaks)) if peaks else None,
                "mean": float(np.nanmean(peaks)) if peaks else None,
                "limit": THRESHOLDS["penetration_soft_limit_mm"],
                "n_over_limit": int(sum(1 for p in peaks
                                        if p == p and
                                        p > THRESHOLDS["penetration_soft_limit_mm"]))},
            "max_step_disp_m": max([r.get("max_step_disp_m", 0.0) for r in rows],
                                   default=None),
            "trials": rows,
        }
        result["groups"][gname] = grp
        log("  → %s：成功 %d/%d %s  闸门失败=%s  峰值穿透 max=%s mm（限 %.1f）"
            % (gname, n_ok, len(rows), "PASS" if grp["pass"] else "FAIL",
               gate_fail or "{}", _f(grp["penetration_peak_pad_mm"]["max"], 3),
               THRESHOLDS["penetration_soft_limit_mm"]))

    # ---------------- 重定向后抓取组 ----------------
    n_re = 0 if a.reorient_trials < 0 else (a.reorient_trials or (2 * len(widths)))
    if a.mode not in ("full", "reorient"):
        n_re = 0
    log("\n" + "=" * 78)
    log("[group reorient_grasp] 先命令式重定向（±90° 偏航，全程 OSC 动作），再完成整条链  n=%d"
        % n_re)
    re_rows = []
    for wi, w in enumerate(widths):
        yaws = [np.pi / 2, -np.pi / 2]
        bcls = block_object_class(w / 1000.0, BLOCK_DEPTH_M, BLOCK_HEIGHT_M)
        install_object_substitution(cat, bcls, log)
        renv, _ = make_env(a, log)
        for yi, yaw in enumerate(yaws):
            i = wi * len(yaws) + yi
            if i >= n_re:
                break
            spec = {"trial": i + 1, "trial_label": "reorient_w%d_%s" % (w, "+" if yaw > 0 else "-"),
                    "tag": "reorient_grasp", "width_mm": w, "yaw_rad": float(yaw),
                    "seed": a.seed_base + 900 + i}
            try:
                rec, tr = run_trial(renv, a, spec, log)
            except Exception as e:                             # noqa: BLE001
                rec, tr = {"trial": i + 1, "trial_label": spec["trial_label"],
                           "success": False, "failed_gates": ["exception"],
                           "notes": [str(e)], "gates": {}}, None
            rec["reorient_yaw_deg"] = float(np.degrees(yaw))
            re_rows.append(rec)
            if tr is not None:
                traces[spec["trial_label"]] = tr
            pe = rec.get("phase_end", {}) or {}
            log("  [reorient %2d/%d] 宽度=%.0f mm 偏航=%+.0f° 重定向残差=%s rad  %s  "
                "抬升=%s mm  失败闸门=%s"
                % (i + 1, n_re, w, np.degrees(yaw),
                   _f(pe.get("reorient_residual_rad"), 4),
                   "OK  " if rec.get("success") else "FAIL",
                   _f(pe.get("lift_mm")),
                   ",".join(rec.get("failed_gates", [])) or "-"))
        renv.close()
    n_re_ok = sum(1 for r in re_rows if r.get("success"))
    re_fail = {}
    for r in re_rows:
        for g in r.get("failed_gates", []):
            re_fail[g] = re_fail.get(g, 0) + 1
    if n_re == 0:
        log("  (跳过 reorient_grasp：本次 mode=%s，不写该组，保留既有结论)" % a.mode)
    result["groups"]["reorient_grasp"] = result["groups"].get("reorient_grasp", {}) \
        if n_re == 0 else {
        "n_total": len(re_rows), "n_success": n_re_ok,
        "criterion": ">= %d/%d" % (THRESHOLDS["reorient_group_min_pass"], n_re),
        "pass": bool(n_re_ok >= THRESHOLDS["reorient_group_min_pass"]),
        "gate_failure_counts": re_fail,
        "penetration_peak_pad_mm": {
            "max": max([r.get("penetration_peak_pad_mm", 0.0) for r in re_rows],
                       default=None)},
        "trials": re_rows,
    }
    if n_re:
        log("  → reorient_grasp：成功 %d/%d %s  闸门失败=%s"
            % (n_re_ok, n_re,
               "PASS" if result["groups"]["reorient_grasp"]["pass"] else "FAIL",
               re_fail or "{}"))

    # ---------------- 总判据 ----------------
    width_keys = sorted([k for k in result["groups"] if k.startswith("width_")],
                        key=lambda k: float(k.split("_")[1]))
    width_pass = bool(width_keys) and all(
        result["groups"][k]["pass"] for k in width_keys)
    probes_pass = (not a.probes) or result.get("probe_summary", {}).get("all_pass", False)
    re_pass = bool(result["groups"].get("reorient_grasp", {}).get("pass", False))
    result["widths_verdict"] = "PASS" if width_pass else "FAIL"
    result["verdict"] = "PASS" if (width_pass and probes_pass and re_pass) else "FAIL"
    bits = []
    if not width_pass:
        bits.append("宽度组未全部达标：" + ", ".join(
            "%s %d/%d" % (k, result["groups"][k]["n_success"],
                          result["groups"][k]["n_total"]) for k in width_keys))
    if not probes_pass:
        bits.append("位姿跟踪探针未全通过")
    if not re_pass:
        bits.append("重定向抓取组未达标")
    result["verdict_reason"] = "；".join(bits) if bits else (
        "所有宽度组 ≥%d/%d，12 个位姿探针全通过，重定向抓取组达标"
        % (THRESHOLDS["pass_min_per_width"], a.trials))
    result["wall_seconds"] = round(time.time() - t_start, 1)

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        np.savez_compressed(OUT_TRACE, **traces)
        log("[trace] %s" % OUT_TRACE)
    except Exception as e:                                     # noqa: BLE001
        log("[trace] 保存失败：%s" % e)

    log("\n" + "=" * 78)
    for k in width_keys:
        g = result["groups"][k]
        log("[verdict] %-10s 成功 %2d/%2d %s  峰值穿透 %s mm  闸门失败 %s"
            % (k, g["n_success"], g["n_total"], "PASS" if g["pass"] else "FAIL",
               _f(g["penetration_peak_pad_mm"]["max"], 3), g["gate_failure_counts"] or "{}"))
    g = result["groups"].get("reorient_grasp", {"n_success": 0, "n_total": 0,
                                                "pass": False,
                                                "gate_failure_counts": {}})
    log("[verdict] %-10s 成功 %2d/%2d %s  闸门失败 %s"
        % ("reorient", g["n_success"], g["n_total"], "PASS" if g["pass"] else "FAIL",
           g["gate_failure_counts"] or "{}"))
    if a.probes:
        log("[verdict] probes     %2d/%2d %s"
            % (result["probe_summary"]["n_pass"], result["probe_summary"]["n"],
               "PASS" if result["probe_summary"]["all_pass"] else "FAIL"))
    log("[verdict] %s —— %s" % (result["verdict"], result["verdict_reason"]))
    log("[out] %s" % OUT_JSON)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
