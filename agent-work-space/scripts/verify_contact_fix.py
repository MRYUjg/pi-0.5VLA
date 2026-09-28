"""验证正式 XML 的碰撞排除在编译模型里实际生效，且 base↔link_1 不再产生接触。
使用 robosuite 自己的 MjModel/MjData 包装接口（不能直接传 mujoco.mj_name2id）。"""
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                        robots=["ECO65"], gripper_types="PandaGripper")
env.seed(7)
env.reset()
sim = env.env.sim
m, d = sim.model, sim.data

print(f"nbody={m.nbody} njnt={m.njnt} nexclude={m.nexclude} npair={m.npair}")

b_base = m.body_name2id("robot0_base")
b_l1 = m.body_name2id("robot0_link_1")
g_base = m.geom_name2id("robot0_base_link_collision")
g_l1 = m.geom_name2id("robot0_link_1_collision")
print(f"robot0_base id={b_base} (weldid={m.body_weldid[b_base]})  robot0_link_1 id={b_l1}")
print(f"geom ids: base_col={g_base} link1_col={g_l1}")
print(f"exclude_signature[0] = {int(m.exclude_signature[0]) if m.nexclude else 'n/a'}")

# 步进 200 步统计接触
hits = other = 0
pairs = {}
for _ in range(200):
    sim.step()
    for c in range(d.ncon):
        con = d.contact[c]
        a, b = int(con.geom1), int(con.geom2)
        if {a, b} == {g_base, g_l1}:
            hits += 1
        else:
            other += 1
            key = tuple(sorted((a, b)))
            pairs[key] = pairs.get(key, 0) + 1
print(f"200 步内接触计数：base<->link_1 = {hits} 次；其它接触 = {other} 次")
for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:5]:
    print(f"    其它接触 {m.geom_id2name(a)} <-> {m.geom_id2name(b)}: {n} 次")

# 几何间距
try:
    import mujoco
    raw = m  # robosuite 包装对象在 3.2.3 下可直接传给部分函数；失败则跳过
    dist = mujoco.mj_geomDistance(m, d, g_base, g_l1, 1.0, None)
    print(f"base<->link_1 几何距离 = {dist*1000:.4f} mm（正数=已分离）")
except Exception as exc:
    print("mj_geomDistance 不可用:", str(exc)[:80])

env.close()
print("FIX_VERIFY_OK" if hits == 0 else "FIX_VERIFY_FAIL")
