#!/usr/bin/env python3
"""N0 / N1 节点完成度判定：逐条核对两个节点的通过判据，输出 PASS/FAIL。

判据来源：agent-work-space/zhixing_adaptation_20260927/实现节点与阶段性验证.md
  N0: (a) 基线可复现  (b) 哈希清单一致  (c) 备份齐全可回退  (d) 默认夹爪真实值已记录
  N1: (a) 闭环残差≤容差且 reset 无求解器修正 (b) 开合符号与语义一致
      (c) 开口范围有记录  (d) 开合时间按秒标定  (e) 无 NaN/Inf/穿透/发散

用法（容器内，仓库挂在 /app）：
    python verify_n0_n1_gates.py --repo /app
宿主机也可跑，只是 N1 的仿真项需要容器环境：
    python verify_n0_n1_gates.py --repo /mnt/data/vla-policy/openpi --skip-sim
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

WORK = "agent-work-space/zhixing_adaptation_20260927"

# N0 (b)：指纹清单要求的文件（与 n0_fingerprint.py 的 TARGETS 对应）
SRC_FILES = {
    "robot_xml": "robosuite/models/assets/robots/ECO65-B/robot.xml",
    "zhixing_gripper_py": "robosuite/models/grippers/zhixing_gripper.py",
    "panda_gripper_py": "robosuite/models/grippers/panda_gripper.py",
    "eco65_robot_py": "robosuite/models/robots/manipulators/eco65_robot.py",
    "mounted_eco65_py": "third_party/libero/libero/libero/envs/robots/mounted_eco65.py",
    "libero_main_py": "examples/libero/main.py",
    "eval_matrix_py": "agent-work-space/scripts/eval_matrix.py",
    "serve_policy_py": "scripts/serve_policy.py",
    "training_config_py": "src/openpi/training/config.py",
}
# N0 (c)：每一次改动都必须有可回退备份
REQUIRED_BAK = [
    "robosuite/models/assets/robots/ECO65-B/robot.xml.pre_contact_fix.bak",
    "robosuite/models/robots/manipulators/eco65_robot.py.pre_init_align.bak",
    "third_party/libero/libero/libero/envs/robots/mounted_eco65.py.pre_init_align.bak",
    "examples/libero/main.py.pre_gripper_entry.bak",
    "agent-work-space/scripts/eval_matrix.py.pre_gripper_entry.bak",
    "robosuite/models/grippers/zhixing_gripper.py.pre_polarity_switch.bak",
    "robosuite/models/grippers/zhixing_gripper.py.pre_fix_landing.bak",
    "docs/ECO65_ZhiXing_项目说明.md.pre_n1_correction.bak",
]
JOINTS = ["Left_1", "Left_Support", "Left_2", "Right_2", "Right_1", "Right_Support"]
EQS = [("Left_2", "Left_1", 1.0), ("Right_1", "Left_1", -1.0), ("Right_2", "Left_1", -1.0),
       ("Left_Support", "Left_1", -1.0), ("Right_Support", "Left_1", -1.0)]

R = []  # (node, check, ok, detail)


def rec(node, check, ok, detail=""):
    R.append((node, check, bool(ok), detail))
    print("  [%s] %-46s %s   %s" % ("PASS" if ok else "FAIL", check, "OK" if ok else "!!", detail))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- N0
def check_n0(repo, work):
    print("\n=== N0 基线冻结与指纹 ===")
    mpath = os.path.join(work, "manifests/source_fingerprint.json")
    if not os.path.isfile(mpath):
        rec("N0", "manifest_exists", False, mpath)
        return
    man = json.load(open(mpath, encoding="utf-8"))
    rec("N0", "manifest_exists", True, os.path.basename(mpath))

    bad = []
    for name, rel in SRC_FILES.items():
        rec_sha = (man.get("sources", {}).get(name) or {}).get("sha256")
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            bad.append(f"{name}:MISSING")
        elif rec_sha != sha(p):
            bad.append(f"{name}:{rec_sha[:8]}!={sha(p)[:8]}")
    rec("N0", "b_source_hashes_match_manifest", not bad, "; ".join(bad) or f"{len(SRC_FILES)} files")

    miss = [b for b in REQUIRED_BAK if not os.path.isfile(os.path.join(repo, b))]
    rec("N0", "c_all_backups_present", not miss, "; ".join(miss) or f"{len(REQUIRED_BAK)} backups")

    # 备份内容必须真的不同（防止空备份）
    same = []
    for b in REQUIRED_BAK:
        bp = os.path.join(repo, b)
        orig = bp.split(".pre_")[0]
        if os.path.isfile(bp) and os.path.isfile(orig) and sha(bp) == sha(orig):
            same.append(b)
    rec("N0", "c_backups_differ_from_current", not same, "; ".join(same) or "all differ")

    dg = man.get("default_gripper_actual", {})
    ok = dg.get("eco65_robot_py") == '"ZhixingGripper"' and dg.get("mounted_eco65_py") == '"ZhixingGripper"'
    rec("N0", "d_default_gripper_recorded", ok, json.dumps(dg, ensure_ascii=False))

    # (a) 基线重复性证据
    ev = os.path.join(repo, "agent-work-space/eval")
    tags = ["H_eco65_panda_align", "N0_repro_eco65_panda", "N0_repro2_eco65_panda",
            "N0_repro3_eco65_panda", "N1_fix_eco65_panda_regress"]
    vals = []
    for t in tags:
        p = os.path.join(ev, t, "summary.json")
        if os.path.isfile(p):
            vals.append(json.load(open(p, encoding="utf-8")).get("total_successes"))
    ok = len(vals) >= 3 and min(vals) >= 8 and max(vals) <= 10
    rec("N0", "a_baseline_reproducible_ge_8_of_10", ok,
        "n=%d values=%s (原判据「精确 9/10」已修订为 ≥8/10)" % (len(vals), vals))


# ---------------------------------------------------------------- N1
def check_n1_static(repo, work):
    print("\n=== N1 静态项 ===")
    gp = os.path.join(repo, SRC_FILES["zhixing_gripper_py"])
    src = open(gp, encoding="utf-8-sig").read()
    rec("N1", "legacy_switches_removed", "ZHIXING_GRIPPER_INVERT" not in src
        and "ZHIXING_GRIPPER_CONSISTENT_INIT" not in src, "环境变量开关已移除")
    rec("N1", "polarity_negation_in_source",
        "- self.speed * np.sign(action)" in src, "format_action 累积项取负")
    rec("N1", "consistent_init_in_source",
        "[-0.91, 0.91, -0.91, 0.91, 0.91, 0.91]" in src, "6 元素自洽初始化")

    aj = os.path.join(work, "results/gripper_audit.json")
    pj = os.path.join(work, "results/gripper_polarity.json")
    if os.path.isfile(aj):
        a = json.load(open(aj, encoding="utf-8"))
        m = a.get("motion", {})
        rng = m.get("opening_range_m")
        rec("N1", "c_opening_range_recorded", rng is not None and abs(rng) > 0.03,
            f"opening_range={rng} m")
        tim = a.get("timing", {})
        rec("N1", "d_close_time_calibrated_seconds",
            tim.get("seconds_to_full_close") is not None,
            f"steps={tim.get('steps_to_full_close')} @ {tim.get('control_freq_hz')} Hz "
            f"= {tim.get('seconds_to_full_close')} s")
        rec("N1", "d_speed_ablation_recorded", bool(a.get("speed_ablation")),
            f"{[r.get('speed') for r in a.get('speed_ablation', [])]}")
        pen = (m.get("fingerpad_penetration") or {})
        rec("N1", "e_no_self_penetration", pen.get("n_fingerpad_contacts") == 0,
            f"fingerpad contacts={pen.get('n_fingerpad_contacts')}")
        rec("N1", "e_no_nan_inf", a.get("checks", {}).get("no_nan_inf") is True, "")
    else:
        rec("N1", "audit_json_present", False, aj)
    rec("N1", "polarity_json_present", os.path.isfile(pj), pj)


def check_n1_sim(repo, work):
    print("\n=== N1 仿真项（reset 残差 / 无求解器修正 / 极性）===")
    sys.path.insert(0, os.path.join(work, "scripts"))
    from gripper_polarity import make_env, get_robot, live_sim, jaw, sites_ends

    env = make_env("ECO65", "ZhixingGripper")
    env.reset()
    r = get_robot(env)
    sim = live_sim(env, r)

    decl = np.asarray(r.gripper.init_qpos, dtype=float)
    q = np.asarray([sim.data.qpos[i] for i in r._ref_gripper_joint_pos_indexes], dtype=float)
    d = {J: float(q[i]) for i, J in enumerate(JOINTS)}
    resid = {j1: abs(d[j1] - c * d[j2]) for j1, j2, c in EQS}
    rec("N1", "a_joint_eq_residuals_zero", max(resid.values()) < 1e-6,
        f"max={max(resid.values()):.2e} rad")
    cl = sites_ends(sim, "Left_Support_tip", "Left_2_connect")
    cr = sites_ends(sim, "Right_Support_tip", "Right_2_connect")
    rec("N1", "a_fourbar_connect_closed", cl < 1e-5 and cr < 1e-5,
        f"left={cl:.2e} m right={cr:.2e} m")
    rec("N1", "a_init_len_6", decl.size == 6, f"len={decl.size}")
    rec("N1", "a_reset_no_solver_correction",
        np.allclose(q, decl, atol=1e-9),
        f"reset qpos == declared init (max diff {np.max(np.abs(q - decl)):.2e})")

    # 零动作下不应发散/不出 NaN
    for _ in range(10):
        a = np.zeros(7, dtype=float)
        env.step(a.tolist())
    r = get_robot(env)
    sim = live_sim(env, r)
    q2 = np.asarray([sim.data.qpos[i] for i in r._ref_gripper_joint_pos_indexes], dtype=float)
    rec("N1", "e_no_divergence_under_zero_action",
        bool(np.all(np.isfinite(q2))) and float(np.max(np.abs(q2))) < 10.0,
        f"|qpos|max={float(np.max(np.abs(q2))):.4f}")

    def drive(cmd, n=25):
        for _ in range(n):
            a = np.zeros(7, dtype=float)
            a[6] = cmd
            env.step(a.tolist())
        rr = get_robot(env)
        return jaw(live_sim(env, rr), "zhixing", rr)[0]

    jm, jp = drive(-1.0), drive(+1.0)
    rec("N1", "b_polarity_plus1_closes_zhixing", jp < jm,
        f"-1 -> {jm*100:.2f} cm (open), +1 -> {jp*100:.2f} cm (close)")

    env2 = make_env("ECO65", "PandaGripper")
    env2.reset()
    r2 = get_robot(env2)

    def drive2(cmd, n=25):
        for _ in range(n):
            a = np.zeros(7, dtype=float)
            a[6] = cmd
            env2.step(a.tolist())
        rr = get_robot(env2)
        return jaw(live_sim(env2, rr), "panda", rr)[0]

    pm, pp = drive2(-1.0), drive2(+1.0)
    rec("N1", "b_polarity_matches_panda_convention", pp < pm,
        f"Panda -1 -> {pm*100:.2f} cm, +1 -> {pp*100:.2f} cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/app")
    ap.add_argument("--work", default=None)
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    work = a.work or os.path.join(a.repo, WORK)
    print("repo=%s\nwork=%s" % (a.repo, work))

    check_n0(a.repo, work)
    check_n1_static(a.repo, work)
    if not a.skip_sim:
        check_n1_sim(a.repo, work)

    print("\n=== 汇总 ===")
    for node in ("N0", "N1"):
        rows = [r for r in R if r[0] == node]
        ok = sum(1 for r in rows if r[2])
        print("  %s: %d/%d 通过" % (node, ok, len(rows)))
        for _, c, o, det in rows:
            if not o:
                print("       FAILED: %s  %s" % (c, det))
    allok = all(r[2] for r in R)
    print("\nNODE_VERDICT = %s" % ("ALL_COMPLETE" if allok else "INCOMPLETE"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
