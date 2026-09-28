#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"重定向机理诊断"、"TCP 偏移核对"与"分段来源"写回结果 JSON，并生成诊断 Markdown。

只读写 `results/n3_osc_*`，不触碰任何历史 `n3_*` / `n3r_*` 文件。
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927")
P = ROOT / "results" / "n3_osc_full_chain.json"
MD = ROOT / "results" / "n3_osc_reorient_diagnosis.md"

REORIENT_DIAGNOSIS = {
    "question_1_closing_axis_vs_block_width": {
        "answer": ("重定向后闭合轴与试块宽度方向**不平行**：+90° 一侧偏 4.66°，−90° 一侧"
                   "偏 4.43°（6/6 trial 完全一致，是确定性的系统偏差，不是随机噪声）。"),
        "residual_rad": {"plus90": 0.0796, "minus90": 0.0787},
        "misalignment_deg": {"plus90": 4.66, "minus90": 4.43},
        "effective_section_width_mm": {
            "note": "w_eff = w·cosθ + depth·sinθ（depth=60 mm）——指垫要跨住的等效截面宽",
            "w30": 34.6, "w50": 54.6, "w70": 74.5},
        "consequence": ("开口约 90 mm 时单侧余量由 ~10 mm 降到 ~7.6 mm（w=70），且指垫"
                        "面相对试块侧面**倾斜 4.5°**，指尖内角先于垫面接触试块顶棱。"),
        "converged_rot": False,
    },
    "question_2_penetration_phase_and_surface": {
        "answer": ("指垫↔试块峰值穿透 3.068 mm（trial 1）出现在 **transport** 阶段；"
                   "另一处 1.759 mm 在 close。物体↔任意 geom 峰值 10.700 mm 同样在 transport"
                   "（trial 1 物体在夹持中转动/滑落）。"),
        "contact_localisation": ("逐步接触打印（n3_osc_diag_reorient.py，带世界坐标的 "
                                 "`d.contact[i].pos`）：失败 trial 的指垫接触点 z≈0.0695，"
                                 "试块顶面 z=0.0699 —— 落在**顶面/顶棱**而不是侧面；"
                                 "侧面接触（pos_z 低于顶面 8 mm 以上）在成功 trial 才出现。"),
        "not_side_face": True,
    },
    "question_3_pinch_target": {
        "answer": ("`grasp_pinch_target` **完全不受重定向影响**：它只由物体 bbox 决定"
                   "（pin = [bbox 中心 xy, z_g]，z_g = 底面 + 0.035），并作为 grip_site 的"
                   "**世界系**目标下发。实测重定向前后 pin 数值逐位相同。"),
        "pin_formula": "pin = [0.5*(lo+hi) 的 x,y ; min(lo_z + max(0.028, min(h/2, h-0.006)), hi_z-0.004)]",
        "affected_by_reorientation": False,
        "what_did_change": ("变的是**夹爪姿态**、以及由此变差的 OSC 位置跟踪：+90° 侧 3/3 "
                            "trial 的下降伺服不收敛（残差 11.9–42.4 mm，实测停止位置比试块"
                            "顶面还高 4.3–32.6 mm）。"),
    },
    "question_4_attribution": {
        "verdict": "两者都有，且可用证据分开",
        "script_defects_found_and_fixed": [
            "① 试块类共用一个临时 XML：`MujocoXMLObject.__init__` 在实例化时才读文件，"
            "width_70 写下的 70 mm 尺寸污染了 reorient 组缓存的 30/50 mm 类 —— 实测 3 个"
            "宽度都建成了 60×70×70 的试块。改为**每个尺寸一个独立临时 XML**。",
            "② 试块宽度对齐到了**错误的闭合轴**：`phi + yaw` 对『绕本体 z』的旋转是错的"
            "（应为 `R_ref·(R_home^T·axis_w)`，与左乘/右乘无关）。改为按**重定向后的真实"
            "闭合轴**反解物体偏航，并把『实测闭合轴 / 试块宽度方向 / 夹角 / 等效截面宽』"
            "逐 trial 写进 JSON。",
            "③ 旋转步长限制器没有按 OSC 单位换算：`ar = er/0.05` 会被 OSC 解释为"
            "**0.4 rad/step** 的旋转指令（output_max[3:]=0.5 rad），姿态误差一旦因可达性"
            "无法消除，这个饱和指令会长期存在并把位置通道带偏。改为把限制落在"
            "**指令增量**上（max_step/output_max）。",
            "④ `env._check_success()` 在 `OffScreenRenderEnv` 上不存在（只有 "
            "`check_success()`），AttributeError 被 except 吞掉 → 原生谓词恒为 False。",
        ],
        "effect_of_fixes": {"before": "0/6 成功（且 3 个宽度都用了错误的 70 mm 试块）",
                            "after": "2/6 成功，6/6 全部完成了『重定向→接近→下降→闭合→抬升』"},
        "real_remaining_limitation": [
            "±90° 工具轴重定向在本复位位姿下**收敛不了**：400 步后姿态残差稳定在 "
            "0.079 rad (4.53°)，`converged_rot=False`，且与试块宽度无关（确定性）。",
            "+90° 一侧的 OSC **位置跟踪**在该腕部构型下明显更差：3/3 trial 下降不收敛"
            "（残差 11.9–42.4 mm），而 −90° 一侧 2/3 收敛（残差 5.6 mm）。",
            "由此产生的后果是**抓在试块顶棱**而不是侧面：抬升仍可能成功（37–65 mm），"
            "但搬运中物体在夹爪内转动/滑落（穿透 3.068 mm，超 2.0 mm 限）。",
        ],
        "conclusion": ("修好脚本后，『带姿态偏转的抓取』不是完全不可用（2/6 全闸门通过、"
                       "6/6 完成抓取链），但当前 OSC 在 ±90° 重定向下无法保证姿态精度与"
                       "位置跟踪，因此**不能**按 ≥5/6 的判据宣称可用。此结论不因放宽阈值而改变。"),
    },
    "evidence_files": ["results/n3_osc_full_chain.json",
                       "results/n3_osc_full_chain_traces.npz",
                       "scripts/n3_osc_diag_reorient.py"],
}

TCP_CHECK = {
    "eef_body_pos": [0.0, 0.0, 0.20254],
    "grip_site_world": [-0.14828, 6e-05, 0.26488],
    "pinch_midpoint_world": [-0.14763, 6e-05, 0.27593],
    "pinch_minus_grip_site_world_m": [0.00065, 0.0, 0.01105],
    "pinch_offset_norm_mm": 11.07,
    "stale_GRASP_OFFSET_CLOSED_mm": 82.54,
    "mismatch_mm": 93.6,
    "note": ("历史常数 GRASP_OFFSET_CLOSED=[-6e-05,2e-05,0.08254] 在当前资产上已经错位："
             "用旧式换算 gs = pinch - R·offset 会把 grip_site 目标点打偏 **93.6 mm**"
             "（远超试块 70 mm 高度），必然抓空。本返工脚本**全程不使用该常数**：所有目标点"
             "由 `N.gripsite()` 的实测世界位姿表达。"),
    "used_in_this_script": False,
    "script": "scripts/n3_osc_tcp_offset.py",
}


def build_md(d):
    g = d["groups"]
    L = []
    L.append("# N3 返工（审核项 R7）：重定向抓取组的机理诊断\n")
    L.append("生成时间：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    L.append("结果文件：`results/n3_osc_full_chain.json`（本文件是它的可读摘要）\n")
    L.append("## 1. 总览\n")
    L.append("| 组 | 成功 | 判据 | 结果 | 峰值指垫穿透 |")
    L.append("|---|---|---|---|---|")
    for k in sorted(g, key=lambda x: (not x.startswith("width_"), x)):
        gg = g[k]
        if "n_total" not in gg:
            continue
        pen = gg.get("penetration_peak_pad_mm", {})
        pen = pen.get("max") if isinstance(pen, dict) else pen
        L.append("| `%s` | %s/%s | %s | %s | %s mm |"
                 % (k, gg.get("n_success"), gg.get("n_total"), gg.get("criterion", "-"),
                    "PASS" if gg.get("pass") else "FAIL",
                    "n/a" if pen is None else round(pen, 3)))
    ps = d.get("probe_summary", {})
    L.append("| `probes` | %s/%s | 收敛且 err≤0.05 rad / ≤5 mm | %s | — |"
             % (ps.get("n_pass"), ps.get("n"),
                "PASS" if ps.get("all_pass") else "FAIL"))
    L.append("\n**总 verdict：%s** —— %s\n" % (d.get("verdict"), d.get("verdict_reason")))
    L.append("## 2. 旋转跟踪探针（命令 vs 实达）\n")
    L.append("| 探针 | 命令 | 实达 | 误差 | 收敛(pos/rot) | 判据 |")
    L.append("|---|---|---|---|---|---|")
    for r in d.get("probes", []):
        if r["kind"] == "rotation":
            L.append("| %s | %.3f rad (%.2f°) | %+.4f rad (%+.2f°) | **%.4f rad (%.2f°)** | %s / %s | %s |"
                     % (r["probe"], r["commanded_rad"], r["commanded_deg"],
                        r["achieved_rad"], r["achieved_deg"], r["err_rad"], r["err_deg"],
                        r.get("converged_pos"), r.get("converged_rot"),
                        "PASS" if r["pass"] else "FAIL"))
        else:
            L.append("| %s | %.1f mm | %+.2f mm | **%.2f mm** | %s / %s | %s |"
                     % (r["probe"], r["commanded_mm"], r["achieved_mm"], r["err_mm"],
                        r.get("converged_pos"), r.get("converged_rot"),
                        "PASS" if r["pass"] else "FAIL"))
    L.append("\n判据：`converged_pos/rot == True` **且** 误差 ≤ 0.05 rad（旋转）/ 5 mm（平移）；"
             "收敛容差（0.005 m / 0.03 rad）故意严于通过闸门，避免循环论证；"
             "误差缺失一律判 FAIL。\n")
    L.append("## 3. 重定向抓取组逐 trial\n")
    for t in g.get("reorient_grasp", {}).get("trials", []):
        pe = t.get("phase_end", {})
        L.append("### trial %s — w=%s mm, 偏航 %+.0f° → %s"
                 % (t.get("trial"), t.get("block_extent_mm", {}).get("y_width_along_closing_axis"),
                    t.get("gripper_yaw_deg", 0), "成功" if t.get("success") else "失败"))
        L.append("- 重定向：%s 步，收敛(rot)=%s，姿态残差 **%.4f rad (%.2f°)**"
                 % (pe.get("reorient_steps"), pe.get("reorient_converged_rot"),
                    pe.get("reorient_residual_rad") or 0,
                    np.degrees(pe.get("reorient_residual_rad") or 0)))
        L.append("- 闭合轴(实测)=%s，试块宽度方向=%s，**夹角 %.2f°**，等效截面宽 %.1f mm"
                 % ([round(v, 3) for v in (pe.get("closing_axis_after_reorient") or [])],
                    [round(v, 3) for v in (pe.get("block_width_dir_world") or [])],
                    pe.get("closing_axis_vs_block_width_deg") or 0,
                    pe.get("effective_section_width_mm") or 0))
        L.append("- 下降：收敛(pos)=%s，残差 %.1f mm，指垫中点高于试块顶 %.1f mm"
                 % (pe.get("descend_converged_pos"),
                    (pe.get("descend_residual_m") or 0) * 1000,
                    -(pe.get("descend_pad_below_block_top_mm") or 0)))
        L.append("- 闭合：%s 步，Left_1=%.3f，净距 %.1f mm，物体被推 %.1f mm"
                 % (pe.get("close_steps"), pe.get("left1_after_close") or 0,
                    pe.get("jaw_after_close_mm") or 0, pe.get("obj_shift_on_close_mm") or 0))
        L.append("- 抬升 %.1f mm，保持 min %.1f mm，最大步位移 %.4f m，峰值指垫穿透 **%.3f mm**"
                 % (pe.get("lift_mm") or 0, pe.get("hold_min_mm") or 0,
                    t.get("max_step_disp_m") or 0, t.get("penetration_peak_pad_mm") or 0))
        L.append("- 失败闸门：%s；备注：%s\n"
                 % (", ".join(t.get("failed_gates", [])) or "-", t.get("notes") or "-"))
    L.append("## 4. 机理结论\n")
    q = REORIENT_DIAGNOSIS
    L.append("1. **闭合轴 vs 试块宽度**：%s\n" % q["question_1_closing_axis_vs_block_width"]["answer"])
    L.append("2. **穿透发生在哪**：%s\n" % q["question_2_penetration_phase_and_surface"]["answer"])
    L.append("   接触定位：%s\n" % q["question_2_penetration_phase_and_surface"]["contact_localisation"])
    L.append("3. **pinch target 是否被带偏**：%s\n" % q["question_3_pinch_target"]["answer"])
    L.append("4. **归因**：%s\n" % q["question_4_attribution"]["verdict"])
    for x in q["question_4_attribution"]["script_defects_found_and_fixed"]:
        L.append("   - 脚本缺陷：%s" % x)
    for x in q["question_4_attribution"]["real_remaining_limitation"]:
        L.append("   - 真实局限：%s" % x)
    L.append("\n   %s\n" % q["question_4_attribution"]["conclusion"])
    L.append("## 5. 陈旧 GRASP_OFFSET_CLOSED 的核对\n")
    L.append("当前实测：pinch 相对 grip_site 偏移 **%.2f mm**（旧常数 %.2f mm）；"
             "继续使用旧常数会把目标点打偏 **%.1f mm**。本脚本全程不使用该常数。\n"
             % (TCP_CHECK["pinch_offset_norm_mm"], TCP_CHECK["stale_GRASP_OFFSET_CLOSED_mm"],
                TCP_CHECK["mismatch_mm"]))
    return "\n".join(L) + "\n"


def main():
    d = json.loads(P.read_text(encoding="utf-8"))
    d["reorient_diagnosis"] = REORIENT_DIAGNOSIS
    d["tcp_offset_check"] = TCP_CHECK
    d["run_segments"] = {
        "note": ("本 JSON 由多段运行合并而成（`--merge`）：宽度组来自一次完整运行，"
                 "探针与重定向组在修正脚本缺陷后各自重跑并合并。**宽度组的逐 trial 结果"
                 "未被改动**，也未被重跑。"),
        "segments": [
            {"segment": "widths", "produced": "groups.width_30/50/70",
             "servo_clamp_revision": "v1（旋转步长限制未按 OSC 单位换算）"},
            {"segment": "probes", "produced": "probes, probe_summary",
             "servo_clamp_revision": "v2（OSC 单位换算后的步长限制）"},
            {"segment": "reorient", "produced": "groups.reorient_grasp",
             "servo_clamp_revision": "v2"}],
        "gate_thresholds_unchanged_across_segments": True,
        "annotated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    P.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    MD.write_text(build_md(d), encoding="utf-8")
    print("[ok] %s" % P)
    print("[ok] %s" % MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
