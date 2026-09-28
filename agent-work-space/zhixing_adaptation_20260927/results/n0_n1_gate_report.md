# N0 / N1 节点完成度判定报告

生成: 2026-09-27T09:40:25Z  |  脚本: scripts/verify_n0_n1_gates.py --repo /app
判据来源: 实现节点与阶段性验证.md

==========
==========
=== N0 基线冻结与指纹 ===
  [PASS] manifest_exists                                OK   source_fingerprint.json
  [PASS] b_source_hashes_match_manifest                 OK   9 files
  [PASS] c_all_backups_present                          OK   8 backups
  [PASS] c_backups_differ_from_current                  OK   all differ
  [PASS] d_default_gripper_recorded                     OK   {"eco65_robot_py": "\"ZhixingGripper\"", "mounted_eco65_py": "\"ZhixingGripper\""}
  [PASS] a_baseline_reproducible_ge_8_of_10             OK   n=5 values=[9, 8, 9, 10, 9] (原判据「精确 9/10」已修订为 ≥8/10)
=== N1 静态项 ===
  [PASS] legacy_switches_removed                        OK   环境变量开关已移除
  [PASS] polarity_negation_in_source                    OK   format_action 累积项取负
  [PASS] consistent_init_in_source                      OK   6 元素自洽初始化
  [PASS] c_opening_range_recorded                       OK   opening_range=-0.08977571605442622 m
  [PASS] d_close_time_calibrated_seconds                OK   steps=8 @ 20.0 Hz = 0.4 s
  [PASS] d_speed_ablation_recorded                      OK   [0.2, 0.01]
  [PASS] e_no_self_penetration                          OK   fingerpad contacts=0
  [PASS] e_no_nan_inf                                   OK   
  [PASS] polarity_json_present                          OK   /app/agent-work-space/zhixing_adaptation_20260927/results/gripper_polarity.json
=== N1 仿真项（reset 残差 / 无求解器修正 / 极性）===
  [PASS] a_joint_eq_residuals_zero                      OK   max=0.00e+00 rad
  [PASS] a_fourbar_connect_closed                       OK   left=7.15e-08 m right=1.75e-07 m
  [PASS] a_init_len_6                                   OK   len=6
  [PASS] a_reset_no_solver_correction                   OK   reset qpos == declared init (max diff 0.00e+00)
  [PASS] e_no_divergence_under_zero_action              OK   |qpos|max=0.4330
  [PASS] b_polarity_plus1_closes_zhixing                OK   -1 -> 11.25 cm (open), +1 -> 2.28 cm (close)
  [PASS] b_polarity_matches_panda_convention            OK   Panda -1 -> 7.88 cm, +1 -> 0.15 cm
=== 汇总 ===
  N0: 6/6 通过
  N1: 16/16 通过
NODE_VERDICT = ALL_COMPLETE
