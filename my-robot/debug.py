import robosuite as suite
env = suite.make(env_name="Lift", robots="ECO65", has_offscreen_renderer=True, use_camera_obs=True)
obs = env.reset()
print("相机名:", list(obs.keys()))
print("末端位姿:", obs.get("robot0_eef_pos"))