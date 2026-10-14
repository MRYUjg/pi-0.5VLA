"""Verify and publish existing 800 rollouts after shutdown; never rerun a policy."""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess

V = pathlib.Path(__file__).resolve().parents[1]
ROOT = V.parents[1]
REPO = ROOT.parents[1]
SOURCE = 'closed_loop_20261008_r1'
RUN = 'closed_loop_readback_20261008_r2'


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    state = ROOT / 'reports/closed_loop_recovery_20261008_r2.json'
    assert not state.exists()
    save(state, {'phase': 'verifying', 'pid': os.getpid(), 'source_run': SOURCE})
    for version in [V.name, 'rollout_20261008_r1', 'media_20261008_r1']:
        manifest = json.loads((ROOT / 'releases' / version / 'manifest.json').read_text())
        for name, wanted in manifest['files'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == wanted, name
    source = ROOT / 'runs' / SOURCE
    previous = json.loads((source / 'verified_results.json').read_text())
    assert previous['complete'] and previous['verification_pass'] and len(previous['videos']) == 40
    assert all(json.loads(p.read_text())['process_exit'] == 0 for p in source.glob('server*execution.json'))
    out = ROOT / 'runs' / RUN
    out.mkdir(exist_ok=False)
    for condition in previous['conditions']:
        (out / condition).symlink_to('../' + SOURCE + '/' + condition, target_is_directory=True)
    common = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '2', '--memory', '3g',
              '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OPENBLAS_NUM_THREADS=1',
              '-v', str(REPO) + ':/app:ro', '-v', str(ROOT) + ':/app/agent-work-space/5-final-train',
              '--entrypoint', '/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
              'libero:latest']
    for label, script in [('verify', 'versions/rollout_20261008_r1/scripts/verify_results.py'),
                          ('package', 'versions/media_20261008_r1/scripts/package.py')]:
        command = common + ['/app/agent-work-space/5-final-train/' + script, '--run', RUN]
        with (out / (label + '.log')).open('x') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        with (out / (label + '_execution.json')).open('x') as f:
            json.dump({'process_exit': code, 'command': command}, f, indent=2)
        assert code == 0, label
        save(state, {'phase': 'packaging' if label == 'verify' else 'writing_record', 'pid': os.getpid()})
    report = json.loads((out / 'verified_results.json').read_text())
    index = json.loads((ROOT / 'videos' / RUN / 'index.json').read_text())
    assert report['complete'] and report['verification_pass'] and not report['errors']
    assert len([v for v in index['videos'] if v['kind'] == 'predeclared_first_attempt']) == 40
    assert all(v['decode_pass'] for v in index['videos'])
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    lines = ['# 闭环测试恢复与完成记录 — 20261008 r2', '',
             '核对时间：' + now + '。关机中断后的只读恢复核验完成，核验和视频打包进程实际退出码均为0。',
             '本记录复核原800次运行；没有重跑策略、增加测试分母或更改候选。旧状态文件保留关机前状态，不能据其旧PID判断当前进程。', '',
             '| 组合 | 成功/200 | 成功率 | 进程异常 |', '|---|---:|---:|---:|']
    for name, c in report['conditions'].items():
        lines.append(f'| {name} | {c["successes"]} | {c["success_rate"]:.1%} | {c["process_failures"]} |')
    lines += ['', '微调组合0/200，≥90%适配验收未通过。已训练五任务和未训练五任务各0/100。主口径220步，首条650步延长诊断不计主成功；初态拒绝保留固定分母。', '',
              '| Task | 微调ECO65+ZhiXing | Panda基座 | ECO65+ZhiXing基座 | ECO65+Panda基座 |', '|---:|---:|---:|---:|---:|']
    order = ['eco65_zhixing_lora', 'panda_base', 'eco65_zhixing_base', 'eco65_panda_base']
    for task in range(10):
        values = [str(report['conditions'][c]['tasks'][str(task)]['successes']) + '/20' for c in order]
        lines.append('| ' + str(task) + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '四组对象reset哈希逐seed匹配：' + json.dumps(report['paired_native_scenes'], ensure_ascii=False), '',
              f'共{len(index["videos"])}段MP4：40段预登记首条双视角录像，以及初态拒绝任务的第一条实际策略输入录像。完整解码、帧数和SHA-256检查通过，失败录像保留。',
              '[视频索引](videos/' + RUN + '/视频索引.md)。', '',
              '训练教师最早528步达到目标，与220步主预算有差异；task06首条微调延长诊断559步达到原生目标，不作为释放稳定认证。',
              '训练与本评测均显示碰撞+视觉网格，属于相同手动渲染域；不是历史纯视觉Panda基准的直接复现。随机reset种子协议也不是官方.pruned初态基准。执行接触观察不等于完整r6教师物理认证。', '',
              '原始证据位于runs/' + SOURCE + '；恢复核验位于runs/' + RUN + '，其条件目录仅是原记录的相对路径符号链接。',
              '新的靠近工作区试验属于独立开发诊断，不能回写本批次成功率。未自动再训练，未实物部署。']
    doc = ROOT / '闭环测试恢复与完成记录_20261008_r2.md'
    with doc.open('x') as f:
        f.write('\n'.join(lines) + '\n')
    save(state, {'phase': 'complete', 'completed_at': now, 'source_run': SOURCE, 'readback_run': RUN,
                 'verification_pass': True, 'video_count': len(index['videos']),
                 'adaptation_acceptance_pass': False, 'report': str(doc)})
    print('RECOVERY COMPLETE', len(index['videos']), 'MP4', flush=True)


if __name__ == '__main__':
    main()
