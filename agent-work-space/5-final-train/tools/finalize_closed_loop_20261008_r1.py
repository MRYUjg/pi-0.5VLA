"""One-shot result/video finalizer; waits for the frozen evaluation, never reruns tests."""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
RUN='closed_loop_20261008_r1'


def main():
    out=ROOT/'runs'/RUN
    state=ROOT/'reports/closed_loop_finalization_20261008_r1.json'
    assert not state.exists()
    state.write_text(json.dumps({'phase':'waiting_for_evaluation','pid':os.getpid()},indent=2))
    while True:
        s=json.loads((out/'status.json').read_text())
        if s['phase']=='verification_failed':
            state.write_text(json.dumps({'phase':'blocked_by_verification_failure'},indent=2))
            raise RuntimeError('Evaluation verification failed; no release or success declaration')
        if s['phase']=='evaluation_and_verification_complete':break
        time.sleep(15)
    report=json.loads((out/'verified_results.json').read_text())
    assert report['verification_pass'] and report['complete']
    for version in ['rollout_20261008_r1','media_20261008_r1']:
        for name,want in json.loads((ROOT/'releases'/version/'manifest.json').read_text())['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name
    state.write_text(json.dumps({'phase':'packaging_mp4','pid':os.getpid()},indent=2))
    cmd=['docker','run','--rm','--network','none','--cpus','2','--memory','3g',
         '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
         '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/app/agent-work-space/5-final-train',
         '--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
         'libero:latest','/app/agent-work-space/5-final-train/versions/media_20261008_r1/scripts/package.py','--run',RUN]
    with (out/'media_packaging.log').open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0, 'MP4 packaging failed'
    videos=ROOT/'videos'/RUN
    subprocess.run(['docker','run','--rm','--network','none','-v',str(videos)+':/output',
                    '--entrypoint','chown','libero:latest','-R',f'{os.getuid()}:{os.getgid()}','/output'],check=True)
    index=json.loads((videos/'index.json').read_text())
    assert len([v for v in index['videos'] if v['kind']=='predeclared_first_attempt'])==40
    assert all(v['decode_pass'] for v in index['videos'])
    conditions=report['conditions'];ft=conditions['eco65_zhixing_lora']
    when=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    lines=['# π0.5 闭环测试完成记录 — 20261008 r1','',
        f'核对时间：{when}。四组各200次、共800次预登记测试已完成，独立记录/动作/视频复核通过。',
        f'微调ECO65+ZhiXing主成功率：{ft["successes"]}/200（{ft["success_rate"]:.1%}），'+('达到' if ft['acceptance_pass'] else '未达到')+'本轮≥90%验收目标。训练完成、闭环执行完成和适配验收是不同结论。',
        '', '| 组合 | 成功/200 | 成功率 | Wilson95%区间 | 进程异常 |', '|---|---:|---:|---|---:|']
    for name,c in conditions.items():
        ci=c['wilson_95'];lines.append(f'| {name} | {c["successes"]} | {c["success_rate"]:.1%} | {ci[0]:.1%}–{ci[1]:.1%} | {c["process_failures"]} |')
    lines+=['','| Task | 已训练 | 微调ECO65+ZhiXing | Panda基座 | ECO65+ZhiXing基座 | ECO65+Panda基座 |','|---:|---|---:|---:|---:|---:|']
    order=['eco65_zhixing_lora','panda_base','eco65_zhixing_base','eco65_panda_base']
    for task in range(10):
        values=[str(conditions[c]['tasks'][str(task)]['successes'])+'/20' for c in order]
        lines.append('| '+str(task)+' | '+('是' if task in [0,2,3,5,6] else '否')+' | '+' | '.join(values)+' |')
    lines+=['',f'微调已训练五任务：{ft["groups"]["trained"]["successes"]}/100；未训练五任务：{ft["groups"]["untrained"]["successes"]}/100。没有用测试结果重新选模、调控制器、改动作或补替换种子。',
        '', '主评测220步；准备120步；每5步推理；首条每任务最多650步的结果是独立延长诊断，不能加到主成功率。初态审核拒绝、物理异常、进程异常均保留在固定分母中。该协议使用冻结随机reset测试种子，不是官方.pruned初态成绩；执行观察也不是完整r6教师轨迹物理认证。']
    for name,c in conditions.items():
        invalid=sum(t['failure_categories'].get('invalid_native_initialization',0) for t in c['tasks'].values())
        native_count=sum(t['initialization_pass_count'] for t in c['tasks'].values())
        lines.append(f'{name}：初态拒绝{invalid}/200；初态通过{native_count}/200；机器人↔对象穿透超过2mm的尝试{c["robot_object_penetration_over_2mm_episodes"]}。这些物理诊断不自动升级为教师物理验收。')
    lines+=['','各组reset对象哈希对比：'+json.dumps(report['paired_native_scenes'],ensure_ascii=False),
        '',f'已生成、核验并索引{len(index["videos"])}段MP4，其中40段是预登记首条真实双视角录像。初态拒绝的任务另补第一条实际策略运行输入录像（4fps，无插帧），不重跑测试、不按成功挑选。',
        f'[视频索引](videos/{RUN}/视频索引.md)。视频内和文件名标明主评测结果；失败视频没有删除。',
        '', '微调首条延长诊断：']
    for v in report['videos']:
        if v['condition']=='eco65_zhixing_lora':
            r=json.loads((out/v['condition']/f'task{v["task"]:02d}_s{v["seed"]}'/'result.json').read_text())
            lines.append(f'- task{v["task"]:02d} seed{v["seed"]}：初态'+('通过' if v['initialization_pass'] else '拒绝')+f'；220步内成功={v["primary_success"]}；延长原生成功={v["extended_success"]}；首次成功步={r.get("first_success_step")}。')
    timing=ROOT/'reports/teacher_timing_diagnostic_20261008_r1.json'
    if timing.exists():
        teacher=json.loads(timing.read_text());lines+=['','独立读取已准入训练/验证教师时序（没有读取test来改预算）：']
        for task,t in teacher['tasks'].items():
            lines.append(f'- task{task}：{t["episodes"]}条；首次原生成功中位数{t["first_native_median"]:.0f}步，最早{t["first_native_min"]}步；220步内达到目标{t["native_by_220"]}条。')
        lines+=['这说明教师时间分布与220步主协议存在差异。它是本轮结果的限制之一，不能据此原地改测试预算或宣称延长诊断等于通过。']
    lines+=['','原始证据：runs/'+RUN+'/verified_results.json、summary.json、各episode的execution.json/initialization.json/trajectory.h5/steps.jsonl；源码冻结清单releases/rollout_20261008_r1/manifest.json。',
            '没有自动再训练，没有部署到真实硬件。全部旧冻结源与失败烟测保留。']
    doc=ROOT/'闭环测试完成记录_20261008_r1.md'
    with doc.open('x') as f:f.write('\n'.join(lines)+'\n')
    state.write_text(json.dumps({'phase':'complete','completed_at':when,'video_count':len(index['videos']),
                                'evaluation_verification_pass':True,'adaptation_acceptance_pass':ft['acceptance_pass'],
                                'report':str(doc),'video_index':str(videos/'视频索引.md')},ensure_ascii=False,indent=2))
    print('FINALIZATION COMPLETE',str(doc),flush=True)


if __name__=='__main__':main()
