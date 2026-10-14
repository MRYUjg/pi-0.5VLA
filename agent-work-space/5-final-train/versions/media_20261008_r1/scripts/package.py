"""Publish honest representative MP4s; retain first attempts, supplement rejected initializations."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

V=pathlib.Path(__file__).resolve().parents[1]
ROOT=V.parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    assert pathlib.Path(a.run).name==a.run
    run=ROOT/'runs'/a.run
    report=json.loads((run/'verified_results.json').read_text())
    assert report['verification_pass'] and report['complete']
    out=ROOT/'videos'/a.run;out.mkdir(parents=True,exist_ok=False)
    rows=[];lines=['# π0.5 闭环测试 MP4 索引','',
        '首条尝试视频按预登记规则保留，不按成功挑选。若首条初态被拒绝，补充该任务按seed排序第一条实际运行策略的输入录像；不重跑测试、不改变结果。',
        '','原始闭环录像20fps，双相机；补充视频取HDF5中真实推理输入帧，4fps（每5控制步一帧），不插帧、不模拟新动作。影片末尾标明主评测结果。',
        '', '| 组合 | 任务 | seed | 主评测 | 初态 | 视频 |', '|---|---:|---:|---|---|---|']
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for video in report['videos']:
        condition=video['condition'];task=video['task'];seed=video['seed']
        dest=out/condition;dest.mkdir(exist_ok=True)
        status='success' if video['primary_success'] else 'failure'
        path=dest/f'task{task:02d}_s{seed}_{status}.mp4'
        source=ROOT/video['path'];shutil.copyfile(source,path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==video['sha256']
        row={**video,'path':str(path.relative_to(ROOT)),'kind':'predeclared_first_attempt','fps':20}
        rows.append(row)
        lines.append(f"| {condition} | {task} | {seed} | {status} | {'通过' if video['initialization_pass'] else '拒绝'} | [首条录像]({condition}/{path.name}) |")
        if video['initialization_pass']:continue
        candidates=[]
        for episode in sorted((run/condition).glob(f'task{task:02d}_s*')):
            if not episode.is_dir() or not (episode/'execution.json').exists():continue
            execution=json.loads((episode/'execution.json').read_text());result=execution.get('result',{})
            if execution['process_exit']==0 and result.get('initialization_pass') and result.get('queries',0)>0:
                candidates.append((int(result['seed']),episode,result))
        if not candidates:continue
        selected,episode,result=min(candidates,key=lambda x:x[0])
        status='success' if result['primary_success'] else 'failure'
        path=dest/f'task{task:02d}_s{selected}_{status}_policy_inputs.mp4'
        encoder=subprocess.Popen([ffmpeg,'-nostdin','-v','error','-f','rawvideo','-pixel_format','rgb24',
                                  '-video_size','512x288','-framerate','4','-i','pipe:0','-an',
                                  '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
        count=0;last=None
        with h5py.File(episode/'trajectory.h5','r') as f:
            for name,g in sorted(f['queries'].items()):
                frame=Image.new('RGB',(512,288),'#171c25');draw=ImageDraw.Draw(frame)
                draw.text((5,3),f'{condition} task{task:02d} seed{selected} step{g.attrs["step"]}',fill='white')
                draw.text((5,17),'RECORDED POLICY INPUTS | 4fps | agentview + wrist',fill='#d1d9ea')
                frame.paste(Image.fromarray(g['image'][...]),(0,32))
                frame.paste(Image.fromarray(g['wrist_image'][...]),(256,32))
                encoder.stdin.write(np.asarray(frame).tobytes());count+=1;last=frame
        draw=ImageDraw.Draw(last);draw.rectangle((0,0,512,32),fill='#171c25')
        draw.text((5,4),f'PRIMARY {status.upper()} | {result.get("failure") or "native goal within220"}',fill='white')
        for _ in range(4):encoder.stdin.write(np.asarray(last).tobytes());count+=1
        encoder.stdin.close();assert encoder.wait(timeout=60)==0
        frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(path))
        decoded=subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],capture_output=True)
        assert frames==count and decoded.returncode==0
        row={'condition':condition,'task':task,'seed':selected,'primary_success':result['primary_success'],
             'initialization_pass':True,'path':str(path.relative_to(ROOT)),'kind':'first_policy_episode_input_replay',
             'selection':'first initialization-pass episode by seed, independent of success',
             'fps':4,'frames':frames,'seconds':seconds,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decode_pass':True}
        rows.append(row)
        lines.append(f'| {condition} | {task} | {selected} | {status} | 通过 | [策略输入录像]({condition}/{path.name}) |')
    (out/'index.json').write_text(json.dumps({'run':a.run,'videos':rows,'test_rerun':False},ensure_ascii=False,indent=2))
    (out/'视频索引.md').write_text('\n'.join(lines)+'\n')
    print('PACKAGED',len(rows),'MP4',flush=True)


if __name__=='__main__':main()
