"""Actual recorded RGB -> labeled H264, with complete decoder verification."""
import argparse,pathlib,json,hashlib,textwrap
import h5py,numpy as np,imageio_ffmpeg as ff
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episode',required=True);ap.add_argument('--out',required=True);ap.add_argument('--label',required=True);a=ap.parse_args()
    ep=pathlib.Path(a.episode);out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    result=json.loads((ep/'teacher/result.json').read_text())
    audit=json.loads((ep/'audit.json').read_text())
    assert audit['accepted'], 'only admitted formal demonstrations'
    physical=audit['physical_audit'];replay_path=ep/'replay.json'
    replay=json.loads(replay_path.read_text())
    passed=result['success'] and physical['audit_pass'] and replay['pass'] and audit['initialization_pass']
    assert passed
    status='ACCEPTED FORMAL TEACHER DEMONSTRATION'
    failure=result.get('failure');reason='' if not failure else failure['stage']+': '+failure['reason']
    writer=ff.write_frames(str(out),(512,384),fps=20,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=8,output_params=['-movflags','+faststart']);writer.send(None)
    h5=ep/'trajectory.h5';h5=h5 if h5.exists() else ep/'trajectory.partial.h5'
    frame_count=0;selected=[]
    with h5py.File(h5,'r') as f:
        groups=[]
        if 'preparation_images' in f:groups.append(('PREPARATION',f['preparation_images']))
        if 'observations' in f:groups.append(('TEACHER',f['observations']))
        total=sum(len(g['image']) for _,g in groups)
        assert total>0,'no actual recorded frames'
        for section,g in groups:
            for i in range(len(g['image'])):
                canvas=Image.new('RGB',(512,384),'#121820');d=ImageDraw.Draw(canvas)
                d.text((8,5),a.label+' | seed '+str(result['seed']),fill='white')
                d.text((8,20),status,fill=('#70ef9c' if passed else '#ff9c75'))
                title=str(f.attrs['language'])
                for k,line in enumerate(textwrap.wrap(title,width=80)):d.text((8,35+12*k),line,fill='white')
                phase=section
                if section=='TEACHER' and 'transitions/phase' in f and len(f['transitions/phase']):phase=f['transitions/phase'][min(i,len(f['transitions/phase'])-1)].decode().strip()
                d.text((8,65),f'agentview                         wrist | {phase} | {frame_count/20:.2f}s',fill='#b0cde0')
                canvas.paste(Image.fromarray(g['image'][i]),(0,84));canvas.paste(Image.fromarray(g['wrist_image'][i]),(256,84))
                for k,line in enumerate(textwrap.wrap(reason,width=80)[:3]):d.text((8,344+12*k),line,fill='#ff9c75')
                writer.send(np.asarray(canvas));
                if frame_count in {0,total//2,total-1}:selected.append(canvas.copy())
                frame_count+=1
    writer.close()
    reader=ff.read_frames(str(out),pix_fmt='rgb24');meta=next(reader);decoded=0
    for frame in reader:
        assert len(frame)==512*384*3;decoded+=1
    assert decoded==frame_count and abs(meta['fps']-20)<1e-6
    sheet=Image.new('RGB',(512*len(selected),384))
    for i,im in enumerate(selected):sheet.paste(im,(i*512,0))
    sheet.save(out.with_suffix('.jpg'))
    record={'episode':str(ep),'video':str(out),'label':a.label,'status':status,'teacher_success':result['success'],'physical_audit_pass':physical.get('audit_pass'),'replay_pass':replay.get('pass'),'replay_source':str(replay_path),'failure':failure,'frames':frame_count,'decoded_frames':decoded,'fps':20,'duration_seconds':frame_count/20,'size':[512,384],'video_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'hdf5_sha256':hashlib.sha256(h5.read_bytes()).hexdigest(),'metadata':meta,'includes_preparation':any(section=='PREPARATION' for section,_ in groups)}
    with out.with_suffix('.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
