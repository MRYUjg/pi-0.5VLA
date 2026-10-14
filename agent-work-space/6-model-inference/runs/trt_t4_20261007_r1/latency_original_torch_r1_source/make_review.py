"""Scientific episode contact sheets from recorded video, without altering frames."""
import argparse
import json
import pathlib

import imageio.v2 as imageio
from PIL import Image, ImageDraw


def main(root,out):
    out.mkdir(exist_ok=False)
    registry=[]
    for task,index in [(1,0),(4,0)]:
        canvas=Image.new('RGB',(1536,4*282+32),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),f'Task {task}, init {index}: initial / middle / final; both cameras; identical initial state and seed',fill='black')
        for row,variant in enumerate(['original_torch','trt_float','awq','gptq']):
            folder=root/f'closedloop_{variant}_r1/task{task:02d}'
            data=json.loads((folder/f'init{index:02d}.json').read_text())['episode']
            reader=imageio.get_reader(folder/data['video'])
            selected=[0,data['steps']//2,data['steps']]
            for column,frame in enumerate(selected):
                pixels=Image.fromarray(reader.get_data(frame));canvas.paste(pixels,(column*512,32+row*282+24))
                draw.text((column*512+8,32+row*282+6),f'{variant} success={data["success"]} frame={frame} sim_t={frame/20:.2f}s',fill='black')
            reader.close()
        name=f'task{task:02d}_init{index:02d}_comparison.png';canvas.save(out/name)
        registry.append({'file':name,'task':task,'init_index':index,'source':'recorded original dual-camera MP4 frames',
            'selection':'fixed original/float discrepancy examples, not success-rate denominator filtering'})
    (out/'registration.json').write_text(json.dumps(registry,indent=2)+'\n')
    print('REVIEW CONTACT SHEETS SAVED',len(registry),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();main(a.root,a.out)
