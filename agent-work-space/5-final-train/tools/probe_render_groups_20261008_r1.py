"""Read-only validation-scene rendering audit; never alters the evaluation or dataset."""
import hashlib
import json
import pathlib
import h5py
import numpy as np
from PIL import Image, ImageDraw
from robosuite import macros
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/render_groups_diagnostic_20261008_r1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    path=ROOT/'runs/formal_20261005_r1/task02_s630201/trajectory.h5'
    with h5py.File(path,'r') as f:
        sim=MjSim.from_xml_string(f.attrs['environment_xml'])
        sim.data.qpos[:]=f['observations/qpos'][0]
        sim.data.qvel[:]=f['observations/qvel'][0]
    sim.forward()
    context=MjRenderContextOffscreen(sim,-1,max_width=256,max_height=256)
    default=context.vopt.geomgroup.copy().tolist()
    qpos=sim.data.qpos.copy();qvel=sim.data.qvel.copy()
    conv=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    def render():
        return {k:np.ascontiguousarray(sim.render(256,256,camera_name=name)[::conv][::-1,::-1])
                for k,name in [('image','agentview'),('wrist_image','robot0_eye_in_hand')]}
    original=render()
    context.vopt.geomgroup[0]=0
    context.vopt.geomgroup[1]=1
    visual_only=render()
    assert np.array_equal(qpos,sim.data.qpos) and np.array_equal(qvel,sim.data.qvel)
    sheet=Image.new('RGB',(512,576),'#171c25');draw=ImageDraw.Draw(sheet)
    for row,(caption,views) in enumerate([('Capture/manual context default: collision + visual',original),
                                        ('Native OffScreenRenderEnv default: visual only',visual_only)]):
        draw.text((4,row*288+8),caption,fill='white')
        for col,key in enumerate(('image','wrist_image')):sheet.paste(Image.fromarray(views[key]),(col*256,row*288+32))
    sheet.save(OUT/'comparison.png')
    result={'scope':'static reconstruction from validation seed630201; rendering flags only, no policy/test rerun',
            'manual_context_default_geomgroups':default,
            'native_visual_only_geomgroups':context.vopt.geomgroup.tolist(),
            'qpos_changed':False,'qvel_changed':False,'ongoing_evaluation_changed':False,
            'differences':{k:{'mean_abs_pixel_error':float(np.mean(abs(original[k].astype(float)-visual_only[k].astype(float)))),
                             'max_abs_pixel_error':int(np.max(abs(original[k].astype(int)-visual_only[k].astype(int)))),
                             'different_pixel_fraction':float(np.mean(np.any(original[k]!=visual_only[k],axis=-1)))} for k in original},
            'implication':'formal capture and this evaluation match each other; their manual default renderer also shows collision meshes. Base references in this run therefore are not identical to historical visual-only camera references. No causal success-rate effect has been established.'}
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
