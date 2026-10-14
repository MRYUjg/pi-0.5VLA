"""Preserve valid floating artifacts after the r4 filename bookkeeping failure."""
import os
import pathlib
import json
import onnx
from common import ROOT,sha,write_json

run=ROOT/'runs/trt_t0_t2_20261007_r1'
source=run/'t1_attempt_r4';out=run/'t1_salvage_r1';out.mkdir(exist_ok=False)
for name in ['vision','prefix']:
    graph=source/(name+'.onnx')
    m=onnx.load(graph,load_external_data=False)
    files={v.value for t in m.graph.initializer for v in t.external_data if v.key=='location'}
    for file in [name+'.onnx',*files]:os.link(source/file,out/file)
    engine_name=name+'.engine' if name=='vision' else 'paligemma_with_expert.paligemma.model.language_model.layers.17.self_attn.v_proj.engine'
    os.link(source/engine_name,out/(name+'.engine'))
    meta={'opset':m.opset_import[0].version,'source_onnx_sha256':sha(graph),'engine_bytes':(source/engine_name).stat().st_size,
          'recovery':'Engine constructed successfully before filename lookup KeyError; no binary modification',
          'source_engine_file':str(source/engine_name),'source_engine_sha256':sha(source/engine_name),
          'source_attempt_status':'failed','numerical_acceptance':'not_yet_evaluated'}
    write_json(out/(name+'_export.json'),meta)
    onnx.checker.check_model(str(out/(name+'.onnx')))
print('SALVAGE PASS')
