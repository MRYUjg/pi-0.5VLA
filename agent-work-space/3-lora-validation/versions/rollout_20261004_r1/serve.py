"""Loopback-only diagnostic RPC using actual OpenPI Policy transforms and saved checkpoint."""
import argparse,base64,json,pathlib,sys,threading
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--training-version',default='lora_20261004_r1');p.add_argument('--port',type=int,default=8871);a=p.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'versions'/a.training_version/'scripts'))
from configuration import make,TRAIN_ID
from openpi.models import model
from openpi.policies.policy import Policy
from openpi import transforms
from openpi.shared import normalize
cfg=make(a.out);net=cfg.model.load(model.restore_params(a.out/'checkpoint/300/params'));net.eval();dc=cfg.data.create(cfg.assets_dirs,cfg.model)
stats=normalize.load(a.out/'checkpoint/300/assets'/TRAIN_ID)
policy=Policy(net,transforms=[*dc.data_transforms.inputs,transforms.Normalize(stats,use_quantiles=True),*dc.model_transforms.inputs],output_transforms=[*dc.model_transforms.outputs,transforms.Unnormalize(stats,use_quantiles=True),*dc.data_transforms.outputs],sample_kwargs={'num_steps':10})
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{"ready":true}')
 def do_POST(self):
  if self.path=='/shutdown':
   self.send_response(200);self.end_headers();threading.Thread(target=self.server.shutdown).start();return
  try:
   x=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   obs={'observation/state':np.asarray(x['state'],dtype=np.float32),'prompt':x['prompt']}
   for k in ('image','wrist_image'):obs['observation/'+k]=np.frombuffer(base64.b64decode(x[k]),np.uint8).reshape(256,256,3)
   actions=policy.infer(obs)['actions'];assert actions.shape==(10,7) and np.isfinite(actions).all()
   body=json.dumps({'actions':actions.tolist()}).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
  except Exception as e:
   self.send_response(500);self.end_headers();self.wfile.write(repr(e).encode());raise
print('POLICY READY',a.port,flush=True);HTTPServer(('127.0.0.1',a.port),Handler).serve_forever();print('POLICY CLOSED',flush=True)
