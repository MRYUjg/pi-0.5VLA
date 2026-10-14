"""Local-only floating reference teacher for Panda calibration, not a deployment service."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pathlib
import time
import threading
import os

import numpy as np
import torch

from common import load_model, prepare, transforms, write_json, SOURCE_SHA

p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18877);p.add_argument('--ready',type=pathlib.Path,required=True);a=p.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
net,cfg=load_model()
_,output=transforms(cfg)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def do_POST(self):
        if self.path=='/shutdown':
            self.send_response(200);self.end_headers();self.wfile.write(b'{}')
            threading.Thread(target=self.server.shutdown,daemon=True).start();return
        try:
            obj=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            record={'image':np.frombuffer(base64.b64decode(obj['image']),dtype=np.uint8).reshape(224,224,3),
                    'wrist_image':np.frombuffer(base64.b64decode(obj['wrist_image']),dtype=np.uint8).reshape(224,224,3),
                    'state':np.asarray(obj['state'],np.float32),'prompt':obj['prompt']}
            observation=prepare(record,cfg)
            torch.manual_seed(int(obj['noise_seed']))
            noise=torch.randn((1,10,32),device='cuda',dtype=torch.float32)
            start=time.perf_counter()
            with torch.inference_mode():
                actions=net.sample_actions('cuda',observation,noise=noise,num_steps=10)
                result=output({'state':observation.state[0].cpu().numpy(),'actions':actions[0].cpu().numpy()})
            body=json.dumps({'actions':result['actions'].tolist(),'seconds':time.perf_counter()-start}).encode()
            self.send_response(200);self.end_headers();self.wfile.write(body)
        except Exception as exc:
            self.send_response(500);self.end_headers();self.wfile.write(repr(exc).encode());raise


write_json(a.ready,{'status':'ready','port':a.port,'pid':os.getpid(),'source_sha256':SOURCE_SHA,'denoise_steps':10,'calibration_only':True})
print('TEACHER READY',flush=True)
HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
