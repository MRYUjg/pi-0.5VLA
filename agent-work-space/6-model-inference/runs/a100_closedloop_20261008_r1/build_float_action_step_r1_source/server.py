"""Local container service with exactly one resident policy and serial requests."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import pathlib
import threading
import time
import traceback

import numpy as np

from common import ROOT, verified, write_json
from models import InferenceModel


def record(obj):
    return {'image':np.frombuffer(base64.b64decode(obj['image']),np.uint8).reshape(224,224,3),
        'wrist_image':np.frombuffer(base64.b64decode(obj['wrist_image']),np.uint8).reshape(224,224,3),
        'state':np.asarray(obj['state'],np.float32),'prompt':obj['prompt']}


def main(out, variant, port):
    out.mkdir(parents=True,exist_ok=False)
    begin = time.perf_counter()
    model = InferenceModel(variant)
    pilot = ROOT / 'runs/trt_t0_t2_20261007_r1/calibration_r1/pilot.json'
    verified(pilot)
    row = json.loads(pilot.read_text())[0]
    verified(pathlib.Path(row['path']))
    raw = dict(np.load(row['path']))
    raw['prompt'] = str(raw['prompt'])
    _, warm = model.infer(raw,row['noise_seed'])
    registration = {'status':'ready','pid':os.getpid(),'variant':variant,'metadata':model.metadata,
        'load_and_first_complete_warmup_seconds':time.perf_counter()-begin,
        'first_warmup':warm,'denoising_steps':10,'resident_models':1,
        'serial_requests':True,'JAX_PERFORMANCE_BASELINE':False,'actual_4070_accepted':False}
    write_json(out/'ready.json',registration)
    counter = [0]
    log = (out/'requests.jsonl').open('x',buffering=1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_GET(self):
            if self.path != '/health':
                self.send_error(404);return
            data=json.dumps(registration).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_POST(self):
            if self.path == '/shutdown':
                self.send_response(200);self.end_headers();self.wfile.write(b'{}')
                threading.Thread(target=self.server.shutdown,daemon=True).start();return
            if self.path != '/infer':
                self.send_error(404);return
            started=time.perf_counter()
            try:
                obj=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                raw=record(obj)
                decoded=time.perf_counter()
                actions, timing=model.infer(raw,int(obj['noise_seed']))
                timing['decode_ms']=(decoded-started)*1000
                timing['server_processing_ms']=(time.perf_counter()-started)*1000
                counter[0]+=1
                result={'actions':actions.tolist(),'internal_actions':model.last_internal.tolist(),'timing':timing,'request':counter[0],'variant':variant}
                log.write(json.dumps({'request':counter[0],'noise_seed':int(obj['noise_seed']),
                    'client_id':obj.get('client_id'),'timing':timing,'finite':True})+'\n')
                data=json.dumps(result).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            except Exception:
                error=traceback.format_exc()
                log.write(json.dumps({'error':error})+'\n')
                self.send_response(500);self.end_headers();self.wfile.write(error.encode())
                raise

    class SerialServer(HTTPServer):
        request_queue_size = 1

    print('POLICY READY',variant,flush=True)
    server = SerialServer(('0.0.0.0',port),Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close();log.close()
        write_json(out/'shutdown.json',{'requests':counter[0],'clean_shutdown':True,'variant':variant})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--variant',choices=['trt_float','awq','gptq','full_awq','full_gptq'],required=True)
    p.add_argument('--port',type=int,default=18879)
    a=p.parse_args()
    main(a.out,a.variant,a.port)
