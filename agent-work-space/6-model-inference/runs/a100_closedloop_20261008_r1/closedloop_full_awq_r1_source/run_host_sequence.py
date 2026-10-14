"""Capture the real exit of the five-model host sequence without shell pipelines."""
import json
import pathlib
import subprocess
import sys
import time

from resources import ROOT, RUN, VERSION


command = [sys.executable, str(ROOT / 'versions' / VERSION / 'run_closedloop.py'),
           '--variant', 'trt_float', 'awq', 'gptq', 'full_awq', 'full_gptq',
           '--attempt', 'r1']
started = time.time()
with (RUN / 'closedloop_sequence_r1.log').open('x') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    code = process.wait()
with (RUN / 'closedloop_sequence_r1_exit.json').open('x') as receipt:
    json.dump({'command': command, 'exit_code': code,
               'started_unix': started, 'seconds': time.time() - started},
              receipt, indent=2)
    receipt.write('\n')
raise SystemExit(code)
