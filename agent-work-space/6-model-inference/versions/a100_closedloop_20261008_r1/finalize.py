"""Audit all 150 saved episodes, report, then freeze the completed new run."""
import json
import subprocess
import sys

from resources import ROOT, RUN, VERSION, write

source = ROOT / 'versions' / VERSION
assert json.loads((RUN / 'closedloop_sequence_r1_exit.json').read_text())['exit_code'] == 0
operations = []
for name, script, out in [('artifact_audit_r1', 'audit_artifacts.py', 'artifact_audit_r1'),
                          ('report_r1', 'report.py', 'report_r1')]:
    command = [sys.executable, str(source / 'execute.py'), '--name', name, script,
               '--root', '/work/runs/' + VERSION,
               '--out', '/work/runs/' + VERSION + '/' + out]
    code = subprocess.run(command).returncode
    operations.append({'command': command, 'exit_code': code})
    assert code == 0, operations
command = [sys.executable, str(source / 'write_documentation.py'), '--root', str(ROOT)]
code = subprocess.run(command).returncode
operations.append({'command': command, 'exit_code': code})
assert code == 0
container = 'pi05-a100-build-20261008-r1'
command = ['docker', 'stop', '-t', '2', container]
code = subprocess.run(command).returncode
assert code == 0
state = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]['State']
assert not state['Running'] and not state['OOMKilled']
write(RUN / 'build_container_shutdown.json', {
    'command': command, 'docker_stop_exit_code': code, 'state': state,
    'idle_container_only': True,
    'idle_sleep_PID1_was_intentionally_stopped_after_all_build_and_analysis_execs_completed': True,
})
operations.append({'command': command, 'exit_code': code, 'container_state': state})
write(RUN / 'finalization_steps_before_freeze_r1.json', {'operations': operations, 'all_exit_zero': True})
code = subprocess.run([sys.executable, str(source / 'freeze_release.py')]).returncode
assert code == 0
print('A100 CLOSEDLOOP FINALIZATION COMPLETE', flush=True)
