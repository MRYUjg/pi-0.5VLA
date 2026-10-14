"""Final archive entry point. Print the command by default; --execute runs it."""
import argparse
import importlib.util
import pathlib
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = {
    'can-collect': ('expansion_20261004_r1', 'collect.py'),
    'can-replay': ('expansion_20261004_r1', 'replay.py'),
    'can-audit': ('expansion_20261004_r1', 'audit.py'),
    'can-preflight': ('expansion_20261004_r1', 'preflight.py'),
    'can-negative-check': ('expansion_20261004_r1', 'test_rejections.py'),
    'spatial-collect': ('spatial_pilot_20261004_r1', 'collect.py'),
    'spatial-physical-audit': ('spatial_pilot_20261004_r1', 'physical_audit.py'),
    'spatial-replay': ('spatial_replay_20261004_r1', 'replay.py'),
    'video': ('spatial_video_20261004_r2', 'video.py'),
    'lerobot-convert': ('lerobot_expansion_20261004_r1', 'convert.py'),
    'lerobot-validate': ('lerobot_expansion_20261004_r1', 'validate.py'),
}

def command(module, arguments):
    version, script = MODULES[module]
    directory = ROOT / 'versions' / version / 'scripts'
    if not (directory / script).is_file():
        raise FileNotFoundError(directory / script)
    spec = importlib.util.spec_from_file_location('_archive_launcher', directory / 'run.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    args = arguments[1:] if arguments[:1] == ['--'] else arguments
    if version.startswith('lerobot_'):
        return launcher.command(script, args)
    cmd = launcher.container_command(script, args)
    # The unchanged r6 physical auditor also resolves historical host-root paths.
    cmd[2:2] = ['-v', str(launcher.REPO) + ':' + str(launcher.REPO) + ':ro']
    return cmd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('module', choices=MODULES)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = command(args.module, args.arguments)
    print(shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode if args.execute else 0

if __name__ == '__main__':
    raise SystemExit(main())
