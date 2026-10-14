"""Freeze this engineering baseline only after its declared checks pass."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from launch import REPO, ROOT


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    audit = json.loads((ROOT / 'reports/spatial10_r1_audit.json').read_text())
    final = json.loads((ROOT / 'reports/final_checks_r1.json').read_text())
    workflow = json.loads((ROOT / 'runs/spatial10_r1/workflow_summary.json').read_text())
    assert audit['trials'] == 10 and audit['successes'] == 10 and audit['all_audits_passed']
    assert final['reused_sources_unchanged'] and final['pilot_and_parallel_task0_actions_states_exact']
    assert workflow['all_task_processes_exit_zero'] and workflow['policy_exited_zero']
    assert not workflow['policy_running_after_cleanup']
    files = [path for path in ROOT.rglob('*') if path.is_file()
             and not any(part in {'runtime', '__pycache__', 'releases'} for part in path.relative_to(ROOT).parts)]
    secret = (REPO / 'gpt-6-control/deepseek_api_key').read_bytes().strip()
    assert secret and not any(secret in path.read_bytes() for path in files), 'credential isolation check failed'
    scan_path = ROOT / 'reports/credential_scan_final_r1.json'
    with scan_path.open('x') as file:
        json.dump({'files_checked': len(files), 'credential_value_found': False,
                   'credential_not_disclosed': True}, file, indent=2)
    files.append(scan_path)
    entries = {str(path.relative_to(ROOT)): {'sha256': sha(path), 'bytes': path.stat().st_size}
               for path in sorted(files)}
    release = ROOT / 'releases/panda_libero_20261007_r1'
    release.mkdir(parents=True, exist_ok=False)
    manifest = {'version': 'panda_libero_20261007_r1',
                'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'files_relative_to_gpt_rpent_control': entries,
                'excluded': ['runtime', '__pycache__', 'releases', 'external model weights / credentials'],
                'acceptance': {'RPent_API_toolkit_RPC_real_policy_chain': True,
                               'Panda_PandaGripper_Spatial_10x1_native_tasks': True,
                               'formal_multi_initial_benchmark': False,
                               'all_RPent_tools_CLI_Dashboard': False,
                               'LLM_improvement_over_pure_VLA': False,
                               'ECO65_ZhiXing_physical_robot': False}}
    with (release / 'manifest.json').open('x') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    for path in (ROOT / 'versions/panda_libero_20261007_r1').iterdir():
        if path.is_file():
            path.chmod(0o444)
    print('frozen', len(entries), 'files; engineering scope only')


if __name__ == '__main__':
    main()
