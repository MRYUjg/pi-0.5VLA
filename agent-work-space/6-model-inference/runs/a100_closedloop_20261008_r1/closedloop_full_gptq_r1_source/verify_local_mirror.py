"""Verify downloaded evidence explicitly as a partial local mirror."""
import argparse
import hashlib
import json
import pathlib

VERSION = 'a100_closedloop_20261008_r1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main(root):
    release = root / 'releases' / VERSION
    manifest_path = release / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    server = json.loads((release / 'server_verify_r1.json').read_text())
    assert server['passed'] and sha(manifest_path) == server['manifest_sha256']
    verified = []
    missing = []
    mismatches = []
    for key, entry in manifest['files_relative_to_6_model_inference'].items():
        path = root / key
        if not path.exists():
            missing.append(key)
        elif path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            mismatches.append(key)
        else:
            verified.append(key)
            path.chmod(0o444)
    required = [
        f'versions/{VERSION}/A100统一闭环测试记录.md',
        f'versions/{VERSION}/protocol.json',
        f'runs/{VERSION}/report_r1/summary.json',
        f'runs/{VERSION}/report_r1/episodes.csv',
        f'runs/{VERSION}/artifact_audit_r1/verified.json',
        f'runs/{VERSION}/closedloop_sequence_r1_exit.json',
        '当前实施索引_20261008_r1.md',
    ]
    assert not mismatches and all(key in verified for key in required), (mismatches, required)
    receipt = {
        'passed_for_downloaded_files': True,
        'partial_local_mirror': True, 'manifest_sha256': server['manifest_sha256'],
        'verified_current_file_count': len(verified), 'not_downloaded_current_file_count': len(missing),
        'not_downloaded_current_files': missing,
        'server_is_authoritative_for_engines_weights_and_all_videos': True,
        'local_verification_does_not_replace_server_content_and_video_audit': True,
        'mismatches': mismatches,
    }
    with (release / 'local_mirror_verify_r1.json').open('x') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
        f.write('\n')
    for path in release.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print('LOCAL PARTIAL MIRROR VERIFIED', len(verified), 'downloaded files;', len(missing), 'not downloaded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    main(parser.parse_args().root)
