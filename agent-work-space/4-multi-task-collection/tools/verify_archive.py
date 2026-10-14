"""Read-only full archive/dependency SHA256 check; no task acceptance claim."""
import argparse
import ast
import hashlib
import json
import pathlib
import re
import sys
from urllib.parse import unquote

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEAL = ROOT / 'archive/final_20261005_r1'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify():
    manifest_path = SEAL / 'manifest.json'
    errors = []
    manifest = json.loads(manifest_path.read_text())
    expected_seal = (SEAL / 'manifest.sha256').read_text().split()[0]
    if sha(manifest_path) != expected_seal:
        errors.append('manifest checksum')
    for name, digest in manifest['files'].items():
        p = ROOT / name
        if not p.is_file() or sha(p) != digest:
            errors.append('archive file: ' + name)
    ignored = {'archive/final_20261005_r1/' + n for n in ('manifest.json', 'manifest.sha256', 'verification.json')}
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    unexpected = sorted(actual - set(manifest['files']) - ignored)
    errors.extend('unexpected file: ' + n for n in unexpected)
    dependencies = json.loads((SEAL / 'dependencies.json').read_text())
    for name, digest in dependencies['files'].items():
        p = REPO / name
        if not p.is_file() or sha(p) != digest:
            errors.append('dependency: ' + name)
    python_files = [p for p in ROOT.rglob('*.py')]
    for p in python_files:
        try:
            ast.parse(p.read_text(), filename=str(p))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))
    links = 0
    for p in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            links += 1
            if not (p.parent / unquote(target.split('#')[0])).exists():
                errors.append('broken Markdown link: ' + str(p.relative_to(ROOT)) + ' -> ' + target)
    videos = list(ROOT.rglob('*.mp4'))
    if len(videos) != 11:
        errors.append('expected exactly 11 representative MP4 files')
    for p in videos:
        meta = json.loads(p.with_suffix('.json').read_text())
        if sha(p) != meta['video_sha256'] or meta['frames'] != meta['decoded_frames']:
            errors.append('video evidence: ' + p.name)
    ledger = json.loads((SEAL / 'cleanup_ledger.json').read_text())
    for record in ledger['deleted']:
        if (ROOT / record['path']).exists():
            errors.append('obsolete file still exists: ' + record['path'])
    return {
        'pass': not errors, 'archive_version': manifest['archive_version'],
        'files_checked': len(manifest['files']), 'external_dependencies_checked': len(dependencies['files']),
        'python_files_parsed': len(python_files), 'markdown_links_checked': links,
        'mp4_files_verified': len(videos), 'deleted_files_confirmed': len(ledger['deleted']),
        'unexpected_files': unexpected, 'errors': errors,
        'scope': 'file identity, inventory, syntax, links and existing video evidence; not a new task/policy evaluation',
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=pathlib.Path)
    args = parser.parse_args()
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with args.out.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    return 0 if result['pass'] else 1

if __name__ == '__main__':
    sys.exit(main())
