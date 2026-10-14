"""Host-side source freeze; does not import numerical runtime or overwrite releases."""
import argparse,hashlib,json,pathlib,tarfile
V=pathlib.Path(__file__).resolve().parents[1];C=V.parents[1];REPO=C.parents[1];R=C/'runtime/lerobot_py310_r1';RELEASE=C/'releases'/V.name

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['create','verify']);a=p.parse_args()
    if a.mode=='create':
        RELEASE.mkdir(parents=True,exist_ok=False)
        src=sorted(p for p in V.rglob('*') if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('._'))
        deps=list((R/'venv/lib/python3.10/site-packages/lerobot').rglob('*.py'))+[R/'pip_freeze.txt',R/'environment.json',R/'upstream.tar.gz']
        deps += [REPO/n for n in ['pyproject.toml','examples/libero/convert_libero_data_to_lerobot.py','src/openpi/policies/libero_policy.py','src/openpi/training/config.py','src/openpi/training/data_loader.py']]
        with tarfile.open(RELEASE/'source.tar.gz','x:gz') as t:
            for p in src:t.add(p,arcname=str(p.relative_to(C)))
        manifest={'version':V.name,'files':{str(p.relative_to(REPO)):sha(p) for p in src+deps},'archive_sha256':sha(RELEASE/'source.tar.gz'),'scope':'conversion source and installed LeRobot Python source; runtime package versions recorded, not all binary dependencies fingerprinted'}
        with (RELEASE/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    m=json.loads((RELEASE/'manifest.json').read_text());errors=[n for n,v in m['files'].items() if not (REPO/n).is_file() or sha(REPO/n)!=v]
    if sha(RELEASE/'source.tar.gz')!=m['archive_sha256']:errors.append('archive')
    print(json.dumps({'pass':not errors,'checked_files':len(m['files']),'errors':errors}));assert not errors
if __name__=='__main__':main()
