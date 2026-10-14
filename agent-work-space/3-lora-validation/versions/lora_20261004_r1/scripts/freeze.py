import argparse,hashlib,json,pathlib,tarfile
V=pathlib.Path(__file__).resolve().parents[1];ROOT=V.parents[1];REPO=ROOT.parents[1];RELEASE=ROOT/'releases'/V.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['create','verify']);a=p.parse_args()
 if a.mode=='create':
  RELEASE.mkdir(parents=True,exist_ok=False);src=sorted(p for p in V.rglob('*') if p.is_file() and '__pycache__' not in str(p));deps=sorted((REPO/'src/openpi').rglob('*.py'))+[REPO/'scripts/train.py',REPO/'pyproject.toml']
  with tarfile.open(RELEASE/'source.tar.gz','x:gz') as t:
   for p in src:t.add(p,arcname=str(p.relative_to(ROOT)))
  with (RELEASE/'manifest.json').open('x') as f:json.dump({'files':{str(p.relative_to(REPO)):sha(p) for p in src+deps},'archive_sha256':sha(RELEASE/'source.tar.gz')},f,indent=2)
 m=json.loads((RELEASE/'manifest.json').read_text());bad=[n for n,h in m['files'].items() if sha(REPO/n)!=h];assert sha(RELEASE/'source.tar.gz')==m['archive_sha256'];print(json.dumps({'pass':not bad,'files':len(m['files']),'errors':bad}));assert not bad
if __name__=='__main__':main()
