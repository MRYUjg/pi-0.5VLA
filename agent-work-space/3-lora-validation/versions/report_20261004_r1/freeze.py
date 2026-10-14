"""Freeze report and small validation evidence, excluding checkpoints and JAX cache."""
import argparse,hashlib,json,pathlib,tarfile
V=pathlib.Path(__file__).resolve().parent;ROOT=V.parents[1];REPO=ROOT.parents[1];RELEASE=ROOT/'releases'/V.name
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['create','verify']);a=p.parse_args()
 if a.mode=='create':
  files=[ROOT/'README.md'];files.extend(p for p in V.rglob('*') if p.is_file() and '__pycache__' not in str(p))
  for sub in ('reports','rollouts/validation_r1','runs/smoke_train_r1_execution','runs/smoke_train_r2_execution','runs/smoke_train_r2/assets'):
   files.extend(p for p in (ROOT/sub).rglob('*') if p.is_file())
  files.extend(p for p in (ROOT/'runs/smoke_train_r2').iterdir() if p.is_file())
  files.extend(p for p in (ROOT/'releases').glob('*/manifest.json'))
  RELEASE.mkdir(exist_ok=False)
  with tarfile.open(RELEASE/'evidence.tar.gz','x:gz') as t:
   for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(ROOT)))
  with (RELEASE/'manifest.json').open('x') as f:json.dump({'files':{str(p.relative_to(REPO)):sha(p) for p in sorted(set(files))},'archive_sha256':sha(RELEASE/'evidence.tar.gz')},f,indent=2)
 m=json.loads((RELEASE/'manifest.json').read_text());bad=[n for n,h in m['files'].items() if sha(REPO/n)!=h];assert sha(RELEASE/'evidence.tar.gz')==m['archive_sha256'];print(json.dumps({'pass':not bad,'files':len(m['files']),'errors':bad}));assert not bad
if __name__=='__main__':main()
