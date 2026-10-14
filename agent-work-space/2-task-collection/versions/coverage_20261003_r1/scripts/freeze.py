"""Seal a new source version and verify drift; never overwrite a seal."""
import argparse,hashlib,json,pathlib,tarfile
from run import VERSION,COLLECTION,REPO
SEAL=COLLECTION/'releases'/VERSION.name

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def source_files():
    return sorted(p for p in VERSION.rglob('*') if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('._'))

def dependencies():
    files=[]
    for root in (REPO/'robosuite',REPO/'third_party/libero/libero/libero'):
        files.extend(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.py','.xml','.json','.stl','.obj','.msh','.png','.jpg','.bddl'))
    return sorted(files)

def verify():
    manifest=json.loads((SEAL/'manifest.json').read_text());errors=[]
    for rel,want in manifest['files'].items():
        p=REPO/rel
        if not p.is_file() or sha(p)!=want:errors.append(rel)
    if sha(SEAL/'source.tar.gz')!=manifest['archive_sha256']:errors.append('source archive')
    result={'pass':not errors,'checked_files':len(manifest['files']),'errors':errors,'scope':'source identity; not task acceptance'}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if errors:raise RuntimeError('Frozen source drift; create a reviewed new version')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['create','verify']);a=ap.parse_args()
    if a.mode=='create':
        SEAL.mkdir(parents=True,exist_ok=False)
        files=source_files();hashes={str(p.relative_to(REPO)):sha(p) for p in files+dependencies()}
        with tarfile.open(SEAL/'source.tar.gz','x:gz') as t:
            for p in files:t.add(p,arcname=str(p.relative_to(COLLECTION)))
        with (SEAL/'manifest.json').open('x') as f:json.dump({'version':VERSION.name,'files':hashes,'archive_sha256':sha(SEAL/'source.tar.gz'),'archive_scope':'new collection version including exact upstream teacher copies; repository dependencies fingerprinted separately'},f,indent=2,ensure_ascii=False)
    verify()
if __name__=='__main__':main()
