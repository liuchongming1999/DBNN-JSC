"""Download author-published NeuronIO test data and a paper TCN checkpoint.

Individual public Kaggle file URLs need no account credentials. Keep files on
/data rather than filling the system disk. Validate sizes and retain hashes.
"""
import argparse, hashlib, json, time, urllib.parse, urllib.request, zipfile
from pathlib import Path

SLUG = 'selfishgene/single-neurons-as-deep-nets-nmda-test-data'

def save(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(p)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def listing(out):
    cache=out/'listing.json'
    if cache.exists():return json.loads(cache.read_text())['files']
    items=[];token=None
    while True:
        url=f'https://www.kaggle.com/api/v1/datasets/list/{SLUG}'
        if token:url+='?'+urllib.parse.urlencode({'pageToken':token})
        with urllib.request.urlopen(url,timeout=60) as f:d=json.load(f)
        items.extend(d['datasetFiles']);token=d.get('nextPageToken')
        if not token:break
    save(cache,dict(files=items));return items

def download(row,out):
    name=row['name'];dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
    n=int(row['totalBytes']);receipt=dest.with_suffix(dest.suffix+'.download.json')
    if dest.exists() and dest.stat().st_size==n:
        if not receipt.exists():save(receipt,dict(name=name,bytes=n,sha256=digest(dest)))
        return
    url=f'https://www.kaggle.com/api/v1/datasets/download/{SLUG}/'+urllib.parse.quote(name,safe='')
    tmp=dest.with_suffix(dest.suffix+'.part')
    for attempt in range(5):
        try:
            print('DOWNLOAD',name,flush=True)
            with urllib.request.urlopen(url,timeout=120) as response,tmp.open('wb') as f:
                count=0;last=time.monotonic()
                while True:
                    b=response.read(4<<20)
                    if not b:break
                    f.write(b);count+=len(b)
                    if time.monotonic()-last>20:
                        save(out/'progress.json',dict(file=name,downloaded=count,expected_uncompressed=n,attempt=attempt))
                        print('BYTES',name,count,flush=True);last=time.monotonic()
            if zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as z:
                    candidates=[i for i in z.infolist() if not i.is_dir() and Path(i.filename).name==dest.name]
                    if len(candidates)!=1:raise ValueError(('Unexpected archive',z.namelist()))
                    with z.open(candidates[0]) as src,dest.open('wb') as dst:
                        while b:=src.read(4<<20):dst.write(b)
                tmp.unlink()
            else:tmp.replace(dest)
            if dest.stat().st_size!=n:raise ValueError(('Size mismatch',name,dest.stat().st_size,n))
            save(receipt,dict(name=name,bytes=n,sha256=digest(dest),url=url,completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            print('READY',name,n,flush=True);return
        except Exception as e:
            save(out/'download_error.json',dict(file=name,attempt=attempt,error=repr(e)))
            print('RETRY',name,repr(e),flush=True)
            if attempt==4:raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--test-limit',type=int,default=0)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);rows=listing(out)
    checkpoints=['Models/NMDA_TCN__DWT_7_128_153__model.h5','Models/NMDA_TCN__DWT_7_128_153__training.pickle']
    tests=sorted([r for r in rows if r['name'].startswith('Data_test/')],key=lambda r:r['name'])
    if a.test_limit:tests=tests[:a.test_limit]
    selected=[r for n in checkpoints for r in rows if r['name']==n]+tests
    assert len(selected)==len(checkpoints)+len(tests)
    for i,r in enumerate(selected,1):
        download(r,out);save(out/'status.json',dict(completed=i,total=len(selected),last=r['name']))
    save(out/'complete.json',dict(finished=True,test_files=len(tests),full_public_test=not a.test_limit,files=[r['name'] for r in selected]))

if __name__=='__main__':main()
