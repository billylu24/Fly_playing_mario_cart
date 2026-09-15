"""Exact image overlap audit only; no use of held-out actions for training."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/recovery_demonstrations_v1'

def run():
    assert json.loads((OUT/'status.json').read_text())['state']=='completed'
    manifest=json.loads((OUT/'selected_manifest.json').read_text())
    supervision=set();histories=set()
    for r in manifest:
        with np.load(OUT/'data'/r['filename']) as d:
            for frame,label in zip(d['observations'],d['actions']):
                h=hashlib.sha256(frame.tobytes()).hexdigest()
                (supervision if label>=0 else histories).add(h)
    original=ROOT/'runs/supervised_architecture_v1'
    rows=json.loads((original/'manifest.json').read_text())
    groups={s:[original/'data'/r['filename'] for r in rows if r['split']==s and r['accepted']] for s in ['validation','test']}
    for p in sorted((ROOT/'runs').glob('*/new_test')):
        groups[p.parent.name]=sorted(p.glob('*.npz'))
    report={}
    for name,files in groups.items():
        hashes=set()
        for p in files:
            with np.load(p) as d:
                hashes.update(hashlib.sha256(x.tobytes()).hexdigest() for x in d['observations'])
        report[name]=dict(files=len(files),distinct_frames=len(hashes),supervised_overlap=len(hashes&supervision),prefix_overlap=len(hashes&histories))
    result=dict(supervised_distinct_frames=len(supervision),prefix_distinct_frames=len(histories),comparisons=report,
        interpretation='Exact pixel equality only. Zero overlap does not imply independent scenes or tracks. Audit does not inspect held-out action labels.')
    (OUT/'overlap_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
