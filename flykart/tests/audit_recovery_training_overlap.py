"""Audit exact new-test frame overlap with actual training observations."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/recovery_training_v1'

def hashes(paths):
    result=set()
    for p in paths:
        with np.load(p) as d:result.update(hashlib.sha256(x.tobytes()).hexdigest() for x in d['observations'])
    return result

def run():
    original=ROOT/'runs/supervised_architecture_v1'
    manifest=json.loads((original/'manifest.json').read_text())
    groups={split:hashes([original/'data'/r['filename'] for r in manifest if r['accepted'] and r['split']==split]) for split in ['train','validation','test']}
    recovery=ROOT/'runs/recovery_demonstrations_v1'
    groups['recovery_training']=hashes([recovery/'data'/r['filename'] for r in json.loads((recovery/'training_manifest.json').read_text())])
    new=hashes([OUT/'new_test'/r['filename'] for r in json.loads((OUT/'new_test_manifest.json').read_text()) if r['accepted']])
    report=dict(new_test_distinct_frames=len(new),overlap={k:len(v&new) for k,v in groups.items()},
        caveat='Pixel equality only, not scene or track independence. No action-label inspection and no outcome-based test filtering.')
    (OUT/'overlap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
