"""Read-only exact frame leakage audit for the head-only experiment."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/head_adaptation_v1'

def hashes(paths):
    h=set()
    for p in paths:
        with np.load(p) as d:h.update(hashlib.sha256(x.tobytes()).hexdigest() for x in d['observations'])
    return h

def run():
    original=ROOT/'runs/supervised_architecture_v1';recovery=ROOT/'runs/recovery_demonstrations_v1'
    rows=json.loads((original/'manifest.json').read_text())
    old_train=hashes([original/'data'/r['filename'] for r in rows if r['split']=='train' and r['accepted']])
    rec_train=hashes([recovery/'data'/r['filename'] for r in json.loads((recovery/'training_manifest.json').read_text())])
    old_val=hashes([original/'data'/r['filename'] for r in rows if r['split']=='validation' and r['accepted']])
    rec_val=hashes([OUT/'validation_data'/r['filename'] for r in json.loads((OUT/'validation_manifest.json').read_text())])
    test=hashes([OUT/'new_test'/r['filename'] for r in json.loads((OUT/'new_test_manifest.json').read_text()) if r['accepted']])
    report=dict(exact_frame_overlaps=dict(recovery_validation_vs_training=len(rec_val&(old_train|rec_train)),test_vs_training=len(test&(old_train|rec_train)),test_vs_validation=len(test&(old_val|rec_val))),
        distinct_frames=dict(old_train=len(old_train),recovery_train=len(rec_train),old_validation=len(old_val),recovery_validation=len(rec_val),new_test=len(test)),
        caveat='Exact pixel comparison only. Same track and adjacent starts remain correlated.')
    (OUT/'overlap_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':run()
