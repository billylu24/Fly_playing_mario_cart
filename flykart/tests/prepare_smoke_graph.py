"""Prepare an explicit Traced-only real graph for a resource smoke test."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

root=Path(__file__).resolve().parents[1]
data=root/'data'/'malecns'
start=time.perf_counter()
a=pd.read_feather(data/'body-annotations-male-cns-v1.0-minconf-0.5.feather')
a=a[a.status.eq('Traced')].sort_values('bodyId')
ids=a.bodyId.to_numpy();n=len(ids)
visual=a.superclass.fillna('').str.startswith(('ol_','visual_')).to_numpy()
output=a.superclass.fillna('').isin(['descending_neuron','vnc_motor','cb_motor']).to_numpy()
assert visual.any() and output.any()
rows=[];cols=[];weights=[];count=0
with pa.memory_map(str(data/'connectome-weights-male-cns-v1.0-minconf-0.5.feather'),'r') as source:
    reader=ipc.open_file(source)
    assert reader.schema.names==['body_pre','body_post','weight']
    for i in range(reader.num_record_batches):
        b=reader.get_batch(i)
        pre=b.column(0).to_numpy();post=b.column(1).to_numpy();w=b.column(2).to_numpy()
        src=np.searchsorted(ids,pre);dst=np.searchsorted(ids,post)
        keep=(src<n)&(dst<n)
        src=np.minimum(src,n-1);dst=np.minimum(dst,n-1)
        keep &= (ids[src]==pre)&(ids[dst]==post)
        rows.append(dst[keep].astype(np.int32));cols.append(src[keep].astype(np.int32))
        weights.append(w[keep].astype(np.float32));count+=len(pre)
        if i%200==0:print(f'batches {i}/{reader.num_record_batches}, scanned {count}',flush=True)
rows=np.concatenate(rows);cols=np.concatenate(cols);weights=np.concatenate(weights)
order=np.argsort(rows.astype(np.int64)*n+cols,kind='stable')
rows=rows[order];cols=cols[order];weights=weights[order];del order
keys=rows.astype(np.int64)*n+cols
assert np.all(np.diff(keys)>0),'duplicate graph edges require explicit aggregation'
del keys
incoming=np.bincount(rows,weights=weights,minlength=n)
raw_sum=float(weights.sum(dtype=np.float64))
weights*=np.float32(0.8)/incoming[rows].astype(np.float32)
path=data/'traced_smoke_graph.npz'
np.savez(path,body_ids=ids,rows=rows,cols=cols,values=weights,
         visual_mask=visual,readout_ids=np.flatnonzero(output))
report={'selection':'status == Traced; retain edges with both endpoints in selected IDs',
        'nodes':n,'edges':len(rows),'raw_weight_sum':raw_sum,
        'visual_injection_nodes':int(visual.sum()),'readout_nodes':int(output.sum()),
        'isolated_nodes':int((np.bincount(rows,minlength=n)+np.bincount(cols,minlength=n)==0).sum()),
        'initialization':'positive count weights normalized per incoming row to sum 0.8; no NT sign assignment',
        'input_mapping':'64 CNN features cyclically assigned to visual/ol nodes; smoke-only mapping',
        'readout_superclasses':['descending_neuron','vnc_motor','cb_motor'],
        'elapsed_seconds':time.perf_counter()-start,'artifact_bytes':path.stat().st_size}
(root/'tests'/'real_smoke_graph.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
