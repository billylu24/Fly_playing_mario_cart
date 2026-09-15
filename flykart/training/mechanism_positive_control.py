"""Evaluate the prespecified archived CNN bypass on new mechanism-test starts."""
import json
import numpy as np
import torch
from . import supervised as base
from . import layer_probes as lp
from .mechanism_audit import OUT,save

def run():
    torch.set_num_threads(4);assert (OUT/'status.json').exists()
    m,path=lp.backbone('k2');f=lp.Features(m);probe=lp.Probe(torch.zeros(2,64,device='cuda'),'mlp32');file=lp.OUT/'k2_cnn_mlp32.pt';original=base.digest(file);probe.load_state_dict(torch.load(file,weights_only=True));policy=lp.Policy(f,'cnn',probe).eval()
    time=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');time.load_state_dict(torch.load(lp.OUT/'time_probe.pt',weights_only=True))
    with torch.no_grad():clock=list(time(torch.arange(225,device='cuda')[:,None]/225).softmax(-1).cpu())
    manifest=json.loads((OUT/'new_test_manifest.json').read_text());rows=[]
    for r in manifest:
        if r['accepted']:
            with np.load(OUT/'new_test'/r['filename']) as d:rows.append(dict(meta=r,obs=torch.tensor(d['observations'],device='cuda'),labels=torch.tensor(d['actions'],device='cuda')))
    report=dict(candidate='archived CNN bypass',offline={mode:base.offline(policy,rows,mode) for mode in ['normal','frozen']},episodes=[]);pool=base.EvaluationPool()
    try:
        for at,r in enumerate(rows):
            es=pool.rollout(policy,r['meta']['config'],clock)
            if at==0:
                ref=base.closed_loop(policy,r['meta']['config'],'normal',8100,clock);actual=next(e for e in es if e['mode']=='normal' and e['sample_seed']==8100)
                for key in ['actions','frames','reason','progress','jumps']:assert ref[key]==actual[key],key
                report['batched_sequential_match']=True
            report['episodes'].extend(es);save('cnn_new_evaluation.json',report);print('CNN_NEW',at+1,flush=True)
    finally:pool.close()
    assert base.digest(file)==original;report['checkpoint_unchanged']=True;save('cnn_new_evaluation.json',report);f.hook.remove()
if __name__=='__main__':run()
