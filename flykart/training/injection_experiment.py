"""Group-specific trainable visual projection, initialized to the existing mapping."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import supervised as base
from .model import GainPolicy,FixedMultiply
from .task import ROOT
ORIGINAL=ROOT/'runs/supervised_architecture_v1'
OUTPUT=ROOT/'runs/visual_injection_v1'

class InjectionPolicy(GainPolicy):
    def __init__(self):
        super().__init__(actions=3)
        with np.load(OUTPUT/'groups.npz') as d:
            groups=torch.as_tensor(d['groups'],device='cuda',dtype=torch.long);self.group_names=d['names'].tolist()
        self.register_buffer('ports',groups*64+self.channels)
        self.register_buffer('identity',torch.eye(64,device='cuda')[None])
        self.projection_delta=nn.Parameter(torch.zeros(len(self.group_names),64,64,device='cuda'))
        self.critic.requires_grad_(False)
    def forward(self,obs,h,starts):
        h=h*(1-starts[None,:]);z=self.encoder(obs[:,None].float()/255)
        mapped=torch.einsum('gij,bj->gib',self.identity+self.projection_delta,z)
        signal=mapped.flatten(0,1).index_select(0,self.ports)*self.visual[:,None]
        h=.5*h+.5*torch.tanh((.5+self.a.sigmoid())[:,None]*FixedMultiply.apply(h,self.w,self.wt)+signal)
        out=h.index_select(0,self.readout).T
        return self.actor(out),self.critic(out).squeeze(-1),h

def make_model(arch,seed):
    assert arch=='graph';torch.manual_seed(seed);np.random.seed(seed);m=InjectionPolicy()
    return m,base.tensor_hash(m.encoder.state_dict())

def prepare():
    import pandas as pd
    OUTPUT.mkdir(exist_ok=False)
    with np.load(ROOT/'data/malecns/traced_smoke_graph.npz') as d:ids=d['body_ids'];visual=d['visual_mask'].astype(bool)
    annotations=ROOT/'data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather'
    a=pd.read_feather(annotations).set_index('bodyId').loc[ids]
    keys=(a.superclass.fillna('unknown').astype(str)+'|'+a.somaSide.fillna('unknown').astype(str)).to_numpy()
    names=sorted(set(keys[visual]));lookup={n:i for i,n in enumerate(names)}
    groups=np.array([lookup[k] if v else 0 for k,v in zip(keys,visual)],dtype=np.int64)
    np.savez(OUTPUT/'groups.npz',groups=groups,names=np.array(names))
    (OUTPUT/'data').symlink_to(ORIGINAL/'data',target_is_directory=True)
    for file in ['manifest.json','data_summary.json']:shutil.copyfile(ORIGINAL/file,OUTPUT/file)
    plan=json.loads((ORIGINAL/'plan.json').read_text())
    plan.update(experiment='group-specific identity-initialized visual projection; original tensors paired',groups=len(names),added_parameters=len(names)*4096,
        grouping='visual superclass x somaSide; original channel slot within each group',baseline=str(ORIGINAL),
        reuse_test_warning='same diagnostic test as previous experiment, not a new independent test',
        new_source_sha256=base.digest(Path(__file__)),training_source_sha256=base.digest(Path(base.__file__)),
        annotations_sha256=base.digest(annotations),groups_sha256=base.digest(OUTPUT/'groups.npz'))
    (OUTPUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ['groups','added_parameters','grouping']},indent=2))

def checks():
    torch.manual_seed(0);old=GainPolicy(actions=3);torch.manual_seed(0);new=InjectionPolicy()
    for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
    torch.manual_seed(10);obs=torch.randint(0,256,(10,3,84,84),device='cuda',dtype=torch.uint8)
    h1=torch.zeros(old.n,3,device='cuda');h2=h1.clone();max_error=0.
    with torch.no_grad():
        for t,o in enumerate(obs):
            starts=torch.tensor([float(t==0),float(t in [0,5]),float(t==0)],device='cuda')
            l1,_,h1=old(o,h1,starts);l2,_,h2=new(o,h2,starts);max_error=max(max_error,float((l1-l2).abs().max()))
            torch.testing.assert_close(h1,h2,atol=2e-6,rtol=1e-5);torch.testing.assert_close(l1,l2,atol=2e-6,rtol=1e-5)
    h=torch.zeros(new.n,3,device='cuda');loss=0
    for t,o in enumerate(obs):
        l,_,h=new(o,h,torch.full((3,),float(t==0),device='cuda'));loss+=torch.nn.functional.cross_entropy(l,torch.tensor([0,1,2],device='cuda'))
    loss.backward();norm=float(new.projection_delta.grad.norm());assert np.isfinite(norm) and norm>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.encoder.parameters())
    result=dict(shared_initial_tensors_identical=True,initial_forward_max_error=max_error,projection_gradient_norm=norm)
    (OUTPUT/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

def finish(seed):
    run=OUTPUT/f'graph_seed{seed}';original=ORIGINAL/f'graph_seed{seed}'
    cfg=json.loads((run/'config.json').read_text());ref=json.loads((original/'config.json').read_text())
    assert cfg['encoder_initial_sha256']==ref['encoder_initial_sha256'];assert cfg['manifest_sha256']==ref['manifest_sha256']
    actual=json.loads((run/'status.json').read_text());prior=json.loads((original/'status.json').read_text());assert actual['valid_presentations']==prior['valid_presentations']
    checkpoint=torch.load(run/'best.pt',weights_only=False);delta=checkpoint['model']['projection_delta']
    (run/'injection_summary.json').write_text(json.dumps(dict(encoder_initialization_matched=True,data_and_presentations_matched=True,
        projection_delta_rms=float(delta.square().mean().sqrt()),projection_delta_max=float(delta.abs().max())),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','check','train','evaluate','all']);p.add_argument('--seed',type=int,default=0);a=p.parse_args();torch.set_num_threads(4)
    if a.phase=='prepare':prepare();return
    base.EXPERIMENT=OUTPUT;base.make_model=make_model
    if a.phase=='check':checks()
    elif a.phase=='train':base.train('graph',a.seed);finish(a.seed)
    elif a.phase=='evaluate':base.evaluate('graph',a.seed)
    else:
        checks()
        for seed in [0,1,2]:base.train('graph',seed);finish(seed);base.evaluate('graph',seed)
if __name__=='__main__':main()
