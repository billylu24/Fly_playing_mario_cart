"""CPU-only post-training diagnostic: frozen teacher vs hard demonstration labels.
Does not select models or modify training/evaluation protocols.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/recovery_training_v1'

@torch.no_grad()
def run():
    torch.set_num_threads(2)
    path=ROOT/'runs/distillation_replication_v1/seed0/joint_best.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    encoder=nn.Sequential(nn.Conv2d(1,16,8,4),nn.ReLU(),nn.Conv2d(16,32,4,2),nn.ReLU(),nn.Flatten(),nn.Linear(2592,64),nn.Tanh())
    encoder.load_state_dict({k.removeprefix('b.encoder.'):v for k,v in state.items() if k.startswith('b.encoder.')})
    head=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,3))
    head.load_state_dict({k.removeprefix('head.net.'):v for k,v in state.items() if k.startswith('head.net.')})
    original=ROOT/'runs/supervised_architecture_v1';recovery=ROOT/'runs/recovery_demonstrations_v1'
    groups={'original_train':[original/'data'/r['filename'] for r in json.loads((original/'manifest.json').read_text()) if r['split']=='train' and r['accepted']],
            'recovery_train':[recovery/'data'/r['filename'] for r in json.loads((recovery/'training_manifest.json').read_text())]}
    results={}
    for name,paths in groups.items():
        logits=[];labels=[]
        for p in paths:
            with np.load(p) as d:
                mask=d['actions']>=0;x=torch.tensor(d['observations'][mask]);labels.extend(d['actions'][mask].tolist())
            for b in x.split(64):
                z=encoder(b[:,None].float()/255);logits.append(head((z-state['head.mean'])/state['head.scale']))
        l=torch.cat(logits);y=torch.tensor(labels);pred=l.argmax(-1);cm=torch.zeros(3,3,dtype=torch.int64)
        for a,b in zip(y,pred):cm[a,b]+=1
        results[name]=dict(decisions=len(y),teacher_agreement=float((pred==y).float().mean()),
            teacher_hard_label_ce=float(F.cross_entropy(l,y)),confusion_true_rows_pred_columns=cm.tolist())
    result=dict(source=str(path),groups=results,interpretation='Post-hoc frozen-teacher target consistency, training-only. At exact CNN reconstruction, frozen head makes these errors on hard demonstration labels. This reveals objective mismatch, not proof of infeasibility: decoder can alter latent values. No model changes or new-test selection.')
    (OUT/'teacher_target_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
