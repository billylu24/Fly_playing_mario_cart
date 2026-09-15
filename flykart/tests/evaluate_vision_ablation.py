"""Closed-loop input/gain interventions; same seeds and bounded-context policy."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from training.task import Task
from training.model import GainPolicy
from training.train_gain import evaluate


def rollout(model,cfg,memory,seed,mode):
    task=Task(**cfg)
    try:
        obs=task.reset();initial=obs.copy()
        h=torch.zeros(model.n,1,device='cuda');starts=torch.ones(1,device='cuda')
        history=[];step=0;probabilities=[];features=[]
        rng=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            while True:
                frame=initial if mode=='frozen_image' else np.zeros_like(obs) if mode=='black_image' else obs
                o=torch.as_tensor(frame[None],device='cuda')
                if step%32==0:
                    h=torch.zeros_like(h)
                    for old_obs,old_start in history[-memory:]:_,_,h=model(old_obs,h,old_start)
                history.append((o,starts.clone()));history=history[-memory:]
                logits,_,h=model(o,h,starts);p=logits.softmax(-1)[0].cpu()
                probabilities.append(p.numpy())
                features.append(model.encoder(o[:,None].float()/255)[0].cpu().numpy())
                action=int(torch.multinomial(p,1,generator=rng))
                m=task.step(action);obs=m['frame'];starts.fill_(0);step+=1
                if m['done']:
                    return {'seed':seed,**m['episode'],'decisions':step,
                            'action_probability_temporal_std':float(np.std(probabilities,axis=0).mean()),
                            'encoder_temporal_std':float(np.std(features,axis=0).mean())}
    finally:task.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--episodes',type=int,default=10);p.add_argument('--seed-base',type=int,default=3000)
    p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0)
    report={'seed_base':a.seed_base,'episodes_per_mode':a.episodes,'runs':{}}
    model=None
    for name in a.runs:
        run=Path(name);cfg=json.loads((run/'config.json').read_text())
        if model is None:model=GainPolicy(actions=3 if cfg['action_set']=='drive' else 7).eval()
        checkpoint=torch.load(run/'last.pt',weights_only=False);model.load_state_dict(checkpoint['model'])
        original=model.a.detach().clone();result={}
        for mode in ['normal','frozen_image','black_image','gain_reset']:
            with torch.no_grad():model.a.copy_(original if mode!='gain_reset' else torch.zeros_like(original))
            episodes=[rollout(model,cfg['task_config'],cfg['memory'],seed,mode)
                      for seed in range(a.seed_base,a.seed_base+a.episodes)]
            result[mode]={'episodes':episodes,'target_rate':float(np.mean([e['target_reached'] for e in episodes])),
                          'mean_progress':float(np.mean([e['progress'] for e in episodes])),
                          'mean_frames':float(np.mean([e['frames'] for e in episodes]))}
            print(json.dumps({'run':name,'mode':mode,**{k:v for k,v in result[mode].items() if k!='episodes'}}),flush=True)
            if mode=='normal':
                reference=evaluate(model,seed_base=a.seed_base,episodes=1,task_config=cfg['task_config'],memory=cfg['memory'])[0]
                for key in ['reason','frames','progress','reward','jumps']:
                    assert reference[key]==episodes[0][key],('normal evaluation mismatch',key)
        with torch.no_grad():model.a.copy_(original)
        for mode in ['frozen_image','black_image','gain_reset']:
            paired=list(zip(result['normal']['episodes'],result[mode]['episodes']))
            result[mode]['paired_changed_success_count']=sum(x['target_reached']!=y['target_reached'] for x,y in paired)
        report['runs'][name]={'config':cfg,'results':result}
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
