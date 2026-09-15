"""Existing-model sensitivity, legal reset gates, and closed-loop feedback controls."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.bend_task import BendTask
from training.model import GainPolicy
from training.task import ROOT
POINTS=np.array([(165,70),(100,42),(64,67),(40,113),(40,300)],float)*8

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def config(f,t):
    ident=f'f{f}_{t}_p36'
    return dict(start_id=ident,perturb_frames=36,start_specs=[dict(id=ident,prefix_frame=f,turn=t)])
def oracle(task,w):
    info=task.env.data.lookup_all();pos=np.array([info['kart1_X'],info['kart1_Y']],float)
    if np.linalg.norm(POINTS[w]-pos)<220:w=min(w+1,len(POINTS)-1)
    d=POINTS[w]-pos;e=(math.atan2(d[0],-d[1])-info['kart1_direction']*2*math.pi/256+math.pi)%(2*math.pi)-math.pi
    return (0 if abs(e)<=.12 else 2 if e>0 else 1),w

def control(cfg,mode,sequence=None):
    task=BendTask(**cfg)
    try:
        first=task.reset();info={k:task.start_info[k] for k in ['kart1_X','kart1_Y','kart1_direction','kart1_speed','surface','current_checkpoint']}
        np.testing.assert_array_equal(first,task.reset());actions=[];w=0
        while True:
            if mode=='oracle':action,w=oracle(task,w)
            elif mode=='sequence':action=sequence[min(len(actions),len(sequence)-1)]
            else:action={'straight':0,'left':1,'right':2}[mode]
            actions.append(action);m=task.step(action)
            if m['done']:
                np.testing.assert_array_equal(first,m['frame'])
                return dict(policy=mode,config=cfg,start_info=info,reset_verified=True,actions=actions,**m['episode'])
    finally:task.close()

class State:
    """Match training: zero-rooted 64-frame history, refreshed every 32 decisions."""
    def __init__(self,model,batch):
        self.model=model;self.h=torch.zeros(model.n,batch,device='cuda');self.history=[];self.step=0
    def forward(self,frames,features=False):
        o=torch.as_tensor(np.stack(frames),device='cuda');s=torch.full((len(frames),),float(self.step==0),device='cuda')
        if self.step%32==0:
            self.h.zero_()
            for old,mask in self.history:_,_,self.h=self.model(old,self.h,mask)
        self.history.append((o,s));self.history=self.history[-64:]
        logits,_,self.h=self.model(o,self.h,s);self.step+=1;p=logits.softmax(-1)
        if features:return p,self.model.encoder(o[:,None].float()/255),self.h[self.model.readout].T
        return p

def sensitivity(model,cfg):
    task=BendTask(**cfg);zs=[];outs=[];ps=[];pixels=[]
    try:
        obs=task.reset();initial=obs.copy();state=State(model,3);w=0
        with torch.no_grad():
            while True:
                p,z,out=state.forward([obs,initial,np.zeros_like(obs)],True)
                zs.append(z.cpu().numpy());outs.append(out.cpu().numpy());ps.append(p.cpu().numpy())
                pixels.append(float(np.abs(obs.astype(float)-initial).mean()/255))
                action,w=oracle(task,w);m=task.step(action);obs=m['frame']
                if m['done']:break
        cut=min(8,len(zs)-1);z=np.stack(zs)[cut:];out=np.stack(outs)[cut:];p=np.stack(ps)[cut:]
        r=dict(start_id=cfg['start_id'],steps=len(zs),warmup_excluded=cut,pixel_frozen_mae=float(np.mean(pixels[cut:])),encoder_temporal_std=float(z[:,0].std(0).mean()),readout_temporal_std=float(out[:,0].std(0).mean()),encoder_saturated_fraction=float((abs(z[:,0])>.99).mean()))
        for name,a in [('encoder',z),('readout',out)]:
            r[name+'_normal_rms']=float(np.sqrt(np.mean(a[:,0]**2)))
            for i,mode in [(1,'frozen'),(2,'black')]:
                r[name+'_'+mode+'_rms_diff']=float(np.sqrt(np.mean((a[:,0]-a[:,i])**2)))
                r[name+'_'+mode+'_relative_rms']=r[name+'_'+mode+'_rms_diff']/(r[name+'_normal_rms']+1e-12)
        for i,mode in [(1,'frozen'),(2,'black')]:
            tv=.5*np.abs(p[:,0]-p[:,i]).sum(-1);r[mode+'_action_tv_mean']=float(tv.mean());r[mode+'_action_tv_max']=float(tv.max())
        return r
    finally:task.close()

def clock_probs(model):
    task=BendTask(start_id='f540_center',perturb_frames=24)
    try:
        initial=task.reset();state=State(model,1)
        with torch.no_grad():return [state.forward([initial])[0].cpu() for _ in range(225)]
    finally:task.close()

def rollout(model,cfg,mode,seed,clock):
    task=BendTask(**cfg);rng=torch.Generator().manual_seed(seed)
    try:
        obs=task.reset();initial=obs.copy();frozen=None;state=State(model,1);actions=[]
        with torch.no_grad():
            while True:
                step=len(actions)
                if step==16:frozen=obs.copy()
                frame=initial if mode=='frozen' else frozen if mode=='midfreeze' and step>=16 else obs
                p=clock[step] if mode=='clock' else state.forward([frame])[0].cpu()
                action=int(torch.multinomial(p,1,generator=rng));actions.append(action)
                m=task.step(action);obs=m['frame']
                if m['done']:return dict(policy=mode,seed=seed,actions=actions,freeze_step=16 if mode=='midfreeze' else None,**m['episode'])
    finally:task.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['gates','models'],required=True);a=parser.parse_args()
    out=ROOT/'tests/bend_feedback_v1';out.mkdir(exist_ok=True);run=ROOT/'runs/bend_gain_32k_v1'
    if a.phase=='gates':
        candidates=[config(f,t) for f in (480,510,630,660) for t in ('center','left','right')]
        report=dict(candidates=candidates,controls=[],errors=[],selection_rule='Per prefix retain oracle-passing left/right; center only if neither passes. Max 8 in grid order.')
        ref=control(dict(start_id='f540_center',perturb_frames=24),'oracle');report['reference_sequence']=ref
        for cfg in candidates:
            try:
                for mode in ['oracle','straight','left','right','sequence']:report['controls'].append(control(cfg,mode,ref['actions']))
                print('GATE',cfg['start_id'],[(x['policy'],x['target_reached']) for x in report['controls'][-5:]],flush=True)
            except (RuntimeError,AssertionError) as e:report['errors'].append(dict(config=cfg,error=repr(e)))
            (out/'gates.json').write_text(json.dumps(report,indent=2)+'\n')
        selected=[]
        for f in (480,510,630,660):
            valid=[r['config'] for r in report['controls'] if r['policy']=='oracle' and r['target_reached'] and r['jumps']==0 and r['config']['start_specs'][0]['prefix_frame']==f and not any(e['config']==r['config'] for e in report['errors'])]
            selected.extend([c for c in valid if c['start_specs'][0]['turn']!='center'] or valid[:1])
        report['selected']=selected[:8];report['prefix_sha256']=sha(ROOT/'tests/track_validation/race_trace.json')
        (out/'gates.json').write_text(json.dumps(report,indent=2)+'\n');return
    torch.set_num_threads(4);torch.manual_seed(0);gates=json.loads((out/'gates.json').read_text());assert gates['selected']
    model=GainPolicy(actions=3).eval();report=dict(diagnostics=[],episodes=[],checkpoints={},evaluation_configs=gates['selected'],seed_base=6100,seeds=5,freeze_step=16)
    original=[dict(start_id=f'f570_{t}',perturb_frames=24) for t in ('center','left','right')]
    for file in ['initial.pt','best_validation.pt','last.pt']:
        path=run/file;report['checkpoints'][file]=sha(path);model.load_state_dict(torch.load(path,weights_only=False)['model'])
        for cfg in original+gates['selected']:report['diagnostics'].append(dict(checkpoint=file,**sensitivity(model,cfg)))
        (out/'models.json').write_text(json.dumps(report,indent=2)+'\n');print('SENSITIVITY',file,'completed',flush=True)
    model.load_state_dict(torch.load(run/'best_validation.pt',weights_only=False)['model']);clock=clock_probs(model);report['clock_probabilities']=[p.tolist() for p in clock]
    from training.train_gain import evaluate
    actual=rollout(model,original[0],'normal',6100,clock);expected=evaluate(model,seed_base=6100,episodes=1,task_config=dict(kind='bend',**original[0]))[0]
    for key in ['reason','frames','progress','reward','jumps']:assert actual[key]==expected[key],key
    report['normal_evaluator_regression_passed']=True
    for cfg in gates['selected']:
        for seed in range(6100,6105):
            paired={mode:rollout(model,cfg,mode,seed,clock) for mode in ['normal','frozen','midfreeze','clock']}
            assert paired['normal']['actions'][:16]==paired['midfreeze']['actions'][:16]
            report['episodes'].extend(paired.values())
        (out/'models.json').write_text(json.dumps(report,indent=2)+'\n');print('EVALUATED',cfg['start_id'],flush=True)
    assert all(sha(run/file)==digest for file,digest in report['checkpoints'].items());report['checkpoint_hashes_unchanged']=True
    (out/'models.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
