"""Matched behavior-cloning experiment: fixed connectome versus CNN+GRU.
No RAM or expert labels enter either policy at inference time.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .bend_task import BendTask
from .model import GainPolicy
from .task import ROOT

EXPERIMENT=ROOT/'runs/supervised_architecture_v1'
POINTS=np.array([(165,70),(100,42),(64,67),(40,113),(40,300)],float)*8
MODES=('normal','frozen','clock')


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h=hashlib.sha256()
    for key,x in sorted(state.items()):h.update(key.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_config(frame,turn,perturb):
    ident=f'f{frame}_{turn}_p{perturb}'
    return dict(start_id=ident,perturb_frames=perturb,
                start_specs=[dict(id=ident,prefix_frame=frame,turn=turn)])


def oracle(task,waypoint):
    info=task.env.data.lookup_all();pos=np.array([info['kart1_X'],info['kart1_Y']],float)
    if np.linalg.norm(POINTS[waypoint]-pos)<220:waypoint=min(waypoint+1,len(POINTS)-1)
    d=POINTS[waypoint]-pos
    error=(math.atan2(d[0],-d[1])-info['kart1_direction']*2*math.pi/256+math.pi)%(2*math.pi)-math.pi
    return (0 if abs(error)<=.12 else 2 if error>0 else 1),waypoint


def collect():
    EXPERIMENT.mkdir(exist_ok=False)
    data_dir=EXPERIMENT/'data';data_dir.mkdir()
    plan=dict(train_frames=[450,480,540,600],validation_frames=[510,570],test_frames=[495,555,615],
              train_perturb=[12,24,36,48],validation_perturb=[24],test_perturb=[18,30],
              seeds=[0,1,2],updates=256,batch=8,bptt=32,memory=64,learning_rate=.0003,
              optimizer='Adam eps1e-5, clip norm1.0',loss='training-frequency inverse weighted cross entropy',
              select='maximum validation normal balanced accuracy, then minimum normal cross entropy; every64 updates',
              evaluate='selected checkpoint only; test untouched during training; greedy + seeds8100,8101,8102; normal/frozen/clock',
              selection='retain successful, jumps0, repeatable oracle trajectories, independent of model outcomes',
              notes='one track, disjoint prefix frames; trajectories may converge; same data/order/CNN initialization per paired seed')
    plan['source_hashes']={p.name:digest(p) for p in [Path(__file__),ROOT/'training/model.py',ROOT/'training/bend_task.py',ROOT/'training/task.py']}
    plan['graph_sha256']=digest(ROOT/'data/malecns/traced_smoke_graph.npz')
    plan['prefix_sha256']=digest(ROOT/'tests/track_validation/race_trace.json')
    plan['rom_sha256']=digest(ROOT/'mario_kart/SuperMarioKart-Snes/rom.sfc')
    (EXPERIMENT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest=[];begin=time.perf_counter()
    for split in ('train','validation','test'):
        for f in plan[split+'_frames']:
            for perturb in plan[split+'_perturb']:
                for turn in ('center','left','right'):
                    cfg=make_config(f,turn,perturb);task=BendTask(**cfg)
                    try:
                        obs=task.reset();initial=obs.copy();info={k:task.start_info[k] for k in ['kart1_X','kart1_Y','kart1_direction','kart1_speed','surface','current_checkpoint']}
                        np.testing.assert_array_equal(initial,task.reset());observations=[];labels=[];waypoint=0
                        while True:
                            action,waypoint=oracle(task,waypoint);observations.append(obs);labels.append(action)
                            m=task.step(action);obs=m['frame']
                            if m['done']:break
                        np.testing.assert_array_equal(initial,obs)
                        accepted=bool(m['episode']['target_reached'] and m['episode']['jumps']==0)
                        filename=cfg['start_id']+'.npz'
                        np.savez_compressed(data_dir/filename,observations=np.stack(observations),actions=np.array(labels,dtype=np.int64))
                        record=dict(split=split,config=cfg,accepted=accepted,filename=filename,sha256=digest(data_dir/filename),
                                    decisions=len(labels),action_counts=np.bincount(labels,minlength=3).tolist(),start_info=info,**m['episode'])
                        manifest.append(record)
                        print('COLLECT',split,cfg['start_id'],accepted,len(labels),flush=True)
                        (EXPERIMENT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                    finally:task.close()
    stats={}
    hashes={}
    for split in ('train','validation','test'):
        rows=[r for r in manifest if r['split']==split and r['accepted']]
        counts=np.sum([r['action_counts'] for r in rows],axis=0);assert (counts>0).all(),(split,counts)
        hashes[split]=set()
        for r in rows:
            with np.load(data_dir/r['filename']) as d:
                hashes[split].update(hashlib.sha256(o.tobytes()).hexdigest() for o in d['observations'])
        stats[split]=dict(accepted=len(rows),candidates=sum(r['split']==split for r in manifest),decisions=sum(r['decisions'] for r in rows),action_counts=counts.tolist())
    stats['exact_frame_overlap']={a+'_'+b:len(hashes[a]&hashes[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    stats['wall_seconds']=time.perf_counter()-begin
    (EXPERIMENT/'data_summary.json').write_text(json.dumps(stats,indent=2)+'\n');print('DATA_READY',json.dumps(stats),flush=True)


class GRUPolicy(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.n=64;self.encoder=copy.deepcopy(encoder)
        self.gru=nn.GRUCell(64,64,device='cuda');self.actor=nn.Linear(64,3,device='cuda')
    def forward(self,obs,h,starts):
        h=h*(1-starts[None,:]);z=self.encoder(obs[:,None].float()/255)
        h=self.gru(z,h.T).T
        return self.actor(h.T),torch.zeros(obs.shape[0],device='cuda'),h


def make_model(arch,seed):
    torch.manual_seed(seed);np.random.seed(seed)
    graph=GainPolicy(actions=3)
    enc_hash=tensor_hash(graph.encoder.state_dict())
    if arch=='graph':
        graph.critic.requires_grad_(False);model=graph
    else:model=GRUPolicy(graph.encoder);del graph
    return model,enc_hash


def dataset(split):
    manifest=json.loads((EXPERIMENT/'manifest.json').read_text());rows=[]
    for r in manifest:
        if r['split']==split and r['accepted']:
            path=EXPERIMENT/'data'/r['filename'];assert digest(path)==r['sha256']
            with np.load(path) as d:rows.append(dict(meta=r,obs=torch.as_tensor(d['observations'],device='cuda'),labels=torch.as_tensor(d['actions'],device='cuda')))
    return rows


def zero(model,batch):return torch.zeros(model.n,batch,device='cuda')


def batch_sequence(rows,choices,model):
    h=zero(model,len(choices));max_prefix=max(min(64,t) for i,t in choices)
    with torch.no_grad():
        for b in range(max_prefix):
            active=[b>=max_prefix-min(64,t) for i,t in choices]
            idx=[max(0,t-min(64,t)+b-(max_prefix-min(64,t))) for i,t in choices]
            ob=torch.stack([rows[i]['obs'][j] for (i,t),j in zip(choices,idx)])
            starts=torch.tensor([float(j==0) for j in idx],device='cuda')
            _,_,candidate=model(ob,h,starts)
            h=torch.where(torch.tensor(active,device='cuda')[None,:],candidate,h)
    logits=[];labels=[]
    for step in range(min(32,max(len(rows[i]['labels'])-t for i,t in choices))):
        idx=[min(t+step,len(rows[i]['labels'])-1) for i,t in choices]
        ob=torch.stack([rows[i]['obs'][j] for (i,t),j in zip(choices,idx)])
        starts=torch.tensor([float(j==0) for j in idx],device='cuda')
        l,_,h=model(ob,h,starts);logits.append(l)
        labels.append(torch.stack([rows[i]['labels'][j] if t+step<len(rows[i]['labels']) else torch.tensor(-100,device='cuda') for (i,t),j in zip(choices,idx)]))
    return torch.stack(logits).flatten(0,1),torch.stack(labels).flatten()


class Context:
    def __init__(self,model):self.model=model;self.h=zero(model,1);self.history=[];self.t=0
    def step(self,obs):
        o=torch.as_tensor(obs,device='cuda').unsqueeze(0);s=torch.tensor([float(self.t==0)],device='cuda')
        if self.t%32==0:
            self.h=zero(self.model,1)
            for a,b in self.history:_,_,self.h=self.model(a,self.h,b)
        self.history.append((o,s));self.history=self.history[-64:]
        l,_,self.h=self.model(o,self.h,s);self.t+=1
        return l[0]


def scores(labels,logits):
    y=np.asarray(labels);p=np.asarray(logits);pred=p.argmax(-1);cm=np.zeros((3,3),dtype=int);np.add.at(cm,(y,pred),1)
    recalls=np.diag(cm)/np.maximum(cm.sum(1),1)
    return dict(accuracy=float((y==pred).mean()),balanced_accuracy=float(recalls.mean()),recall=recalls.tolist(),confusion_matrix=cm.tolist(),cross_entropy=float(F.cross_entropy(torch.as_tensor(p),torch.as_tensor(y))))


def offline(model,rows,mode='normal'):
    labels=[];logits=[]
    with torch.no_grad():
        for r in rows:
            ctx=Context(model)
            for t,o in enumerate(r['obs']):
                logits.append(ctx.step(r['obs'][0] if mode=='frozen' else o).cpu().numpy());labels.append(int(r['labels'][t]))
    return scores(labels,logits)


def train(arch,seed):
    plan=json.loads((EXPERIMENT/'plan.json').read_text());run=EXPERIMENT/f'{arch}_seed{seed}';run.mkdir(exist_ok=False)
    model,enc_hash=make_model(arch,seed);train_rows=dataset('train');val_rows=dataset('validation')
    counts=torch.bincount(torch.cat([r['labels'] for r in train_rows]),minlength=3);weight=counts.sum()/(3*counts.float())
    choices=[(i,t) for i,r in enumerate(train_rows) for t in range(0,len(r['labels']),32)]
    rng=np.random.default_rng(seed+10000)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],eps=1e-5)
    config=dict(architecture=arch,seed=seed,encoder_initial_sha256=enc_hash,plan_sha256=digest(EXPERIMENT/'plan.json'),manifest_sha256=digest(EXPERIMENT/'manifest.json'),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),class_weights=weight.tolist(),initial_model_sha256=tensor_hash(model.state_dict()))
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def save(name,step):torch.save(dict(model=model.state_dict(),step=step,config=config),run/name)
    save('initial.pt',0);best=None;begin=time.perf_counter();records=[];valid_presentations=0
    fixed_hash=tensor_hash({'w':model.w.values()}) if arch=='graph' else None
    initial=offline(model,val_rows);records.append(dict(step=0,validation=initial));print('INITIAL',arch,seed,initial['balanced_accuracy'],flush=True)
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,plan['updates']+1):
        batch=[choices[i] for i in rng.integers(len(choices),size=plan['batch'])]
        opt.zero_grad(set_to_none=True);logits,labels=batch_sequence(train_rows,batch,model)
        loss=F.cross_entropy(logits,labels,weight=weight);assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        valid_presentations+=int((labels>=0).sum())
        if step%16==0:print('TRAIN',arch,seed,step,'loss',round(float(loss.detach()),4),flush=True)
        if step%64==0:
            with torch.no_grad():val=offline(model,val_rows)
            record=dict(step=step,training_batch_loss=float(loss.detach()),gradient_norm=float(norm),validation=val,valid_presentations=valid_presentations,wall_seconds=time.perf_counter()-begin)
            records.append(record);key=(val['balanced_accuracy'],-val['cross_entropy'])
            if best is None or key>best:best=key;save('best.pt',step)
            (run/'metrics.json').write_text(json.dumps(records,indent=2)+'\n');save('last.pt',step)
            print('VALIDATION',arch,seed,step,round(val['balanced_accuracy'],4),flush=True)
    assert arch!='graph' or tensor_hash({'w':model.w.values()})==fixed_hash
    torch.cuda.synchronize()
    status=dict(state='completed',updates=plan['updates'],valid_presentations=valid_presentations,wall_seconds=time.perf_counter()-begin,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,fixed_w_unchanged=True if arch=='graph' else None)
    (run/'status.json').write_text(json.dumps(status,indent=2)+'\n')


def closed_loop(model,cfg,mode,sample_seed,clock):
    task=BendTask(**cfg)
    try:
        obs=task.reset();initial=obs.copy();ctx=Context(model);actions=[]
        rng=None if sample_seed is None else torch.Generator().manual_seed(sample_seed)
        with torch.no_grad():
            while True:
                probs=clock[len(actions)] if mode=='clock' else ctx.step(initial if mode=='frozen' else obs).softmax(-1).cpu()
                a=int(probs.argmax()) if rng is None else int(torch.multinomial(probs,1,generator=rng))
                actions.append(a);m=task.step(a);obs=m['frame']
                if m['done']:return dict(mode=mode,sample_seed=sample_seed,actions=actions,**m['episode'])
    finally:task.close()


def evaluation_worker(pipe):
    torch.set_num_threads(1);task=None
    try:
        while True:
            command=pipe.recv()
            if command=='close':break
            if isinstance(command,dict):
                if task is not None:task.close()
                task=BendTask(**command);pipe.send(dict(frame=task.reset()))
            else:pipe.send(task.step(command))
    except BaseException as exc:
        pipe.send(dict(error=repr(exc)))
    finally:
        if task is not None:task.close()
        pipe.close()


class EvaluationPool:
    def __init__(self):
        import multiprocessing as mp
        context=mp.get_context('spawn');self.pipes=[];self.processes=[]
        self.specs=[(mode,seed) for seed in [None,8100,8101,8102] for mode in MODES]
        for _ in self.specs:
            p,c=context.Pipe();proc=context.Process(target=evaluation_worker,args=(c,));proc.start();c.close()
            self.pipes.append(p);self.processes.append(proc)
    def receive(self,i):
        p=self.pipes[i]
        if not p.poll(60):raise TimeoutError('evaluation worker timeout')
        result=p.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def rollout(self,model,cfg,clock):
        for p in self.pipes:p.send(cfg)
        obs=[self.receive(i)['frame'] for i in range(len(self.pipes))];initial=[o.copy() for o in obs]
        visual=[i for i,(mode,seed) in enumerate(self.specs) if mode!='clock']
        rngs=[None if seed is None else torch.Generator().manual_seed(seed) for mode,seed in self.specs]
        histories=[];h=zero(model,len(visual));finished=[False]*len(self.pipes);actions=[[] for _ in self.pipes];results=[None]*len(self.pipes);step=0
        with torch.no_grad():
            while not all(finished):
                o=torch.as_tensor(np.stack([initial[i] if self.specs[i][0]=='frozen' else obs[i] for i in visual]),device='cuda')
                starts=torch.full((len(visual),),float(step==0),device='cuda')
                if step%32==0:
                    h=zero(model,len(visual))
                    for old,mask in histories:_,_,h=model(old,h,mask)
                histories.append((o,starts));histories=histories[-64:]
                logits,_,h=model(o,h,starts);prob=logits.softmax(-1).cpu();by_index={i:prob[j] for j,i in enumerate(visual)}
                active=[i for i,done in enumerate(finished) if not done]
                for i in active:
                    p=clock[step] if self.specs[i][0]=='clock' else by_index[i]
                    action=int(p.argmax()) if rngs[i] is None else int(torch.multinomial(p,1,generator=rngs[i]))
                    actions[i].append(action);self.pipes[i].send(action)
                for i in active:
                    m=self.receive(i);obs[i]=m['frame']
                    if m['done']:
                        finished[i]=True;mode,seed=self.specs[i]
                        results[i]=dict(mode=mode,sample_seed=seed,actions=actions[i],**m['episode'])
                step+=1
        return results
    def close(self):
        for p in self.pipes:
            try:p.send('close')
            except (BrokenPipeError,EOFError):pass
        for proc in self.processes:
            proc.join(5)
            if proc.is_alive():proc.terminate();proc.join(5)
        for p in self.pipes:p.close()


def evaluate(arch,seed):
    run=EXPERIMENT/f'{arch}_seed{seed}';model,_=make_model(arch,seed);checkpoint=torch.load(run/'best.pt',weights_only=False);model.load_state_dict(checkpoint['model']);model.eval()
    ckhash=digest(run/'best.pt');test_rows=dataset('test');train_rows=dataset('train');report=dict(architecture=arch,seed=seed,selected_step=checkpoint['step'],checkpoint_sha256=ckhash,offline={},episodes=[])
    for split,rows in [('train',train_rows),('test',test_rows)]:
        report['offline'][split]={mode:offline(model,rows,mode) for mode in ['normal','frozen']}
    with torch.no_grad():
        ctx=Context(model);first=train_rows[0]['obs'][0];clock=[ctx.step(first).softmax(-1).cpu() for _ in range(225)]
    report['clock_source']=train_rows[0]['meta']['start_id'];report['clock_probabilities']=[p.tolist() for p in clock]
    pool=EvaluationPool()
    try:
        for at,r in enumerate(test_rows):
            cfg=r['meta']['config'];episodes=pool.rollout(model,cfg,clock)
            if at==0:
                reference=closed_loop(model,cfg,'normal',8100,clock)
                actual=next(e for e in episodes if e['mode']=='normal' and e['sample_seed']==8100)
                for key in ['actions','reason','frames','progress','jumps']:assert actual[key]==reference[key],key
                report['batched_evaluator_regression_passed']=True
            report['episodes'].extend(episodes)
            (run/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print('EVALUATED',arch,seed,cfg['start_id'],flush=True)
    finally:pool.close()
    assert digest(run/'best.pt')==ckhash;report['checkpoint_unchanged']=True
    (run/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['collect','train','evaluate']);parser.add_argument('--arch',choices=['graph','gru']);parser.add_argument('--seed',type=int,default=0);a=parser.parse_args()
    torch.set_num_threads(4)
    if a.phase=='collect':collect()
    else:
        assert a.arch
        (train if a.phase=='train' else evaluate)(a.arch,a.seed)
if __name__=='__main__':main()
