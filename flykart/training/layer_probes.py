"""Frozen-backbone, trajectory-split layer decoding diagnostic."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import supervised as base
from .microsteps_experiment import MicroPolicy, ORIGINAL
OUT=ORIGINAL.parent/'layer_probes_v1'

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')

def backbone(kind):
    if kind=='gru':m,_=base.make_model('gru',0);p=ORIGINAL/'gru_seed0/best.pt'
    else:
        m=MicroPolicy(1 if kind=='k1' else 2)
        p=(ORIGINAL/'graph_seed0/best.pt' if kind=='k1' else ORIGINAL.parent/'neural_microsteps_v1/k2/graph_seed0/best.pt')
    m.load_state_dict(torch.load(p,weights_only=False)['model']);m.requires_grad_(False);m.eval()
    return m,p

class Features(nn.Module):
    def __init__(self,m):
        super().__init__();self.backbone=m;self.n=m.n;self.z=None
        self.hook=m.encoder.register_forward_hook(self.capture)
        self.indices={}
        if self.n>64:
            rng=np.random.default_rng(20260914);visual=m.visual.bool().cpu().numpy();motor=m.readout.cpu().numpy()
            middle=~visual;middle[motor]=False
            for name,ids in [('visual',np.flatnonzero(visual)),('intermediate',np.flatnonzero(middle)),('motor256',motor)]:
                self.indices[name]=torch.tensor(np.sort(rng.choice(ids,256,replace=False)),device='cuda')
            self.indices['motor']=m.readout
        self.names=['cnn']+list(self.indices) if self.n>64 else ['cnn','recurrent']
        dims=[64]+[len(v) for v in self.indices.values()] if self.n>64 else [64,64]
        self.slices={};at=0
        for name,d in zip(self.names,dims):self.slices[name]=slice(at,at+d);at+=d
    def capture(self,module,args,out):self.z=out
    def forward(self,obs,h,starts):
        _,v,h=self.backbone(obs,h,starts)
        parts=[self.z]+[h.index_select(0,i).T for i in self.indices.values()] if self.n>64 else [self.z,h.T]
        return torch.cat(parts,-1),v,h

@torch.no_grad()
def extract(model,rows,mode):
    xs=[];ys=[];times=[]
    for row in rows:
        ctx=base.Context(model)
        for t,o in enumerate(row['obs']):
            xs.append(ctx.step(row['obs'][0] if mode=='frozen' else o).clone());ys.append(row['labels'][t]);times.append(t/225.)
    return torch.stack(xs),torch.stack(ys),torch.tensor(times,device='cuda')[:,None]

class Probe(nn.Module):
    def __init__(self,x,kind):
        super().__init__();self.register_buffer('mean',x.mean(0));self.register_buffer('scale',x.std(0).clamp_min(1e-5))
        d=x.shape[1];self.net=(nn.Linear(d,3) if kind=='linear' else nn.Sequential(nn.Linear(d,32),nn.Tanh(),nn.Linear(32,3))).cuda()
    def forward(self,x):return self.net((x-self.mean)/self.scale)

def score(p,x,y):
    with torch.no_grad():return base.scores(y.cpu().numpy(),p(x).cpu().numpy())

def fit(x,y,v,vy,kind):
    torch.manual_seed(123);p=Probe(x,kind);opt=torch.optim.Adam(p.parameters(),lr=.003)
    weights=len(y)/(3*torch.bincount(y,minlength=3).float());best=None;history=[]
    for step in range(601):
        if step%25==0:
            s=score(p,v,vy);history.append(dict(step=step,**s));key=(s['balanced_accuracy'],-s['cross_entropy'])
            if best is None or key>best[0]:best=(key,copy.deepcopy(p.state_dict()),step)
        if step==600:break
        opt.zero_grad();loss=nn.functional.cross_entropy(p(x),y,weight=weights);loss.backward();opt.step()
    p.load_state_dict(best[1]);p.requires_grad_(False)
    return p,dict(selected_step=best[2],history=history,parameters=sum(t.numel() for t in p.parameters()))

class Policy(nn.Module):
    def __init__(self,features,layer,probe):
        super().__init__();self.features=features;self.layer=layer;self.probe=probe;self.n=features.n
    def forward(self,obs,h,starts):
        x,v,h=self.features(obs,h,starts);return self.probe(x[:,self.features.slices[self.layer]]),v,h

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=False)
    save('protocol.json',dict(seed=0,backbones=['k1','k2','gru'],layers='CNN64; random visual256; random nonvisual nonmotor256; motor256; full motor2129; GRU64',node_seed=20260914,probe_types=['linear','mlp32'],updates=600,selection_every=25,selection='validation normal balanced accuracy, tie CE; best graph and best GRU; evaluate only if BA >= .45 and normal-frozen >= .05',normalization='training normal features only; std floor 1e-5',controls='same trained probe on frozen first image for all history; separately trained time linear/MLP; constant majority',warning='Seed0 exploratory diagnostic, reused same-track test starts. Random intermediate samples are not anatomically staged layers.',source_sha256=base.digest(Path(__file__))))
    rows={s:base.dataset(s) for s in ['train','validation','test']};all_results={};candidates=[];hashes={}
    y=torch.cat([r['labels'] for r in rows['train']]);vy=torch.cat([r['labels'] for r in rows['validation']])
    def times(split):return torch.cat([torch.arange(len(r['labels']),device='cuda')[:,None]/225. for r in rows[split]])
    time_probes={};time_results={}
    for typ in ['linear','mlp32']:
        p,info=fit(times('train'),y,times('validation'),vy,typ);time_probes[typ]=p
        time_results[typ]={s:score(p,times(s),torch.cat([r['labels'] for r in rows[s]])) for s in rows}
    chosen_time=max(time_probes,key=lambda k:(time_results[k]['validation']['balanced_accuracy'],-time_results[k]['validation']['cross_entropy']))
    timep=time_probes[chosen_time];torch.save(timep.state_dict(),OUT/'time_probe.pt');save('time_results.json',dict(results=time_results,selected=chosen_time))
    for kind in ['k1','k2','gru']:
        m,path=backbone(kind);hashes[str(path)]=base.digest(path);before=base.tensor_hash(m.state_dict());f=Features(m)
        save(f'{kind}_nodes.json',{k:v.cpu().tolist() for k,v in f.indices.items()})
        data={}
        for split in ['train','validation']:
            for mode in ['normal','frozen']:
                data[split,mode]=extract(f,rows[split],mode);print('EXTRACTED',kind,split,mode,flush=True)
        results={};trained={}
        for layer,sl in f.slices.items():
            for typ in ['linear','mlp32']:
                name=layer+'_'+typ;p,info=fit(data['train','normal'][0][:,sl],y,data['validation','normal'][0][:,sl],vy,typ)
                trained[name]=p;results[name]=dict(layer=layer,type=typ,dimension=sl.stop-sl.start,**info)
                for split in ['train','validation']:
                    results[name][split]={mode:score(p,data[split,mode][0][:,sl],data[split,mode][1]) for mode in ['normal','frozen']}
                torch.save(p.state_dict(),OUT/f'{kind}_{name}.pt')
        chosen=max(results,key=lambda k:(results[k]['validation']['normal']['balanced_accuracy'],-results[k]['validation']['normal']['cross_entropy']))
        save(f'{kind}_selection.json',dict(chosen=chosen,selected_before_test=True,validation=results[chosen]['validation']))
        for mode in ['normal','frozen']:
            x,yy,_=extract(f,rows['test'],mode)
            for name,p in trained.items():results[name].setdefault('test',{})[mode]=score(p,x[:,f.slices[results[name]['layer']]],yy)
        # Verify extraction and deployed wrapper agree under history refresh.
        p=trained[chosen];policy=Policy(f,results[chosen]['layer'],p)
        with torch.no_grad():
            ctx=base.Context(policy);actual=torch.stack([ctx.step(o) for o in rows['validation'][0]['obs']]);expected=p(data['validation','normal'][0][:len(actual),f.slices[results[chosen]['layer']]])
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-5)
        assert before==base.tensor_hash(m.state_dict());assert all(p.grad is None for p in m.parameters())
        results['checks']=dict(backbone_unchanged=True,no_backbone_gradients=True,extracted_deployed_logits_match=True)
        save(f'{kind}_results.json',results);all_results[kind]=results
        candidates.append((kind,chosen,results[chosen]['validation']))
        print('PROBED',kind,chosen,results[chosen]['test'],flush=True)
        f.hook.remove();del policy,f,m,data,trained
    selected=[]
    graph=max([c for c in candidates if c[0]!='gru'],key=lambda c:(c[2]['normal']['balanced_accuracy'],-c[2]['normal']['cross_entropy']))
    for c in [graph,next(c for c in candidates if c[0]=='gru')]:
        if c[2]['normal']['balanced_accuracy']>=.45 and c[2]['normal']['balanced_accuracy']-c[2]['frozen']['balanced_accuracy']>=.05:selected.append(c)
    save('closed_loop_selection.json',dict(candidates=candidates,selected=selected,rule_uses_validation_only=True))
    with torch.no_grad():clock=list(timep(torch.arange(225,device='cuda')[:,None]/225.).softmax(-1).cpu())
    for kind,name,_ in selected:
        m,_=backbone(kind);f=Features(m);r=all_results[kind][name];p=Probe(torch.zeros(2,r['dimension'],device='cuda'),r['type']);p.load_state_dict(torch.load(OUT/f'{kind}_{name}.pt',weights_only=True));policy=Policy(f,r['layer'],p).eval();report=dict(kind=kind,probe=name,episodes=[],clock='validation-selected trained time-only probe',clock_probabilities=[x.tolist() for x in clock]);pool=base.EvaluationPool()
        try:
            for at,row in enumerate(rows['test']):
                episodes=pool.rollout(policy,row['meta']['config'],clock)
                if at==0:
                    ref=base.closed_loop(policy,row['meta']['config'],'normal',8100,clock);actual=next(e for e in episodes if e['mode']=='normal' and e['sample_seed']==8100)
                    for key in ['actions','reason','frames','progress','jumps']:assert ref[key]==actual[key],key
                    report['batched_evaluator_regression_passed']=True
                report['episodes'].extend(episodes);save(f'{kind}_evaluation.json',report);print('EVALUATED',kind,at+1,flush=True)
        finally:pool.close()
        f.hook.remove();del policy,f,m
    assert all(base.digest(Path(p))==h for p,h in hashes.items());save('status.json',dict(state='completed',backbone_checkpoint_hashes=hashes,checkpoints_unchanged=True))
if __name__=='__main__':run()
