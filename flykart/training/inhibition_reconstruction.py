"""Matched full-trajectory reconstruction with positive, GABA and shuffled signs."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import supervised as base
from . import layer_probes as lp
from .model import FixedMultiply
OUT=lp.OUT.parent/'inhibition_reconstruction_v1'
META=lp.OUT.parent/'architecture_mechanisms_v1/metadata.npz'
SOURCE=lp.ORIGINAL.parent/'neural_microsteps_v1/k2/graph_seed0/best.pt'
KINDS=['positive','gaba','shuffled']
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
class Model(nn.Module):
    def __init__(self,kind,stats):
        super().__init__();self.b,_=lp.backbone('k2');self.b.requires_grad_(False);self.b.a.data.zero_();self.b.a.requires_grad_(True);self.n=self.b.n;self.kind=kind;self.raw=False;self.action=False
        with np.load(META) as d:sign=np.ones(self.n,np.float32) if kind=='positive' else d['signs' if kind=='gaba' else 'shuffled_signs']
        self.register_buffer('signs',torch.tensor(sign,device='cuda'))
        for k,v in stats.items():self.register_buffer(k,v.clone())
        gen=torch.Generator(device='cuda').manual_seed(8181);self.register_buffer('ports',torch.randn(self.n,8,generator=gen,device='cuda')/8**.5)
        self.interface=nn.Parameter(torch.zeros(8,64,device='cuda'));self.decoder=nn.Linear(len(self.b.readout),64,device='cuda');nn.init.zeros_(self.decoder.weight);nn.init.zeros_(self.decoder.bias)
        self.head=lp.Probe(torch.zeros(2,64,device='cuda'),'mlp32');self.head.load_state_dict(torch.load(lp.OUT/'k2_cnn_mlp32.pt',weights_only=True));self.head.requires_grad_(False)
    def forward(self,obs,h,starts):
        with torch.no_grad():z=self.b.encoder(obs[:,None].float()/255) if obs.ndim==3 else obs
        h=h*(1-starts[None]);signal=z.T.index_select(0,self.b.channels)*self.b.visual[:,None]
        signal=signal+(self.ports@(self.interface@((z-self.zmean)/self.zscale).T))*self.b.visual[:,None]
        gain=(.5+self.b.a.sigmoid())[:,None]
        for _ in range(2):h=.5*h+.5*torch.tanh(gain*FixedMultiply.apply(h*self.signs[:,None],self.b.w,self.b.wt)+signal)
        motor=h[self.b.readout].T
        out=motor if self.raw else self.decoder((motor-self.hmean)/self.hscale)
        if self.action:out=self.head(out*self.zscale+self.zmean)
        return out,torch.zeros(len(obs),device='cuda'),h
@torch.no_grad()
def encode(m,rows):
    return [dict(meta=r['meta'],obs=m.b.encoder(r['obs'][:,None].float()/255),labels=r['labels']) for r in rows]
@torch.no_grad()
def offline(m,rows,mode='normal'):
    predictions=[];targets=[];labels=[];sat=[];disagreement=[]
    for row in rows:
        ctx=base.Context(m)
        for t,z in enumerate(row['obs']):
            predictions.append(ctx.step(row['obs'][0] if mode=='frozen' else z));targets.append((z-m.zmean)/m.zscale);labels.append(row['labels'][t]);sat.append((ctx.h.abs()>.99).float().mean())
    x=torch.stack(predictions);y=torch.stack(targets);yy=torch.stack(labels);loss=float((x-y).square().mean());var=float((y-y.mean(0)).square().mean());actions=m.head(x*m.zscale+m.zmean);reference=m.head(y*m.zscale+m.zmean)
    return dict(normalized_mse=loss,r2_global=1-loss/max(var,1e-12),action_scores=base.scores(yy.cpu().numpy(),actions.cpu().numpy()),direct_cnn_scores=base.scores(yy.cpu().numpy(),reference.cpu().numpy()),agreement_with_cnn=float((actions.argmax(-1)==reference.argmax(-1)).float().mean()),mean_state_saturation=float(torch.stack(sat).mean()))
def batch(m,rows,choices):
    logits,labels=base.batch_sequence(rows,choices,m);targets=[]
    for step in range(min(32,max(len(rows[i]['labels'])-t for i,t in choices))):
        targets.append(torch.stack([rows[i]['obs'][min(t+step,len(rows[i]['labels'])-1)] for i,t in choices]))
    target=(torch.stack(targets).flatten(0,1)-m.zmean)/m.zscale;mask=labels>=0
    return (logits[mask]-target[mask]).square().mean(),int(mask.sum())
def collect():
    out=OUT/'new_test';out.mkdir();manifest=[]
    for frame in [475,535,595]:
        for perturb in [16,32]:
            for turn in ['center','left','right']:
                cfg=base.make_config(frame,turn,perturb);task=base.BendTask(**cfg)
                try:
                    obs=task.reset();first=obs.copy();np.testing.assert_array_equal(first,task.reset());xs=[];ys=[];waypoint=0
                    while True:
                        a,waypoint=base.oracle(task,waypoint);xs.append(obs);ys.append(a);msg=task.step(a);obs=msg['frame']
                        if msg['done']:break
                    name=cfg['start_id']+'.npz';np.savez_compressed(out/name,observations=np.stack(xs),actions=np.array(ys,np.int64));manifest.append(dict(config=cfg,filename=name,sha256=base.digest(out/name),accepted=bool(msg['episode']['target_reached'] and msg['episode']['jumps']==0),episode=msg['episode']));save('new_test_manifest.json',manifest);print('COLLECT',cfg['start_id'],manifest[-1]['accepted'],flush=True)
                finally:task.close()

def checks(stats,rows):
    from .microsteps_experiment import MicroPolicy
    m=Model('positive',stats);ref=MicroPolicy(2);ref.load_state_dict(torch.load(SOURCE,weights_only=False)['model']);ref.a.data.zero_();obs=rows[0]['obs'][:2];h=torch.zeros(m.n,2,device='cuda');starts=torch.ones(2,device='cuda')
    with torch.no_grad():
        _,_,actual=m(obs,h,starts);_,_,expected=ref(obs,h,starts);torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
    # Same context reconstruction, target alignment, and padding as deployment.
    encoded=encode(m,rows[:2]);m.decoder.weight.data.normal_(0,.0001)
    choices=[(0,0),(1,32)];p,labels=base.batch_sequence(encoded,choices,m)
    with torch.no_grad():
        contexts=[base.Context(m),base.Context(m)];refout=[]
        for i,t in choices:
            ctx=base.Context(m);refout.append(torch.stack([ctx.step(z) for z in encoded[i]['obs']]))
        at=0
        for step in range(min(32,max(len(encoded[i]['obs'])-t for i,t in choices))):
            for j,(i,t) in enumerate(choices):
                if t+step<len(refout[j]):torch.testing.assert_close(p[at],refout[j][t+step],rtol=1e-4,atol=2e-6)
                else:assert labels[at]==-100
                at+=1
    loss,n=batch(m,encoded,choices);loss.backward();assert m.interface.grad.norm()>0 and m.b.a.grad.norm()>0;assert all(p.grad is None for p in m.b.encoder.parameters());save('checks.json',dict(initial_positive_dynamics_match=True,context_and_padding_match=True,interface_and_gain_gradients_nonzero=True,encoder_gradients_none=True))
def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=False)
    save('protocol.json',dict(kinds=KINDS,seed=0,updates=384,batch=8,bptt=32,burn_in=64,optimizer='Adam lr .0003 eps1e-5 clip1',data='all43 training trajectories/2184 decisions, all6 validation trajectories/251 decisions; no random frame split',initialization='same frozen trained CNN; a=0, residual interface=0, decoder=0. No previously trained graph gains/readout.',normalization='CNN stats from all training decisions; common motor stats from positive zero-gain-initial graph on all training decisions with Context semantics, fixed for all groups',only_difference='presynaptic sign vector; W absolute magnitudes identical; interface/decoder/gain initial tensors, latent samples and optimizer matched',sign_definition='high confidence >=.8 body GABA agreeing with consensus negative; other neurons positive. Matched shuffle within superclass x side. Not a full biological E/I model.',selection='best validation normal reconstruction MSE at0/64/.../384; tie earlier. Closed loop positive and best signed variant by validation MSE; if signed qualifies, also its paired signed control. Qualification: MSE <= .9*positive and normal downstream BA >= frozen BA+.10. No replication within this pilot.',positive_control='archived direct CNN head, always evaluate on new starts',new_test='prefix475/535/595 x perturb16/32 x center/left/right; oracle accepted before model evaluation',sources={str(p):base.digest(p) for p in [Path(__file__),SOURCE,META,lp.OUT/'k2_cnn_mlp32.pt']}))
    collect();rows={s:base.dataset(s) for s in ['train','validation']};dummy=dict(zmean=torch.zeros(64,device='cuda'),zscale=torch.ones(64,device='cuda'),hmean=torch.zeros(2129,device='cuda'),hscale=torch.ones(2129,device='cuda'));m=Model('positive',dummy);encoded={s:encode(m,r) for s,r in rows.items()};zz=torch.cat([r['obs'] for r in encoded['train']]);stats=dict(zmean=zz.mean(0),zscale=zz.std(0).clamp_min(1e-5));m.raw=True;motors=[]
    with torch.no_grad():
        for r in encoded['train']:
            ctx=base.Context(m)
            for z in r['obs']:motors.append(ctx.step(z))
    hh=torch.stack(motors);stats.update(hmean=hh.mean(0),hscale=hh.std(0).clamp_min(1e-5));torch.save({k:v.cpu() for k,v in stats.items()},OUT/'stats.pt');del m,motors,hh;checks(stats,rows['train'])
    chunks=[(i,t) for i,r in enumerate(encoded['train']) for t in range(0,len(r['obs']),32)];rng=np.random.default_rng(10000);schedule=[[chunks[j] for j in rng.integers(len(chunks),size=8)] for _ in range(384)];save('schedule.json',schedule);results={};initial_hash=None
    for kind in KINDS:
        m=Model(kind,stats);m.head.requires_grad_(False);params=[p for p in m.parameters() if p.requires_grad];initial={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad};ih=base.tensor_hash(initial)
        if initial_hash is None:initial_hash=ih
        else:assert initial_hash==ih
        cnn_hash=base.tensor_hash(m.b.encoder.state_dict());opt=torch.optim.Adam(params,lr=.0003,eps=1e-5);history=[];best=None;presentations=0;begin=time.perf_counter()
        for step in range(385):
            if step%64==0:
                val=offline(m,encoded['validation']);history.append(dict(step=step,validation=val,valid_presentations=presentations));key=val['normalized_mse']
                if best is None or key<best[0]:best=(key,step);torch.save(m.state_dict(),OUT/(kind+'_best.pt'))
                save(kind+'_metrics.json',history);print('VALIDATION',kind,step,val['normalized_mse'],val['action_scores']['balanced_accuracy'],flush=True)
            if step==384:break
            opt.zero_grad(set_to_none=True);loss,n=batch(m,encoded['train'],schedule[step]);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();presentations+=n
            if (step+1)%32==0:print('TRAIN',kind,step+1,float(loss.detach()),float(norm),flush=True)
        torch.save(m.state_dict(),OUT/(kind+'_last.pt'));m.load_state_dict(torch.load(OUT/(kind+'_best.pt'),weights_only=True));result=dict(selected_step=best[1],history=history,train={mode:offline(m,encoded['train'],mode) for mode in ['normal','frozen']},validation={mode:offline(m,encoded['validation'],mode) for mode in ['normal','frozen']},parameter_updates={n:float((p.detach()-initial[n]).norm()) for n,p in m.named_parameters() if p.requires_grad},last_gradient_norms={n:float(p.grad.norm()) if p.grad is not None else None for n,p in m.named_parameters() if p.requires_grad},valid_presentations=presentations,wall_seconds=time.perf_counter()-begin,initial_trainable_sha256=ih)
        assert base.tensor_hash(m.b.encoder.state_dict())==cnn_hash and all(p.grad is None for p in m.b.encoder.parameters());result['cnn_unchanged']=True;save(kind+'_results.json',result);results[kind]=result;del m,opt
    assert len({r['valid_presentations'] for r in results.values()})==1
    signed=min(['gaba','shuffled'],key=lambda k:results[k]['validation']['normal']['normalized_mse']);v=results[signed]['validation'];qualified=v['normal']['normalized_mse']<=.9*results['positive']['validation']['normal']['normalized_mse'] and v['normal']['action_scores']['balanced_accuracy']>=v['frozen']['action_scores']['balanced_accuracy']+.10
    selected=['positive',signed]+([k for k in ['gaba','shuffled'] if k!=signed] if qualified else []);save('selection.json',dict(selected=selected,best_signed=signed,qualified=qualified,selected_before_new_test=True))
    manifest=json.loads((OUT/'new_test_manifest.json').read_text());test=[]
    for r in manifest:
        if r['accepted']:
            with np.load(OUT/'new_test'/r['filename']) as d:test.append(dict(meta=r,obs=torch.tensor(d['observations'],device='cuda'),labels=torch.tensor(d['actions'],device='cuda')))
    reports={}
    for kind in KINDS:
        m=Model(kind,stats);m.load_state_dict(torch.load(OUT/(kind+'_best.pt'),weights_only=True));et=encode(m,test);reports[kind]={mode:offline(m,et,mode) for mode in ['normal','frozen']};del m
    save('new_test_offline.json',reports)
    timehead=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');timehead.load_state_dict(torch.load(lp.OUT/'time_probe.pt',weights_only=True))
    with torch.no_grad():clock=list(timehead(torch.arange(225,device='cuda')[:,None]/225).softmax(-1).cpu())
    pool=base.EvaluationPool()
    try:
        for kind in selected+['cnn']:
            m=Model('positive' if kind=='cnn' else kind,stats)
            if kind=='cnn':
                f=lp.Features(m.b);policy=lp.Policy(f,'cnn',m.head)
            else:m.load_state_dict(torch.load(OUT/(kind+'_best.pt'),weights_only=True));m.action=True;policy=m
            report=dict(kind=kind,episodes=[])
            for at,r in enumerate(test):
                es=pool.rollout(policy,r['meta']['config'],clock)
                if at==0:
                    ref=base.closed_loop(policy,r['meta']['config'],'normal',8100,clock);actual=next(e for e in es if e['mode']=='normal' and e['sample_seed']==8100)
                    for key in ['actions','reason','frames','progress','jumps']:assert ref[key]==actual[key],key
                    report['batched_sequential_match']=True
                report['episodes'].extend(es);save(kind+'_evaluation.json',report);print('EVALUATED',kind,at+1,flush=True)
            if kind=='cnn':f.hook.remove()
            del policy,m
    finally:pool.close()
    for p,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():assert base.digest(Path(p))==h
    save('status.json',dict(state='completed',sources_unchanged=True,matched_initial_parameters=True,matched_presentations=True,new_test_accepted=len(test),selection=selected))
if __name__=='__main__':run()
