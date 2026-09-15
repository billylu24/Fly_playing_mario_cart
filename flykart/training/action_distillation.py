"""Matched warm-start reconstruction versus reconstruction plus action distillation."""
import json,time
from pathlib import Path
import numpy as np
import torch
from . import supervised as base
from . import layer_probes as lp
from . import inhibition_reconstruction as prior
SOURCE=prior.OUT
OUT=SOURCE.parent/'action_distillation_v1'
KINDS=['reconstruction','joint']
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def model(stats):
    m=prior.Model('positive',stats);m.load_state_dict(torch.load(SOURCE/'positive_best.pt',weights_only=True));return m
@torch.no_grad()
def evaluate(m,rows,mode='normal'):
    r=prior.offline(m,rows,mode);student=[];teacher=[]
    for row in rows:
        ctx=base.Context(m)
        for z in row['obs']:
            recon=ctx.step(row['obs'][0] if mode=='frozen' else z);student.append(m.head(recon[None]*m.zscale+m.zmean)[0]);teacher.append(m.head(z[None])[0])
    s=torch.stack(student);t=torch.stack(teacher);r['teacher_kl']=float(torch.nn.functional.kl_div(s.log_softmax(-1),t.softmax(-1),reduction='batchmean'));return r

def batch(m,rows,choices):
    recon,labels=base.batch_sequence(rows,choices,m);zs=[]
    for step in range(min(32,max(len(rows[i]['labels'])-t for i,t in choices))):zs.append(torch.stack([rows[i]['obs'][min(t+step,len(rows[i]['labels'])-1)] for i,t in choices]))
    z=torch.stack(zs).flatten(0,1);mask=labels>=0;recon=recon[mask];z=z[mask]
    mse=(recon-(z-m.zmean)/m.zscale).square().mean();student=m.head(recon*m.zscale+m.zmean)
    with torch.no_grad():teacher=m.head(z).softmax(-1)
    kl=torch.nn.functional.kl_div(student.log_softmax(-1),teacher,reduction='batchmean');return mse,kl,int(mask.sum())
def collect():
    out=OUT/'new_test';out.mkdir();manifest=[]
    for frame in [485,545,605]:
        for perturb in [22,34]:
            for turn in ['center','left','right']:
                cfg=base.make_config(frame,turn,perturb);task=base.BendTask(**cfg)
                try:
                    obs=task.reset();first=obs.copy();np.testing.assert_array_equal(first,task.reset());xs=[];ys=[];waypoint=0
                    while True:
                        action,waypoint=base.oracle(task,waypoint);xs.append(obs);ys.append(action);msg=task.step(action);obs=msg['frame']
                        if msg['done']:break
                    file=cfg['start_id']+'.npz';np.savez_compressed(out/file,observations=np.stack(xs),actions=np.array(ys,np.int64));manifest.append(dict(config=cfg,filename=file,sha256=base.digest(out/file),accepted=bool(msg['episode']['target_reached'] and msg['episode']['jumps']==0),episode=msg['episode']));save('new_test_manifest.json',manifest);print('COLLECT',cfg['start_id'],manifest[-1]['accepted'],flush=True)
                finally:task.close()

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=False);save('protocol.json',dict(kinds=KINDS,seed=0,updates=256,batch=8,lr=.0003,optimizer='Adam eps1e-5 clip1',initialization='both warm-start identical inhibition_reconstruction_v1/positive_best; reset optimizer for both',trainable='gain, visual interface512, reconstruction decoder; CNN and teacher action head frozen',objectives='reconstruction: standardized latent MSE; joint: MSE + KL(teacher_action || student_action), coefficient1 temperature1',teacher='existing frozen CNN + archived cnn_mlp32; no expert action labels in objective except padding mask',selection='both by minimum normal validation teacher KL at0/64/128/192/256; ties earlier; all models evaluated regardless of result',new_starts='prefix485/545/605 x perturb22/34 x center/left/right; oracle feasibility before training; held out from tuning',evaluation='all selected graph heads plus archived direct CNN; greedy and seeds8100/8101/8102 x normal/frozen/time',limits='single seed; current short track task; teacher errors can be inherited',sources={str(p):base.digest(p) for p in [Path(__file__),Path(prior.__file__),SOURCE/'positive_best.pt',SOURCE/'stats.pt',lp.OUT/'k2_cnn_mlp32.pt']}))
    collect();stats={k:v.cuda() for k,v in torch.load(SOURCE/'stats.pt',weights_only=True).items()};m=model(stats);originalrows={s:base.dataset(s) for s in ['train','validation']};rows={s:prior.encode(m,r) for s,r in originalrows.items()};chunks=[(i,t) for i,r in enumerate(rows['train']) for t in range(0,len(r['obs']),32)];rng=np.random.default_rng(12000);schedule=[[chunks[i] for i in rng.integers(len(chunks),size=8)] for _ in range(256)];save('schedule.json',schedule)
    # Explicitly verify frozen teacher still propagates gradients to reconstructed inputs.
    mse,kl,n=batch(m,rows['train'],schedule[0]);gs=torch.autograd.grad(kl,[m.interface,m.b.a,m.decoder.weight]);assert all(torch.isfinite(g).all() and g.norm()>0 for g in gs);assert all(p.grad is None for p in m.b.encoder.parameters());save('gradient_checks.json',dict(distillation_reaches_interface_gain_decoder=[float(g.norm()) for g in gs],teacher_weights_frozen=all(not p.requires_grad for p in m.head.parameters()),cnn_weights_frozen=all(not p.requires_grad for p in m.b.encoder.parameters())));del m,mse,kl,gs
    results={};hashes=[]
    for kind in KINDS:
        m=model(stats);initial={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad};hashes.append(base.tensor_hash(initial));cnn=base.tensor_hash(m.b.encoder.state_dict());head=base.tensor_hash(m.head.state_dict());opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.0003,eps=1e-5);history=[];best=None;presentations=0;begin=time.perf_counter()
        for step in range(257):
            if step%64==0:
                val=evaluate(m,rows['validation']);history.append(dict(step=step,validation=val,valid_presentations=presentations));key=val['teacher_kl']
                if best is None or key<best[0]:best=(key,step);torch.save(m.state_dict(),OUT/(kind+'_best.pt'))
                save(kind+'_metrics.json',history);print('VALIDATION',kind,step,'KL',key,'BA',val['action_scores']['balanced_accuracy'],'MSE',val['normalized_mse'],flush=True)
            if step==256:break
            opt.zero_grad(set_to_none=True);mse,kl,n=batch(m,rows['train'],schedule[step]);loss=mse+(kl if kind=='joint' else 0);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();presentations+=n
            if (step+1)%32==0:print('TRAIN',kind,step+1,float(mse.detach()),float(kl.detach()),flush=True)
        torch.save(m.state_dict(),OUT/(kind+'_last.pt'));m.load_state_dict(torch.load(OUT/(kind+'_best.pt'),weights_only=True));r=dict(selected_step=best[1],train={mode:evaluate(m,rows['train'],mode) for mode in ['normal','frozen']},validation={mode:evaluate(m,rows['validation'],mode) for mode in ['normal','frozen']},initial_trainable_hash=hashes[-1],valid_presentations=presentations,wall_seconds=time.perf_counter()-begin,parameter_update_l2={n:float((p.detach()-initial[n]).norm()) for n,p in m.named_parameters() if p.requires_grad});assert base.tensor_hash(m.b.encoder.state_dict())==cnn and base.tensor_hash(m.head.state_dict())==head;assert all(p.grad is None for p in m.b.encoder.parameters()) and all(p.grad is None for p in m.head.parameters());r['cnn_and_teacher_unchanged']=True;save(kind+'_results.json',r);results[kind]=r;del m,opt
    assert hashes[0]==hashes[1] and results[KINDS[0]]['valid_presentations']==results[KINDS[1]]['valid_presentations'];save('selection.json',{k:r['selected_step'] for k,r in results.items()})
    manifest=json.loads((OUT/'new_test_manifest.json').read_text());test=[]
    for r in manifest:
        if r['accepted']:
            with np.load(OUT/'new_test'/r['filename']) as d:test.append(dict(meta=r,obs=torch.tensor(d['observations'],device='cuda'),labels=torch.tensor(d['actions'],device='cuda')))
    offline={}
    for kind in KINDS:
        m=model(stats);m.load_state_dict(torch.load(OUT/(kind+'_best.pt'),weights_only=True));encoded=prior.encode(m,test);offline[kind]={mode:evaluate(m,encoded,mode) for mode in ['normal','frozen']};save('new_test_offline.json',offline);del m
    th=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');th.load_state_dict(torch.load(lp.OUT/'time_probe.pt',weights_only=True))
    with torch.no_grad():clock=list(th(torch.arange(225,device='cuda')[:,None]/225).softmax(-1).cpu())
    pool=base.EvaluationPool()
    try:
        for kind in KINDS+['cnn']:
            m=model(stats)
            if kind=='cnn':f=lp.Features(m.b);policy=lp.Policy(f,'cnn',m.head)
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
            del m,policy
    finally:pool.close()
    for path,digest in json.loads((OUT/'protocol.json').read_text())['sources'].items():assert base.digest(Path(path))==digest
    save('status.json',dict(state='completed',sources_unchanged=True,matched_initial_parameters=True,matched_training_presentations=True,new_test_accepted=len(test)))
if __name__=='__main__':run()
