"""Loss-matched recovery imitation, three paired warm starts, new-start controls."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import supervised as base
from . import action_distillation as pilot
from . import layer_probes as lp

OUT=pilot.OUT.parent/'recovery_training_v1'
DATA=pilot.OUT.parent/'recovery_demonstrations_v1'
SOURCE=pilot.OUT.parent/'distillation_replication_v1'
KINDS=['distill','old_ce','recovery_ce']
UPDATES=128


def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')


def model(stats,seed):
    m=pilot.model(stats)
    m.load_state_dict(torch.load(SOURCE/f'seed{seed}/joint_best.pt',weights_only=True))
    return m


def prepare():
    assert json.loads((DATA/'quality_gate.json').read_text())['passed']
    OUT.mkdir(exist_ok=False)
    for s in range(3):(OUT/f'seed{s}').mkdir()
    paths=[Path(__file__),Path(base.__file__),Path(pilot.__file__),Path(pilot.prior.__file__),DATA/'training_manifest.json',pilot.SOURCE/'stats.pt',lp.OUT/'k2_cnn_mlp32.pt']
    paths += [SOURCE/f'seed{s}/joint_best.pt' for s in range(3)]
    save('protocol.json',dict(kinds=KINDS,seeds=[0,1,2],updates=UPDATES,
        warm_start='Within each seed, all three arms start at identical archived joint_best; the three starts differ by prior minibatch order. Same CNN/teacher; no independent pretraining claim.',
        optimizer='Reset Adam lr .0003 eps1e-5 clip1, 128 updates, batch8/BPTT32/burn64, gain/interface/decoder only.',
        objective='Average of two equally weighted groups of 4 sequences. Group1 original data MSE+teacher KL for all arms. Group2: A distill original MSE+KL; B old_ce original MSE+action CE; C recovery_ce successful recovery MSE+action CE. All coefficients1, temperature1, unweighted CE. CNN/head frozen.',
        matching='Group1 choices shared; A/B share group2 original full32 chunks. C uses recovery chunks. Original group2 loss mask is copied from paired recovery chunk, including ignored failed prefix and tail, so valid supervision counts match exactly. MSE and action losses use that same mask. Dynamics still process full actual observed history.',
        sampling='Uniform chunks at t=0,32,... with >=1 supervised action; original group2 candidates require32 valid frames to cover any paired recovery mask. Original group1 uses all original chunks. Schedule fixed RNG13000+seed, original and recovery choices stored before training.',
        recovery='All74 deduplicated successful recoveries from training-only starts; prefix actions -100. Pool collected from three archived seeds; shared across arms/seeds.',
        selection='Minimum original validation normal hard-oracle action cross entropy at0/32/64/96/128; ties earlier. Same rule all arms, no new-test selection.',
        new_test='prefix445/515/575/635 x perturb10/26 x center/left/right; oracle success+jumps0 accepted before training. Same-track new starts, not independent-track generalization.',
        evaluation='All9 selected models +3 unchanged warm starts +common CNN, greedy and sampled8100/8101/8102, normal/frozen/clock. Every first-start sampled normal rollout checked against sequential implementation.',
        success_gate='Recovery-data effect requires each seed C greedy normal>=B, aggregate C-B >=10 percentage points, each C normal-frozen>=20pp and normal-clock>=10pp, aggregate C sampled normal>B. Also report versus A and unchanged starts. Descriptive gate; shared starts are not independent samples.',
        sources={str(p):base.digest(p) for p in paths}))
    collect()


def collect():
    out=OUT/'new_test';out.mkdir();manifest=[]
    for frame in [445,515,575,635]:
        for perturb in [10,26]:
            for turn in ['center','left','right']:
                cfg=base.make_config(frame,turn,perturb);task=base.BendTask(**cfg)
                try:
                    obs=task.reset();np.testing.assert_array_equal(obs,task.reset());xs=[];ys=[];waypoint=0
                    while True:
                        a,waypoint=base.oracle(task,waypoint);xs.append(obs);ys.append(a);msg=task.step(a);obs=msg['frame']
                        if msg['done']:break
                    file=cfg['start_id']+'.npz';np.savez_compressed(out/file,observations=np.stack(xs),actions=np.asarray(ys,np.int64))
                    manifest.append(dict(config=cfg,filename=file,sha256=base.digest(out/file),accepted=bool(msg['episode']['target_reached'] and msg['episode']['jumps']==0),episode=msg['episode']))
                    save('new_test_manifest.json',manifest);print('COLLECT',cfg['start_id'],manifest[-1]['accepted'],flush=True)
                finally:task.close()


def recovery_rows():
    rows=[]
    for r in json.loads((DATA/'training_manifest.json').read_text()):
        p=DATA/'data'/r['filename'];assert base.digest(p)==r['sha256']
        with np.load(p) as d:rows.append(dict(meta=r,obs=torch.tensor(d['observations'],device='cuda'),labels=torch.tensor(d['actions'],device='cuda')))
    return rows


def schedule(rows,seed):
    old=[(i,t) for i,r in enumerate(rows['train']) for t in range(0,len(r['labels']),32)]
    full=[(i,t) for i,t in old if len(rows['train'][i]['labels'])-t>=32]
    rec=[(i,t) for i,r in enumerate(rows['recovery']) for t in range(0,len(r['labels']),32) if (r['labels'][t:t+32]>=0).any()]
    rng=np.random.default_rng(13000+seed)
    return [dict(common=[old[j] for j in rng.integers(len(old),size=4)],original=[full[j] for j in rng.integers(len(full),size=4)],recovery=[rec[j] for j in rng.integers(len(rec),size=4)]) for _ in range(UPDATES)]


def batch(m,rows,choice,kind,details=False):
    auxiliary=rows['recovery'] if kind=='recovery_ce' else rows['train']
    combined=rows['train']+auxiliary
    chosen=choice['common']+[(i+len(rows['train']),t) for i,t in choice['recovery' if kind=='recovery_ce' else 'original']]
    recon,labels=base.batch_sequence(combined,chosen,m)
    steps=recon.shape[0]//8
    z=torch.stack([torch.stack([combined[i]['obs'][min(t+s,len(combined[i]['obs'])-1)] for i,t in chosen]) for s in range(steps)])
    recon=recon.reshape(steps,8,64);labels=labels.reshape(steps,8)
    masks=[]
    for s in range(steps):
        masks.append(torch.tensor([t+s<len(rows['recovery'][i]['labels']) and int(rows['recovery'][i]['labels'][min(t+s,len(rows['recovery'][i]['labels'])-1)])>=0 for i,t in choice['recovery']],device='cuda'))
    shared=torch.stack(masks);mask0=labels[:,:4]>=0;mask1=(labels[:,4:]>=0)&shared
    if kind=='recovery_ce':assert torch.equal(mask1,labels[:,4:]>=0)
    n0=int(mask0.sum());n1=int(mask1.sum());assert n0>0 and n1>0
    losses=[]
    for lo,hi,mask,action_kind in [(0,4,mask0,'kl'),(4,8,mask1,'kl' if kind=='distill' else 'ce')]:
        rr=recon[:,lo:hi][mask];zz=z[:,lo:hi][mask]
        sl=m.head(rr*m.zscale+m.zmean)
        mse=(rr-(zz-m.zmean)/m.zscale).square().mean()
        action=F.kl_div(sl.log_softmax(-1),m.head(zz).softmax(-1),reduction='batchmean') if action_kind=='kl' else F.cross_entropy(sl,labels[:,lo:hi][mask])
        losses.append(mse+action)
    loss=.5*(losses[0]+losses[1])
    if details:return loss,(n0,n1),recon,labels,chosen
    return loss,(n0,n1)


def checks(stats,rows):
    m=model(stats,0);choice=schedule(rows,0)[0];counts=[]
    for kind in KINDS:
        loss,n,p,y,chosen=batch(m,rows,choice,kind,True);counts.append(n)
        assert torch.isfinite(loss)
        if kind=='recovery_ce':
            combined=rows['train']+rows['recovery']
            # Direct Context vs padded BPTT output, including masked recovery prefixes.
            with torch.no_grad():
                for col,(i,t) in enumerate(chosen):
                    ctx=base.Context(m);reference=[]
                    for j,z in enumerate(combined[i]['obs'][:t+32]):
                        pred=ctx.step(z)
                        if j>=t:reference.append(pred)
                    if reference:torch.testing.assert_close(p[:len(reference),col],torch.stack(reference),rtol=2e-4,atol=3e-6)
            grads=torch.autograd.grad(loss,[m.interface,m.b.a,m.decoder.weight])
            assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
    assert len(set(counts))==1
    assert all(not p.requires_grad for p in m.head.parameters()) and all(not p.requires_grad for p in m.b.encoder.parameters())
    save('checks.json',dict(matched_valid_counts=counts,context_matches_bptt=True,finite_nonzero_gradients=True,cnn_and_head_frozen=True))


def train(stats,rows):
    for seed in range(3):
        plan=schedule(rows,seed);save(f'seed{seed}/schedule.json',plan);hashes=[];counts=[]
        for kind in KINDS:
            m=model(stats,seed);initial=base.tensor_hash({n:p for n,p in m.named_parameters() if p.requires_grad});hashes.append(initial)
            fixed=(base.tensor_hash(m.b.encoder.state_dict()),base.tensor_hash(m.head.state_dict()))
            opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.0003,eps=1e-5);history=[];best=None;presentations=[0,0];begin=time.perf_counter()
            for step in range(UPDATES+1):
                if step%32==0:
                    val=pilot.evaluate(m,rows['validation']);key=val['action_scores']['cross_entropy'];history.append(dict(step=step,validation=val,presentations=list(presentations)))
                    if best is None or key<best[0]:
                        best=(key,step);torch.save(m.state_dict(),OUT/f'seed{seed}/{kind}_best.pt')
                    save(f'seed{seed}/{kind}_metrics.json',history);print('VALIDATION',seed,kind,step,'CE',round(key,4),'BA',round(val['action_scores']['balanced_accuracy'],4),flush=True)
                if step==UPDATES:break
                opt.zero_grad(set_to_none=True);loss,nn=batch(m,rows,plan[step],kind);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                presentations=[a+b for a,b in zip(presentations,nn)]
            assert fixed==(base.tensor_hash(m.b.encoder.state_dict()),base.tensor_hash(m.head.state_dict()))
            assert all(p.grad is None for p in m.head.parameters()) and all(p.grad is None for p in m.b.encoder.parameters())
            counts.append(presentations);torch.save(m.state_dict(),OUT/f'seed{seed}/{kind}_last.pt')
            save(f'seed{seed}/{kind}_results.json',dict(selected_step=best[1],initial_trainable_hash=initial,presentations=presentations,wall_seconds=time.perf_counter()-begin,frozen_parameters_unchanged=True))
            del m,opt
        assert len(set(hashes))==1 and all(c==counts[0] for c in counts)


def evaluate(stats):
    selections={f'{k}_seed{s}':json.loads((OUT/f'seed{s}/{k}_results.json').read_text())['selected_step'] for s in range(3) for k in KINDS}
    save('selection.json',dict(models=selections,all_selected_before_new_test_evaluation=True))
    test=[r for r in json.loads((OUT/'new_test_manifest.json').read_text()) if r['accepted']]
    th=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');th.load_state_dict(torch.load(lp.OUT/'time_probe.pt',weights_only=True))
    with torch.no_grad():clock=list(th(torch.arange(225,device='cuda')[:,None]/225).softmax(-1).cpu())
    pool=base.EvaluationPool()
    try:
        for seed,kind in [(s,k) for s in range(3) for k in KINDS+['warm']]+[(0,'cnn')]:
            m=model(stats,seed);name='cnn' if kind=='cnn' else f'{kind}_seed{seed}'
            if kind=='cnn':f=lp.Features(m.b);policy=lp.Policy(f,'cnn',m.head)
            else:
                if kind!='warm':m.load_state_dict(torch.load(OUT/f'seed{seed}/{kind}_best.pt',weights_only=True))
                m.action=True;policy=m
            report=dict(seed=seed,kind=kind,episodes=[])
            for i,r in enumerate(test):
                es=pool.rollout(policy,r['config'],clock)
                if i==0:
                    ref=base.closed_loop(policy,r['config'],'normal',8100,clock);actual=next(e for e in es if e['mode']=='normal' and e['sample_seed']==8100)
                    for key in ['actions','reason','frames','progress','jumps']:assert ref[key]==actual[key],(name,key)
                    report['batched_sequential_match']=True
                report['episodes'].extend(es);save(name+'_evaluation.json',report);print('EVALUATED',name,i+1,'/',len(test),flush=True)
            if kind=='cnn':f.hook.remove()
            del m,policy
    finally:pool.close()
    return len(test)


def run():
    torch.set_num_threads(4);prepare()
    stats={k:v.cuda() for k,v in torch.load(pilot.SOURCE/'stats.pt',weights_only=True).items()}
    m=model(stats,0);rows={s:pilot.prior.encode(m,base.dataset(s)) for s in ['train','validation']};rows['recovery']=pilot.prior.encode(m,recovery_rows());del m
    checks(stats,rows);train(stats,rows);n=evaluate(stats)
    for p,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():assert base.digest(Path(p))==h
    save('status.json',dict(state='completed',sources_unchanged=True,new_test_accepted=n,trained_models=9,evaluated_policies=13,episodes=13*n*12,sequential_regressions=13))


if __name__=='__main__':run()
