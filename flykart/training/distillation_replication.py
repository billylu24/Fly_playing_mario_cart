"""Fixed-protocol replication across minibatch-order seeds and common new starts."""
import json,shutil,time
from pathlib import Path
import numpy as np
import torch
from . import supervised as base
from . import layer_probes as lp
from . import action_distillation as pilot
OUT=pilot.OUT.parent/'distillation_replication_v1'
KINDS=pilot.KINDS

def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def prepare():
    OUT.mkdir(exist_ok=False)
    files=[Path(__file__),Path(pilot.__file__),Path(pilot.prior.__file__),pilot.SOURCE/'positive_best.pt',pilot.SOURCE/'stats.pt',lp.OUT/'k2_cnn_mlp32.pt']
    for kind in KINDS:files += [pilot.OUT/(kind+'_best.pt'),pilot.OUT/(kind+'_results.json')]
    save('protocol.json',dict(seeds=[0,1,2],seed_semantics='minibatch schedule RNG12000+seed; same frozen teacher/CNN and identical warm-start graph checkpoint. This is optimizer-order robustness, not independently pretrained backbones.',kinds=KINDS,reuse='seed0 selected checkpoints from action_distillation_v1; new seeds1/2 each objective 256 updates',training='same lr.0003 Adam eps1e-5 clip1 batch8 BPTT32 burn-in64; reconstruction MSE or MSE+KL coefficient1 temperature1; no modifications',selection='same minimum normal validation teacher KL at0/64/128/192/256, ties earlier. All six selected graph models evaluated; no test-driven selection.',test='prefix455/505/565/625 x perturb14/38 x center/left/right =24 prespecified candidates; oracle accepted independent of policies',success_rule='For each objective: all three seeds greedy normal >=.60, normal-frozen >=.20, normal-clock >=.10; and each seed sampled normal strictly exceeds both controls. Report full counts even if gate fails. Shared test starts and teacher mean aggregate episodes not independent evidence.',evaluate='greedy and sampling8100/8101/8102 x normal/frozen/clock; same new starts for all models including seed0 and one common direct-CNN control',sources={str(p):base.digest(p) for p in files}))
    for seed in [0,1,2]:(OUT/f'seed{seed}').mkdir()
    for kind in KINDS:
        for suffix in ['_best.pt','_results.json']:shutil.copyfile(pilot.OUT/(kind+suffix),OUT/'seed0'/(kind+suffix))
    collect()

def collect():
    out=OUT/'new_test';out.mkdir();manifest=[]
    for frame in [455,505,565,625]:
        for perturb in [14,38]:
            for turn in ['center','left','right']:
                cfg=base.make_config(frame,turn,perturb);task=base.BendTask(**cfg)
                try:
                    obs=task.reset();np.testing.assert_array_equal(obs,task.reset());xs=[];ys=[];waypoint=0
                    while True:
                        action,waypoint=base.oracle(task,waypoint);xs.append(obs);ys.append(action);msg=task.step(action);obs=msg['frame']
                        if msg['done']:break
                    file=cfg['start_id']+'.npz';np.savez_compressed(out/file,observations=np.stack(xs),actions=np.array(ys,np.int64));manifest.append(dict(config=cfg,filename=file,sha256=base.digest(out/file),accepted=bool(msg['episode']['target_reached'] and msg['episode']['jumps']==0),episode=msg['episode']));save('new_test_manifest.json',manifest);print('COLLECT',cfg['start_id'],manifest[-1]['accepted'],flush=True)
                finally:task.close()

def train(seed,stats,rows):
    run=OUT/f'seed{seed}';chunks=[(i,t) for i,r in enumerate(rows['train']) for t in range(0,len(r['obs']),32)];rng=np.random.default_rng(12000+seed);schedule=[[chunks[j] for j in rng.integers(len(chunks),size=8)] for _ in range(256)];save(f'seed{seed}/schedule.json',schedule);matched=[];hashes=[]
    for kind in KINDS:
        m=pilot.model(stats);initial={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad};ih=base.tensor_hash(initial);hashes.append(ih);assert ih==json.loads((pilot.OUT/(kind+'_results.json')).read_text())['initial_trainable_hash'];cnn=base.tensor_hash(m.b.encoder.state_dict());head=base.tensor_hash(m.head.state_dict());opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.0003,eps=1e-5);best=None;history=[];presentations=0;begin=time.perf_counter()
        for step in range(257):
            if step%64==0:
                v=pilot.evaluate(m,rows['validation']);history.append(dict(step=step,validation=v,valid_presentations=presentations));key=v['teacher_kl']
                if best is None or key<best[0]:best=(key,step);torch.save(m.state_dict(),run/(kind+'_best.pt'))
                save(f'seed{seed}/{kind}_metrics.json',history);print('VALIDATION',seed,kind,step,'KL',round(key,4),'BA',round(v['action_scores']['balanced_accuracy'],4),flush=True)
            if step==256:break
            opt.zero_grad(set_to_none=True);mse,kl,n=pilot.batch(m,rows['train'],schedule[step]);loss=mse+(kl if kind=='joint' else 0);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();presentations+=n
            if (step+1)%32==0:print('TRAIN',seed,kind,step+1,float(mse.detach()),float(kl.detach()),flush=True)
        torch.save(m.state_dict(),run/(kind+'_last.pt'));m.load_state_dict(torch.load(run/(kind+'_best.pt'),weights_only=True));r=dict(seed=seed,selected_step=best[1],validation={mode:pilot.evaluate(m,rows['validation'],mode) for mode in ['normal','frozen']},initial_trainable_hash=ih,valid_presentations=presentations,wall_seconds=time.perf_counter()-begin)
        assert cnn==base.tensor_hash(m.b.encoder.state_dict()) and head==base.tensor_hash(m.head.state_dict());assert all(p.grad is None for p in m.b.encoder.parameters()) and all(p.grad is None for p in m.head.parameters());r['cnn_and_teacher_unchanged']=True;save(f'seed{seed}/{kind}_results.json',r);matched.append(presentations);del m,opt
    assert matched[0]==matched[1] and hashes[0]==hashes[1]

def evaluate_all(stats):
    selections={f'{kind}_seed{seed}':json.loads((OUT/f'seed{seed}'/(kind+'_results.json')).read_text())['selected_step'] for seed in [0,1,2] for kind in KINDS};save('selection.json',dict(checkpoints=selections,all_selected_before_new_test=True))
    manifest=json.loads((OUT/'new_test_manifest.json').read_text());test=[]
    for r in manifest:
        if r['accepted']:
            with np.load(OUT/'new_test'/r['filename']) as d:test.append(dict(meta=r,obs=torch.tensor(d['observations'],device='cuda'),labels=torch.tensor(d['actions'],device='cuda')))
    timehead=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');timehead.load_state_dict(torch.load(lp.OUT/'time_probe.pt',weights_only=True))
    with torch.no_grad():clock=list(timehead(torch.arange(225,device='cuda')[:,None]/225).softmax(-1).cpu())
    pool=base.EvaluationPool()
    try:
        for seed,kind in [(s,k) for s in [0,1,2] for k in KINDS]+[(0,'cnn')]:
            m=pilot.model(stats);name='cnn' if kind=='cnn' else f'{kind}_seed{seed}'
            if kind=='cnn':f=lp.Features(m.b);policy=lp.Policy(f,'cnn',m.head);offline=None
            else:
                path=OUT/f'seed{seed}'/(kind+'_best.pt');ckhash=base.digest(path);m.load_state_dict(torch.load(path,weights_only=True));encoded=pilot.prior.encode(m,test);offline={mode:pilot.evaluate(m,encoded,mode) for mode in ['normal','frozen']};m.action=True;policy=m
            report=dict(kind=kind,seed=seed,offline=offline,episodes=[])
            for at,r in enumerate(test):
                es=pool.rollout(policy,r['meta']['config'],clock)
                if at==0:
                    ref=base.closed_loop(policy,r['meta']['config'],'normal',8100,clock);actual=next(e for e in es if e['mode']=='normal' and e['sample_seed']==8100)
                    for key in ['actions','reason','frames','progress','jumps']:assert ref[key]==actual[key],key
                    report['batched_sequential_match']=True
                report['episodes'].extend(es);save(name+'_evaluation.json',report);print('EVALUATED',name,at+1,'/',len(test),flush=True)
            if kind=='cnn':f.hook.remove()
            else:assert base.digest(path)==ckhash
            del m,policy
    finally:pool.close()
    return len(test)

def run():
    torch.set_num_threads(4);prepare();stats={k:v.cuda() for k,v in torch.load(pilot.SOURCE/'stats.pt',weights_only=True).items()};m=pilot.model(stats);rows={s:pilot.prior.encode(m,base.dataset(s)) for s in ['train','validation']};del m
    for seed in [1,2]:train(seed,stats,rows)
    n=evaluate_all(stats)
    for path,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():assert base.digest(Path(path))==h
    save('status.json',dict(state='completed',sources_unchanged=True,new_test_accepted=n,training_seeds=[0,1,2],seed_scope='minibatch order only, common warm start',all_models_evaluated=True))
if __name__=='__main__':run()
