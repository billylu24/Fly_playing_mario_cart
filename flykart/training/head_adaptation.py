"""Frozen-feature action-head adaptation with separate recovery validation.
Only the 64->32->3 action MLP changes; all input normalization stays fixed.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import supervised as base
from . import action_distillation as pilot
from . import recovery_demonstrations as demos
from . import layer_probes as lp

OUT=pilot.OUT.parent/'head_adaptation_v1'
DATA=demos.OUT
SOURCE=demos.SOURCE
STEPS=512


def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')


def model(stats,seed):
    m=pilot.model(stats)
    m.load_state_dict(torch.load(SOURCE/f'seed{seed}/joint_best.pt',weights_only=True))
    m.requires_grad_(False);m.eval()
    return m


class CNNPolicy(nn.Module):
    def __init__(self,encoder,head):
        super().__init__();self.encoder=encoder;self.head=head;self.n=1
    def forward(self,obs,h,starts):
        return self.head(self.encoder(obs[:,None].float()/255)),torch.zeros(len(obs),device=obs.device),h*0


def prepare():
    OUT.mkdir(exist_ok=False);(OUT/'validation_data').mkdir();(OUT/'new_test').mkdir()
    paths=[Path(__file__),Path(base.__file__),Path(demos.__file__),DATA/'training_manifest.json',pilot.SOURCE/'stats.pt',lp.OUT/'k2_cnn_mlp32.pt']
    paths += [SOURCE/f'seed{s}/joint_best.pt' for s in range(3)]
    save('protocol.json',dict(graph_seeds=[0,1,2],kinds=['frozen','old','mixed'],updates=STEPS,
        hypothesis='Can the same trainable action head learn recovery labels from frozen graph reconstruction versus frozen direct CNN? No gradient through CNN, graph, gain, interface, decoder, or normalization.',
        training='Existing frozen head weights initialize every arm. Adam lr.001 eps1e-5 clip1, 512 updates, batch128 observations. Old: two groups of64, each loss weight0.5; both groups old in old arm, second group recovery in mixed arm. Unweighted CE only, no MSE or teacher KL. Schedule same across feature sources and arms, RNG15000.',
        features='Graph features are reconstructed64 in original CNN units, computed using actual full observation history and deployment Context; labels -100 discarded only after history replay. Direct CNN64 shares same frames and labels. Frozen normalization buffers of original head preserved.',
        recovery_validation='Original six accepted validation starts only (prefix510/570), all three archived graph policies. At decisions8/32, compare waypoint and CNN; on baseline failures choose successful waypoint else successful CNN, deduplicate full observed sequences plus labels. No validation data in gradients. If no successful failure-recovery examples, stop without training.',
        selection='Every32 updates including0. Minimize 0.5*old-validation CE +0.5*recovery-validation CE among checkpoints with old-validation CE <=1.05*its own step0 CE; ties earlier. Step0 always eligible. Same selection for old and mixed arms.',
        controls='Old-data-only trainable head and unchanged frozen head for each of3 graph feature sources; same old/mixed/frozen arms for one common direct CNN. Graph seeds reflect earlier minibatch schedules, not independent pretrained backbones or head-training schedules.',
        new_test='prefix460/520/580/640 x perturb28 x center/left/right,12 candidates; accept oracle target+jumps0 before training. All12 policies x accepted starts x greedy/sampling8100/8101/8102 x normal/frozen/clock.',
        gates='First: mixed-head recovery-validation CE improves >=10% vs own frozen head while satisfying old-validation CE retention. Closed loop graph mixed must not lose greedy success vs old in any seed and aggregate gain>=10pp; each mixed normal-frozen>=20pp, normal-clock>=10pp; aggregate sampled normal>old. Descriptive, correlated starts. Always report complete CNN controls.',
        sources={str(p):base.digest(p) for p in paths}))


def collect_validation(stats):
    manifest=json.loads((base.EXPERIMENT/'manifest.json').read_text())
    rows=[r for r in manifest if r['split']=='validation' and r['accepted']]
    baselines=[];branches=[];chosen=[];seen=set();skipped=[]
    for seed in range(3):
        m=model(stats,seed);m.action=True
        for row in rows:
            cfg=row['config'];start=row['start_id'];original,_=demos.baseline(m,cfg)
            original.update(seed=seed,config=cfg);baselines.append(original)
            for takeover in [8,32]:
                if len(original['actions'])<=takeover:
                    skipped.append(dict(seed=seed,start_id=start,takeover=takeover));continue
                candidates=[]
                for teacher in ['waypoint','cnn']:
                    r,xs,ys=demos.branch(m,cfg,original,takeover,teacher)
                    r.update(seed=seed,config=cfg,baseline_success=bool(original['target_reached']));branches.append(r)
                    if r['accepted'] and not original['target_reached']:candidates.append((r,xs,ys))
                if candidates:
                    r,xs,ys=candidates[0]
                    import hashlib
                    x=np.stack(xs);y=np.asarray(ys,np.int64);fingerprint=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
                    if fingerprint not in seen:
                        seen.add(fingerprint);file=f'seed{seed}_{start}_t{takeover}_{r["teacher"]}.npz'
                        np.savez_compressed(OUT/'validation_data'/file,observations=x,actions=y)
                        chosen.append(dict(r,filename=file,sha256=base.digest(OUT/'validation_data'/file)))
            save('validation_baselines.json',baselines);save('validation_branches.json',branches);save('validation_manifest.json',chosen);save('validation_skipped.json',skipped)
            print('VALIDATION_COLLECT',seed,start,'baseline',original['target_reached'],'chosen',len(chosen),flush=True)
        del m
    save('validation_status.json',dict(baseline_episodes=len(baselines),branch_episodes=len(branches),selected=len(chosen),distinct_starts=len({r['start_id'] for r in chosen})))
    if not chosen:raise RuntimeError('No recovery validation examples; do not train or substitute test data.')


def collect_test():
    manifest=[]
    for frame in [460,520,580,640]:
        for turn in ['center','left','right']:
            cfg=base.make_config(frame,turn,28);task=base.BendTask(**cfg)
            try:
                obs=task.reset();np.testing.assert_array_equal(obs,task.reset());xs=[];ys=[];waypoint=0
                while True:
                    a,waypoint=base.oracle(task,waypoint);xs.append(obs);ys.append(a);msg=task.step(a);obs=msg['frame']
                    if msg['done']:break
                file=cfg['start_id']+'.npz';np.savez_compressed(OUT/'new_test'/file,observations=np.stack(xs),actions=np.asarray(ys,np.int64))
                manifest.append(dict(config=cfg,filename=file,accepted=bool(msg['episode']['target_reached'] and msg['episode']['jumps']==0),episode=msg['episode'],sha256=base.digest(OUT/'new_test'/file)))
                save('new_test_manifest.json',manifest);print('TEST_COLLECT',cfg['start_id'],manifest[-1]['accepted'],flush=True)
            finally:task.close()


def read_rows(folder,manifest):
    rows=[]
    for r in json.loads(manifest.read_text()):
        p=folder/r['filename'];assert base.digest(p)==r['sha256']
        with np.load(p) as d:rows.append(dict(meta=r,obs=torch.tensor(d['observations'],device='cuda'),labels=torch.tensor(d['actions'],device='cuda')))
    return rows


@torch.no_grad()
def features(m,rows,graph):
    xx=[];yy=[]
    for r in rows:
        if graph:
            ctx=base.Context(m)
            x=torch.stack([ctx.step(o) for o in r['obs']])*m.zscale+m.zmean
        else:x=torch.cat([m.b.encoder(o[None,None].float()/255) for o in r['obs']])
        mask=r['labels']>=0;xx.append(x[mask]);yy.append(r['labels'][mask])
    return torch.cat(xx),torch.cat(yy)


@torch.no_grad()
def score(head,data):
    x,y=data
    return base.scores(y.cpu().numpy(),head(x).cpu().numpy())


def train_head(m,data,source,schedule):
    initial=base.tensor_hash(m.head.state_dict());results={}
    baseline={s:score(m.head,d) for s,d in data.items()}
    save(source+'_frozen_metrics.json',baseline)
    for kind in ['old','mixed']:
        head=copy.deepcopy(m.head);head.net.requires_grad_(True)
        opt=torch.optim.Adam(head.net.parameters(),lr=.001,eps=1e-5);best=None;history=[]
        assert base.tensor_hash(head.state_dict())==initial
        for step in range(STEPS+1):
            if step%32==0:
                v={s:score(head,data[s]) for s in ['validation','recovery_validation']}
                eligible=v['validation']['cross_entropy']<=1.05*baseline['validation']['cross_entropy']
                key=.5*(v['validation']['cross_entropy']+v['recovery_validation']['cross_entropy'])
                history.append(dict(step=step,metrics=v,eligible=eligible,selection_ce=key))
                if eligible and (best is None or key<best[0]):
                    best=(key,step);torch.save(head.state_dict(),OUT/f'{source}_{kind}_head.pt')
                print('HEAD_TRAIN',source,kind,step,'oldCE',round(v['validation']['cross_entropy'],3),'recoveryCE',round(v['recovery_validation']['cross_entropy'],3),eligible,flush=True)
            if step==STEPS:break
            plan=schedule[step];a=data['train'];b=data['train' if kind=='old' else 'recovery_train']
            ia=torch.tensor(plan['common'],device='cuda');ib=torch.tensor(plan['old' if kind=='old' else 'recovery'],device='cuda')
            opt.zero_grad(set_to_none=True);loss=.5*F.cross_entropy(head(a[0][ia]),a[1][ia])+.5*F.cross_entropy(head(b[0][ib]),b[1][ib]);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
        torch.save(head.state_dict(),OUT/f'{source}_{kind}_last.pt')
        head.load_state_dict(torch.load(OUT/f'{source}_{kind}_head.pt',weights_only=True))
        assert torch.equal(head.mean,m.head.mean) and torch.equal(head.scale,m.head.scale)
        results[kind]=dict(selected_step=best[1],history=history,initial_head_hash=initial,selected={s:score(head,d) for s,d in data.items()})
        save(source+'_'+kind+'_metrics.json',results[kind])
    assert base.tensor_hash(m.head.state_dict())==initial
    return results


def train(stats):
    rows={'train':base.dataset('train'),'validation':base.dataset('validation'),
          'recovery_train':read_rows(DATA/'data',DATA/'training_manifest.json'),
          'recovery_validation':read_rows(OUT/'validation_data',OUT/'validation_manifest.json')}
    train_ids={r['meta'].get('start_id',r['meta'].get('config',{}).get('start_id')) for split in ['train','recovery_train'] for r in rows[split]}
    val_ids={r['meta'].get('start_id',r['meta'].get('config',{}).get('start_id')) for split in ['validation','recovery_validation'] for r in rows[split]}
    assert train_ids.isdisjoint(val_ids)
    plan=None;selections={}
    for source in ['cnn','graph0','graph1','graph2']:
        seed=0 if source=='cnn' else int(source[-1]);m=model(stats,seed);before=base.tensor_hash(m.state_dict())
        data={s:features(m,r,source!='cnn') for s,r in rows.items()}
        if plan is None:
            rng=np.random.default_rng(15000);n=len(data['train'][1]);nr=len(data['recovery_train'][1])
            plan=[dict(common=rng.integers(n,size=64).tolist(),old=rng.integers(n,size=64).tolist(),recovery=rng.integers(nr,size=64).tolist()) for _ in range(STEPS)]
            save('schedule.json',plan)
        # Cache/deployment equality: first original trajectory's logits agree.
        with torch.no_grad():
            policy=CNNPolicy(m.b.encoder,m.head) if source=='cnn' else m
            if source!='cnn':m.action=True
            ctx=base.Context(policy);ref=torch.stack([ctx.step(o) for o in rows['train'][0]['obs']])
            torch.testing.assert_close(ref,m.head(data['train'][0][:len(ref)]),rtol=1e-4,atol=3e-6)
            m.action=False
        result=train_head(m,data,source,plan)
        assert before==base.tensor_hash(m.state_dict());assert all(p.grad is None for p in m.parameters())
        for kind,r in result.items():selections[f'{source}_{kind}']=r['selected_step']
        save('selection.json',dict(models=selections,all_selected_before_new_test_evaluation=len(selections)==8))
        print('FEATURE_SOURCE_DONE',source,flush=True);del m,data
    save('checks.json',dict(frozen_features_and_original_heads_unchanged=True,cache_deployment_logits_match=True,train_validation_start_ids_disjoint=True,shared_head_initialization_and_sampling=True))


def evaluate(stats):
    test=[r for r in json.loads((OUT/'new_test_manifest.json').read_text()) if r['accepted']]
    th=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');th.load_state_dict(torch.load(lp.OUT/'time_probe.pt',weights_only=True))
    with torch.no_grad():clock=list(th(torch.arange(225,device='cuda')[:,None]/225).softmax(-1).cpu())
    pool=base.EvaluationPool()
    try:
        for source in ['cnn','graph0','graph1','graph2']:
            for kind in ['frozen','old','mixed']:
                seed=0 if source=='cnn' else int(source[-1]);m=model(stats,seed)
                if kind!='frozen':m.head.load_state_dict(torch.load(OUT/f'{source}_{kind}_head.pt',weights_only=True))
                if source=='cnn':policy=CNNPolicy(m.b.encoder,m.head)
                else:m.action=True;policy=m
                report=dict(source=source,kind=kind,episodes=[]);name=source+'_'+kind
                for i,r in enumerate(test):
                    es=pool.rollout(policy,r['config'],clock)
                    if i==0:
                        ref=base.closed_loop(policy,r['config'],'normal',8100,clock);actual=next(e for e in es if e['mode']=='normal' and e['sample_seed']==8100)
                        for key in ['actions','reason','frames','progress','jumps']:assert ref[key]==actual[key],(name,key)
                        report['batched_sequential_match']=True
                    report['episodes'].extend(es);save(name+'_evaluation.json',report);print('EVALUATED',name,i+1,'/',len(test),flush=True)
                del m,policy
    finally:pool.close()
    return len(test)


def run():
    import sys
    torch.set_num_threads(4)
    resume='--resume-after-cache-fix' in sys.argv
    if not resume:prepare()
    else:
        assert not list(OUT.glob('*_head.pt')) and not (OUT/'selection.json').exists()
        for p,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():assert base.digest(Path(p))==h
    stats={k:v.cuda() for k,v in torch.load(pilot.SOURCE/'stats.pt',weights_only=True).items()}
    if not resume:collect_validation(stats);collect_test()
    train(stats);n=evaluate(stats)
    for p,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():assert base.digest(Path(p))==h
    save('status.json',dict(state='completed',sources_unchanged=True,trained_heads=8,evaluated_policies=12,new_test_accepted=n,episodes=12*n*12,sequential_regressions=12))

if __name__=='__main__':run()
